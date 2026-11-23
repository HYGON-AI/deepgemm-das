# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Tune gfx938 contiguous pair32_v2 persistent FP8 kernels."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PREFIX = "DEEPGEMM_W8A8_FP8_CONTIGUOUS"
LAYOUT = Path("w8a8_contiguous/hipc_pair32_v2")
TILES = {1000: (256, 256), 1001: (128, 256), 1002: (128, 384)}
FIELDS = ["arch", "num_cus", "E", "N", "K", "M", "mode", "repeat",
          "cost_us", "median_cost_us", "status", "best"]


def supported(mode, m, n, k):
    bm, bn = TILES[mode]
    return m > 0 and n > 0 and k > 0 and m % bm == n % bn == k % 128 == 0


def load_config(path):
    path = Path(path).resolve()
    defaults = dict(gpu=0, modes=[1000, 1001, 1002], warmup=8, iters=50,
                    replays=20, repeats=5, resume=False, emit_config=True,
                    emit_only=False, aggregate_only=False, arch="gfx938",
                    cases="contiguous_fp8_cases.yml", out="contiguous_fp8.csv",
                    config_root="../../deepgemm/configs")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError("unknown config keys or config is not a mapping")
    cfg = {**defaults, **raw}
    if cfg["arch"] != "gfx938":
        raise ValueError("pair32_v2 contiguous FP8 supports gfx938 only")
    for key in ("cases", "out", "config_root"):
        cfg[key] = (path.parent / cfg[key]).resolve()
    cfg["modes"] = list(dict.fromkeys(int(v) for v in cfg["modes"]))
    if not cfg["modes"] or set(cfg["modes"]) - set(TILES):
        raise ValueError("modes must be a nonempty subset of [1000, 1001, 1002]")
    for key in ("iters", "replays", "repeats"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(cfg["warmup"]) < 0:
        raise ValueError("warmup must be nonnegative")
    return cfg


def write_table(path, configs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"configs": {str(m): {"MODE": mode}
                    for m, mode in sorted(configs.items())}}, indent=4) + "\n")
    print(f"WRITE {path}", flush=True)


def aggregate(directory):
    """Vote across E, then across N/K, separately for each CU count."""
    shape_votes = defaultdict(lambda: defaultdict(list))
    for path in sorted(directory.glob(f"{PREFIX}_E*_N*_K*_CU*.json")):
        # E-level table names always contain the full N/K/CU suffix.
        import re
        match = re.fullmatch(PREFIX + r"_E\d+_N(\d+)_K(\d+)_CU(\d+)\.json", path.name)
        if not match:
            continue
        n, k, cu = map(int, match.groups())
        for m, item in json.loads(path.read_text())["configs"].items():
            mode = int(item["MODE"])
            if mode in TILES and supported(mode, int(m), n, k):
                shape_votes[n, k, cu][int(m)].append(mode)
    default_votes = defaultdict(lambda: defaultdict(list))
    for (n, k, cu), votes in sorted(shape_votes.items()):
        picked = {m: min(Counter(values), key=lambda mode: (-values.count(mode), mode))
                  for m, values in votes.items()}
        write_table(directory / f"{PREFIX}_N{n}_K{k}_CU{cu}.json", picked)
        for m, mode in picked.items():
            group = "BN384" if n % 384 == 0 else "BN256"
            default_votes[cu, group][m].append(mode)
    for (cu, group), votes in sorted(default_votes.items()):
        picked = {m: min(Counter(values), key=lambda mode: (-values.count(mode), mode))
                  for m, values in votes.items()}
        write_table(directory / f"{PREFIX}_DEFAULT_{group}_CU{cu}.json", picked)


def emit(cfg):
    best = {}
    with cfg["out"].open(newline="") as f:
        for row in csv.DictReader(f):
            if row["arch"] != cfg["arch"] or row["status"] != "ok":
                continue
            cu, e, n, k, m, mode = (int(row[key]) for key in
                                     ("num_cus", "E", "N", "K", "M", "mode"))
            cost = float(row["median_cost_us"])
            if cu <= 0 or mode not in cfg["modes"] or not math.isfinite(cost) or cost <= 0:
                continue
            if not supported(mode, m, n, k):
                continue
            key = cu, e, n, k, m
            if key not in best or (cost, mode) < best[key]:
                best[key] = cost, mode
    tables = defaultdict(dict)
    for (cu, e, n, k, m), (_, mode) in best.items():
        tables[cu, e, n, k][m] = mode
    directory = cfg["config_root"] / cfg["arch"] / LAYOUT
    for (cu, e, n, k), table in sorted(tables.items()):
        write_table(directory / f"{PREFIX}_E{e}_N{n}_K{k}_CU{cu}.json", table)
    aggregate(directory)


def tune(cfg):
    import torch
    import deepgemm
    from deepgemm import m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2 as run
    from deepgemm import pack_fp8_weight_pair32_v2 as pack

    torch.cuda.set_device(cfg["gpu"])
    prop = torch.cuda.get_device_properties(cfg["gpu"])
    if prop.gcnArchName.split(":")[0] != cfg["arch"]:
        raise ValueError(f"configured arch differs from selected device: {prop}")
    cu = prop.multi_processor_count
    print(f"deepgemm={deepgemm.__file__}\nextension={deepgemm.op.__file__}\n{prop}", flush=True)
    cases = yaml.safe_load(cfg["cases"].read_text())
    if not isinstance(cases, dict) or set(cases) != {"e_list", "shapes", "m_list"}:
        raise ValueError("cases requires exactly e_list, shapes, m_list")
    done = defaultdict(set)
    append = cfg["resume"] and cfg["out"].exists()
    if append:
        with cfg["out"].open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError("resume CSV schema differs; use a new output file")
            for row in reader:
                if row["arch"] == cfg["arch"] and row["status"] in ("ok", "unsupported"):
                    key = tuple(int(row[key]) for key in ("num_cus", "E", "N", "K", "M"))
                    done[key].add(int(row["mode"]))
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    with cfg["out"].open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not append:
            writer.writeheader()
        for e in map(int, cases["e_list"]):
            if e <= 0:
                raise ValueError("E must be positive")
            for n, k in cases["shapes"]:
                n, k = int(n), int(k)
                if n <= 0 or n % 256 or k <= 0 or k % 128:
                    raise ValueError(f"FP8 packing requires N%256=0 and K%128=0: {(n, k)}")
                torch.manual_seed(123)
                weight = torch.randn(e, n, k, device="cuda").to(torch.float8_e4m3fn)
                packed = pack(weight)
                del weight
                sb = torch.rand(e, n, device="cuda")
                for m in map(int, cases["m_list"]):
                    if m <= 0 or m % 128:
                        raise ValueError("total M must be a positive multiple of 128")
                    key = cu, e, n, k, m
                    if set(cfg["modes"]) <= done[key]:
                        continue
                    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
                    sa = torch.rand(m, device="cuda")
                    output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                    # Divide consecutive M256 blocks into contiguous expert segments.
                    # A final M128 block is legal for the two BM128 modes only.
                    blocks = (m + 255) // 256
                    ids = torch.arange(m, device="cuda") // 256
                    indices = (ids * e // blocks).to(torch.int32).contiguous()
                    graphs = {}
                    samples = {}
                    for mode in cfg["modes"]:
                        if not supported(mode, m, n, k):
                            continue
                        def invoke():
                            run((a, sa), (packed, sb), output, indices, {"MODE": mode})
                        # Runtime/launch failures abort tuning instead of producing a false winner.
                        for _ in range(cfg["warmup"]):
                            invoke()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(cfg["iters"]):
                                invoke()
                        graphs[mode] = graph
                        samples[mode] = []
                    for repeat in range(cfg["repeats"]):
                        order = list(graphs)
                        if repeat % 2:
                            order.reverse()
                        for mode in order:
                            graph = graphs[mode]
                            graph.replay()
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(cfg["replays"]):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            samples[mode].append(start.elapsed_time(end) * 1000 /
                                                 (cfg["iters"] * cfg["replays"]))
                    medians = {mode: statistics.median(values) for mode, values in samples.items()}
                    if not medians or any(not math.isfinite(v) or v <= 0 for v in medians.values()):
                        raise RuntimeError(f"no valid timings for E={e}, M={m}, N={n}, K={k}")
                    winner = min(medians, key=lambda mode: (medians[mode], mode))
                    for mode in cfg["modes"]:
                        values = samples.get(mode, [float("inf")])
                        for repeat, value in enumerate(values):
                            writer.writerow(dict(arch=cfg["arch"], num_cus=cu, E=e, N=n, K=k,
                                M=m, mode=mode, repeat=repeat, cost_us=value,
                                median_cost_us=medians.get(mode, float("inf")),
                                status="ok" if mode in medians else "unsupported", best=int(mode == winner)))
                    f.flush()
                    print(f"E={e} M={m} N={n} K={k}: {medians}, best={winner}", flush=True)
                    graphs.clear()
                    del a, sa, output, indices
                del packed, sb
    if cfg["emit_config"]:
        emit(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "tune_contiguous_fp8_cfg.yml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["aggregate_only"]:
        aggregate(cfg["config_root"] / cfg["arch"] / LAYOUT)
    elif cfg["emit_only"]:
        emit(cfg)
    else:
        tune(cfg)


if __name__ == "__main__":
    main()
