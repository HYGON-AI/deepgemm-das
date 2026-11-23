# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OP_TEST = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(OP_TEST)
if ROOT not in sys.path and os.path.isdir(os.path.join(ROOT, "deepgemm")) and any(
    name.startswith("op.") and name.endswith(".so")
    for name in os.listdir(os.path.join(ROOT, "deepgemm"))
):
    sys.path.insert(0, ROOT)
if OP_TEST not in sys.path:
    sys.path.insert(0, OP_TEST)

E_LIST = []
SHAPES = []
EXPECTED = []
MODES = []
PERSISTENT_MODES = set()
MODE_DEFAULT_MULT = {}

DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "tune_w8a8_int8_cfg.yml")
DEFAULT_CASES_PATH = "masked_cases.yml"
CASES_KEYS = ("e_list", "shapes", "expected")
CONFIG_DEFAULTS = {
    "modes": [1, 2],
    "persistent_modes": [],
    "mode_default_mult": {},
    "cases": DEFAULT_CASES_PATH,
    "gpu": 3,
    "m": 2048,
    "warmup": 8,
    "iters": 40,
    "repeats": 3,
    "shuffle_unique": 0,
    "out": "w8a8_int8_masked_sweep_stable.csv",
    "resume": False,
    "emit_config": False,
    "emit_only": False,
    "aggregate_only": False,
    "arch": None,
    "layout": "",
    "config_root": None,
}
ARCH_CHOICES = ("gfx936", "gfx938", "gfx92a")
# 只有这些架构才区分 CU（文件名带 _CU 后缀、聚合时按 CU 分组），其余架构忽略 CU
CU_DISTINGUISH_ARCHS = ("gfx938",)

W8A8_INT8_MASKED_CONFIG_DIR = "w8a8_masked"
W8A8_INT8_MASKED_CONFIG_PREFIX = "DEEPGEMM_W8A8_INT8_MASKED"
EXPERT_CONFIG_PATTERN = re.compile(
    rf"^{W8A8_INT8_MASKED_CONFIG_PREFIX}_E(?P<e>\d+)_N(?P<n>\d+)_K(?P<k>\d+)"
    r"(?:_CU(?P<cus>\d+))?\.json$"
)
SHAPE_CONFIG_PATTERN = re.compile(
    rf"^{W8A8_INT8_MASKED_CONFIG_PREFIX}_N(?P<n>\d+)_K(?P<k>\d+)"
    r"(?:_CU(?P<cus>\d+))?\.json$"
)

torch = None
triton = None
m_grouped_i8_gemm_nt_masked = None
marlin_i8_masked_weight = None


def import_runtime_deps() -> None:
    global torch
    global triton
    global m_grouped_i8_gemm_nt_masked
    global marlin_i8_masked_weight

    import torch as torch_module
    import triton as triton_module

    from deepgemm import (
        m_grouped_i8_gemm_nt_masked as masked_i8_op,
        marlin_i8_masked_weight as marlin_i8_pack,
    )

    torch = torch_module
    triton = triton_module
    m_grouped_i8_gemm_nt_masked = masked_i8_op
    marlin_i8_masked_weight = marlin_i8_pack


def resolve_path(value, base_dir):
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _load_yaml_mapping(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")
    with open(path) as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{label} file must be a YAML mapping, got {type(raw).__name__}")
    return raw


def _require_keys(raw: dict, keys, label: str):
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError(f"{label} file missing required keys: {missing}")
    unknown = set(raw) - set(keys)
    if unknown:
        raise ValueError(f"unknown keys in {label} file: {sorted(unknown)}")


def load_cases(cases_path: str):
    raw = _load_yaml_mapping(cases_path, "cases")
    _require_keys(raw, CASES_KEYS, "cases")
    return raw


def load_config(config_path: str):
    raw = _load_yaml_mapping(config_path, "config")
    unknown = set(raw) - set(CONFIG_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown keys in config: {sorted(unknown)}")

    merged = {**CONFIG_DEFAULTS, **raw}

    if merged["arch"] and merged["arch"] not in ARCH_CHOICES:
        raise ValueError(f"arch must be one of {ARCH_CHOICES}, got {merged['arch']!r}")

    base_dir = os.path.dirname(os.path.abspath(config_path))
    merged["out"] = resolve_path(merged["out"], base_dir)
    merged["config_root"] = resolve_path(merged["config_root"], base_dir) or base_dir
    merged["cases"] = resolve_path(merged["cases"], base_dir)

    cases_data = load_cases(merged["cases"])

    global E_LIST, SHAPES, EXPECTED, MODES, PERSISTENT_MODES, MODE_DEFAULT_MULT
    MODES = [int(x) for x in merged["modes"]]
    PERSISTENT_MODES = {int(x) for x in merged["persistent_modes"]}
    MODE_DEFAULT_MULT = {int(k): int(v) for k, v in merged["mode_default_mult"].items()}
    E_LIST = [int(x) for x in cases_data["e_list"]]
    SHAPES = [(int(pair[0]), int(pair[1])) for pair in cases_data["shapes"]]
    EXPECTED = [int(x) for x in cases_data["expected"]]

    class Config:
        pass

    cfg = Config()
    for key, value in merged.items():
        setattr(cfg, key, value)
    cfg.config_path = os.path.abspath(config_path)
    return cfg


def gpu_use(device: int):
    try:
        output = subprocess.check_output(
            ["/opt/dtk/bin/rocm-smi", "--showuse"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    prefix = f"HCU[{device}]"
    for line in output.splitlines():
        if prefix in line and "HCU use" in line:
            try:
                return float(line.rsplit(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def wait_gpu_idle(device: int, timeout_s: int = 600) -> None:
    start = time.time()
    while True:
        use = gpu_use(device)
        if use == 0.0:
            return
        if time.time() - start > timeout_s:
            raise TimeoutError(f"GPU {device} did not become idle, last use={use}")
        print(f"[wait] GPU{device} use={use}, sleep 5s", flush=True)
        time.sleep(5)


def detect_arch() -> str:
    override = os.environ.get("DEEPGEMM_W8A8_INT8_MASKED_ARCH")
    if override:
        return override.lower()

    try:
        import torch

        if torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
            arch = arch.split(":", 1)[0].lower()
            if arch:
                return arch
    except Exception:
        pass
    return "unknown"


def current_num_cus() -> int | None:
    try:
        override = int(os.environ.get("DEEPGEMM_GPU_CUS", "0"))
        if override > 0:
            return override
    except ValueError:
        pass

    try:
        if torch.cuda.is_available():
            return int(
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count
            )
    except Exception:
        pass
    return None


def alloc_case(e: int, m: int, n: int, k: int, shuffle_unique: int):
    torch.manual_seed(1)
    q_a = torch.randint(-4, 5, (e, m, k), device="cuda", dtype=torch.int8).contiguous()
    a_scale = torch.rand((e, m, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)
    weight = torch.randint(-4, 5, (e, n, k), device="cuda", dtype=torch.int8)
    w_scale = torch.rand((e, n, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)
    packed_w = marlin_i8_masked_weight(weight, shuffle_unique=shuffle_unique)
    out = torch.zeros((e, m, n), device="cuda", dtype=torch.bfloat16)
    return q_a, a_scale, packed_w, w_scale, out


def run_once(
    tensors,
    e: int,
    n: int,
    k: int,
    expected: int,
    mode: int,
    warmup: int,
    iters: int,
    shuffle_unique: int,
):
    q_a, a_scale, packed_w, w_scale, out = tensors
    masked_m = torch.full((e,), expected, device="cuda", dtype=torch.int32)
    cfg = config_for_mode(mode)
    m_grouped_i8_gemm_nt_masked(
        (q_a, a_scale),
        (packed_w, w_scale),
        out,
        masked_m,
        expected,
        cfg,
        shuffle_unique=shuffle_unique,
    )
    torch.cuda.synchronize()
    for _ in range(warmup):
        m_grouped_i8_gemm_nt_masked(
            (q_a, a_scale),
            (packed_w, w_scale),
            out,
            masked_m,
            expected,
            cfg,
            shuffle_unique=shuffle_unique,
        )
    torch.cuda.synchronize()
    cost_us = triton.testing.do_bench(
        lambda: m_grouped_i8_gemm_nt_masked(
            (q_a, a_scale),
            (packed_w, w_scale),
            out,
            masked_m,
            expected,
            cfg,
            shuffle_unique=shuffle_unique,
        ),
        warmup=warmup,
        rep=iters,
        quantiles=None,
        return_mode="mean",
    ) * 1000
    valid_m = e * expected
    # int8 activation: 1 byte, int8 weight: 1 byte, bf16 output: out.element_size() bytes
    data_bytes = valid_m * k + e * n * k + valid_m * n * out.element_size()
    bandwidth = data_bytes / (cost_us / 1e6) / 1e9
    return cost_us, bandwidth


def config_for_mode(mode: int):
    config = {"MODE": int(mode)}
    if mode in PERSISTENT_MODES:
        config["PERSISTENT_MULTIPLIER"] = MODE_DEFAULT_MULT[mode]
    return config


def config_output_dir(config_root: str, arch: str, layout: str) -> str:
    path = os.path.join(config_root, arch, W8A8_INT8_MASKED_CONFIG_DIR)
    if layout:
        path = os.path.join(path, layout)
    return path


def emit_configs_from_csv(csv_path: str, arch: str, config_root: str, layout: str) -> None:
    use_cu = arch in CU_DISTINGUISH_ARCHS
    output_dir = config_output_dir(config_root, arch, layout)
    best_by_case = {}
    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        if use_cu and "num_cus" not in (reader.fieldnames or []):
            raise ValueError("CSV requires a num_cus column for this arch; rerun tuning")
        for row in reader:
            try:
                num_cus = int(row["num_cus"]) if use_cu else 0
                key = (
                    num_cus,
                    int(row["E"]),
                    int(row["N"]),
                    int(row["K"]),
                    int(row["expected_m"]),
                )
                mode = int(row["mode"])
                median_cost = float(row["median_cost_us"])
            except (KeyError, TypeError, ValueError):
                continue
            if (use_cu and num_cus <= 0) or not math.isfinite(median_cost):
                continue
            previous = best_by_case.get(key)
            if previous is None or median_cost < previous[0]:
                best_by_case[key] = (median_cost, mode)

    configs_by_file = {}
    for (num_cus, e, n, k, expected), (_, mode) in best_by_case.items():
        suffix = f"_CU{num_cus}" if use_cu else ""
        path = os.path.join(
            output_dir,
            f"{W8A8_INT8_MASKED_CONFIG_PREFIX}_E{e}_N{n}_K{k}{suffix}.json",
        )
        configs_by_file.setdefault(path, {})[str(expected)] = config_for_mode(mode)

    for path, configs in sorted(configs_by_file.items()):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ordered = {key: configs[key] for key in sorted(configs, key=lambda item: int(item))}
        with open(path, "w") as file:
            json.dump({"configs": ordered}, file, indent=4)
            file.write("\n")
        print(f"WRITE {path}", flush=True)


def load_config_file(config_path: str) -> dict:
    with open(config_path) as file:
        data = json.load(file)
    configs = data.get("configs", data) if isinstance(data, dict) else {}
    return configs if isinstance(configs, dict) else {}


def modal_value(values):
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def modal_config(configs) -> dict | None:
    modes = []
    for config in configs:
        if not isinstance(config, dict):
            continue
        try:
            modes.append(int(config["MODE"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not modes:
        return None

    mode = modal_value(modes)
    multipliers = []
    for config in configs:
        if not isinstance(config, dict):
            continue
        try:
            if int(config["MODE"]) != mode:
                continue
            multipliers.append(int(config["PERSISTENT_MULTIPLIER"]))
        except (KeyError, TypeError, ValueError):
            continue

    result = {"MODE": mode}
    if multipliers:
        result["PERSISTENT_MULTIPLIER"] = modal_value(multipliers)
    return result


def merge_configs(configs_list) -> dict:
    by_expected = {}
    for configs in configs_list:
        for key, config in configs.items():
            try:
                expected = int(key)
            except (TypeError, ValueError):
                continue
            by_expected.setdefault(expected, []).append(config)

    merged = {}
    for expected, configs in by_expected.items():
        picked = modal_config(configs)
        if picked is not None:
            merged[str(expected)] = picked
    return {key: merged[key] for key in sorted(merged, key=int)}


def write_config_file(config_path: str, configs: dict) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as file:
        json.dump({"configs": configs}, file, indent=4)
        file.write("\n")
    print(f"WRITE {config_path}", flush=True)


def emit_aggregate_configs(arch: str, config_root: str, layout: str) -> None:
    use_cu = arch in CU_DISTINGUISH_ARCHS
    config_dir = config_output_dir(config_root, arch, layout)
    if not os.path.isdir(config_dir):
        print(f"SKIP aggregate configs, directory not found: {config_dir}", flush=True)
        return

    expert_files = {}
    for name in sorted(os.listdir(config_dir)):
        match = EXPERT_CONFIG_PATTERN.match(name)
        if not match:
            continue
        if not use_cu and match["cus"] is not None:
            continue
        key = (int(match["n"]), int(match["k"]), match["cus"])
        expert_files.setdefault(key, []).append(os.path.join(config_dir, name))

    # XXXX_Nn_Kk.json: 同形状、同 CU 下的各 E 配置文件逐 expected_m 取众数
    for (n, k, cus), paths in sorted(
        expert_files.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
    ):
        merged = merge_configs([load_config_file(path) for path in paths])
        if not merged:
            continue
        suffix = f"_CU{cus}" if cus else ""
        write_config_file(
            os.path.join(
                config_dir, f"{W8A8_INT8_MASKED_CONFIG_PREFIX}_N{n}_K{k}{suffix}.json"
            ),
            merged,
        )

    shape_files = {}
    for name in sorted(os.listdir(config_dir)):
        match = SHAPE_CONFIG_PATTERN.match(name)
        if not match:
            continue
        if not use_cu and match["cus"] is not None:
            continue
        shape_files.setdefault(match["cus"], []).append(os.path.join(config_dir, name))

    # XXXX_DEFAULT.json: 同 CU 下的各 Nn_Kk 配置文件逐 expected_m 取众数
    for cus, paths in sorted(shape_files.items(), key=lambda item: item[0] or ""):
        merged = merge_configs([load_config_file(path) for path in paths])
        if not merged:
            continue
        suffix = f"_CU{cus}" if cus else ""
        write_config_file(
            os.path.join(
                config_dir, f"{W8A8_INT8_MASKED_CONFIG_PREFIX}_DEFAULT{suffix}.json"
            ),
            merged,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.arch:
        cfg.arch = detect_arch()
    if cfg.arch not in ARCH_CHOICES:
        raise ValueError(
            f"unsupported arch {cfg.arch!r} (supported: {ARCH_CHOICES}); "
            "set DEEPGEMM_W8A8_INT8_MASKED_ARCH to override"
        )
    print(f"CONFIG {cfg.config_path}", flush=True)
    print(f"  cases={cfg.cases}", flush=True)
    print(
        f"  gpu={cfg.gpu} m={cfg.m} warmup={cfg.warmup} iters={cfg.iters} repeats={cfg.repeats}",
        flush=True,
    )
    print(f"  shuffle_unique={cfg.shuffle_unique}", flush=True)
    print(f"  out={cfg.out}", flush=True)
    print(
        f"  arch={cfg.arch} layout={cfg.layout or '(none)'} config_root={cfg.config_root}",
        flush=True,
    )
    print(
        f"  resume={cfg.resume} emit_config={cfg.emit_config} "
        f"emit_only={cfg.emit_only} aggregate_only={cfg.aggregate_only}",
        flush=True,
    )

    if cfg.aggregate_only:
        emit_aggregate_configs(cfg.arch, cfg.config_root, cfg.layout)
        return

    if cfg.emit_only:
        emit_configs_from_csv(cfg.out, cfg.arch, cfg.config_root, cfg.layout)
        emit_aggregate_configs(cfg.arch, cfg.config_root, cfg.layout)
        return

    import_runtime_deps()
    torch.cuda.set_device(cfg.gpu)
    num_cus = current_num_cus() if cfg.arch in CU_DISTINGUISH_ARCHS else 0
    if cfg.arch in CU_DISTINGUISH_ARCHS and (num_cus is None or num_cus <= 0):
        raise RuntimeError("could not determine the CU count for tuning")
    print(f"  num_cus={num_cus}", flush=True)

    fields = [
        "num_cus",
        "E",
        "N",
        "K",
        "expected_m",
        "mode",
        "repeat",
        "cost_us",
        "bandwidth",
        "median_cost_us",
        "median_bandwidth",
        "best",
    ]
    completed = set()
    if cfg.resume and os.path.exists(cfg.out):
        with open(cfg.out, newline="") as file:
            reader = csv.DictReader(file)
            if "num_cus" not in (reader.fieldnames or []):
                raise ValueError("resume CSV is missing the num_cus column; use a new output file")
            for row in reader:
                try:
                    key = (
                        int(row["num_cus"]),
                        int(row["E"]),
                        int(row["N"]),
                        int(row["K"]),
                        int(row["expected_m"]),
                    )
                    completed.add(key)
                except (KeyError, TypeError, ValueError):
                    continue

    file_mode = "a" if cfg.resume and os.path.exists(cfg.out) else "w"
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, file_mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file_mode == "w":
            writer.writeheader()
        for e in E_LIST:
            for n, k in SHAPES:
                wait_gpu_idle(cfg.gpu)
                print(f"ALLOC E={e} N={n} K={k}", flush=True)
                tensors = alloc_case(e, cfg.m, n, k, cfg.shuffle_unique)
                for expected in EXPECTED:
                    if (num_cus, e, n, k, expected) in completed:
                        print(f"SKIP E={e} N={n} K={k} exp={expected}", flush=True)
                        continue
                    wait_gpu_idle(cfg.gpu)
                    mode_runs = {}
                    for mode in MODES:
                        runs = []
                        for repeat in range(cfg.repeats):
                            wait_gpu_idle(cfg.gpu)
                            try:
                                cost_us, bandwidth = run_once(
                                    tensors,
                                    e,
                                    n,
                                    k,
                                    expected,
                                    mode,
                                    cfg.warmup,
                                    cfg.iters,
                                    cfg.shuffle_unique,
                                )
                            except Exception as error:
                                print(
                                    f"FAIL E={e} N={n} K={k} exp={expected} mode={mode} "
                                    f"repeat={repeat}: {error}",
                                    flush=True,
                                )
                                cost_us, bandwidth = float("inf"), 0.0
                            runs.append((repeat, cost_us, bandwidth))
                        finite_costs = [cost for _, cost, _ in runs if math.isfinite(cost)]
                        median_cost = statistics.median(finite_costs) if finite_costs else float("inf")
                        finite_bws = [bw for _, _, bw in runs if bw > 0.0]
                        median_bw = statistics.median(finite_bws) if finite_bws else 0.0
                        mode_runs[mode] = (runs, median_cost, median_bw)
                    best_mode = min(MODES, key=lambda current: mode_runs[current][1])
                    best_cost = mode_runs[best_mode][1]
                    best_bw = mode_runs[best_mode][2]
                    print(
                        f"E={e:2d} N={n} K={k} exp={expected:4d} "
                        f"best={best_mode:4d} median={best_cost:8.2f}us {best_bw:8.1f}GB/s",
                        flush=True,
                    )
                    for mode, (runs, median_cost, median_bw) in mode_runs.items():
                        for repeat, cost_us, bandwidth in runs:
                            writer.writerow(
                                {
                                    "num_cus": num_cus,
                                    "E": e,
                                    "N": n,
                                    "K": k,
                                    "expected_m": expected,
                                    "mode": mode,
                                    "repeat": repeat,
                                    "cost_us": cost_us,
                                    "bandwidth": bandwidth,
                                    "median_cost_us": median_cost,
                                    "median_bandwidth": median_bw,
                                    "best": int(
                                        mode == best_mode and math.isfinite(best_cost)
                                    ),
                                }
                            )
                    file.flush()
                del tensors
                torch.cuda.empty_cache()
    print(f"DONE {cfg.out}", flush=True)
    if cfg.emit_config:
        emit_configs_from_csv(cfg.out, cfg.arch, cfg.config_root, cfg.layout)
        emit_aggregate_configs(cfg.arch, cfg.config_root, cfg.layout)


if __name__ == "__main__":
    main()
