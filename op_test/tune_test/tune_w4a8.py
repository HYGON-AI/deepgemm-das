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
if ROOT not in sys.path and any(
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

DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "tune_w4a8_cfg.yml")
DEFAULT_CASES_PATH = "masked_cases.yml"
CASES_KEYS = ("e_list", "shapes", "expected")
CONFIG_DEFAULTS = {
    "modes": [1161, 1162, 1321, 1322, 1642, 1644, 2282],
    "persistent_modes": [1161, 1162, 1321, 1322, 1642],
    "mode_default_mult": {1161: 2, 1162: 2, 1321: 2, 1322: 2, 1642: 2},
    "multipliers": [1, 2, 3, 4, 8],
    "cases": DEFAULT_CASES_PATH,
    "gpu": 3,
    "m": 2048,
    "warmup": 8,
    "iters": 40,
    "repeats": 3,
    "out": "w4a8_masked_multiplier_sweep.csv",
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
CU_DISTINGUISH_ARCHS = ("gfx938")

W4A8_HIPC_CONFIG_DIR = "w4a8_masked_hipc"
W4A8_HIPC_CONFIG_PREFIX = "DEEPGEMM_W4A8_MASKED_HIPC"
EXPERT_CONFIG_PATTERN = re.compile(
    rf"^{W4A8_HIPC_CONFIG_PREFIX}_E(?P<e>\d+)_N(?P<n>\d+)_K(?P<k>\d+)"
    r"(?:_CU(?P<cus>\d+))?\.json$"
)
SHAPE_CONFIG_PATTERN = re.compile(
    rf"^{W4A8_HIPC_CONFIG_PREFIX}_N(?P<n>\d+)_K(?P<k>\d+)"
    r"(?:_CU(?P<cus>\d+))?\.json$"
)

torch = None
triton = None
m_grouped_w4a8_gemm_nt_masked_hipc = None
pack_w4a8_moe_hipc_weight = None
pack_w4a8_weight_k_contiguous = None


def import_runtime_deps() -> None:
    global torch
    global triton
    global m_grouped_w4a8_gemm_nt_masked_hipc
    global pack_w4a8_moe_hipc_weight
    global pack_w4a8_weight_k_contiguous

    import torch as torch_module
    import triton as triton_module

    from deepgemm import (
        m_grouped_w4a8_gemm_nt_masked_hipc as masked_w4a8_op,
        pack_w4a8_moe_hipc_weight as pack_moe_weight,
    )
    from deepgemm.group_gemm_weight_pack import (
        pack_w4a8_weight_k_contiguous as pack_weight_k_contiguous,
    )

    torch = torch_module
    triton = triton_module
    m_grouped_w4a8_gemm_nt_masked_hipc = masked_w4a8_op
    pack_w4a8_moe_hipc_weight = pack_moe_weight
    pack_w4a8_weight_k_contiguous = pack_weight_k_contiguous


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
    merged["multipliers"] = list(dict.fromkeys(int(x) for x in merged["multipliers"]))
    if not merged["multipliers"] or any(x <= 0 for x in merged["multipliers"]):
        raise ValueError("multipliers must be a nonempty list of positive integers")
    if not MODES or not E_LIST or not SHAPES or not EXPECTED:
        raise ValueError("modes and case lists must not be empty")
    if any(k < 512 or k % 512 for _, k in SHAPES):
        raise ValueError("K must be a positive multiple of 512")
    if any(x <= 0 or x > merged["m"] for x in EXPECTED):
        raise ValueError("expected M must be between 1 and m")
    if merged["repeats"] <= 0:
        raise ValueError("repeats must be positive")

    class Config:
        pass

    cfg = Config()
    for key, value in merged.items():
        setattr(cfg, key, value)
    cfg.config_path = os.path.abspath(config_path)
    return cfg


def detect_arch() -> str:
    override = os.environ.get("DEEPGEMM_W4A8_MASKED_HIPC_ARCH")
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


def alloc_case(e: int, m: int, n: int, k: int):
    torch.manual_seed(1)
    q_a = torch.randint(-8, 8, (e, m, k), device="cuda", dtype=torch.int8).contiguous()
    a_scale = torch.rand((e, m, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)
    w_scale = torch.rand((e, n, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)
    kernel_w_scale = w_scale / 16.0
    weight_u4 = torch.randint(0, 15, (e, n, k), device="cuda", dtype=torch.int8)
    packed_w = pack_w4a8_moe_hipc_weight(pack_w4a8_weight_k_contiguous(weight_u4))
    out = torch.empty((e, m, n), device="cuda", dtype=torch.bfloat16)
    return q_a, a_scale, packed_w, kernel_w_scale, out


def run_once(tensors, e: int, n: int, k: int, expected: int, mode: int, warmup: int, iters: int,
             persistent_multiplier: int = 0):
    q_a, a_scale, packed_w, kernel_w_scale, out = tensors
    masked_m = torch.full((e,), expected, device="cuda", dtype=torch.int32)
    cfg = config_for_mode(mode, persistent_multiplier)
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (q_a, a_scale), (packed_w, kernel_w_scale), out, masked_m, expected, cfg
    )
    torch.cuda.synchronize()
    for _ in range(warmup):
        m_grouped_w4a8_gemm_nt_masked_hipc(
            (q_a, a_scale), (packed_w, kernel_w_scale), out, masked_m, expected, cfg
        )
    torch.cuda.synchronize()
    cost_us = triton.testing.do_bench(
        lambda: m_grouped_w4a8_gemm_nt_masked_hipc(
            (q_a, a_scale), (packed_w, kernel_w_scale), out, masked_m, expected, cfg
        ),
        warmup=warmup,
        rep=iters,
        quantiles=None,
        return_mode="mean",
    ) * 1000
    valid_m = e * expected
    data_bytes = valid_m * k + e * n * k // 2 + valid_m * n * out.element_size()
    bandwidth = data_bytes / (cost_us / 1e6) / 1e9
    return cost_us, bandwidth


def config_for_mode(mode: int, persistent_multiplier=None):
    config = {"MODE": int(mode)}
    if mode in PERSISTENT_MODES:
        config["PERSISTENT_MULTIPLIER"] = int(
            MODE_DEFAULT_MULT[mode] if persistent_multiplier is None else persistent_multiplier
        )
    return config


def tuning_candidates(multipliers):
    return [(mode, multiplier)
            for mode in MODES
            for multiplier in (multipliers if mode in PERSISTENT_MODES else [0])]


def config_output_dir(config_root: str, arch: str, layout: str) -> str:
    path = os.path.join(config_root, arch, W4A8_HIPC_CONFIG_DIR)
    if layout:
        path = os.path.join(path, layout)
    return path


def emit_configs_from_csv(csv_path: str, arch: str, config_root: str, layout: str) -> None:
    use_cu = arch in CU_DISTINGUISH_ARCHS
    output_dir = config_output_dir(config_root, arch, layout)
    latest_by_candidate = {}
    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        if not {"num_cus", "persistent_multiplier"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires num_cus and persistent_multiplier columns; rerun tuning")
        for row in reader:
            try:
                key = (
                    int(row["num_cus"]) if use_cu else 0,
                    int(row["E"]),
                    int(row["N"]),
                    int(row["K"]),
                    int(row["expected_m"]),
                )
                mode = int(row["mode"])
                multiplier = int(row["persistent_multiplier"])
                median_cost = float(row["median_cost_us"])
            except (KeyError, TypeError, ValueError):
                continue
            if (use_cu and key[0] <= 0) or mode not in MODES or multiplier < 0:
                continue
            latest_by_candidate[key + (mode, multiplier)] = median_cost

    best_by_case = {}
    for (num_cus, e, n, k, expected, mode, multiplier), cost in latest_by_candidate.items():
        if not math.isfinite(cost):
            continue
        key = (num_cus, e, n, k, expected)
        previous = best_by_case.get(key)
        if previous is None or cost < previous[0]:
            best_by_case[key] = (cost, mode, multiplier)

    configs_by_file = {}
    for (num_cus, e, n, k, expected), (_, mode, multiplier) in best_by_case.items():
        suffix = f"_CU{num_cus}" if use_cu else ""
        path = os.path.join(
            output_dir,
            f"{W4A8_HIPC_CONFIG_PREFIX}_E{e}_N{n}_K{k}{suffix}.json",
        )
        configs_by_file.setdefault(path, {})[str(expected)] = config_for_mode(mode, multiplier)

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
            os.path.join(config_dir, f"{W4A8_HIPC_CONFIG_PREFIX}_N{n}_K{k}{suffix}.json"), merged
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
            os.path.join(config_dir, f"{W4A8_HIPC_CONFIG_PREFIX}_DEFAULT{suffix}.json"), merged
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
            "set DEEPGEMM_W4A8_MASKED_HIPC_ARCH to override"
        )
    print(f"CONFIG {cfg.config_path}", flush=True)
    print(f"  cases={cfg.cases}", flush=True)
    print(
        f"  gpu={cfg.gpu} m={cfg.m} warmup={cfg.warmup} iters={cfg.iters} repeats={cfg.repeats}",
        flush=True,
    )
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

    if int(os.environ.get("DEEPGEMM_W4A8_MASKED_PERSISTENT_MULT", "0")) > 0:
        raise ValueError("unset DEEPGEMM_W4A8_MASKED_PERSISTENT_MULT before tuning; it overrides candidates")
    import_runtime_deps()
    torch.cuda.set_device(cfg.gpu)
    if cfg.arch in CU_DISTINGUISH_ARCHS:
        from deepgemm.configs.w4a8_masked_hipc import _current_num_cus

        num_cus = _current_num_cus()
        if num_cus is None or num_cus <= 0:
            raise RuntimeError("could not determine the CU count for tuning")
    else:
        num_cus = 0
    print(f"  num_cus={num_cus}", flush=True)
    candidates = tuning_candidates(cfg.multipliers)

    fields = [
        "num_cus",
        "E",
        "N",
        "K",
        "expected_m",
        "mode",
        "persistent_multiplier",
        "repeat",
        "cost_us",
        "bandwidth",
        "median_cost_us",
        "median_bandwidth",
        "best",
    ]
    completed = {}
    if cfg.resume and os.path.exists(cfg.out):
        with open(cfg.out, newline="") as file:
            reader = csv.DictReader(file)
            if not set(fields).issubset(reader.fieldnames or []):
                raise ValueError("resume CSV is missing tuning columns (including num_cus); use a new output file")
            fields = reader.fieldnames
            for row in reader:
                try:
                    key = (
                        int(row["num_cus"]),
                        int(row["E"]),
                        int(row["N"]),
                        int(row["K"]),
                        int(row["expected_m"]),
                        int(row["mode"]),
                        int(row["persistent_multiplier"]),
                    )
                    cost, bandwidth = float(row["cost_us"]), float(row["bandwidth"])
                    if math.isfinite(cost):
                        completed.setdefault(key, {})[int(row["repeat"])] = (cost, bandwidth)
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
                tensors = alloc_case(e, cfg.m, n, k)
                for expected in EXPECTED:
                    case_key = (num_cus, e, n, k, expected)
                    cached = {
                        candidate: completed.get(case_key + candidate, {})
                        for candidate in candidates
                    }
                    if all(all(repeat in cached[candidate] for repeat in range(cfg.repeats))
                           for candidate in candidates):
                        print(f"SKIP E={e} N={n} K={k} exp={expected}", flush=True)
                        continue
                    wait_gpu_idle(cfg.gpu)
                    mode_runs = {}
                    for mode, multiplier in candidates:
                        runs = []
                        for repeat in range(cfg.repeats):
                            previous = cached[(mode, multiplier)].get(repeat)
                            if previous is not None:
                                runs.append((repeat, *previous))
                                continue
                            wait_gpu_idle(cfg.gpu)
                            try:
                                cost_us, bandwidth = run_once(
                                    tensors, e, n, k, expected, mode, cfg.warmup, cfg.iters,
                                    multiplier,
                                )
                            except Exception as error:
                                print(f"FAIL mode={mode} multiplier={multiplier}: {error}", flush=True)
                                cost_us, bandwidth = float("inf"), 0.0
                            runs.append((repeat, cost_us, bandwidth))
                        finite_costs = [cost for _, cost, _ in runs if math.isfinite(cost)]
                        median_cost = statistics.median(finite_costs) if len(finite_costs) == cfg.repeats else float("inf")
                        finite_bws = [bw for _, _, bw in runs if bw > 0.0]
                        median_bw = statistics.median(finite_bws) if finite_bws else 0.0
                        mode_runs[(mode, multiplier)] = (runs, median_cost, median_bw)
                    best_candidate = min(candidates, key=lambda current: mode_runs[current][1])
                    best_mode, best_multiplier = best_candidate
                    best_cost = mode_runs[best_candidate][1]
                    best_bw = mode_runs[best_candidate][2]
                    print(
                        f"E={e:2d} N={n} K={k} exp={expected:4d} "
                        f"best={best_mode:4d} multiplier={best_multiplier} "
                        f"median={best_cost:8.2f}us {best_bw:8.1f}GB/s",
                        flush=True,
                    )
                    for (mode, multiplier), (runs, median_cost, median_bw) in mode_runs.items():
                        if all(repeat in cached[(mode, multiplier)] for repeat in range(cfg.repeats)):
                            continue
                        for repeat, cost_us, bandwidth in runs:
                            writer.writerow(
                                {
                                    "num_cus": num_cus,
                                    "E": e,
                                    "N": n,
                                    "K": k,
                                    "expected_m": expected,
                                    "mode": mode,
                                    "persistent_multiplier": multiplier,
                                    "repeat": repeat,
                                    "cost_us": cost_us,
                                    "bandwidth": bandwidth,
                                    "median_cost_us": median_cost,
                                    "median_bandwidth": median_bw,
                                    "best": int((mode, multiplier) == best_candidate and math.isfinite(best_cost)),
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
