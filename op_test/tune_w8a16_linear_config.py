import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

import deepgemm


# ABC_SHAPES = {
#     "A": (12288, 2048),
#     "B": (2048, 4096),
#     "C": (9216, 2048),
# }

ABC_SHAPES = {
    "A": (12288, 2560),
    "B": (2560, 4096),
    "C": (18432, 2560),
    "D": (2560, 9216),
    "E": (10240, 2560),
}

DEFAULT_ATOL = 0.5
DEFAULT_RTOL = 0.02


Config = Dict[str, int]


def parse_m_values(text: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("--m-values is empty")
    return sorted(set(values))


def range_text(values: Iterable[int]) -> str:
    xs = sorted(set(values))
    if not xs:
        return ""
    ranges: List[str] = []
    start = prev = xs[0]
    for value in xs[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def make_config(
    kernel_type: int,
    num_k_warps: int,
    n_per_warp: int,
    k_per_warp: int,
    num_n_warps: int,
    splitk: int,
    block_m: int,
    weight_layout: int = 0,
) -> Config:
    return {
        "NUM_K_WARPS": int(num_k_warps),
        "N_PER_WARP": int(n_per_warp),
        "K_PER_WARP": int(k_per_warp),
        "NUM_N_WARPS": int(num_n_warps),
        "SPLITK": int(splitk),
        "BLOCK_M": int(block_m),
        "KERNEL_TYPE": int(kernel_type),
        "WEIGHT_LAYOUT": int(weight_layout),
    }


def force_type2_config(m: int, n: int, k: int) -> Config:
    cfg = dict(deepgemm.get_w8a16_linear_config(m, n, k))
    cfg["KERNEL_TYPE"] = 2
    cfg["WEIGHT_LAYOUT"] = 0
    return normalize_config(cfg, k)


def normalize_config(cfg: Dict[str, Any], k: int) -> Config:
    out = {
        "NUM_K_WARPS": int(cfg.get("NUM_K_WARPS", 1)),
        "N_PER_WARP": int(cfg.get("N_PER_WARP", 32)),
        "K_PER_WARP": int(cfg.get("K_PER_WARP", 128)),
        "NUM_N_WARPS": int(cfg.get("NUM_N_WARPS", 2)),
        "SPLITK": int(cfg.get("SPLITK", 1)),
        "BLOCK_M": int(cfg.get("BLOCK_M", 16)),
        "KERNEL_TYPE": int(cfg.get("KERNEL_TYPE", 0)),
        "WEIGHT_LAYOUT": int(cfg.get("WEIGHT_LAYOUT", 0)),
    }
    if out["KERNEL_TYPE"] == 0:
        out["BLOCK_M"] = 16
    if out["KERNEL_TYPE"] in (1, 2) and out["SPLITK"] <= 0:
        out["SPLITK"] = 1
    if out["K_PER_WARP"] > 0 and out["SPLITK"] > 1:
        out["SPLITK"] = min(out["SPLITK"], max(1, int(k) // out["K_PER_WARP"]))
    return out


def config_key(cfg: Config) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((k, int(v)) for k, v in cfg.items()))


def config_tag(cfg: Config) -> str:
    return (
        f"kt{cfg['KERNEL_TYPE']}:"
        f"KW={cfg['NUM_K_WARPS']},"
        f"NPW={cfg['N_PER_WARP']},"
        f"KPW={cfg['K_PER_WARP']},"
        f"NW={cfg['NUM_N_WARPS']},"
        f"SK={cfg['SPLITK']},"
        f"BM={cfg['BLOCK_M']},"
        f"WL={cfg['WEIGHT_LAYOUT']}"
    )


def parse_config_json(values: List[str], k: int) -> List[Config]:
    configs: List[Config] = []
    for value in values:
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            configs.append(normalize_config(loaded, k))
        elif isinstance(loaded, list):
            for row in loaded:
                if not isinstance(row, dict):
                    raise ValueError("--config-json list must contain objects")
                configs.append(normalize_config(row, k))
        else:
            raise ValueError("--config-json must be a JSON object or list")
    return configs


def candidate_configs(n: int, k: int, preset: str, allow_1024_threads: bool) -> List[Config]:
    if k % 16 != 0:
        return []

    max_wave_product = 16 if allow_1024_threads else 8

    if preset == "tiny":
        kt0_space = {
            "kw": [1, 2, 4],
            "npw": [32],
            "kpw": [64, 128],
            "nw": [2],
            "sk": [1],
        }
        kt1_space = {
            "kw": [1, 2],
            "npw": [32, 64],
            "kpw": [64, 128],
            "nw": [2],
            "bm": [32],
        }
    elif preset == "quick":
        kt0_space = {
            "kw": [1, 2, 4],
            "npw": [32, 64],
            "kpw": [64, 128, 256],
            "nw": [2, 4],
            "sk": [1, 2, 4, 8],
        }
        kt1_space = {
            "kw": [1, 2, 4],
            "npw": [32, 64],
            "kpw": [64, 128, 256],
            "nw": [2, 4],
            "bm": [32],
        }
    else:
        kt0_space = {
            "kw": [1, 2, 4],
            "npw": [32, 64],
            "kpw": [64, 128, 256],
            "nw": [2, 4],
            "sk": [1, 2, 4, 8],
        }
        kt1_space = {
            "kw": [1, 2, 4],
            "npw": [32, 64],
            "kpw": [64, 128, 256],
            "nw": [2, 4],
            "bm": [32, 64],
        }

    rows: List[Config] = []
    for kw in kt0_space["kw"]:
        for npw in kt0_space["npw"]:
            for kpw in kt0_space["kpw"]:
                for nw in kt0_space["nw"]:
                    if kw * nw > max_wave_product:
                        continue
                    for sk in kt0_space["sk"]:
                        sk_eff = min(sk, max(1, k // kpw))
                        rows.append(make_config(0, kw, npw, kpw, nw, sk_eff, 16))

    for kw in kt1_space["kw"]:
        for npw in kt1_space["npw"]:
            for kpw in kt1_space["kpw"]:
                for nw in kt1_space["nw"]:
                    if kw * nw > max_wave_product:
                        continue
                    for bm in kt1_space["bm"]:
                        rows.append(make_config(1, kw, npw, kpw, nw, 1, bm))

    deduped: List[Config] = []
    seen = set()
    for row in rows:
        key = config_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def make_inputs(shape_name: str, m_max: int, n: int, k: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed = 20260907 + sum(ord(ch) for ch in shape_name) * 1009 + n * 3 + k
    torch.manual_seed(seed)
    x = torch.randn((m_max, k), dtype=dtype, device="cuda") * 0.01
    weight = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="cuda")
    scale = (torch.randn((n,), dtype=dtype, device="cuda").abs() + 0.5) * 0.01
    return x, weight, scale


def validate(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> Tuple[float, float, int]:
    diff = (out.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    allowed = atol + rtol * ref_abs
    max_abs = float(diff.max().item())
    max_rel = float((diff / torch.clamp(ref_abs, min=1e-6)).max().item())
    fails = int((diff > allowed).sum().item())
    return max_abs, max_rel, fails


def flush_cache(flusher: torch.Tensor, value: int) -> None:
    if flusher.numel() == 0:
        return
    flusher.fill_(value & 0x7F)
    torch.cuda.synchronize()


def trimmed_mean(values: List[float], trim: int) -> float:
    if not values:
        return math.inf
    if len(values) <= 2 * trim:
        return statistics.mean(values)
    xs = sorted(values)
    return statistics.mean(xs[trim : len(xs) - trim])


def bench_cold(
    fn,
    warmup: int,
    rep: int,
    rounds: int,
    trim: int,
    flusher: torch.Tensor,
    clear_dequant_cache_each_call: bool,
) -> Tuple[float, List[float]]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    seq = 1

    for _ in range(warmup):
        if clear_dequant_cache_each_call:
            deepgemm.clear_w8a16_dequant_cache()
        flush_cache(flusher, seq)
        seq += 1
        fn()
        torch.cuda.synchronize()

    round_medians: List[float] = []
    for _ in range(rounds):
        vals: List[float] = []
        for _ in range(rep):
            if clear_dequant_cache_each_call:
                deepgemm.clear_w8a16_dequant_cache()
            flush_cache(flusher, seq)
            seq += 1
            start.record()
            fn()
            end.record()
            end.synchronize()
            vals.append(start.elapsed_time(end) * 1000.0)
        round_medians.append(statistics.median(vals))
    return trimmed_mean(round_medians, trim), round_medians


def run_with_config(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, cfg: Config) -> torch.Tensor:
    return deepgemm.w8a16_linear(x, weight, scale, config=cfg)


def measure_one(
    name: str,
    cfg: Config,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    ref: torch.Tensor,
    args,
    flusher: torch.Tensor,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": name,
        "config": dict(cfg),
        "config_tag": config_tag(cfg),
        "ok": False,
        "error": "",
        "us": math.inf,
        "samples": [],
        "max_abs": math.inf,
        "max_rel": math.inf,
        "fails": -1,
    }
    try:
        if cfg["KERNEL_TYPE"] == 2:
            deepgemm.clear_w8a16_dequant_cache()
        out = run_with_config(x, weight, scale, cfg)
        torch.cuda.synchronize()
        max_abs, max_rel, fails = validate(out, ref, args.atol, args.rtol)
        result.update({"max_abs": max_abs, "max_rel": max_rel, "fails": fails})
        if fails != 0:
            return result

        clear_cache = cfg["KERNEL_TYPE"] == 2 and cfg["WEIGHT_LAYOUT"] == 0
        us, samples = bench_cold(
            lambda: run_with_config(x, weight, scale, cfg),
            args.warmup,
            args.rep,
            args.rounds,
            args.trim,
            flusher,
            clear_cache,
        )
        result.update({"ok": True, "us": us, "samples": samples})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def select_shapes(args) -> List[Tuple[str, int, int]]:
    rows: List[Tuple[str, int, int]] = []
    for shape in args.shape or []:
        n, k = ABC_SHAPES[shape]
        rows.append((shape, n, k))
    if args.n is not None or args.k is not None:
        if args.n is None or args.k is None:
            raise ValueError("--n and --k must be provided together")
        rows.append((args.shape_name, int(args.n), int(args.k)))
    if not rows:
        n, k = ABC_SHAPES["C"]
        rows.append(("C", n, k))
    return rows


def best_rows_by_range(results: List[Dict[str, Any]]) -> List[Tuple[str, int, int, Config]]:
    grouped: Dict[Tuple[str, int, int, Tuple[Tuple[str, int], ...]], List[int]] = {}
    cfg_by_key: Dict[Tuple[str, int, int, Tuple[Tuple[str, int], ...]], Config] = {}
    for row in results:
        best = row.get("best")
        if not best or not best.get("ok"):
            continue
        cfg = best["config"]
        key = (row["shape"], row["n"], row["k"], config_key(cfg))
        grouped.setdefault(key, []).append(row["m"])
        cfg_by_key[key] = cfg

    ranges: List[Tuple[str, int, int, Config]] = []
    for key, ms in grouped.items():
        shape, n, k, _cfg_key = key
        cfg = cfg_by_key[key]
        ranges.append((f"{shape} M={range_text(ms)}", n, k, cfg))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune deepgemm.w8a16_linear configs for an existing or new shape."
    )
    parser.add_argument("--shape", choices=sorted(ABC_SHAPES), action="append", help="ABC shortcut shape; can repeat")
    parser.add_argument("--shape-name", default="Custom", help="name used when --n/--k is provided")
    parser.add_argument("--n", type=int, help="custom weight N")
    parser.add_argument("--k", type=int, help="custom weight K")
    parser.add_argument("--m-values", default="32", help="M list/range, e.g. 17-32 or 17,18,24,32")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--preset", choices=("tiny", "quick", "full"), default="quick")
    parser.add_argument("--allow-1024-threads", action="store_true", help="allow NUM_K_WARPS*NUM_N_WARPS=16 candidates")
    parser.add_argument("--config-json", action="append", default=[], help="extra config JSON object/list to try")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trim", type=int, default=1)
    parser.add_argument("--flush-mb", type=int, default=64)
    parser.add_argument(
        "--show-top",
        "--top-configs",
        "--topk",
        dest="show_top",
        type=int,
        default=8,
        help="number of valid configs to print for each M; --topk is kept as a deprecated alias",
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    m_values = parse_m_values(args.m_values)
    shapes = select_shapes(args)
    flush_elems = max(0, args.flush_mb) * 1024 * 1024
    flusher = torch.empty((flush_elems,), dtype=torch.uint8, device="cuda")

    print(f"torch={torch.__version__} hip={torch.version.hip} gpu={torch.cuda.get_device_name(0)}")
    print(
        f"dtype={args.dtype} preset={args.preset} M={range_text(m_values)} "
        f"cold_cache_flush={args.flush_mb}MB warmup={args.warmup} rep={args.rep} rounds={args.rounds}"
    )
    print(f"atol={args.atol} rtol={args.rtol}")
    print()
    print(
        f"{'shape':>8} {'M':>6} {'baseline':>10} {'default':>10} {'best':>10} "
        f"{'vs_base':>8} {'vs_def':>8} {'max_abs':>9} {'fails':>7}  best_config"
    )
    print("-" * 160)

    all_results: List[Dict[str, Any]] = []
    for shape_name, n, k in shapes:
        if k % 4 != 0:
            raise ValueError(f"{shape_name}: K must be divisible by 4 for type2 reference, got K={k}")
        if n % 4 != 0:
            raise ValueError(f"{shape_name}: N must be divisible by 4 for fused vector stores, got N={n}")

        x_full, weight, scale = make_inputs(shape_name, max(m_values), n, k, dtype)
        base_candidates = candidate_configs(n, k, args.preset, args.allow_1024_threads)
        extra_candidates = parse_config_json(args.config_json, k)

        for m in m_values:
            x = x_full[:m]
            baseline_cfg = force_type2_config(m, n, k)
            default_cfg = normalize_config(deepgemm.get_w8a16_linear_config(m, n, k), k)

            deepgemm.clear_w8a16_dequant_cache()
            ref = run_with_config(x, weight, scale, baseline_cfg)
            torch.cuda.synchronize()

            candidates: List[Tuple[str, Config]] = [
                ("type2_cold", baseline_cfg),
                ("default", default_cfg),
            ]
            candidates.extend((f"cand{i}", cfg) for i, cfg in enumerate(base_candidates))
            candidates.extend((f"extra{i}", cfg) for i, cfg in enumerate(extra_candidates))

            deduped: List[Tuple[str, Config]] = []
            seen = set()
            for cname, cfg in candidates:
                key = config_key(cfg)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append((cname, cfg))

            measured = [
                measure_one(cname, cfg, x, weight, scale, ref, args, flusher)
                for cname, cfg in deduped
            ]
            valid = [row for row in measured if row["ok"]]
            valid.sort(key=lambda row: row["us"])
            baseline = next(row for row in measured if row["name"] == "type2_cold")
            default = next((row for row in measured if row["name"] == "default"), baseline)
            best = valid[0] if valid else None

            result_row = {
                "shape": shape_name,
                "n": n,
                "k": k,
                "m": m,
                "baseline": baseline,
                "default": default,
                "best": best,
                "valid_top": valid[: args.show_top],
                "num_candidates": len(deduped),
                "num_valid": len(valid),
            }
            all_results.append(result_row)

            if best is None:
                print(f"{shape_name:>8} {m:6d} {baseline['us']:10.2f} {'FAIL':>10}", flush=True)
                continue

            vs_base = baseline["us"] / best["us"] if best["us"] > 0 else math.inf
            vs_def = default["us"] / best["us"] if best["us"] > 0 else math.inf
            print(
                f"{shape_name:>8} {m:6d} {baseline['us']:10.2f} {default['us']:10.2f} "
                f"{best['us']:10.2f} {vs_base:8.3f} {vs_def:8.3f} "
                f"{best['max_abs']:9.4g} {best['fails']:7d}  {best['name']} {best['config_tag']}",
                flush=True,
            )
            for row in valid[1 : args.show_top]:
                print(
                    f"{'':>8} {m:6d} {'':>10} {'':>10} {row['us']:10.2f} "
                    f"{'':>8} {'':>8} {row['max_abs']:9.4g} {row['fails']:7d}  "
                    f"{row['name']} {row['config_tag']}",
                    flush=True,
                )
            print(flush=True)

    print("Suggested config ranges:")
    for label, n, k, cfg in best_rows_by_range(all_results):
        print(f"  {label} N={n} K={k} -> {config_tag(cfg)}")
        print("    " + json.dumps(cfg, ensure_ascii=False, sort_keys=True))

    if args.output_json:
        payload = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "results": all_results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
