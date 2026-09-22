#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import os
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in [REPO_ROOT, os.path.join(REPO_ROOT, "op_test")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from deepgemm import m_grouped_w8a8_gemm_nt_masked_impl
from deepgemm_marlin_quant import weight8bit_nt_kpack2_marlin2

SEED = 20260715
SIZE_M = 16384
MODE_BIG_TILE = 1000
OUT_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class RealLogCase:
    name: str
    log_line: int
    n: int
    k: int
    masked_m: tuple[int, ...]

    @property
    def expert_count(self) -> int:
        return len(self.masked_m)

    @property
    def valid_m(self) -> int:
        return sum(self.masked_m)

    @property
    def max_count(self) -> int:
        return max(self.masked_m)


DEFAULT_CASES = ['real_hot_single_k4096_line8998', 'real_hot_single_k2048_line9003', 'real_hot_many_k4096_line9076', 'real_dense_top_k4096_line11090', 'real_dense_top_k2048_line11093', 'real_balanced36_k4096_line12196']

CASES = {
    'real_hot_single_k4096_line8998': RealLogCase('real_hot_single_k4096_line8998', 8998, 4096, 4096, (3, 0, 0, 0, 0, 0, 16384, 0, 1, 0, 45, 3, 10, 0, 0, 0, 1479, 0, 0, 0, 0, 1, 7310, 17, 0, 0, 53, 0, 0, 2, 0, 0, 1, 0, 0, 3)),
    'real_hot_single_k2048_line9003': RealLogCase('real_hot_single_k2048_line9003', 9003, 4096, 2048, (3, 0, 0, 0, 0, 0, 16384, 0, 1, 0, 45, 3, 10, 0, 0, 0, 1479, 0, 0, 0, 0, 1, 7310, 17, 0, 0, 53, 0, 0, 2, 0, 0, 1, 0, 0, 3)),
    'real_hot_many_k4096_line9076': RealLogCase('real_hot_many_k4096_line9076', 9076, 4096, 4096, (53, 72, 4, 0, 40, 9, 16384, 521, 12, 5, 0, 793, 180, 31, 1311, 1, 163, 1, 403, 1, 876, 174, 19, 0, 1, 78, 1700, 11, 2, 35, 150, 0, 416, 28, 0, 112)),
    'real_dense_top_k4096_line11090': RealLogCase('real_dense_top_k4096_line11090', 11090, 4096, 4096, (4, 7, 9290, 1716, 1, 0, 0, 0, 0, 8, 693, 15, 126, 6, 14577, 7, 7, 273, 42, 93, 428, 0, 29, 514, 194, 0, 47, 0, 3, 16, 722, 0, 13, 20, 217, 15407)),
    'real_dense_top_k2048_line11093': RealLogCase('real_dense_top_k2048_line11093', 11093, 4096, 2048, (4, 7, 9290, 1716, 1, 0, 0, 0, 0, 8, 693, 15, 126, 6, 14577, 7, 7, 273, 42, 93, 428, 0, 29, 514, 194, 0, 47, 0, 3, 16, 722, 0, 13, 20, 217, 15407)),
    'real_balanced36_k4096_line12196': RealLogCase('real_balanced36_k4096_line12196', 12196, 4096, 4096, (75, 109, 220, 12525, 59, 150, 42, 88, 86, 175, 1643, 51, 119, 80, 8297, 125, 92, 74, 94, 304, 10011, 108, 39, 444, 126, 529, 105, 135, 10, 335, 6091, 236, 19, 119, 207, 10)),
}


def select_device(device_arg: str) -> int:
    if device_arg != "auto":
        device = int(device_arg)
        torch.cuda.set_device(device)
        print(f"使用指定 GPU: cuda:{device}")
        return device
    best_device = torch.cuda.current_device()
    best_free = -1
    best_total = 0
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes > best_free:
            best_device = device
            best_free = free_bytes
            best_total = total_bytes
    torch.cuda.set_device(best_device)
    print(f"自动选择 GPU: cuda:{best_device}, free={best_free / 1024**3:.2f} GiB, total={best_total / 1024**3:.2f} GiB")
    return best_device


def rand_int8(shape):
    return torch.randint(-16, 17, shape, device="cuda", dtype=torch.int8).contiguous()


def build_case(case: RealLogCase):
    if case.max_count > SIZE_M:
        raise ValueError(f"{case.name}: max_count={case.max_count} > SIZE_M={SIZE_M}")
    q_a = rand_int8((case.expert_count, SIZE_M, case.k))
    a_scale = (torch.rand((case.expert_count, SIZE_M, 1), device="cuda", dtype=torch.float32) * 1.5 + 0.5).contiguous()
    weight = rand_int8((case.expert_count, case.n, case.k))
    b_scale = (torch.rand((case.expert_count, case.n, 1), device="cuda", dtype=torch.float32) * 1.5 + 0.5).contiguous()
    masked_m = torch.tensor(case.masked_m, device="cuda", dtype=torch.int32)
    output = torch.empty((case.expert_count, SIZE_M, case.n), device="cuda", dtype=OUT_DTYPE)
    packed_weight = weight8bit_nt_kpack2_marlin2(weight)
    return (q_a, a_scale), (packed_weight, b_scale), output, masked_m, weight


def bench_ms(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(statistics.median(times))


def tflops(valid_m: int, n: int, k: int, elapsed_ms: float) -> float:
    return 2.0 * valid_m * n * k / (elapsed_ms / 1000.0) / 1e12


def torch_reference(q_a, a_scale, weight, b_scale):
    acc = torch.bmm(q_a.to(torch.float32), weight.to(torch.float32).transpose(1, 2))
    return (acc * a_scale * b_scale.transpose(1, 2)).to(OUT_DTYPE)


def check_correctness(ref, output, masked_m, verbose: bool):
    ok = True
    max_abs = 0.0
    sum_abs = 0.0
    max_rel = 0.0
    sum_rel = 0.0
    checked = 0
    ref_abs_max = 0.0
    out_abs_max = 0.0
    failed = None
    for expert in range(masked_m.numel()):
        valid_m = int(masked_m[expert].item())
        if valid_m <= 0:
            continue
        ref_slice = ref[expert, :valid_m].float()
        out_slice = output[expert, :valid_m].float()
        diff = (ref_slice - out_slice).abs()
        denom = ref_slice.abs().clamp_min(1e-6)
        rel = diff / denom
        cur_max_abs = float(diff.max().item()) if diff.numel() else 0.0
        cur_mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        cur_max_rel = float(rel.max().item()) if rel.numel() else 0.0
        cur_mean_rel = float(rel.mean().item()) if rel.numel() else 0.0
        cur_ref_abs_max = float(ref_slice.abs().max().item()) if ref_slice.numel() else 0.0
        cur_out_abs_max = float(out_slice.abs().max().item()) if out_slice.numel() else 0.0
        numel = diff.numel()
        max_abs = max(max_abs, cur_max_abs)
        max_rel = max(max_rel, cur_max_rel)
        ref_abs_max = max(ref_abs_max, cur_ref_abs_max)
        out_abs_max = max(out_abs_max, cur_out_abs_max)
        sum_abs += float(diff.sum().item())
        sum_rel += float(rel.sum().item())
        checked += numel
        cur_ok = bool(torch.allclose(ref_slice, out_slice, rtol=1e-2, atol=1e-2))
        if verbose:
            print(
                f"  expert={expert:2d} valid_m={valid_m:<5d} "
                f"max_abs={cur_max_abs:.6f} mean_abs={cur_mean_abs:.6f} "
                f"max_rel={cur_max_rel:.6f} mean_rel={cur_mean_rel:.6f} "
                f"ref_abs_max={cur_ref_abs_max:.3f} out_abs_max={cur_out_abs_max:.3f} ok={cur_ok}"
            )
        if not cur_ok and failed is None:
            failed = (expert, valid_m, cur_max_abs, cur_max_rel)
            ok = False
    mean_abs = sum_abs / checked if checked else 0.0
    mean_rel = sum_rel / checked if checked else 0.0
    stats = {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "ref_abs_max": ref_abs_max,
        "out_abs_max": out_abs_max,
        "checked": checked,
    }
    return ok, stats, failed


def run_case(case: RealLogCase, warmup: int, iters: int, correctness: bool, verbose: bool):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    a, b, output, masked_m, raw_weight = build_case(case)

    def run_big_tile():
        return m_grouped_w8a8_gemm_nt_masked_impl(a, b, output, masked_m, 512, MODE_BIG_TILE)

    status = "SKIP"
    stats = {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0, "ref_abs_max": 0.0, "out_abs_max": 0.0}
    if correctness:
        run_big_tile()
        torch.cuda.synchronize()
        ref = torch_reference(a[0], a[1], raw_weight, b[1])
        ok, stats, failed = check_correctness(ref, output, masked_m, verbose)
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  first_failed={failed}")
            raise AssertionError(f"big tile correctness failed: {case.name}")
        del ref
        torch.cuda.empty_cache()

    ms = bench_ms(run_big_tile, warmup, iters)
    perf = tflops(case.valid_m, case.n, case.k, ms)
    print(
        "line={:<5d} E={:<2d} size_m={:<6d} valid={:<6d} N={:<4d} K={:<4d} "
        "mode={} {} repo_big_tile={:>8.3f}T ms={:>8.4f} "
        "max_abs={:.6f} mean_abs={:.6f} max_rel={:.6f} mean_rel={:.6f} "
        "ref_abs_max={:.3f} out_abs_max={:.3f} case={}".format(
            case.log_line, case.expert_count, SIZE_M, case.valid_m, case.n, case.k,
            MODE_BIG_TILE, status, perf, ms,
            stats["max_abs"], stats["mean_abs"], stats["max_rel"], stats["mean_rel"],
            stats["ref_abs_max"], stats["out_abs_max"], case.name,
        )
    )
    return {
        "line": case.log_line,
        "case": case.name,
        "valid_m": case.valid_m,
        "n": case.n,
        "k": case.k,
        "status": status,
        "ms": ms,
        "tflops": perf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["default", "all"] + sorted(CASES), default="default")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    select_device(args.device)
    if args.case == "all":
        names = sorted(CASES)
    elif args.case == "default":
        names = DEFAULT_CASES
    else:
        names = [args.case]
    print(f"using DeepGEMM source tree, big tile mode={MODE_BIG_TILE}, SIZE_M={SIZE_M}, cases={len(names)}")
    results = []
    for name in names:
        results.append(run_case(CASES[name], args.warmup, args.iters, not args.skip_correctness, args.verbose))

    print("\nBig tile performance summary:")
    print("line   status valid_m N    K    ms       TFLOPS   case")
    for item in results:
        print(
            "{line:<6d} {status:<6s} {valid_m:<7d} {n:<4d} {k:<4d} "
            "{ms:<8.4f} {tflops:<8.3f} {case}".format(**item)
        )


if __name__ == "__main__":
    main()
