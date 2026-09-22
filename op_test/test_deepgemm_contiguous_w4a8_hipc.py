import argparse
import os
import random
import sys

import pandas as pd
import torch
import triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPGEMM_PACKAGE_DIR = os.path.join(ROOT, "deepgemm")
USE_SOURCE_TREE = (
    os.environ.get("DEEPGEMM_USE_SOURCE_TREE") == "1"
    or (
        os.path.isdir(DEEPGEMM_PACKAGE_DIR)
        and any(
            name.startswith("op.") and name.endswith(".so")
            for name in os.listdir(DEEPGEMM_PACKAGE_DIR)
        )
    )
)
if USE_SOURCE_TREE and ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deepgemm import (
    m_grouped_w4a8_gemm_nt_contiguous_hipc,
    pack_w4a8_moe_hipc_weight,
)
from deepgemm.group_gemm_weight_pack import pack_w4a8_weight_k_contiguous


def align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _build_case(
    e: int,
    total_m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: str = "cuda",
) -> dict:
    base_m = total_m // e
    remainder_m = total_m % e
    actual_ms = [base_m + (1 if expert < remainder_m else 0) for expert in range(e)]
    aligned_ms = [align(actual_m, 128) for actual_m in actual_ms]
    size_m = sum(aligned_ms)

    q_a = torch.randint(-8, 8, (size_m, k), device=device, dtype=torch.int8).contiguous()
    a_scale = torch.rand((size_m, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.001)
    w_scale = torch.rand((e, n, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.001)
    kernel_w_scale = w_scale / 16.0
    weight_u4 = torch.randint(0, 15, (e, n, k), device=device, dtype=torch.int8)

    framework_packed_w = pack_w4a8_weight_k_contiguous(weight_u4)
    packed_w = pack_w4a8_moe_hipc_weight(framework_packed_w)
    w_ref = torch.where(weight_u4 > 7, weight_u4 - 16, weight_u4)

    m_indices = torch.empty((size_m,), device=device, dtype=torch.int32)
    start = 0
    for expert, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        end = start + aligned_m
        m_indices[start:actual_end] = expert
        m_indices[actual_end:end] = -1
        start = end

    out = torch.empty((size_m, n), device=device, dtype=dtype)
    ref = torch.zeros((size_m, n), device=device, dtype=dtype)
    for expert in range(e):
        rows = m_indices == expert
        acc = q_a[rows].float() @ w_ref[expert].float().t()
        acc = acc * a_scale[rows].float() * w_scale[expert].float().t()
        ref[rows] = acc.to(dtype)

    return {
        "q_a": q_a,
        "a_scale": a_scale,
        "packed_w": packed_w,
        "w_scale": kernel_w_scale,
        "m_indices": m_indices,
        "out": out,
        "ref": ref,
        "actual_ms": actual_ms,
        "aligned_ms": aligned_ms,
        "actual_per_group": total_m / e,
        "aligned_per_group": size_m / e,
        "valid_m": total_m,
    }


def _check(ref: torch.Tensor, out: torch.Tensor, m_indices: torch.Tensor) -> tuple[bool, float, int, int, float]:
    valid = (m_indices >= 0).unsqueeze(1).expand_as(out)
    ref_valid = ref.masked_select(valid)
    out_valid = out.masked_select(valid)
    diff = (ref_valid.float() - out_valid.float()).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    close = torch.isclose(ref_valid, out_valid, rtol=1e-2, atol=1e-2)
    mismatch = int((~close).sum().item())
    total = close.numel()
    ratio = mismatch / total if total else 0.0
    return mismatch == 0, max_abs, mismatch, total, ratio


def run_case(
    e: int,
    total_m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    bench: bool = False,
    verify: bool = True,
    warmup: int = 20,
    iters: int = 100,
) -> dict:
    data = _build_case(e, total_m, n, k, dtype)
    fn = lambda: m_grouped_w4a8_gemm_nt_contiguous_hipc(
        (data["q_a"], data["a_scale"]),
        (data["packed_w"], data["w_scale"]),
        data["out"],
        data["m_indices"],
    )
    fn()
    torch.cuda.synchronize()

    max_abs = float("nan")
    correctness = "skipped"
    if verify:
        ok, max_abs, mismatch, total, ratio = _check(data["ref"], data["out"], data["m_indices"])
        correctness = "passed" if ok else "failed"
        if not ok:
            print(
                "run error",
                "max_abs:",
                max_abs,
                "mismatch:",
                mismatch,
                "total:",
                total,
                "ratio:",
                ratio,
            )
            raise AssertionError(
                f"deepgemm contiguous w4a8 hipc precision failed: max_abs={max_abs}"
            )
        print("success deepgemm_contiguous_w4a8_hipc gemm 精度检查合格")

    cost_us = None
    tflops = None
    bandwidth = None
    if bench:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        cost_us = triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=iters,
            quantiles=None,
            return_mode="mean",
        ) * 1000
        valid_m = int((data["m_indices"] >= 0).sum().item())
        tflops = (2 * valid_m * n * k) / (cost_us / 1e6) / 1e12
        weight_bytes = e * n * k // 2
        data_bytes = valid_m * k + weight_bytes + valid_m * n * data["out"].element_size()
        bandwidth = data_bytes / (cost_us / 1e6) / 1e9

    return {
        "E": e,
        "total_m": total_m,
        "actual_m_per_group": data["actual_per_group"],
        "aligned_m_per_group": data["aligned_per_group"],
        "valid_m": data["valid_m"],
        "N": n,
        "K": k,
        "dtype": str(dtype).replace("torch.", ""),
        "correctness": correctness,
        "max_abs": max_abs,
        "cost_us": cost_us,
        "tflops": tflops,
        "bandwidth": bandwidth,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e", type=int, default=16)
    parser.add_argument("--m", type=int, nargs="*", default=[128, 512, 1024, 2048, 4096, 131072, 262144])
    parser.add_argument("--shapes", nargs="*", default=["2048x3584", "3072x3584", "4096x6144", "6144x2048"])
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    df_E = []
    df_total_m = []
    df_n = []
    df_k = []
    df_actual_m = []
    df_aligned_m = []
    df_valid_m = []
    df_per_expert_m_avg = []
    df_costs = []
    df_tflops = []
    df_bandwidth = []
    total_cases = 0
    passed_cases = 0
    failed_cases = []

    for shape in args.shapes:
        n, k = [int(x) for x in shape.lower().split("x")]
        for m in args.m:
            total_cases += 1
            print(f"total_m:{m} n:{n} k:{k} ")
            try:
                result = run_case(
                    args.e,
                    m,
                    n,
                    k,
                    torch.bfloat16,
                    bench=True,
                    verify=not args.no_verify,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except AssertionError:
                if not args.no_verify:
                    failed_case = {
                        "E": args.e,
                        "total_m": m,
                        "n": n,
                        "k": k,
                    }
                    failed_cases.append(failed_case)
                    print(
                        "\033[31mcase correctness failed: "
                        f"E={args.e} total_m={m} N={n} K={k}"
                        "\033[0m"
                    )
                    print("------------------------------------------------------")
                continue

            if not args.no_verify:
                passed_cases += 1

            valid_m = result["valid_m"]
            df_E.append(args.e)
            df_total_m.append(m)
            df_n.append(n)
            df_k.append(k)
            df_actual_m.append(result["actual_m_per_group"])
            df_aligned_m.append(result["aligned_m_per_group"])
            df_valid_m.append(valid_m)
            df_per_expert_m_avg.append(valid_m / args.e)
            df_costs.append(result["cost_us"])
            df_tflops.append(result["tflops"])
            df_bandwidth.append(result["bandwidth"])

            print("valid_m:", valid_m, "actual_m_per_group:", result["actual_m_per_group"], "aligned_m_per_group:", result["aligned_m_per_group"])
            print("------------------------------------------------------")
            print(f"hipc cost: {result['cost_us']:.2f} us, {result['tflops']:.2f} TFlops")
            print(f"hipc bandwidth: {result['bandwidth']:.2f} GB/s")
            print("------------------------------------------------------")

    df = pd.DataFrame(
        {
            "E": df_E,
            "total_m": df_total_m,
            "n": df_n,
            "k": df_k,
            "actual_m_per_group": df_actual_m,
            "aligned_m_per_group": df_aligned_m,
            "valid_m": df_valid_m,
            "per_expert_m_avg": df_per_expert_m_avg,
            "hipc deepgemm 耗时(us)": df_costs,
            "hipc deepgemm TFlops": df_tflops,
            "hipc deepgemm Bandwidth(GB/s)": df_bandwidth,
        }
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("------------------------------------------------------")
    if not args.no_verify:
        if failed_cases:
            print(f"\033[31mcorrectness summary: FAIL {passed_cases}/{total_cases}\033[0m")
            print("\033[31mfailed cases:\033[0m")
            for case in failed_cases:
                print(
                    "\033[31m"
                    f"E={case['E']} total_m={case['total_m']} "
                    f"N={case['n']} K={case['k']}"
                    "\033[0m"
                )
            raise AssertionError(
                f"deepgemm contiguous w4a8 hipc correctness failed: {len(failed_cases)} of {total_cases} cases failed"
            )
        else:
            print(f"\033[32mcorrectness summary: PASS {passed_cases}/{total_cases}\033[0m")
    else:
        print(f"correctness summary: SKIP 0/{total_cases} (run without --no-verify to enable)")
    print("------------------------------------------------------")
    print(df)


if __name__ == "__main__":
    main()
