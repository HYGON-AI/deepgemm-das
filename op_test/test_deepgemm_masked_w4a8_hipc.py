import argparse
import os
import random
import sys

import pandas as pd
import torch
import triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OP_TEST = os.path.join(ROOT, "op_test")
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
if OP_TEST not in sys.path:
    sys.path.insert(0, OP_TEST)

from deepgemm import (
    m_grouped_w4a8_gemm_nt_masked_hipc,
    pack_w4a8_moe_hipc_weight,
)
from deepgemm.group_gemm_weight_pack import pack_w4a8_weight_k_contiguous

try:
    from aiter.moe_c_golden import compare_tensors
except ImportError:
    compare_tensors = None


ASM_RTOL = 1e-2
ASM_ATOL = 1e-2
DEBUG_MISMATCH_SAMPLES = 16


def run_case_impl(
    e: int,
    m: int,
    n: int,
    k: int,
    masked_m: torch.Tensor,
    dtype: torch.dtype,
    expected_m_per_group: int | None = None,
    bench: bool = False,
    verify: bool = True,
    warmup: int = 20,
    iters: int = 100,
) -> dict:
    torch.manual_seed(1)
    device = "cuda"
    if expected_m_per_group is None:
        expected_m_per_group = int(masked_m.max().item())

    q_a = torch.randint(-8, 8, (e, m, k), device=device, dtype=torch.int8).contiguous()
    a_scale = torch.rand((e, m, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.001)

    w_scale = torch.rand((e, n, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.001)
    kernel_w_scale = w_scale / 16.0

    weight_u4 = torch.randint(0, 15, (e, n, k), device=device, dtype=torch.int8)
    
    # 模拟框架传入的权重layout
    framework_packed_w = pack_w4a8_weight_k_contiguous(weight_u4)

    # weight layout处理成hipc需要的layout
    packed_w = pack_w4a8_moe_hipc_weight(framework_packed_w)

    w_ref = torch.where(weight_u4 > 7, weight_u4 - 16, weight_u4)

    ref = None
    if verify:
        ref = torch.bmm(q_a.float(), w_ref.float().transpose(1, 2))
        ref = (ref * a_scale * w_scale.transpose(1, 2)).to(dtype)

    out = torch.empty((e, m, n), device=device, dtype=dtype)
    masked_m = masked_m.to(device=device, dtype=torch.int32).contiguous()

    m_grouped_w4a8_gemm_nt_masked_hipc(
        (q_a, a_scale),
        (packed_w, kernel_w_scale),
        out,
        masked_m,
        expected_m_per_group,
    )
    torch.cuda.synchronize()

    max_abs = float("nan")
    if verify:
        valid_check = True
        max_abs = 0.0
        for expert_idx in range(masked_m.shape[0]):
            cur_valid_m = int(masked_m[expert_idx].item())
            if cur_valid_m == 0:
                continue
            valid = out[expert_idx, :cur_valid_m, :]
            ref_valid = ref[expert_idx, :cur_valid_m, :]
            diff = (valid.float() - ref_valid.float()).abs()
            cur_max_abs = diff.max().item()
            if torch.isnan(diff).any():
                cur_max_abs = float("nan")
            if not torch.allclose(valid, ref_valid, rtol=ASM_RTOL, atol=ASM_ATOL):
                valid_check = False
                close = torch.isclose(valid, ref_valid, rtol=ASM_RTOL, atol=ASM_ATOL)
                mismatch = int((~close).sum().item())
                total = close.numel()
                print(
                    "run error index:",
                    expert_idx,
                    "cur_valid_m:",
                    cur_valid_m,
                    "max_abs:",
                    cur_max_abs,
                    "mismatch:",
                    mismatch,
                    "total:",
                    total,
                    "ratio:",
                    mismatch / total,
                )
                mismatch_indices = torch.nonzero(~close, as_tuple=False)
                sample_indices = mismatch_indices[:DEBUG_MISMATCH_SAMPLES]
                print("mismatch sample coords/value:")
                for local_m, local_n in sample_indices.tolist():
                    actual_val = valid[local_m, local_n].item()
                    ref_val = ref_valid[local_m, local_n].item()
                    print(
                        "  expert:",
                        expert_idx,
                        "m:",
                        local_m,
                        "n:",
                        local_n,
                        "actual:",
                        actual_val,
                        "ref:",
                        ref_val,
                        "delta:",
                        float(actual_val) - float(ref_val),
                    )
                if compare_tensors is not None:
                    compare_tensors(ref_valid, valid, atol=ASM_ATOL, rtol=ASM_RTOL)
                else:
                    print("aiter.moe_c_golden.compare_tensors is not available")
                break
            max_abs = max(max_abs, cur_max_abs)
        if valid_check:
            print("success deepgemm_masked_w4a8_hipc gemm 精度检查合格")
        else:
            raise AssertionError("deepgemm masked w4a8 hipc precision check failed")

    cost_us = None
    if bench:
        for _ in range(warmup):
            m_grouped_w4a8_gemm_nt_masked_hipc(
                (q_a, a_scale),
                (packed_w, kernel_w_scale),
                out,
                masked_m,
                expected_m_per_group,
            )
        torch.cuda.synchronize()
        cost_us = triton.testing.do_bench(
            lambda: m_grouped_w4a8_gemm_nt_masked_hipc(
                (q_a, a_scale),
                (packed_w, kernel_w_scale),
                out,
                masked_m,
                expected_m_per_group,
            ),
            warmup=warmup,
            rep=iters,
            quantiles=None,
            return_mode="mean",
        ) * 1000

    valid_m = int(masked_m.sum().item())
    tflops = None
    bandwidth = None
    if bench:
        computes = 2 * valid_m * n * k
        tflops = computes / (cost_us / 1e6) / 1e12
        weight_bytes = e * n * k // 2
        data_bytes = valid_m * k + weight_bytes + valid_m * n * out.element_size()
        bandwidth = data_bytes / (cost_us / 1e6) / 1e9
    return {
        "E": e,
        "m": m,
        "n": n,
        "k": k,
        "expected_m": expected_m_per_group,
        "valid_m": valid_m,
        "masked_m": masked_m,
        "dtype": dtype,
        "max_abs": max_abs,
        "cost_us": cost_us,
        "tflops": tflops,
        "bandwidth": bandwidth,
    }


def run_cases(verify: bool = True, warmup: int = 20, iters: int = 100) -> None:
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    m_list = 2048
    n_list = [4096,6144]
    k_list = [6144,2048]
    E = 16
    expected_m_per_group_list = [ 1,2,4,8,16,32,64 ,128, 256, 512, 1024, 2048]
    # expected_m_per_group_list = [64]
    out_dtype = torch.bfloat16
    if len(n_list) != len(k_list):
        raise ValueError(f"n_list and k_list must have the same length, got {len(n_list)} and {len(k_list)}")

    df_E = []
    df_m = []
    df_n = []
    df_k = []
    df_expected_m = []
    df_valid_m = []
    df_per_expert_m_avg = []
    deepgemm_w4a8_hipc_costs = []
    df_tflops = []
    df_bandwidth = []
    total_cases = 0
    passed_cases = 0
    failed_cases = []

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(len(k_list)):
            total_cases += 1
            n = n_list[i]
            k = k_list[i]
            print(f"expected_m_per_group:{expected_m_per_group} n:{n} k:{k} ")
            masked_values = [
                int(expected_m_per_group ) # * random.uniform(0.7, 1.3)
                for _ in range(E)
            ]
            masked_m = torch.tensor(masked_values, device="cuda", dtype=torch.int32)
            try:
                result = run_case_impl(
                    e=E,
                    m=m_list,
                    n=n,
                    k=k,
                    masked_m=masked_m,
                    dtype=out_dtype,
                    expected_m_per_group=expected_m_per_group,
                    bench=True,
                    verify=verify,
                    warmup=warmup,
                    iters=iters,
                )
            except AssertionError:
                if verify:
                    failed_case = {
                        "E": E,
                        "capacity_m": m_list,
                        "n": n,
                        "k": k,
                        "expected_m": expected_m_per_group,
                        "valid_m": int(masked_m.sum().item()),
                    }
                    failed_cases.append(failed_case)
                    print(
                        "\033[31mcase correctness failed: "
                        f"E={failed_case['E']} capacity_m={failed_case['capacity_m']} "
                        f"N={failed_case['n']} K={failed_case['k']} "
                        f"expected_m={failed_case['expected_m']} valid_m={failed_case['valid_m']}"
                        "\033[0m"
                    )
                    print("------------------------------------------------------")
                continue
            if verify:
                passed_cases += 1

            hipc_cost_gemm = result["cost_us"]
            deepgemm_w4a8_hipc_costs.append(hipc_cost_gemm)
            df_E.append(E)
            df_m.append(m_list)
            df_n.append(n)
            df_k.append(k)
            df_expected_m.append(expected_m_per_group)
            df_valid_m.append(result["valid_m"])
            df_per_expert_m_avg.append(result["valid_m"] / E)

            print("valid_m:", result["valid_m"], "masked_m:", result["masked_m"])
            print("------------------------------------------------------")
            print(f"hipc cost: {result['cost_us']:.2f} us, {result['tflops']:.2f} TFlops")
            print(f"hipc bandwidth: {result['bandwidth']:.2f} GB/s")
            print("------------------------------------------------------")

            df_tflops.append(result["tflops"])
            df_bandwidth.append(result["bandwidth"])

    df = pd.DataFrame(
        {
            "E": df_E,
            "capacity_m": df_m,
            "n": df_n,
            "k": df_k,
            "expected_m": df_expected_m,
            "valid_m": df_valid_m,
            "per_expert_m_avg": df_per_expert_m_avg,
            "hipc deepgemm 耗时(us)": deepgemm_w4a8_hipc_costs,
            "hipc deepgemm TFlops": df_tflops,
            "hipc deepgemm Bandwidth(GB/s)": df_bandwidth,
        }
    )
    excel_name = "DeepGEMM_masked_groupgemm_w4a8_hipc.xlsx"
    # df.to_excel(excel_name, index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("------------------------------------------------------")
    if verify:
        if failed_cases:
            print(f"\033[31mcorrectness summary: FAIL {passed_cases}/{total_cases}\033[0m")
            print("\033[31mfailed cases:\033[0m")
            for case in failed_cases:
                print(
                    "\033[31m"
                    f"E={case['E']} capacity_m={case['capacity_m']} "
                    f"N={case['n']} K={case['k']} "
                    f"expected_m={case['expected_m']} valid_m={case['valid_m']}"
                    "\033[0m"
                )
            raise AssertionError(
                f"deepgemm masked w4a8 hipc correctness failed: {len(failed_cases)} of {total_cases} cases failed"
            )
        else:
            print(f"\033[32mcorrectness summary: PASS {passed_cases}/{total_cases}\033[0m")
    else:
        print(f"correctness summary: SKIP 0/{total_cases} (run with --verify-sweep to enable)")
    print("------------------------------------------------------")
    print(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    run_cases(verify=not args.no_verify, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
