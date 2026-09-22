import gc
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import triton

from deepgemm.m_group_gemm import grouped_gemm_wfp4a16_nt_masked_entry


_FP4_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    device="cuda",
    dtype=torch.float32,
)


# EP32 table shapes. The table says INT4, but this WFP4A16 test uses
# packed FP4 E2M1 weights with uint8 E8M0 scales.
EXPERTS_LIST = [28]
M_LIST = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 150, 164, 172, 196, 228, 248, 256, 274, 283, 294]
NK_LIST = [
    (6144, 3584),  # moe_gate_up: A [28,293,3584], B [28,3584,6144], C [28,293,6144]
    (3584, 3072),  # moe_down:    A [28,293,3072], B [28,3072,3584], C [28,293,3584]
]
DTYPE = torch.bfloat16
ACTUAL_MODE = "fixed"
WARMUP_ITERS = 10
ACC_CHECK = True
CSV_PATH = Path(__file__).parent / "test_wfp4a16_gw32_masked.csv"


def pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 1::2] << 4).bitwise_or(codes[..., ::2]).contiguous()


def make_fp4_weight(experts: int, n: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    qweight = torch.randint(0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8)
    weight_scale = torch.randint(121, 126, (experts, n, k // 32), device="cuda", dtype=torch.uint8)
    return qweight, weight_scale


def e8m0_to_float(scales: torch.Tensor) -> torch.Tensor:
    return (scales.to(torch.int32) << 23).view(torch.float32)


def dequant_wfp4a16_weight_one_expert(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    codes = torch.empty(
        (qweight.shape[0], qweight.shape[1] * 2),
        device=qweight.device,
        dtype=torch.long,
    )
    codes[:, 0::2] = (qweight & 0x0F).to(torch.long)
    codes[:, 1::2] = (qweight >> 4).to(torch.long)
    values = _FP4_E2M1_VALUES[codes]
    scale_f = e8m0_to_float(scales).repeat_interleave(32, dim=-1)
    return values * scale_f[:, : values.shape[1]]


def make_actual_tokens(experts: int, max_tokens: int, mode: str) -> torch.Tensor:
    if mode == "fixed":
        return torch.full((experts,), max_tokens, device="cuda", dtype=torch.int32)

    low = max(0, int(max_tokens * 0.7))
    high = max(low, min(max_tokens, int(math.ceil(max_tokens * 1.3))))
    actual_tokens = torch.randint(low, high + 1, (experts,), device="cuda", dtype=torch.int32)
    if experts > 0:
        actual_tokens[0] = max_tokens
    if experts > 1:
        actual_tokens[-1] = max(1, max_tokens // 2)
    return actual_tokens


def generate_m_grouped_masked_wfp4a16(
    experts: int,
    max_tokens: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    actual_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix_a = torch.empty((experts, max_tokens, k), device="cuda", dtype=dtype).normal_(
        0.0, 1.0 / math.sqrt(k)
    )
    matrix_b, matrix_b_scale = make_fp4_weight(experts, n, k)
    actual_tokens = make_actual_tokens(experts, max_tokens, actual_mode)
    matrix_c = torch.zeros((experts, max_tokens, n), device="cuda", dtype=dtype)
    return matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c


def reference_grouped_gemm_wfp4a16_masked(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    actual_tokens: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    experts, max_tokens, _ = matrix_a.shape
    size_n = matrix_b.shape[1]
    ref = torch.zeros(experts, max_tokens, size_n, device=matrix_a.device, dtype=output_dtype)
    for expert in range(experts):
        valid_m = int(actual_tokens[expert].item())
        if valid_m <= 0:
            continue
        weight = dequant_wfp4a16_weight_one_expert(matrix_b[expert], matrix_b_scale[expert]).to(torch.float32)
        ref[expert, :valid_m] = (matrix_a[expert, :valid_m].to(torch.float32) @ weight.t()).to(output_dtype)
        del weight
    return ref


def check_wfp4a16_masked_precision(
    ref: torch.Tensor,
    out: torch.Tensor,
    actual_tokens: torch.Tensor,
    rtol: float = 2e-2,
    atol: float = 2e-1,
) -> Tuple[bool, float, float, float, float]:
    correct = True
    max_abs_err = 0.0
    max_rel_err = 0.0
    sum_abs_err = 0.0
    sum_rel_err = 0.0
    elem_count = 0

    for expert in range(int(actual_tokens.numel())):
        valid_m = int(actual_tokens[expert].item())
        if valid_m > 0:
            ref_valid = ref[expert, :valid_m]
            out_valid = out[expert, :valid_m]
            diff = (ref_valid.float() - out_valid.float()).abs()
            rel = diff / ref_valid.float().abs().clamp(min=1e-7)
            max_abs_err = max(max_abs_err, diff.max().item())
            max_rel_err = max(max_rel_err, rel.max().item())
            sum_abs_err += diff.sum().item()
            sum_rel_err += rel.sum().item()
            elem_count += diff.numel()
            if not torch.allclose(ref_valid, out_valid, rtol=rtol, atol=atol):
                correct = False
                mask = diff > (atol + rtol * out_valid.float().abs())
                mismatch = torch.nonzero(mask)
                print(f"error WFP4A16 masked expert={expert}, valid_m={valid_m}")
                print(f"mismatch elements: {mismatch.shape[0]}")
                for idx in mismatch[:20]:
                    m_idx, n_idx = idx.tolist()
                    print(
                        f"[{m_idx}, {n_idx}] ref={ref_valid[m_idx, n_idx].item():.6f}, "
                        f"out={out_valid[m_idx, n_idx].item():.6f}, diff={diff[m_idx, n_idx].item():.6f}"
                    )
                break
        if valid_m < out.shape[1] and torch.count_nonzero(out[expert, valid_m:]).item() != 0:
            correct = False
            print(f"error WFP4A16 masked expert={expert}: inactive rows were written")
            break

    mean_abs_err = sum_abs_err / elem_count if elem_count else 0.0
    mean_rel_err = sum_rel_err / elem_count if elem_count else 0.0
    print(
        f"  WFP4A16 precision: max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}, "
        f"max_rel_err={max_rel_err:.6f}, mean_rel_err={mean_rel_err:.6f}"
    )
    return correct, max_abs_err, mean_abs_err, max_rel_err, mean_rel_err


def benchmark_masked_wfp4a16(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    matrix_c: torch.Tensor,
    actual_tokens: torch.Tensor,
    max_tokens: int,
    warmup_iters: int,
) -> float:
    for _ in range(warmup_iters):
        grouped_gemm_wfp4a16_nt_masked_entry(
            matrix_a, matrix_b, matrix_b_scale, matrix_c, actual_tokens, max_tokens
        )
    torch.cuda.synchronize()
    return triton.testing.do_bench(
        lambda: grouped_gemm_wfp4a16_nt_masked_entry(
            matrix_a, matrix_b, matrix_b_scale, matrix_c, actual_tokens, max_tokens
        ),
        quantiles=None,
        return_mode="mean",
    ) * 1000


def logical_metrics(latency_us: float, valid_m: int, active_experts: int, n: int, k: int) -> Tuple[float, float]:
    computes = 2 * valid_m * n * k
    tflops = computes / (latency_us / 1e6) / 1e12
    data_bytes = valid_m * k * 2 + active_experts * n * (k // 2) + active_experts * n * (k // 32) + valid_m * n * 2
    bandwidth = data_bytes / (latency_us / 1e6) / 1e9
    return tflops, bandwidth


def run_case(
    experts: int,
    max_tokens: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    actual_mode: str,
    warmup_iters: int,
    acc_check: bool,
) -> dict:
    matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c = generate_m_grouped_masked_wfp4a16(
        experts, max_tokens, n, k, dtype, actual_mode
    )
    print(f"E:{experts} M:{max_tokens} N:{n} K:{k} actual_mode:{actual_mode}")
    print("actual_tokens:", actual_tokens)

    grouped_gemm_wfp4a16_nt_masked_entry(
        matrix_a, matrix_b, matrix_b_scale, matrix_c, actual_tokens, max_tokens
    )
    torch.cuda.synchronize()

    status = "SKIPPED"
    max_abs_err = mean_abs_err = max_rel_err = mean_rel_err = 0.0
    if acc_check:
        ref = reference_grouped_gemm_wfp4a16_masked(
            matrix_a, matrix_b, matrix_b_scale, actual_tokens, dtype
        )
        ok, max_abs_err, mean_abs_err, max_rel_err, mean_rel_err = check_wfp4a16_masked_precision(
            ref, matrix_c, actual_tokens
        )
        status = "PASS" if ok else "FAILED"
        print(f"[{status}] deepgemm_masked_wfp4a16")
        del ref

    latency_us = benchmark_masked_wfp4a16(
        matrix_a, matrix_b, matrix_b_scale, matrix_c, actual_tokens, max_tokens, warmup_iters
    )
    valid_m = int(actual_tokens.sum().item())
    active_experts = int((actual_tokens > 0).sum().item())
    tflops, bandwidth = logical_metrics(latency_us, valid_m, active_experts, n, k)

    print("------------------------------------------------------")
    print(f"deepgemm WFP4A16 masked cost: {latency_us:.2f} us, {tflops:.2f} TFlops")
    print(f"deepgemm WFP4A16 masked bandwidth: {bandwidth:.2f} GB/s")
    print("------------------------------------------------------")

    row = {
        "experts": experts,
        "max_m": max_tokens,
        "valid_m": valid_m,
        "active_experts": active_experts,
        "n": n,
        "k": k,
        "dtype": str(dtype).replace("torch.", ""),
        "actual_mode": actual_mode,
        "latency_us": latency_us,
        "tflops": tflops,
        "bandwidth_gbps": bandwidth,
        "status": status,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "max_rel_err": max_rel_err,
        "mean_rel_err": mean_rel_err,
    }

    del matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    seed = 2026
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rows = []
    for experts in EXPERTS_LIST:
        for max_tokens in M_LIST:
            for n, k in NK_LIST:
                row = run_case(
                    experts,
                    max_tokens,
                    n,
                    k,
                    DTYPE,
                    ACTUAL_MODE,
                    WARMUP_ITERS,
                    ACC_CHECK,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"WFP4A16 masked results saved to {CSV_PATH}")
    if ACC_CHECK and any(row.get("status") == "FAILED" for row in rows):
        raise AssertionError("WFP4A16 masked has failed cases")


if __name__ == "__main__":
    main()
