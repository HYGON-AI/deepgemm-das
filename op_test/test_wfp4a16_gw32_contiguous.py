import gc
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import triton

from deepgemm import m_grouped_wfp4a16_gemm_nt_contiguous


_BLOCK_M = 128
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


def align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


# EP32 table shapes. The table says INT4, but this WFP4A16 test uses
# packed FP4 E2M1 weights with uint8 E8M0 scales.
EXPERTS_LIST = [28]
M_PER_EXPERT_LIST = list(range(128, 2049, 128))
NK_LIST = [
    (6144, 3584),  # moe_gate_up: A [28,293,3584], B [28,3584,6144], C [28,293,6144]
    (3584, 3072),  # moe_down:    A [28,293,3072], B [28,3072,3584], C [28,293,3584]
]
DTYPE = torch.bfloat16
ACTUAL_MODE = "fixed"
WARMUP_ITERS = 10
ACC_CHECK = True
CSV_PATH = Path(__file__).parent / "test_wfp4a16_gw32_contiguous.csv"


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


def make_actual_ms(experts: int, m_per_expert: int, mode: str) -> List[int]:
    if mode == "fixed":
        return [m_per_expert for _ in range(experts)]

    low = max(0, int(m_per_expert * 0.7))
    high = max(low, int(math.ceil(m_per_expert * 1.3)))
    actual_ms = torch.randint(low, high + 1, (experts,), device="cpu", dtype=torch.int32).tolist()
    actual_ms = [min(m_per_expert, int(v)) for v in actual_ms]
    if experts > 0:
        actual_ms[0] = m_per_expert
    if experts > 1:
        actual_ms[-1] = max(1, m_per_expert // 2)
    return actual_ms


def build_m_indices(actual_ms: List[int], block_m: int) -> torch.Tensor:
    chunks = []
    for expert, actual_m in enumerate(actual_ms):
        aligned_m = align(actual_m, block_m) if actual_m > 0 else 0
        if aligned_m == 0:
            continue
        values = torch.empty(aligned_m, device="cuda", dtype=torch.int32)
        values[:actual_m] = expert
        values[actual_m:] = -1
        chunks.append(values)
    if not chunks:
        return torch.empty(0, device="cuda", dtype=torch.int32)
    return torch.cat(chunks).contiguous()


def generate_m_grouped_contiguous_wfp4a16(
    experts: int,
    m_per_expert: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    actual_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    actual_ms = make_actual_ms(experts, m_per_expert, actual_mode)
    m_indices = build_m_indices(actual_ms, _BLOCK_M)
    total_m = int(m_indices.numel())
    if total_m == 0:
        raise ValueError("test case produced zero M")

    matrix_a = torch.empty((total_m, k), device="cuda", dtype=dtype).normal_(0.0, 1.0 / math.sqrt(k))
    matrix_b, matrix_b_scale = make_fp4_weight(experts, n, k)
    matrix_c = torch.zeros((total_m, n), device="cuda", dtype=dtype)
    return matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, actual_ms


def reference_grouped_gemm_wfp4a16_contiguous(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    total_m = matrix_a.shape[0]
    experts, size_n, _ = matrix_b.shape
    ref = torch.zeros(total_m, size_n, device=matrix_a.device, dtype=output_dtype)
    for expert in range(experts):
        rows = m_indices == expert
        if not torch.any(rows):
            continue
        weight = dequant_wfp4a16_weight_one_expert(matrix_b[expert], matrix_b_scale[expert]).to(torch.float32)
        ref[rows] = (matrix_a[rows].to(torch.float32) @ weight.t()).to(output_dtype)
        del weight
    return ref


def check_wfp4a16_contiguous_precision(
    ref: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
    rtol: float = 2e-2,
    atol: float = 2e-1,
) -> Tuple[bool, float, float, float, float]:
    valid_rows = m_indices >= 0
    if not torch.any(valid_rows):
        print("  WFP4A16 precision: no valid rows, skipped")
        return True, 0.0, 0.0, 0.0, 0.0

    ref_valid = ref[valid_rows]
    out_valid = out[valid_rows]
    diff = (ref_valid.float() - out_valid.float()).abs()
    rel = diff / ref_valid.float().abs().clamp(min=1e-7)
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    max_rel_err = rel.max().item()
    mean_rel_err = rel.mean().item()
    correct = torch.allclose(ref_valid, out_valid, rtol=rtol, atol=atol)

    if not correct:
        mask = diff > (atol + rtol * out_valid.float().abs())
        mismatch = torch.nonzero(mask)
        print("error WFP4A16 contiguous precision")
        print(f"mismatch elements: {mismatch.shape[0]}")
        for idx in mismatch[:20]:
            row, col = idx.tolist()
            print(
                f"[{row}, {col}] ref={ref_valid[row, col].item():.6f}, "
                f"out={out_valid[row, col].item():.6f}, diff={diff[row, col].item():.6f}"
            )

    invalid_rows = m_indices < 0
    if torch.any(invalid_rows) and torch.count_nonzero(out[invalid_rows]).item() != 0:
        correct = False
        print("error WFP4A16 contiguous: padding rows were written")

    print(
        f"  WFP4A16 precision: max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}, "
        f"max_rel_err={max_rel_err:.6f}, mean_rel_err={mean_rel_err:.6f}"
    )
    return correct, max_abs_err, mean_abs_err, max_rel_err, mean_rel_err


def benchmark_contiguous_wfp4a16(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    matrix_c: torch.Tensor,
    m_indices: torch.Tensor,
    warmup_iters: int,
) -> float:
    for _ in range(warmup_iters):
        m_grouped_wfp4a16_gemm_nt_contiguous(matrix_a, (matrix_b, matrix_b_scale), matrix_c, m_indices)
    torch.cuda.synchronize()
    return triton.testing.do_bench(
        lambda: m_grouped_wfp4a16_gemm_nt_contiguous(matrix_a, (matrix_b, matrix_b_scale), matrix_c, m_indices),
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
    m_per_expert: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    actual_mode: str,
    warmup_iters: int,
    acc_check: bool,
) -> dict:
    matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, actual_ms = generate_m_grouped_contiguous_wfp4a16(
        experts, m_per_expert, n, k, dtype, actual_mode
    )
    total_m = int(m_indices.numel())
    valid_m = int((m_indices >= 0).sum().item())
    active_experts = int(torch.unique(m_indices[m_indices >= 0]).numel())
    print(
        f"E:{experts} M_per_E:{m_per_expert} total_m:{total_m} valid_m:{valid_m} "
        f"N:{n} K:{k} actual_mode:{actual_mode}"
    )
    print("actual_ms:", actual_ms)

    m_grouped_wfp4a16_gemm_nt_contiguous(matrix_a, (matrix_b, matrix_b_scale), matrix_c, m_indices)
    torch.cuda.synchronize()

    status = "SKIPPED"
    max_abs_err = mean_abs_err = max_rel_err = mean_rel_err = 0.0
    if acc_check:
        ref = reference_grouped_gemm_wfp4a16_contiguous(matrix_a, matrix_b, matrix_b_scale, m_indices, dtype)
        ok, max_abs_err, mean_abs_err, max_rel_err, mean_rel_err = check_wfp4a16_contiguous_precision(
            ref, matrix_c, m_indices
        )
        status = "PASS" if ok else "FAILED"
        print(f"[{status}] deepgemm_contiguous_wfp4a16")
        del ref

    latency_us = benchmark_contiguous_wfp4a16(
        matrix_a, matrix_b, matrix_b_scale, matrix_c, m_indices, warmup_iters
    )
    tflops, bandwidth = logical_metrics(latency_us, valid_m, active_experts, n, k)

    print("------------------------------------------------------")
    print(f"deepgemm WFP4A16 contiguous cost: {latency_us:.2f} us, {tflops:.2f} TFlops")
    print(f"deepgemm WFP4A16 contiguous bandwidth: {bandwidth:.2f} GB/s")
    print("------------------------------------------------------")

    row = {
        "experts": experts,
        "m_per_expert": m_per_expert,
        "total_m": total_m,
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

    del matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c
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
        for m_per_expert in M_PER_EXPERT_LIST:
            for n, k in NK_LIST:
                row = run_case(
                    experts,
                    m_per_expert,
                    n,
                    k,
                    DTYPE,
                    ACTUAL_MODE,
                    WARMUP_ITERS,
                    ACC_CHECK,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"WFP4A16 contiguous results saved to {CSV_PATH}")
    if ACC_CHECK and any(row.get("status") == "FAILED" for row in rows):
        raise AssertionError("WFP4A16 contiguous has failed cases")


if __name__ == "__main__":
    main()
