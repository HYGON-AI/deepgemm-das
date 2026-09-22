import csv
from pathlib import Path

import torch
from deepgemm import (
    m_grouped_w4a8_gemm_nt_contiguous_hipc,
    pack_w4a8_moe_hipc_weight,
)
from deepgemm.group_gemm_weight_pack import pack_w4a8_weight_k_contiguous

from contiguous_cases import CASES, correctness_check


def _bench(fn, repeat=9, iters=30):
    values = []
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / iters)

    values.sort()
    return values[0], values[len(values) // 2]


def _bytes_per_case(experts: int, size_n: int, size_k: int, total_m: int) -> int:
    # a: int8 = 1 byte, weight: int4 = 0.5 byte, output: bf16 = 2 bytes
    return total_m * size_k + experts * size_n * size_k // 2 + total_m * size_n * 2


def _tflops(size_n: int, size_k: int, total_m: int, latency_us: float) -> float:
    flops = 2 * total_m * size_n * size_k
    return flops / (latency_us * 1e-6) / 1e12


def _align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _reference_matmul(q_a, w_signed, a_scale, w_scale, m_indices, experts, out_dtype):
    size_m, size_n = q_a.shape[0], w_signed.shape[1]
    ref = torch.zeros((size_m, size_n), device=q_a.device, dtype=out_dtype)
    for expert in range(experts):
        rows = m_indices == expert
        if not bool(rows.any().item()):
            continue
        acc = q_a[rows].float() @ w_signed[expert].float().t()
        acc = acc * a_scale[rows].float() * w_scale[expert].float().t()
        ref[rows] = acc.to(out_dtype)
    return ref


def run_case(name, experts, size_n, size_k, total_m, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out_dtype = torch.bfloat16

    base_m = total_m // experts
    remainder_m = total_m % experts
    actual_ms = [base_m + (1 if expert < remainder_m else 0) for expert in range(experts)]
    aligned_ms = [_align(actual_m, 128) for actual_m in actual_ms]
    size_m = sum(aligned_ms)

    q_a = torch.randint(-4, 5, (size_m, size_k), device="cuda", dtype=torch.int8).contiguous()
    a_scale = torch.rand((size_m, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)
    weight_u4 = torch.randint(4, 12, (experts, size_n, size_k), device="cuda", dtype=torch.int8)
    w_scale = torch.rand((experts, size_n, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)

    w_signed = torch.where(weight_u4 > 7, weight_u4 - 16, weight_u4)

    m_indices = torch.empty((size_m,), device="cuda", dtype=torch.int32)
    start = 0
    for expert, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        end = start + aligned_m
        m_indices[start:actual_end] = expert
        m_indices[actual_end:end] = -1
        start = end

    output = torch.empty((size_m, size_n), device="cuda", dtype=out_dtype)

    ref = None
    if correctness_check:
        ref = _reference_matmul(q_a, w_signed, a_scale, w_scale, m_indices, experts, out_dtype)

    framework_packed_w = pack_w4a8_weight_k_contiguous(weight_u4)
    packed_weight = pack_w4a8_moe_hipc_weight(framework_packed_w)

    m_grouped_w4a8_gemm_nt_contiguous_hipc(
        (q_a, a_scale),
        (packed_weight, w_scale),
        output,
        m_indices,
    )
    torch.cuda.synchronize()

    max_err = 0.0
    status = "SKIPPED"
    if correctness_check:
        valid = (m_indices >= 0).unsqueeze(1).expand_as(output)
        ref_valid = ref.masked_select(valid)
        out_valid = output.masked_select(valid)
        max_err = (ref_valid.float() - out_valid.float()).abs().max().item()
        if not torch.allclose(out_valid, ref_valid, rtol=1e-2, atol=1e-2):
            raise AssertionError(f"{name} rows mismatch: max_err={max_err}")

    best_us, median_us = _bench(
        lambda: m_grouped_w4a8_gemm_nt_contiguous_hipc(
            (q_a, a_scale),
            (packed_weight, w_scale),
            output,
            m_indices,
        )
    )
    bytes_moved = _bytes_per_case(experts, size_n, size_k, total_m)
    tflops = _tflops(size_n, size_k, total_m, best_us)
    bandwidth = bytes_moved / (best_us * 1e-6) / 1e9


    print("------------------------------------------------------")
    print("deepgemm test cases: e: {} n: {} k: {} total_m: {}".format(experts, size_n, size_k, total_m))
    print("deepgemm cost: {:.2f} us, {:.2f} TFlops".format(median_us, tflops))
    print("deepgemm bandwidth: {:.2f} GB/s".format(bandwidth))
    print("------------------------------------------------------")
    return {
        "case": name,
        "E": experts,
        "N": size_n,
        "K": size_k,
        "total_m": total_m,
        "aligned_m": size_m,
        "per_expert_m_avg": total_m / experts,
        "best_us": f"{best_us:.2f}",
        "median_us": f"{median_us:.2f}",
        "tflops": f"{tflops:.2f}",
        "algo_bw_GBps": f"{bandwidth:.2f}",
        "max_err": f"{max_err:.4f}",
        "status": status,
    }


def main():
    rows = []
    for idx, case in enumerate(CASES):
        row = run_case(*case, seed=20260811 + idx)
        rows.append(row)


    out_path = Path(__file__).parent / "test_deepgemm_contiguous_w4a8_hipc.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
