import random
from pathlib import Path

import pandas as pd
import torch
import triton

from deepgemm import m_grouped_fp8_gemm_nt_contiguous, marlin_fp8_contiguous_weight


def align(x, y):
    return ((x + y - 1) // y) * y


def to_fp8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.float8_e4m3fn)


def check_fp8_precision(
    ref_d: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 5e-2,
) -> bool:
    ref_d_cpu = ref_d.detach().cpu()
    d_cpu = d.detach().cpu()
    m_indices_cpu = m_indices.detach().cpu()

    valid_mask = (m_indices_cpu >= 0).unsqueeze(1).expand_as(ref_d_cpu)
    ref_valid = ref_d_cpu.masked_select(valid_mask)
    d_valid = d_cpu.masked_select(valid_mask)
    if ref_valid.numel() == 0:
        print("FP8 CPU 精度检查: 无有效行，跳过")
        return True
    abs_err = (ref_valid.float() - d_valid.float()).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()
    ref_abs = ref_valid.float().abs().clamp(min=1e-7)
    rel_err = abs_err / ref_abs
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()
    print(
        f"  FP8 CPU 精度统计: max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}, "
        f"max_rel_err={max_rel_err:.6f}, mean_rel_err={mean_rel_err:.6f}"
    )
    return torch.allclose(ref_valid, d_valid, rtol=rtol, atol=atol)


def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int):
    actual_ms = [int(expected_m_per_group) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, 256) for actual_m in actual_ms]
    m = sum(aligned_ms)
    print("align_m after m : ", m)

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        start = aligned_end

    a_f8 = to_fp8(a)
    b_f8 = to_fp8(b)
    a_scale = torch.rand(m, device="cuda", dtype=torch.float) * 1.5 + 0.5
    b_scale = torch.rand((num_groups, n), device="cuda", dtype=torch.float) * 1.5 + 0.5
    return m, [a_f8, a_scale], [b_f8, b_scale], m_indices, d, ref_d


def torch_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scaleA: torch.Tensor,
    scaleB: torch.Tensor,
    alpha: float,
    beta: float,
    m_index_h: torch.Tensor = None,
) -> torch.Tensor:
    A_cpu = A.detach().cpu().to(torch.float32)
    B_cpu = B.detach().cpu().to(torch.float32)
    scaleA_cpu = scaleA.detach().cpu().to(torch.float32)
    scaleB_cpu = scaleB.detach().cpu().to(torch.float32)
    m_index_cpu = m_index_h.detach().cpu().to(torch.long)

    M, K = A_cpu.shape
    B_ext, N, _ = B_cpu.shape
    D_cpu = torch.zeros((M, N), device="cpu", dtype=C.dtype)
    C_cpu = C.detach().cpu().to(torch.float32) if beta != 0.0 else None

    for e_id in range(B_ext):
        row_idx = torch.nonzero(m_index_cpu == e_id, as_tuple=False).flatten()
        if row_idx.numel() == 0:
            continue

        out = A_cpu.index_select(0, row_idx).matmul(B_cpu[e_id].transpose(0, 1))
        out = out * scaleA_cpu.index_select(0, row_idx).view(-1, 1)
        out = out * scaleB_cpu[e_id].view(1, -1)
        if alpha != 1.0:
            out = out * alpha
        if C_cpu is not None:
            out = out + beta * C_cpu.index_select(0, row_idx)
        D_cpu[row_idx] = out.to(D_cpu.dtype)

    return D_cpu


def test_m_grouped_prefill_down_fp8_contiguous(warmup_iters=10, acc_check=True) -> None:
    num_groups = 32
    expected_m_per_group_list = [1184]
    n_list = [6144]
    k_list = [2048]
    rows = []
    for k_val in k_list:
        for n_val in n_list:
            for me in expected_m_per_group_list:
                expected_m_per_group, n, k = me, n_val, k_val
                m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                    num_groups, expected_m_per_group, n, k
                )
                print("m:{} n:{} k:{} (prefill-down fp8)".format(m, n, k))
                a_f8, a_scale = a
                b_f8, b_scale = b
                if acc_check:
                    ref_d = torch_gemm(
                        a_f8, b_f8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices
                    )

                w = marlin_fp8_contiguous_weight(b_f8)
                b = [w, b_scale]

                for _ in range(warmup_iters):
                    m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices)
                torch.cuda.synchronize()

                if acc_check:
                    ok = check_fp8_precision(ref_d, d, m_indices, rtol=1e-2, atol=5e-2)
                    status = "PASS" if ok else "FAILED"
                    print(f"[{status}] " + ("FP8 精度检查合格！！！" if ok else "FP8 精度检查不合格！！！"))

                elapsed_time = (
                    triton.testing.do_bench(
                        lambda: m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices),
                        quantiles=None,
                        return_mode="mean",
                    )
                    * 1000
                )
                valid_m = int((m_indices >= 0).sum().item())
                computes = 2 * valid_m * n * k
                tflops = computes / (elapsed_time / 1e6) / 1e12
                weight_bytes = num_groups * n * k // 2
                data_bytes = valid_m * k + weight_bytes + valid_m * n * d.element_size()
                bandwidth = data_bytes / (elapsed_time / 1e6) / 1e9
                print(
                    f"deepGemm prefill-down FP8 - M{m}xN{n}xK{k}, "
                    f"平均耗时: {elapsed_time:.2f} us\n"
                )
                row = {
                    "num_groups": num_groups,
                    "expected_m_per_group": expected_m_per_group,
                    "m": m,
                    "valid_m": int((m_indices >= 0).sum().item()),
                    "n": n,
                    "k": k,
                    "latency_us": elapsed_time,
                    "tflops": tflops,
                    "bandwidth": bandwidth,
                    "status": status,
                }
                if acc_check:
                    row["status"] = status
                rows.append(row)
                pd.DataFrame(rows).to_csv(Path(__file__).parent / "test_deepgemm_contiguous_w8a8_fp8.csv", index=False)

    if acc_check and any(row.get("status") == "FAILED" for row in rows):
        raise AssertionError("FP8 contiguous has failed cases")


if __name__ == "__main__":
    test_m_grouped_prefill_down_fp8_contiguous(acc_check=True)
