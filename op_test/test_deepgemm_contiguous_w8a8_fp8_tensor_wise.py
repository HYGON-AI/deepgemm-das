import random
from pathlib import Path

import pandas as pd
import torch
from deepgemm import marlin_fp8_contiguous_weight
from deepgemm.m_group_gemm_nt_contiguous import m_grouped_fp8_gemm_nt_contiguous_tensor_wise


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
    valid_mask = (m_indices >= 0).unsqueeze(1).expand_as(ref_d)
    ref_valid = ref_d.masked_select(valid_mask)
    d_valid = d.masked_select(valid_mask)
    if ref_valid.numel() == 0:
        print("FP8 精度检查: 无有效行，跳过")
        return True
    abs_err = (ref_valid.float() - d_valid.float()).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()
    ref_abs = ref_valid.float().abs().clamp(min=1e-7)
    rel_err = abs_err / ref_abs
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()
    print(
        f"  FP8 精度统计: max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}, "
        f"max_rel_err={max_rel_err:.6f}, mean_rel_err={mean_rel_err:.6f}"
    )
    return torch.allclose(ref_valid, d_valid, rtol=rtol, atol=atol)


def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int):
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
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
    b_scale = torch.rand((num_groups, 1), device="cuda", dtype=torch.float) * 1.5 + 0.5
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
    M, K = A.shape
    B_ext, N, _ = B.shape
    D = C.clone()
    for m_idx in range(M):
        e_id = int(m_index_h[m_idx].item())
        if e_id == -1:
            continue
        if e_id < 0 or e_id >= B_ext:
            continue
        a_row = A[m_idx, :].to(torch.float32)
        sa = scaleA[m_idx].to(torch.float32)
        for n_idx in range(N):
            b_row = B[e_id, n_idx, :].to(torch.float32)
            sb = scaleB[e_id, 0].to(torch.float32)
            s = torch.dot(a_row, b_row) * sa * sb
            val = alpha * s + beta * C[m_idx, n_idx].to(torch.float32)
            D[m_idx, n_idx] = val.to(D.dtype)
    return D


def test_m_grouped_prefill_down_fp8_contiguous(warmup_iters=10, acc_check=True) -> None:
    num_groups = 4
    expected_m_per_group_list = [64, 128]
    n_list = [512, 1024]
    k_list = [128, 1024]
    rows = []
    for k_val in k_list:
        for n_val in n_list:
            for me in expected_m_per_group_list:
                expected_m_per_group, n, k = me, n_val, k_val
                m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                    num_groups, expected_m_per_group, n, k
                )
                print("m:{} n:{} k:{} (prefill-down fp8 tensor-wise)".format(m, n, k))
                a_f8, a_scale = a
                b_f8, b_scale = b
                if acc_check:
                    ref_d = torch_gemm(
                        a_f8, b_f8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices
                    )

                w = marlin_fp8_contiguous_weight(b_f8)
                b = [w, b_scale]

                for _ in range(warmup_iters):
                    m_grouped_fp8_gemm_nt_contiguous_tensor_wise(a, b, d, m_indices)
                torch.cuda.synchronize()

                if acc_check:
                    ok = check_fp8_precision(ref_d, d, m_indices, rtol=1e-2, atol=5e-2)
                    status = "PASS" if ok else "FAILED"
                    print(f"[{status}] " + ("FP8 精度检查合格！！！" if ok else "FP8 精度检查不合格！！！"))

                row = {
                    "num_groups": num_groups,
                    "expected_m_per_group": expected_m_per_group,
                    "m": m,
                    "valid_m": int((m_indices >= 0).sum().item()),
                    "n": n,
                    "k": k,
                }
                if acc_check:
                    row["status"] = status
                rows.append(row)
                pd.DataFrame(rows).to_csv(Path(__file__).parent / "test_deepgemm_contiguous_w8a8_fp8_tensor_wise.csv", index=False)

    if acc_check and any(row.get("status") == "FAILED" for row in rows):
        raise AssertionError("FP8 contiguous tensor-wise has failed cases")


if __name__ == "__main__":
    test_m_grouped_prefill_down_fp8_contiguous(acc_check=True)
