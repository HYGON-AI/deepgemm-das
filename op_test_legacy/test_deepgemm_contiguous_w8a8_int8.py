# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import torch
from typing import Type
import pandas as pd
import triton
from deepgemm import m_grouped_i8_gemm_nt_contiguous
from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def align(x, y):
    return ((x + y - 1) // y) * y

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int):
    #每个group实际的M
    actual_ms = [int(expected_m_per_group) for _ in range(num_groups)]
    # contiguous接口要求每个group的M按256对齐，padding行用m_indices=-1标记
    aligned_ms = [align(actual_m, 256) for actual_m in actual_ms]
    #group的M的总和
    m = sum(aligned_ms)
    print("align_m after m : ", m)
    # a 是整体input
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)  
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    ref_d = torch.zeros((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        #表示M属于哪个group id
        m_indices[start:actual_end] = i        #[000...-1-1-1, 111...-1-1]           
        m_indices[actual_end:aligned_end] = -1

        # ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    
    # ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)
    a_int8 = to_int8(a)
    b_int8 = to_int8(b)
    a_scale = torch.randn(m, device='cuda', dtype=torch.float)
    b_scale = torch.randn((num_groups, n), device='cuda', dtype=torch.float)

    group_a = [a_int8, a_scale]
    group_b = [b_int8, b_scale]
    return m, group_a, group_b, m_indices, d, ref_d

def torch_gemm(
    A: torch.Tensor,          # [M, K]
    B: torch.Tensor,          # [E, N, K]
    C: torch.Tensor,          # [M, N]
    scaleA: torch.Tensor,     # [M] or [M, 1]
    scaleB: torch.Tensor,     # [E, N] or [E, N, 1]
    alpha: float,
    beta: float,
    m_index_h: torch.Tensor = None  # [M]，元素为 e_id，-1 表示跳过
) -> torch.Tensor:

    if m_index_h is None:
        raise ValueError('m_index_h is required for contiguous grouped GEMM')

    D = C.clone()
    B_ext = B.shape[0]
    for e_id in range(B_ext):
        rows = m_index_h == e_id
        if not torch.any(rows):
            continue
        out = A[rows].to(torch.float32) @ B[e_id].to(torch.float32).t()
        out = out * scaleA[rows].to(torch.float32).reshape(-1, 1)
        out = out * scaleB[e_id].to(torch.float32).reshape(1, -1)
        if beta != 0:
            out = alpha * out + beta * C[rows].to(torch.float32)
        elif alpha != 1:
            out = alpha * out
        D[rows] = out.to(D.dtype)

    return D


def check_w8a8_precision(
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
        print('W8A8 精度检查: 无有效行，跳过')
        return True

    abs_err = (ref_valid.float() - d_valid.float()).abs()
    ref_abs = ref_valid.float().abs().clamp(min=1e-7)
    rel_err = abs_err / ref_abs
    print(
        f'  W8A8 精度统计: max_abs_err={abs_err.max().item():.6f}, '
        f'mean_abs_err={abs_err.mean().item():.6f}, '
        f'max_rel_err={rel_err.max().item():.6f}, '
        f'mean_rel_err={rel_err.mean().item():.6f}'
    )
    return torch.allclose(ref_valid, d_valid, rtol=rtol, atol=atol)

def test_m_grouped_gemm_contiguous(warmup_iters=1, test_iters=100, acc_check=True) -> None:
    num_groups = 16
    expected_m_per_group_list = [128, 256, 512]
    n_k_list = [(2048, 2048)]
    out_dtype = torch.bfloat16

    for expected_m_per_group in expected_m_per_group_list:
        for n, k in n_k_list:
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                num_groups, expected_m_per_group, n, k
            )
            print("m:{} n:{} k:{}".format(m, n, k))
            a_int8, a_scale = a
            b_int8, b_scale = b
            if acc_check:
                ref_d = torch_gemm(
                    a_int8, b_int8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices
                )
            w = pack_int8_weight_enk_to_w6_low_latency(b_int8)
            b = [w, b_scale]

            for _ in range(warmup_iters):
                m_grouped_i8_gemm_nt_contiguous(a, b, d, m_indices)
            torch.cuda.synchronize()

            if acc_check:
                ok = check_w8a8_precision(ref_d, d, m_indices, rtol=1e-2, atol=5e-2)
                if not ok:
                    print("精度检查不合格！！！")
                    raise AssertionError(
                        f"W8A8 contiguous precision check failed: M={m}, N={n}, K={k}"
                    )
                print("精度检查合格！！！")

            print(
                f"deepGemm contiguous W8A8 精度测试 ({out_dtype}) - M{m}xN{n}xK{k}"
            )


def deepgemm_contiguous_w8a8_benchmark(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    out_dtype: Type[torch.dtype] = torch.bfloat16,
    verify_precision: bool = False,
):
    m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m_per_group, n, k
    )
    a_int8, a_scale = a
    b_int8, b_scale = b
    if verify_precision:
        ref_d = torch_gemm(a_int8, b_int8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices)
    w = pack_int8_weight_enk_to_w6_low_latency(b_int8)
    b = [w, b_scale]

    # warm up
    for _ in range(5):
        m_grouped_i8_gemm_nt_contiguous(a, b, d, m_indices)
    torch.cuda.synchronize()

    asm_cost_us = (
        triton.testing.do_bench(
            lambda: m_grouped_i8_gemm_nt_contiguous(a, b, d, m_indices),
            quantiles=None,
            return_mode="mean",
        )
        * 1000
    )

    if verify_precision:
        ok = check_w8a8_precision(ref_d, d, m_indices, rtol=1e-2, atol=5e-2)
        if not ok:
            print('精度检查不合格！！！')
            raise AssertionError(f'W8A8 contiguous precision check failed: M={m}, N={n}, K={k}')
        else:
            print('精度检查合格')

    computes = 2 * m * n * k
    tflops = computes / (asm_cost_us / 1e6) / 1e12
    data_bytes = m * k + num_groups * n * k + m * n * 2
    data_bytes += m * 4 + num_groups * n * 4  # scales float32
    bandwidth_gbs = data_bytes / (asm_cost_us / 1e6) / 1e9

    valid_m = int((m_indices >= 0).sum().item())
    return asm_cost_us, tflops, bandwidth_gbs, m, valid_m


def main_perf():
    num_groups = 16
    expected_m_per_group_list = [128, 256, 512]
    n_list = [2048]
    k_list = [2048]
    out_dtype = torch.bfloat16

    df_E = []
    df_m = []
    df_n = []
    df_k = []
    df_expected_m = []
    df_valid_m = []
    asm_costs = []
    df_tflops = []
    df_bandwidth = []

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(len(k_list)):
            n, k = n_list[i], k_list[i]
            print(
                "expected_m_per_group:{} n:{} k:{} ".format(
                    expected_m_per_group, n, k
                )
            )
            asm_cost_us, tflops, bandwidth_gbs, m, valid_m = deepgemm_contiguous_w8a8_benchmark(
                num_groups=num_groups,
                expected_m_per_group=expected_m_per_group,
                n=n,
                k=k,
                out_dtype=out_dtype,
                verify_precision=True,
            )

            asm_costs.append(asm_cost_us)
            df_E.append(num_groups)
            df_m.append(m)
            df_n.append(n)
            df_k.append(k)
            df_expected_m.append(expected_m_per_group)
            df_valid_m.append(valid_m)
            df_tflops.append(tflops)
            df_bandwidth.append(bandwidth_gbs)

            print("------------------------------------------------------")
            print(
                "deepgemm contiguous asm cost: {:.2f} us, {:.2f} TFlops".format(
                    asm_cost_us, tflops
                )
            )
            print("deepgemm contiguous asm bandwidth: {:.2f} GB/s".format(bandwidth_gbs))
            print("------------------------------------------------------")

    df = pd.DataFrame(
        {
            "E": df_E,
            "m": df_m,
            "n": df_n,
            "k": df_k,
            "expected_m": df_expected_m,
            "valid_m": df_valid_m,
            "asm deepgemm 耗时(us)": asm_costs,
            "asm deepgemm TFlops": df_tflops,
            "asm deepgemm Bandwidth(GB/s)": df_bandwidth,
        }
    )
    excel_name = "DeepGEMM_contiguous_groupgemm_w8a8.xlsx"
    try:
        df.to_excel(excel_name, index=False)
        print("表格已保存到 {} 文件中。".format(excel_name))
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        csv_name = "DeepGEMM_contiguous_groupgemm_w8a8.csv"
        df.to_csv(csv_name, index=False)
        print("openpyxl 未安装，表格已改存到 {} 文件中。".format(csv_name))


if __name__ == "__main__":
    main_perf()
