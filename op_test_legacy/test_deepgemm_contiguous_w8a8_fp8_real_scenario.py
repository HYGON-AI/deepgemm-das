# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import random

import torch
import triton

from deepgemm import m_grouped_fp8_gemm_nt_contiguous, op
from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency
from deepgemm_marlin_quant import weight8bit_nt_kpack2_marlin


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
    """FP8 精度检查：仅对有效行（m_indices >= 0）比较，并输出误差统计。"""
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
    # actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    actual_ms = [int(expected_m_per_group* 1)  for _ in range(num_groups)]
    # actual_ms = [61, 65, 58, 73, 80, 67, 62, 77, 54, 71, 76, 55, 57, 74, 62, 59, 57, 46, 52, 54, 70, 48, 55, 47, 52, 74, 77, 47, 79, 50, 78, 51]


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


def _print_bench(label, m, n, k, elapsed_us):
    tflops = 2 * m * n * k / (elapsed_us / 1e6) / 1e12
    print(f"{label} - M{m}xN{n}xK{k}, 平均耗时: {elapsed_us:.2f} us, TFLOPS: {tflops:.2f}")


def _generate_seeded_case(seed: int, num_groups: int, expected_m_per_group: int, n: int, k: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k)


def test_m_grouped_gemm_contiguous(warmup_iters=10, acc_check=True) -> None:
    num_groups = 32
    expected_m_per_group_list = [ 64,128,256,512,1024]
    n_list = [512, 1024 , 2048]
    k_list = [128, 4096]
    # num_groups = 4
    # expected_m_per_group_list = [64, 128]
    # n_list = [512, 1024]
    # k_list = [128, 1024]
    # num_groups = 16
    # expected_m_per_group_list = [64, 512 , 8192]
    # n_list = [128,1024,2048]
    # k_list = [128,4096]
    for k_val in k_list:
        for n_val in n_list:
            for me in expected_m_per_group_list:
                expected_m_per_group, n, k = me, n_val, k_val
                case_seed = 20260723 + k * 100000 + n * 100 + expected_m_per_group
                m, a, b, m_indices, d, _ = _generate_seeded_case(
                    case_seed, num_groups, expected_m_per_group, n, k
                )
                print("m:{} n:{} k:{} (fp8)".format(m, n, k))
                b_f8, b_scale = b
                b_packed = [pack_int8_weight_enk_to_w6_low_latency(b_f8), b_scale]

                m_grouped_fp8_gemm_nt_contiguous(a, b_packed, d, m_indices)
                torch.cuda.synchronize()

                elapsed_hip = triton.testing.do_bench(
                    lambda: m_grouped_fp8_gemm_nt_contiguous(
                        a, b_packed, d, m_indices
                    ),
                    quantiles=None,
                    return_mode="mean",
                ) * 1000
                del a, b, b_f8, b_scale, b_packed, m_indices
                torch.cuda.empty_cache()

                m_asm, a_asm, b_asm, m_indices_asm, ref_d, _ = _generate_seeded_case(
                    case_seed, num_groups, expected_m_per_group, n, k
                )
                assert m_asm == m
                a_f8, a_scale = a_asm
                b_f8, b_scale = b_asm
                w_asm = weight8bit_nt_kpack2_marlin(b_f8)
                op.m_grouped_fp8_gemm_nt_contiguous(
                    a_f8, w_asm, ref_d, a_scale, b_scale, m_indices_asm, 1000
                )
                torch.cuda.synchronize()

                if acc_check:
                    ok = check_fp8_precision(ref_d, d, m_indices_asm, rtol=1e-2, atol=5e-2)

                    if not ok:
                        print("FP8 精度检查不合格（HIP vs ASM）！！！")
                    else:
                        print("FP8 精度检查合格（HIP vs ASM）！！！")

                elapsed_asm = triton.testing.do_bench(
                    lambda: op.m_grouped_fp8_gemm_nt_contiguous(
                        a_f8, w_asm, ref_d, a_scale, b_scale, m_indices_asm, 1000
                    ),
                    quantiles=None,
                    return_mode="mean",
                ) * 1000
                _print_bench("deepGemm HIP contiguous FP8", m, n, k, elapsed_hip)
                _print_bench("deepGemm ASM contiguous FP8", m, n, k, elapsed_asm)
                ratio = elapsed_asm / elapsed_hip   if elapsed_asm > 0 else float("inf")
                print(f"HIP/ASM 性能比: {ratio:.2f}x\n")


if __name__ == "__main__":
    test_m_grouped_gemm_contiguous(acc_check=True)
