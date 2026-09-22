"""
测试 tf32_hc_pernorm_gemm 接口。

输入 tensor 构造与 DeepGEMM/tests/test_hyperconnection.py 一致：
  a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
  b = torch.randn((n, k), dtype=torch.float, device='cuda')
  d = torch.empty((m, n), dtype=torch.float, device='cuda') if num_splits is None else torch.empty((num_splits, m, n), ...)
  s = torch.empty((m,), dtype=torch.float, device='cuda') if num_splits is None else torch.empty((num_splits, m), ...)
本 kernel 输出为扁平布局，故 d 实际分配 (m,n) 或 (num_splits*m,n)，s 实际分配 align(m,64) 或 num_splits*align(m,64)。
"""

import random
import sys
from pathlib import Path
import pandas as pd
import torch
from typing import Optional
import deepgemm


def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """最大绝对差，用于精度检查（与 DeepGEMM testing.calc_diff 一致）。"""
    return (a.float() - b.float()).abs().max().item()
    
def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total

def main():

    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("需要 CUDA，跳过测试")
        return 1

    # 与 DeepGEMM test_hyperconnection 一致的 m / (n,k) / num_splits；n 取 24 满足 kernel 约束，k 已满足 k%64==0
    print("tf32_hc_pernorm_gemm 正确性测试（构造逻辑参考 DeepGEMM test_hyperconnection）")
    print("-" * 60)
    all_ok = True

    m = 9
    n = 32
    k = 128
    num_splits = 2
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda').contiguous()
    b = torch.randn((n, k), dtype=torch.float, device='cuda').contiguous()
    d = torch.empty((m, n), dtype=torch.float, device='cuda').contiguous() if num_splits is None else torch.empty((num_splits, m, n), dtype=torch.float, device='cuda').contiguous()
    s = torch.empty((m,), dtype=torch.float, device='cuda').contiguous() if num_splits is None else torch.empty((num_splits, m), dtype=torch.float, device='cuda').contiguous()
    deepgemm.tf32_hc_pernorm_gemm(a, b, d, s, num_splits=num_splits)
    final_d = d if num_splits is None else d.sum(0)
    final_s = s if num_splits is None else s.sum(0)
    ref_d = a.float() @ b.T
    ref_s = a.float().square().sum(-1)
    # diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
    torch.set_printoptions(threshold=torch.inf)
    # print("a: ", a[:, 64:128].T)
    # print("final_s: ", final_s)
    # print("ref_s: ", ref_s)
    diff = calc_diff(final_s, ref_s)
    s_max_diff = diff
    # assert diff < 1e-5, f'{m=}, {n=}, {k=}, {diff:.10f}'
    # print("b: ", b)
    # print("final_d: ", final_d)
    # print("ref_d: ", ref_d)
    diff = calc_diff(final_d, ref_d)
    d_max_diff = diff
    # assert diff < 1e-3, f'{m=}, {n=}, {k=}, {diff:.10f}'
            # print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits={(num_splits or 0):2}): '
            #       f'{t * 1e6:4.0f} us | '
            #           f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
            #           f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
    pd.DataFrame([{"m": m, "n": n, "k": k, "num_splits": num_splits,
                   "s_max_diff": s_max_diff, "d_max_diff": d_max_diff}]).to_csv(
        Path(__file__).parent / "test_tf32_hc_pernorm_gemm.csv", index=False)
    print()


if __name__ == "__main__":
    exit(main())
