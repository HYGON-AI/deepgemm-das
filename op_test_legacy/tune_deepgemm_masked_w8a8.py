# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""
DeepGEMM Grouped W8A8 GEMM Tuning Script

简洁的 tuning 脚本，用于寻找最优的 kernel 配置并保存到 JSON 文件
用法: python tune_deepgemm_masked_w8a8.py --n 4096 --k 7168 --expected-m 32 64 128
"""

import argparse
import random
import math
import json
import os
import torch
import triton
from typing import Optional

from deepgemm import m_grouped_w8a8_gemm_nt_masked_impl
from deepgemm.config import get_all_groupgemm_modes, save_groupgemm_config
from lmslim.layers.gemm.int8_utils import per_token_quant_int8
from deepgemm_marlin_quant import weight8bit_nt_kpack2_marlin
from generate_masked_m import generate_masked_m


seed = 0
torch.manual_seed(seed)

# PyTorch GPU随机种子
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 多GPU情况



def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def native_w8a8_perChannel_batch_matmul(q_a1_all, weight13, qa1_scale_all, w13_scale, output_dtype):
    """参考实现，用于精度验证"""
    A = q_a1_all.to(torch.float32)
    B = weight13.to(torch.float32)
    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    C = torch.bmm(A, B.transpose(1, 2))
    C = qa1_scale_all * C * w13_scale.transpose(1, 2)
    return C.to(output_dtype)


def generate_test_data(E: int, m: int, n: int, k: int, 
                       expected_m_per_group: int, 
                       out_dtype: torch.dtype = torch.bfloat16,
                       device: str = "cuda"):
    """生成测试数据"""
    input_data = (torch.randn((E, m, k), device=device) * 5).to(dtype=out_dtype)
    weight = to_int8(torch.randn((E, n, k), device=device))
    weight_scale = torch.randn((E, n, 1), device=device, dtype=torch.float32)
    
    # 生成 masked_m (每个 expert 的 token 数)
    masked_m = torch.empty((E,), device='cuda', dtype=torch.int32)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))

    # masked_ms = generate_masked_m(E)
    # masked_m = torch.tensor(masked_ms, device='cuda', dtype=torch.int32).reshape(E)

    # masked_m_list = [17, 18, 11, 19, 14, 17, 13, 17, 14, 14, 20, 12, 20, 20, 12, 11]
    # masked_m = torch.tensor(masked_m_list, device='cuda', dtype=torch.int)
    
    return input_data, weight, weight_scale, masked_m


def benchmark_single_mode(mode: int,
                          q_input: torch.Tensor,
                          a_scale: torch.Tensor,
                          weight_marlin: torch.Tensor,
                          w_scale: torch.Tensor,
                          output: torch.Tensor,
                          masked_m: torch.Tensor,
                          expected_m_per_group: int,
                          warmup: int = 5,
                          num_iters: int = 10) -> Optional[float]:
    """
    对单个 mode 进行 benchmark
    
    Returns:
        耗时 (us) 或 None (如果 kernel 执行失败)
    """
    try:
        # Warmup
        for _ in range(warmup):
            m_grouped_w8a8_gemm_nt_masked_impl(
                (q_input, a_scale),
                (weight_marlin, w_scale),
                output,
                masked_m,
                expected_m_per_group,
                mode
            )
        torch.cuda.synchronize()
        
        # Benchmark
        cost = triton.testing.do_bench(
            lambda: m_grouped_w8a8_gemm_nt_masked_impl(
                (q_input, a_scale),
                (weight_marlin, w_scale),
                output,
                masked_m,
                expected_m_per_group,
                mode
            ),
            quantiles=None,
            return_mode="mean"
        ) * 1000  # us
        
        return cost
    except Exception as e:
        print(f"  Mode {mode} failed: {e}")
        return None


def verify_correctness(mode: int,
                       q_input: torch.Tensor,
                       a_scale: torch.Tensor,
                       weight: torch.Tensor,
                       weight_marlin: torch.Tensor,
                       w_scale: torch.Tensor,
                       output: torch.Tensor,
                       masked_m: torch.Tensor,
                       expected_m_per_group: int,
                       ref_output: torch.Tensor) -> bool:
    """验证 kernel 输出精度"""
    output.zero_()
    m_grouped_w8a8_gemm_nt_masked_impl(
        (q_input, a_scale),
        (weight_marlin, w_scale),
        output,
        masked_m,
        expected_m_per_group,
        mode
    )
    torch.cuda.synchronize()
    
    # import pdb; pdb.set_trace()

    # valid_m = int(expected_m_per_group * 0.7)
    # print(f"  Verifying mode {mode}...")
    # print(f"    ref_output: {ref_output[:, :valid_m, :]}")
    # print(f"    output: {output[:, :valid_m, :]}")
    # 通过 masked_m 比较 output 和 ref_output 的有效行
    valid = True
    for e in range(masked_m.shape[0]):
        cur_valid_m = int(masked_m[e].item())
        if cur_valid_m == 0:
            continue
        if not torch.allclose(
            ref_output[e, :cur_valid_m, :],
            output[e, :cur_valid_m, :],
            rtol=1e-2, atol=1e-2
        ):
            valid = False
            break
    return valid


def tune_for_expected_m(E: int, m: int, n: int, k: int,
                        expected_m_per_group: int,
                        out_dtype: torch.dtype = torch.bfloat16,
                        verify: bool = True) -> dict:
    """
    对特定 expected_m_per_group 进行 tuning
    
    Returns:
        最优配置字典
    """
    print(f"\n{'='*60}")
    print(f"Tuning: E={E}, m={m}, n={n}, k={k}, expected_m={expected_m_per_group}")
    print(f"{'='*60}")
    
    # 生成测试数据
    input_data, weight, weight_scale, masked_m = generate_test_data(
        E, m, n, k, expected_m_per_group, out_dtype)
    
    # 量化输入
    q_input, a_scale = per_token_quant_int8(input_data)
    
    # 权重转换为 marlin 格式
    weight_marlin_list = []
    for i in range(weight.shape[0]):
        w_marlin = weight8bit_nt_kpack2_marlin(weight[i])
        weight_marlin_list.append(w_marlin)
    weight_marlin = torch.stack(weight_marlin_list, dim=0)
    
    # 创建输出 tensor
    output = torch.zeros((E, m, n), device='cuda', dtype=out_dtype)
    
    # 计算参考输出
    if verify:
        ref_output = native_w8a8_perChannel_batch_matmul(
            q_input, weight.to(torch.int32), a_scale, weight_scale, out_dtype)
        # valid_m = int(expected_m_per_group * 0.7)
        # ref_output[:, valid_m:, :] = 0
    
    # 获取所有可用 modes
    cuda_modes = [i for i in range(85)]         # 不过lds
    cuda_modes1 = [i for i in range(100, 127)]  # bypass kernel
    cuda_modes2 = [i for i in range(200, 285)]  # persistent kernel
    cuda_modes3 = [i for i in range(500, 596)]  # 过lds版本 待完善
    asm_modes = [1000, 1001, 1002]
    
    #all_modes = cuda_modes + cuda_modes1 + cuda_modes2 + asm_modes
    all_modes = cuda_modes + asm_modes


    best_mode = 0
    best_time = float('inf')    
    valid_modes = []
    
    print(f"\nTesting {len(all_modes)} modes...")
    
    for mode in all_modes:
        # if mode >= 152 and mode <= 159:
        #     continue
        # if mode in [182, 183, 184]:
        #     continue
        # 先验证精度
        if verify:
            if not verify_correctness(mode, q_input, a_scale, weight, 
                                     weight_marlin, weight_scale, output,
                                     masked_m, expected_m_per_group, ref_output):
                print(f"  Mode {mode}: FAILED (precision)")
                continue
        
        # Benchmark
        cost = benchmark_single_mode(
            mode, q_input, a_scale, weight_marlin, weight_scale,
            output, masked_m, expected_m_per_group)
        
        if cost is not None:
            valid_modes.append((mode, cost))
            if cost < best_time:
                best_time = cost
                best_mode = mode
            print(f"  Mode {mode}: {cost:.2f} us")
    
    # 排序并打印 top 5
    valid_modes.sort(key=lambda x: x[1])
    print(f"\nTop 5 modes for expected_m={expected_m_per_group}:")
    for i, (mode, cost) in enumerate(valid_modes[:5]):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. Mode {mode}: {cost:.2f} us{marker}")
    
    config = {"MODE": best_mode}
    print(f"\nBest config: {config}")
    
    return config


def main():
    parser = argparse.ArgumentParser(description="DeepGEMM Grouped W8A8 GEMM Tuning")  
    parser.add_argument("--e", type=int, default=16, help="Number of experts")
    parser.add_argument("--m", type=int, default=4096, help="Total M dimension")
    parser.add_argument("--n", type=int, nargs='+', default=[4096, 7168], 
                        help="N dimension(s)")
    parser.add_argument("--k", type=int, nargs='+', default=[7168, 2048], 
                        help="K dimension(s)")
    parser.add_argument("--expected-m", type=int, nargs='+', 
                        default=[8],
                        help="Expected M per group values to tune")
    parser.add_argument("--no-verify", action="store_true", 
                        help="Skip precision verification")
    parser.add_argument("--save", action="store_true", 
                        help="Save config to JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    
    # 获取 GPU 信息
    device_props = torch.cuda.get_device_properties(0)
    device_name = device_props.gcnArchName.split(':')[0] if hasattr(device_props, 'gcnArchName') else device_props.name
    num_cus = device_props.multi_processor_count
    
    print(f"Device: {device_name}, CUs: {num_cus}")
    print(f"E: {args.e}, M: {args.m}")
    print(f"N list: {args.n}")
    print(f"K list: {args.k}")
    print(f"Expected M list: {args.expected_m}")
    
    # 确保 N 和 K 列表长度一致
    if len(args.n) != len(args.k):
        raise ValueError("N and K lists must have the same length")
    
    results = {}
    
    # 对每个 (N, K) 组合进行 tuning
    for n, k in zip(args.n, args.k):
        key = f"{n}_{k}"
        results[key] = {}
        
        for expected_m in args.expected_m:
            config = tune_for_expected_m(
                E=args.e,
                m=args.m,
                n=n,
                k=k,
                expected_m_per_group=expected_m,
                verify=not args.no_verify
            )
            results[key][expected_m] = config
            
            # 保存配置
            if args.save:
                save_groupgemm_config(config, expected_m, n, k, num_cus)
    
    # 打印最终结果汇总
    print(f"\n{'='*60}")
    print("TUNING RESULTS SUMMARY")
    print(f"{'='*60}")
    for key, configs in results.items():
        print(f"\n{key}:")
        for expected_m, config in configs.items():
            print(f"  expected_m={expected_m}: MODE={config['MODE']}")
    
    # 保存完整结果到单独文件
    if args.save:
        gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
        result_file = f"tuning_results_{gfx_version}_CU{num_cus}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nFull results saved to: {result_file}")


if __name__ == "__main__":
    main()
