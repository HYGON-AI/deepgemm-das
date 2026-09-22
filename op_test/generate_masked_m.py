#!/usr/bin/env python3
"""
根据实际日志分析结果生成符合类似分布的 masked_m 数据
"""

import numpy as np
from collections import Counter

# 基于实际统计的分布参数
DISTRIBUTION = {
    0: 1480602,
    1: 91728,
    2: 15262,
    3: 3260,
    4: 2222,
    5: 1646,
    6: 1114,
    7: 662,
    8: 566,
    9: 302,
    10: 870,
    11: 306,
    12: 266,
    13: 102,
    14: 94,
    15: 70,
    16: 130,
    17: 44,
    18: 16,
    19: 18,
    20: 24,
    21: 18,
    22: 10,
    23: 10,
    24: 16,
    25: 10,
    28: 10,
    29: 8,
    31: 16,
    32: 340,
}

def create_distribution():
    """创建概率分布"""
    values = list(DISTRIBUTION.keys())
    counts = list(DISTRIBUTION.values())
    total = sum(counts)
    probs = [c / total for c in counts]
    return values, probs

def generate_masked_m(size=16):
    """
    生成一个 masked_m tensor (大小为 size)
    
    Args:
        size: tensor 大小，默认 16
    
    Returns:
        numpy array
    """
    values, probs = create_distribution()
    return np.random.choice(values, size=size, p=probs)

def generate_batch_masked_m(batch_size, tensor_size=16):
    """
    批量生成 masked_m
    
    Args:
        batch_size: 批次数量
        tensor_size: 每个 tensor 的大小
    
    Returns:
        形状为 (batch_size, tensor_size) 的 numpy array
    """
    values, probs = create_distribution()
    return np.random.choice(values, size=(batch_size, tensor_size), p=probs)

def verify_distribution(samples, num_samples=10000):
    """验证生成的分布是否与目标分布接近"""
    flat_samples = samples.flatten()
    counter = Counter(flat_samples)
    total = len(flat_samples)
    
    print("生成数据的分布验证:")
    print(f"{'值':<8} {'生成占比':<12} {'目标占比':<12}")
    print("-" * 32)
    
    target_total = sum(DISTRIBUTION.values())
    for val in sorted(DISTRIBUTION.keys())[:15]:
        gen_ratio = counter.get(val, 0) / total * 100
        target_ratio = DISTRIBUTION[val] / target_total * 100
        print(f"{val:<8} {gen_ratio:<12.2f}% {target_ratio:<12.2f}%")

if __name__ == "__main__":
    # 示例1: 生成单个 masked_m
    single = generate_masked_m(16)
    print("单个 masked_m:")
    print(f"  tensor({list(single)})")
    
    # 示例2: 批量生成
    batch = generate_batch_masked_m(1000, 16)
    print(f"\n批量生成 1000 个 masked_m, shape: {batch.shape}")
    
    # 示例3: 验证分布
    print("\n" + "=" * 50)
    large_batch = generate_batch_masked_m(10000, 16)
    verify_distribution(large_batch)
    
    # 零值统计
    zero_ratio = np.sum(large_batch == 0) / large_batch.size * 100
    print(f"\n零值占比: {zero_ratio:.2f}% (目标: 92.55%)")