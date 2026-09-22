# python3 op_test/test_fp8_gemm_hipc.py
# python3 op_test/test_fp8_gemm_hipc.py bench
# python3 -m pytest op_test/test_fp8_gemm_hipc.py -v
"""fp8_gemm_hipc 精度/性能测试。

被测算子：csrc/arch/gfx938/fp8_gemm_hipc.cu 的 fp8_gemm（kernel 名
``fp8_gemm_hipc_kernel_gfx938``，warp-specialized producer-consumer，
block tile 256M x 256N x 128K）。

按需求只覆盖 **大 shape 的 2D fp8 GEMM**（bf16 输出、identity scale）：
batch / per-row scale / alpha / beta / fp16 / fp32 输出 等均不在本测试范围。

调用入口：``deepgemm.fp8_gemm_hipc(a, b, d)``（语义 d = A @ B^T，NT 布局）
  a = (A, scale_a)：A fp8_e4m3 [M, K]，scale_a fp32 [M]（0 元素空张量 = identity）
  b = (B, scale_b)：B fp8_e4m3 [N, K]，scale_b fp32 [N]（0 元素空张量 = identity）
  d = 输出 [M, N] bf16（原地写）

精度判定与 test_fp8_gemm.py 一致：与 fp32 参考 GEMM（dequant 后 matmul）对比，
转 bf16 后在 fp32 域比较。容忍度取 max(1, 1% * |ref|max)，可覆盖 bf16 输出
约 2-3 个 ULP 的舍入差异。

benchmark（bench 模式 / test_performance）同时测 fp8_gemm_hipc 与
fp8_gemm（hipBLASLt 参考引擎）作同台对比。hipc 走空 scale 的 identity 路径；
hipBLASLt 没有 identity 分支、必须带 outer-vector scale，故用全 1 scale
（乘 1 不改变结果，数值上等价）。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import torch

from deepgemm import fp8_gemm, fp8_gemm_hipc

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


DEVICE = "cuda"
RESULTS = {"accuracy": [], "performance": []}

# FP32 GEMM 参考保持纯 FP32（关闭 TF32），消除 rocBLAS 隐式 TF32 引入的误差。
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 大 shape 精度用例（M/N 多为 256 的整数倍、K 为 128 的整数倍）；
# 最后一行为"大但不规整"用例，用于覆盖 M/K 尾块（部分 tile / K tail）路径。
BIG_ACCURACY_SHAPES = [
    (4096, 4096, 4096),    # 大方阵
    (8192, 2048, 4096),    # 高瘦
    (4096, 1024, 8192),    # 大 K
    (1024, 4096, 8192),    # 大 N、大 K
    (8192, 8192, 2048),    # 最大 M x N
    (1408, 3072, 2432),    # 大但 M/K 非对齐：M tail + K tail
]

# 性能用例（大 shape，bench 模式）
BIG_PERF_SHAPES = [
    (4096, 4096, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (7168, 7168, 7168,),
    (8192, 4096, 8192),
    (8192, 8192, 8192),
]


def record_result(test_type, **row):
    RESULTS[test_type].append(row)
    if "status" in row:
        print(f"[{row['status']}] {row}")
    pd.DataFrame(RESULTS[test_type]).to_csv(
        Path(__file__).parent / f"test_fp8_gemm_hipc_{test_type}.csv", index=False)


def _fp8_tensor(rows: int, cols: int) -> torch.Tensor:
    """随机 FP8 e4m3 输入（无 scale，值域压到 fp8 可表示范围）。"""
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    x = torch.randn(rows, cols, device=DEVICE, dtype=torch.float32) * 0.5
    x = x.clamp(min=fp8_info.min, max=fp8_info.max)
    return x.to(torch.float8_e4m3fn)


def _identity_scale(rows: int) -> torch.Tensor:
    """空 scale = identity：传给 fp8_gemm_hipc 时内核直接跳过 scale 乘法。"""
    return torch.empty(0, device=DEVICE, dtype=torch.float32)


def torch_fp8_gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """参考实现：FP8 值 exact dequant 成 fp32 后做 A @ B^T，返回 fp32。"""
    return torch.matmul(a.to(torch.float32), b.to(torch.float32).t())


def _run_one_gemm(M: int, N: int, K: int):
    """生成 2D 输入 -> 调用 fp8_gemm_hipc，返回 (d, ref_bf16)。"""
    a = _fp8_tensor(M, K)
    b = _fp8_tensor(N, K)
    d = torch.empty(M, N, device=DEVICE, dtype=torch.bfloat16)
    fp8_gemm_hipc((a, _identity_scale(M)), (b, _identity_scale(N)), d)

    ref = torch_fp8_gemm_ref(a, b).to(torch.bfloat16)  # fp32 -> bf16
    return d, ref


def _test_accuracy_once(M: int, N: int, K: int):
    d, ref = _run_one_gemm(M, N, K)
    torch.cuda.synchronize()

    d_f, ref_f = d.float(), ref.float()
    max_abs = ref_f.abs().max().item()
    # 容差覆盖 bf16 输出的 2-3 ULP（~0.8% * |ref|max），并保证小数值时 >= 1。
    tol = max(1.0, 1e-2 * max_abs)
    max_diff = (d_f - ref_f).abs().max().item()
    passed = max_diff <= tol
    record_result(test_type="accuracy", m=M, n=N, k=K, out_dtype="bfloat16",
                  max_abs=max_abs, max_diff=max_diff, atol=tol,
                  status="PASS" if passed else "FAILED")
    assert passed, (
        f"M={M}, N={N}, K={K}: max_diff={max_diff:.6g} > atol={tol:.6g} "
        f"(|ref|max={max_abs:.6g})")
    print(f"fp8_gemm_hipc M={M}, N={N}, K={K} (bf16, identity scale): OK "
          f"(max_diff={max_diff:.6g}, atol={tol:.6g})")


@pytest.mark.parametrize("M,N,K", BIG_ACCURACY_SHAPES)
def test_accuracy_big_2d(M, N, K):
    """大 shape 2D FP8 GEMM 精度测试：与 fp32 参考 allclose。"""
    _test_accuracy_once(M, N, K)


def _measure_ms(run, M: int, N: int, K: int, warmup: int = 25,
                repeat: int = 100) -> tuple[float, float, float]:
    """反复计时 run()（同输入同输出），返回 (平均毫秒, TFLOPS, GB/s)。"""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    if HAS_TRITON:
        mean_ms, _, _ = triton.testing.do_bench(run, quantiles=[0.5, 0.2, 0.8])
    else:
        start = time.perf_counter()
        for _ in range(repeat):
            run()
        torch.cuda.synchronize()
        mean_ms = (time.perf_counter() - start) / repeat * 1000

    tflops = (2.0 * M * N * K) / (mean_ms * 1e-3) / 1e12
    total_bytes = M * K + N * K + M * N * 2   # A + B (fp8) + D (bf16)
    gbps = (total_bytes / 1e9) / (mean_ms * 1e-3)
    return mean_ms, tflops, gbps


def _bench_fp8_gemm_hipc_ms(M: int, N: int, K: int, warmup: int = 25,
                            repeat: int = 100) -> tuple[float, float, float]:
    """跑 fp8_gemm_hipc（producer-consumer kernel）性能。空 scale = identity。"""
    a = _fp8_tensor(M, K)
    b = _fp8_tensor(N, K)
    d = torch.empty(M, N, device=DEVICE, dtype=torch.bfloat16)

    def run():
        fp8_gemm_hipc((a, _identity_scale(M)), (b, _identity_scale(N)), d)

    return _measure_ms(run, M, N, K, warmup, repeat)


def _bench_fp8_gemm_ms(M: int, N: int, K: int, warmup: int = 25,
                       repeat: int = 100) -> tuple[float, float, float]:
    """跑 fp8_gemm（hipBLASLt 参考引擎）性能，语义同为 d = A @ B^T。

    hipBLASLt 路径总是带 outer-vector fp32 scale，走不了"空 scale"的
    identity 分支，因此传全 1 的 scale（乘以 1 不改变结果），与
    fp8_gemm_hipc 的空 scale identity 数值上等价。
    """
    a = _fp8_tensor(M, K)
    b = _fp8_tensor(N, K)
    sa = torch.ones(M, device=DEVICE, dtype=torch.float32)
    sb = torch.ones(N, device=DEVICE, dtype=torch.float32)
    d = torch.empty(M, N, device=DEVICE, dtype=torch.bfloat16)

    def run():
        fp8_gemm((a, sa), (b, sb), d)

    return _measure_ms(run, M, N, K, warmup, repeat)


@pytest.mark.parametrize("M,N,K", [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
])
def test_performance(M, N, K):
    """性能冒烟：fp8_gemm_hipc 与 fp8_gemm(hipBLASLt) 对比，确认都能跑。"""
    h_ms, h_tf, h_gb = _bench_fp8_gemm_hipc_ms(M, N, K)
    b_ms, b_tf, b_gb = _bench_fp8_gemm_ms(M, N, K)
    record_result(test_type="performance", backend="hipc", m=M, n=N, k=K,
                  latency_ms=h_ms, tflops=h_tf, bandwidth_gbps=h_gb,
                  status="PASS" if h_ms > 0 else "FAILED")
    record_result(test_type="performance", backend="hipblaslt", m=M, n=N, k=K,
                  latency_ms=b_ms, tflops=b_tf, bandwidth_gbps=b_gb,
                  status="PASS" if b_ms > 0 else "FAILED")
    assert h_ms > 0 and b_ms > 0, f"M={M}, N={N}, K={K} 计时失败"
    print(f"M={M}, N={N}, K={K}: "
          f"hipc {h_ms:.3f} ms / {h_tf:.2f} TFLOPS | "
          f"hipblaslt {b_ms:.3f} ms / {b_tf:.2f} TFLOPS | "
          f"speedup {b_ms / h_ms:.2f}x")


def run_performance_benchmark(shapes: list | None = None):
    if shapes is None:
        shapes = BIG_PERF_SHAPES
    print("fp8_gemm_hipc vs fp8_gemm(hipBLASLt) — 2D NT, BF16 out, identity scale")
    print("  hipc: 空 scale identity 路径 | hipblaslt: 全 1 outer-vec scale")
    print("-" * 100)
    print(f"{'M':>8} {'N':>8} {'K':>8} |"
          f"{'hipc ms':>10} {'TFLOPS':>9} |"
          f"{'hipblaslt ms':>12} {'TFLOPS':>9} | {'speedup':>8}")
    print("-" * 100)
    for (M, N, K) in shapes:
        h_ms, h_tf, h_gb = _bench_fp8_gemm_hipc_ms(M, N, K)
        b_ms, b_tf, b_gb = _bench_fp8_gemm_ms(M, N, K)
        record_result(test_type="performance", backend="hipc", m=M, n=N, k=K,
                      latency_ms=h_ms, tflops=h_tf, bandwidth_gbps=h_gb)
        record_result(test_type="performance", backend="hipblaslt",
                      m=M, n=N, k=K, latency_ms=b_ms, tflops=b_tf,
                      bandwidth_gbps=b_gb)
        print(f"{M:>8} {N:>8} {K:>8} |"
              f"{h_ms:>10.3f} {h_tf:>9.1f} |"
              f"{b_ms:>12.3f} {b_tf:>9.1f} | {b_ms / h_ms:>7.2f}x")
    print("-" * 100)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_performance_benchmark()
    else:
        pytest.main([__file__])
