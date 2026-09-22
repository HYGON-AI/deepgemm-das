# python3 test_fp8_gemm.py
# python3 test_fp8_gemm.py bench
"""FP8 GEMM 测试：参考 hipblaslt_fp8_w8a8 / hipblaslt_w8a8_channelwise_gemm 的测试方式。"""
from pathlib import Path

import pandas as pd
import pytest
import time
import torch
from deepgemm import fp8_gemm

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


RESULTS = {"accuracy": [], "performance": []}


def record_result(test_type, **row):
    RESULTS[test_type].append(row)
    if "status" in row:
        print(f"[{row['status']}] {row}")
    pd.DataFrame(RESULTS[test_type]).to_csv(
        Path(__file__).parent / f"test_fp8_gemm_{test_type}.csv", index=False)


def torch_fp8_gemm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16):
    """
    参考实现：FP8 per-channel scale 后做 GEMM，与 hipblaslt_w8a8_channelwise_gemm 一致。
    a: [m, k], b: [k, n] (NT 布局)，scale_a: [m], scale_b: [n]。
    """
    a_fp = a.to(torch.float32) * scale_a.view(-1, 1)
    b_fp = b.to(torch.float32) * scale_b.view(1, -1)
    out = torch.matmul(a_fp, b_fp)
    return out.to(out_dtype)


def _test_accuracy_once(m, n, k, out_dtype, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a = (torch.randn(m, k, device=device, dtype=torch.float32) * 0.5).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device=device, dtype=torch.float32) * 0.5).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    scale_a = torch.randn(m, device=device, dtype=torch.float32).abs() * 0.01 + 0.01
    scale_b = torch.randn(n, device=device, dtype=torch.float32).abs() * 0.01 + 0.01
    b_t = b.t()  # [k, n]

    base = torch_fp8_gemm(a, b_t, scale_a, scale_b, out_dtype=out_dtype)
    d = torch.empty(m, n, dtype=out_dtype, device=device)
    fp8_gemm((a, scale_a), (b, scale_b), d)

    # 与 hipblaslt_fp8_w8a8 一致：转 float32 比较，atol=1
    max_diff = torch.abs(d.to(torch.float32) - base.to(torch.float32)).max().item()
    passed = torch.allclose(d.to(torch.float32), base.to(torch.float32), atol=1)
    record_result(test_type="accuracy", m=m, n=n, k=k, out_dtype=str(out_dtype),
                  max_diff=max_diff, status="PASS" if passed else "FAILED")
    assert passed, (
        f"m={m}, n={n}, k={k}, out_dtype={out_dtype}: max_diff={max_diff}"
    )
    print(f"m={m}, n={n}, k={k}, out_dtype={out_dtype}: OK")


@pytest.mark.parametrize("m,n,k", [
    (168, 342, 512),   # 与 hipblaslt_fp8_w8a8 一致
    (1, 16, 1024),
    (128, 128, 1024),
    (512, 512, 1024),
    (1024, 1024, 4096),
])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy(m, n, k, out_dtype):
    """精度测试：与 torch 参考实现 allclose(atol=1)。"""
    _test_accuracy_once(m, n, k, out_dtype, "cuda")


def _prepare_fp8_gemm_tensors(M, N, K, device, out_dtype=torch.bfloat16):
    """准备 fp8_gemm 所需的 (a, scale_a), (b, scale_b), output。NT: a [M,K], b [K,N] -> out [M,N]。"""
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a = (torch.randn(M, K, device=device, dtype=torch.float32) * 0.5).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device=device, dtype=torch.float32) * 0.5).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    scale_a = torch.randn(M, device=device, dtype=torch.float32).abs() * 0.01 + 0.01
    scale_b = torch.randn(N, device=device, dtype=torch.float32).abs() * 0.01 + 0.01
    b_t = b.t()
    out = torch.empty(M, N, dtype=out_dtype, device=device)
    return (a, scale_a), (b, scale_b), out


def _bench_fp8_gemm_ms(M, N, K, device="cuda", warmup=25, repeat=100):
    """跑 fp8_gemm 性能，返回平均耗时（毫秒）、TFLOPS 和带宽（GB/s）。"""
    a, b, out = _prepare_fp8_gemm_tensors(M, N, K, device)

    def run():
        fp8_gemm(a, b, out)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    quantiles = [0.5, 0.2, 0.8]
    if HAS_TRITON:
        mean_ms,_,_ = triton.testing.do_bench(run, quantiles=quantiles)
    else:
        start = time.perf_counter()
        for _ in range(repeat):
            run()
        torch.cuda.synchronize()
        mean_ms = (time.perf_counter() - start) / repeat * 1000

    tflops = (2.0 * M * N * K) / (mean_ms * 1e-3) / 1e12
    total_bytes = M * K + K * N + (M + N) * 4 + M * N * 2
    gbps = (total_bytes / 1e9) / (mean_ms * 1e-3)
    return mean_ms, tflops, gbps


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 1024),
    (512, 512, 1024),
    (1024, 1024, 4096),
    (4096, 4096, 4096),
])
def test_performance(M, N, K):
    """性能测试：测量 fp8_gemm 延迟、TFLOPS 与带宽。"""
    mean_ms, tflops, gbps = _bench_fp8_gemm_ms(M, N, K, device="cuda")
    passed = mean_ms > 0 and tflops > 0 and gbps > 0
    record_result(test_type="performance", m=M, n=N, k=K, latency_ms=mean_ms,
                  tflops=tflops, bandwidth_gbps=gbps, status="PASS" if passed else "FAILED")
    assert passed
    print(f"fp8_gemm M={M}, N={N}, K={K}: {mean_ms:.3f} ms, {tflops:.2f} TFLOPS, {gbps:.2f} GB/s")


def run_performance_benchmark(shapes=None):
    if shapes is None:
        shapes = [
            (128, 128, 1024),
            (256, 256, 1024),
            (512, 512, 1024),
            (1024, 1024, 2048),
            (1024, 1024, 4096),
            (2048, 2048, 4096),
            (4096, 4096, 4096),
            (16384, 16384, 16384),
        ]
    device = "cuda"
    print("fp8_gemm performance (NT layout, BF16 output)")
    print("-" * 80)
    print(f"{'M':>8} {'N':>8} {'K':>8} {'ms':>10} {'TFLOPS':>10} {'GB/s':>10}")
    print("-" * 80)
    for (M, N, K) in shapes:
        mean_ms, tflops, gbps = _bench_fp8_gemm_ms(M, N, K, device=device)
        record_result(test_type="performance", m=M, n=N, k=K, latency_ms=mean_ms,
                      tflops=tflops, bandwidth_gbps=gbps)
        print(f"{M:>8} {N:>8} {K:>8} {mean_ms:>10.3f} {tflops:>10.2f} {gbps:>10.2f}")
    print("-" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_performance_benchmark()
    else:
        pytest.main([__file__])
