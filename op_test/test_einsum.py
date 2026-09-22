# python3 op_test/test_einsum.py
# python3 -m pytest op_test/test_einsum.py -v
"""Einsum 测试：BF16 走 bmm，FP8 走 fp8_gemm。"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pytest
import torch

from deepgemm import einsum, fp8_einsum

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

DEVICE = "cuda"
RESULTS = {"accuracy": [], "performance": []}


def record_result(test_type, **row):
    RESULTS[test_type].append(row)
    if "status" in row:
        print(f"[{row['status']}] {row}")
    pd.DataFrame(RESULTS[test_type]).to_csv(
        Path(__file__).parent / f"test_einsum_{test_type}.csv", index=False)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x.float() - y.float()).abs().max().item()


def ref_bmk_bnk_mn(
    a: torch.Tensor, b: torch.Tensor, d0: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """与 C++ bmk_bnk_mn 一致：at::bmm(a, b^T) 再对 batch 维求和（BF16 乘加）。"""
    reduced = torch.bmm(a, b.transpose(1, 2)).sum(0)
    if d0 is not None:
        return d0 + reduced
    return reduced


def ref_bhr_hdr_bhd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """与 C++ at::einsum('bhr,hdr->bhd') 一致。"""
    return torch.einsum("bhr,hdr->bhd", x, y)


def ref_bhd_hdr_bhr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhd,hdr->bhr", x, y)


def count_bytes(*args) -> int:
    total = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            total += arg.numel() * arg.element_size()
        elif isinstance(arg, tuple):
            total += count_bytes(*arg)
    return total


def cast_fp8_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """按行（最后一维）量化 FP8 + per-row scale，与 fp8_gemm / hipBLASLt outer-vec 一致。"""
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    x32 = x.float()
    scale = x32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / fp8_info.max
    q = (x32 / scale).clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1)


def cast_fp8_bhr(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [b, h, r] -> fp8 [b, h, r], scale [b, h]"""
    b, h, r = x.shape
    q = torch.empty((b, h, r), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((b, h), device=x.device, dtype=torch.float32)
    for hi in range(h):
        q[:, hi], s[:, hi] = cast_fp8_2d(x[:, hi])
    return q, s


def cast_fp8_hdr(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """y: [h, d, r] -> fp8 [h, d, r], scale [h, d]"""
    h, d, r = y.shape
    q = torch.empty((h, d, r), device=y.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((h, d), device=y.device, dtype=torch.float32)
    for hi in range(h):
        q[hi], s[hi] = cast_fp8_2d(y[hi])
    return q, s


def cast_fp8_bhd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [b, h, d] -> fp8 + scale [b, h]（按 head 内 batch 行）"""
    return cast_fp8_bhr(x)


def cast_fp8_hdr_scale_hr(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """y: [h, d, r] fp8; scale [h, r] for fp8_einsum bhd,hdr->bhr (GEMM n=r).

    向量化：scale[h,r] = 沿 d 维的 max|.|（原先逐列 python 循环较慢）。
    """
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    scale = y.float().abs().amax(dim=1).clamp(min=1e-4) / fp8_info.max  # [h, r]
    q = (y.float() / scale.unsqueeze(1)).clamp(
        min=fp8_info.min, max=fp8_info.max
    ).to(torch.float8_e4m3fn)
    return q, scale


def cast_fp8_bhd_scale_hd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [b, h, d] fp8; scale [h, d] for fp8_einsum bhd,bhr->hdr (GEMM m=d).

    向量化：scale[h,d] = 沿 b 维的 max|.|（原先逐列 python 循环较慢）。
    """
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    scale = x.float().abs().amax(dim=0).clamp(min=1e-4) / fp8_info.max  # [h, d]
    q = (x.float() / scale.unsqueeze(0)).clamp(
        min=fp8_info.min, max=fp8_info.max
    ).to(torch.float8_e4m3fn)
    return q, scale


def ref_fp8_bhr_hdr_bhd(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
) -> torch.Tensor:
    """Per-head FP8 dequant + matmul，与 fp8_einsum bhr,hdr->bhd 一致。"""
    b, h, r = x_fp8.shape
    d = y_fp8.shape[1]
    out = torch.zeros((b, h, d), device=x_fp8.device, dtype=torch.float32)
    for hi in range(h):
        xa = x_fp8[:, hi].float() * x_scale[:, hi].view(-1, 1)
        yb = y_fp8[hi].float() * y_scale[hi].view(-1, 1)
        out[:, hi] = xa @ yb.t()
    return out


def ref_fp8_bhd_hdr_bhr(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
) -> torch.Tensor:
    b, h, d = x_fp8.shape
    r = y_fp8.shape[2]
    out = torch.zeros((b, h, r), device=x_fp8.device, dtype=torch.float32)
    for hi in range(h):
        xa = x_fp8[:, hi].float() * x_scale[:, hi].view(-1, 1)
        yb = y_fp8[hi].float() * y_scale[hi].view(1, -1)  # [d, r], scale [h, r]
        out[:, hi] = xa @ yb
    return out


def ref_fp8_bhd_bhr_hdr(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
) -> torch.Tensor:
    h, d = x_scale.shape
    r = y_scale.shape[1]
    out = torch.zeros((h, d, r), device=x_fp8.device, dtype=torch.float32)
    for hi in range(h):
        xa = x_fp8[:, hi].float() * x_scale[hi].view(1, -1)  # [b, d], scale [h, d]
        yb = y_fp8[:, hi].float() * y_scale[hi].view(1, -1)  # [b, r], scale [h, r]
        out[hi] = xa.t() @ yb
    return out


def cast_fp8_bhr_scale_hr(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [b, h, r] fp8; scale [h, r] for fp8_einsum bhd,bhr->hdr (GEMM n=r).

    向量化：scale[h,r] = 沿 b 维的 max|.|（原先逐列 python 循环较慢）。
    """
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    scale = x.float().abs().amax(dim=0).clamp(min=1e-4) / fp8_info.max  # [h, r]
    q = (x.float() / scale.unsqueeze(0)).clamp(
        min=fp8_info.min, max=fp8_info.max
    ).to(torch.float8_e4m3fn)
    return q, scale


def bench_ms(fn, warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if HAS_TRITON:
        result = triton.testing.do_bench(fn, return_mode="mean")
        if isinstance(result, (list, tuple)):
            result = result[0]
        # triton do_bench already returns mean latency in milliseconds
        return float(result)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat * 1e3


# ---------------------------------------------------------------------------
# BF16 einsum
# ---------------------------------------------------------------------------

def test_bmk_bnk_mn_accuracy() -> None:
    failed_cases = []
    # (s, [(m, n, k), ...])；末尾两行为补充的大 shape 用例
    for s, mnk in [
        (129, [(128, 384, 128), (256, 256, 256)]),
        (512, [(128, 384, 128), (256, 256, 256)]),
        (512, [(512, 512, 512)]),
        (2048, [(256, 256, 256)]),
        (4096, [(128, 128, 128)]),
    ]:
        for m, n, k in mnk:
            for dtype in (torch.float, torch.bfloat16):
                a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEVICE)
                b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEVICE)
                d = torch.randn((m, n), dtype=dtype, device=DEVICE)
                c = d if dtype == torch.float else None

                d0 = d.clone() if dtype == torch.float else None
                ref_d = ref_bmk_bnk_mn(a, b, d0)
                if dtype == torch.bfloat16:
                    ref_d = ref_d.to(torch.bfloat16)

                einsum("bmk,bnk->mn", a, b, d, c=c)
                atol = 1e-5
                diff = calc_diff(d, ref_d)
                status = "PASS" if diff < atol else "FAILED"
                record_result(test_type="accuracy", equation="bmk,bnk->mn",
                              shape=f"S={s}, M={m}, N={n}, K={k}",
                              dtype=str(dtype), max_diff=diff,
                              status=status)
                if status == "FAILED":
                    failed_cases.append(
                        f"s={s}, m={m}, n={n}, k={k}, dtype={dtype}, max_diff={diff:.6g}, atol={atol}")
    assert not failed_cases, "\n".join(failed_cases)


def test_bhr_hdr_bhd_accuracy() -> None:
    failed_cases = []
    # (h, r, d, batch 集合)；末尾两行为补充的大 shape 用例
    for h, r, d, bs in [
        (128, 512, 128, (4, 32, 128)),
        (8, 1024, 512, (4, 32, 128)),
        (8, 4096, 1024, (1024, 4096)),
        (32, 2048, 512, (512, 2048)),
    ]:
        for b in bs:
            x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            ref_z = ref_bhr_hdr_bhd(x, y)
            z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            einsum("bhr,hdr->bhd", x, y, z)
            diff = calc_diff(z, ref_z)
            status = "PASS" if diff < 1e-5 else "FAILED"
            record_result(test_type="accuracy", equation="bhr,hdr->bhd",
                          shape=f"B={b}, H={h}, R={r}, D={d}",
                          dtype=str(torch.bfloat16), max_diff=diff, status=status)
            if status == "FAILED":
                failed_cases.append(f"b={b}, h={h}, r={r}, d={d}, max_diff={diff:.6g}")
    assert not failed_cases, "\n".join(failed_cases)


def test_bhd_hdr_bhr_accuracy() -> None:
    failed_cases = []
    # (h, r, d, batch 集合)；末尾两行为补充的大 shape 用例
    for h, r, d, bs in [
        (128, 512, 128, (4, 32, 128)),
        (8, 1024, 512, (4, 32, 128)),
        (8, 4096, 1024, (1024, 4096)),
        (32, 2048, 512, (512, 2048)),
    ]:
        for b in bs:
            x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            ref_z = ref_bhd_hdr_bhr(x, y)
            z = torch.empty((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            einsum("bhd,hdr->bhr", x, y, z)
            diff = calc_diff(z, ref_z)
            status = "PASS" if diff < 1e-5 else "FAILED"
            record_result(test_type="accuracy", equation="bhd,hdr->bhr",
                          shape=f"B={b}, H={h}, R={r}, D={d}",
                          dtype=str(torch.bfloat16), max_diff=diff, status=status)
            if status == "FAILED":
                failed_cases.append(f"b={b}, h={h}, r={r}, d={d}, max_diff={diff:.6g}")
    assert not failed_cases, "\n".join(failed_cases)


# ---------------------------------------------------------------------------
# FP8 einsum（hipBLASLt fp8_gemm 路径，atol 与 test_fp8_gemm 一致）
# ---------------------------------------------------------------------------

def test_fp8_bhr_hdr_bhd_accuracy() -> None:
    failed_cases = []
    # (h, r, d, batch 集合)；末尾两行为补充的大 shape 用例（大 M=b、多头 h、更大 K=r）
    for h, r, d, bs in [
        (8, 1024, 512, (4, 32, 128)),
        (16, 1024, 1024, (2048, 4096)),
        (8, 2048, 1024, (1024,)),
    ]:
        for b in bs:
            x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            x_fp8, x_scale = cast_fp8_bhr(x)
            y_fp8, y_scale = cast_fp8_hdr(y)  # bhr,hdr->bhd: y scale [h, d]
            ref_z = ref_fp8_bhr_hdr_bhd(x_fp8, x_scale, y_fp8, y_scale)
            z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)

            fp8_einsum("bhr,hdr->bhd", (x_fp8, x_scale), (y_fp8, y_scale), z)
            diff = calc_diff(z, ref_z)
            status = "PASS" if diff < 1.0 else "FAILED"
            record_result(test_type="accuracy", equation="FP8 bhr,hdr->bhd",
                          shape=f"B={b}, H={h}, R={r}, D={d}",
                          dtype=str(torch.bfloat16), max_diff=diff, status=status)
            if status == "FAILED":
                failed_cases.append(f"FP8 bhr,hdr->bhd b={b}, h={h}, r={r}, d={d}")
    assert not failed_cases, "\n".join(failed_cases)


def test_fp8_bhd_hdr_bhr_accuracy() -> None:
    failed_cases = []
    # (h, r, d, batch 集合)；末尾两行为补充的大 shape 用例（大 M=b、多头 h、更大 N=r）
    for h, r, d, bs in [
        (8, 1024, 512, (4, 32, 128)),
        (16, 1024, 1024, (2048, 4096)),
        (8, 1024, 2048, (1024,)),
    ]:
        for b in bs:
            x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            x_fp8, x_scale = cast_fp8_bhd(x)
            y_fp8, y_scale = cast_fp8_hdr_scale_hr(y)
            ref_z = ref_fp8_bhd_hdr_bhr(x_fp8, x_scale, y_fp8, y_scale)
            z = torch.empty((b, h, r), device=DEVICE, dtype=torch.bfloat16)

            fp8_einsum("bhd,hdr->bhr", (x_fp8, x_scale), (y_fp8, y_scale), z)
            diff = calc_diff(z, ref_z)
            status = "PASS" if diff < 1.0 else "FAILED"
            record_result(test_type="accuracy", equation="FP8 bhd,hdr->bhr",
                          shape=f"B={b}, H={h}, R={r}, D={d}",
                          dtype=str(torch.bfloat16), max_diff=diff, status=status)
            if status == "FAILED":
                failed_cases.append(f"FP8 bhd,hdr->bhr b={b}, h={h}, r={r}, d={d}")
    assert not failed_cases, "\n".join(failed_cases)


def test_fp8_bhd_bhr_hdr_accuracy() -> None:
    failed_cases = []
    # (h, r, d, batch 集合)；末尾两行为补充的大 shape 用例（更大 K=b、更大 D）
    for h, r, d, bs in [
        (8, 1024, 512, (128, 512)),
        (8, 1024, 512, (2048,)),
        (8, 1024, 1024, (1024,)),
    ]:
        for b in bs:
            x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            x_fp8, x_scale = cast_fp8_bhd_scale_hd(x)
            y_fp8, y_scale = cast_fp8_bhr_scale_hr(y)
            ref_z = ref_fp8_bhd_bhr_hdr(x_fp8, x_scale, y_fp8, y_scale)
            # BF16 输出、beta=0：float32 累加 (c=z) 在 hipBLASLt 上会产出 NaN
            z = torch.empty((h, d, r), device=DEVICE, dtype=torch.bfloat16)

            fp8_einsum("bhd,bhr->hdr", (x_fp8, x_scale), (y_fp8, y_scale), z, recipe=(1, 1, 128))
            diff = calc_diff(z, ref_z)
            status = "PASS" if diff < 1.0 else "FAILED"
            record_result(test_type="accuracy", equation="FP8 bhd,bhr->hdr",
                          shape=f"B={b}, H={h}, R={r}, D={d}",
                          dtype=str(torch.bfloat16), max_diff=diff, status=status)
            if status == "FAILED":
                failed_cases.append(f"FP8 bhd,bhr->hdr b={b}, h={h}, r={r}, d={d}")
    assert not failed_cases, "\n".join(failed_cases)


# ---------------------------------------------------------------------------
# 可选性能打印（__main__）
# ---------------------------------------------------------------------------

def _bench_engine_note() -> str:
    """当前 FP8 引擎说明（与 einsum.cu 的 DEEPGEMM_EINSUM_FP8_ENGINE 一致）。"""
    if os.environ.get("DEEPGEMM_EINSUM_FP8_ENGINE") == "hipblaslt":
        return "hipBLASLt 参考路径"
    return "hipc kernel (默认)"


def _report_bench(equation: str, note: str, shape_line: str,
                  t_ms: float, flops: float, gbps: float) -> None:
    t_s = t_ms * 1e-3
    print(f'Bench {equation}  [{note}]')
    print(f"  {shape_line}")
    print(f"  latency {t_ms:6.2f} ms | {flops / t_s / 1e12:7.1f} TFLOPS | "
          f"{gbps:6.1f} GB/s | FLOPs={flops:.3e}")


def run_bmk_bnk_mn_bench() -> None:
    s, m, n, k = 4096, 256, 256, 256
    a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEVICE)
    d = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
    t_ms = bench_ms(lambda: einsum("bmk,bnk->mn", a, b, d))
    t_s = t_ms * 1e-3
    flops = 2 * s * m * n * k
    gbps = count_bytes(a, b, d) / t_s / 1e9
    record_result(test_type="performance", equation="bmk,bnk->mn",
                  shape=f"S={s}, M={m}, N={n}, K={k}",
                  latency_ms=t_ms, tflops=flops / t_s / 1e12,
                  bandwidth_gbps=gbps)
    _report_bench(
        equation='"bmk,bnk->mn"',
        note="BF16 · torch at::bmm 参考实现 · 对 s、k 双轴求和",
        shape_line=f"a[{s},{m},{k}] · b[{s},{n},{k}] -> d[{m},{n}]  "
                   f"(每个 s: M={m} N={n} K={k})",
        t_ms=t_ms, flops=flops, gbps=gbps)


def run_bhr_hdr_bhd_bench() -> None:
    b, h, r, d = 4096, 8, 4096, 1024
    x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
    z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)
    t_ms = bench_ms(lambda: einsum("bhr,hdr->bhd", x, y, z))
    t_s = t_ms * 1e-3
    flops = 2 * b * h * r * d
    gbps = count_bytes(x, y, z) / t_s / 1e9
    record_result(test_type="performance", equation="bhr,hdr->bhd",
                  shape=f"B={b}, H={h}, R={r}, D={d}",
                  latency_ms=t_ms, tflops=flops / t_s / 1e12,
                  bandwidth_gbps=gbps)
    _report_bench(
        equation='"bhr,hdr->bhd"',
        note="BF16 · torch at::bmm 参考实现 · 对 r 求和, h 为 batch",
        shape_line=f"x[{b},{h},{r}] · y[{h},{d},{r}] -> z[{b},{h},{d}]  "
                   f"(M={b} N={d} K={r}, batch=H={h})",
        t_ms=t_ms, flops=flops, gbps=gbps)


def run_fp8_bhr_hdr_bhd_bench() -> None:
    b, h, r, d = 4096, 8, 4096, 1024
    x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
    x_fp8, x_scale = cast_fp8_bhr(x)
    y_fp8, y_scale = cast_fp8_hdr(y)  # scale [h, d]
    z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)

    t_ms = bench_ms(lambda: fp8_einsum("bhr,hdr->bhd", (x_fp8, x_scale), (y_fp8, y_scale), z))
    t_s = t_ms * 1e-3
    flops = 2 * b * h * r * d
    gbps = count_bytes(x_fp8, x_scale, y_fp8, y_scale, z) / t_s / 1e9
    record_result(test_type="performance", equation="FP8 bhr,hdr->bhd",
                  shape=f"B={b}, H={h}, R={r}, D={d}",
                  latency_ms=t_ms, tflops=flops / t_s / 1e12,
                  bandwidth_gbps=gbps)
    _report_bench(
        equation='FP8 "bhr,hdr->bhd"',
        note=f"FP8 e4m3 · {_bench_engine_note()} · 对 r 求和, h 为 batch",
        shape_line=f"x_fp8[{b},{h},{r}]+sx[{b},{h}] · y_fp8[{h},{d},{r}]+sy[{h},{d}] "
                   f"-> z[{b},{h},{d}]  (M={b} N={d} K={r}, batch=H={h})",
        t_ms=t_ms, flops=flops, gbps=gbps)


def run_fp8_bhd_hdr_bhr_bench() -> None:
    # x[b,h,d] · y[h,d,r] -> z[b,h,r]（对 d 求和, h=batch）; 内层 GEMM M=b N=r K=d
    b, h, d, r = 4096, 8, 1024, 1024
    x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
    x_fp8, x_scale = cast_fp8_bhd(x)  # scale [b, h]
    y_fp8, y_scale = cast_fp8_hdr_scale_hr(y)  # scale [h, r]
    z = torch.empty((b, h, r), device=DEVICE, dtype=torch.bfloat16)

    t_ms = bench_ms(lambda: fp8_einsum("bhd,hdr->bhr", (x_fp8, x_scale), (y_fp8, y_scale), z))
    t_s = t_ms * 1e-3
    flops = 2 * b * h * d * r
    gbps = count_bytes(x_fp8, x_scale, y_fp8, y_scale, z) / t_s / 1e9
    record_result(test_type="performance", equation="FP8 bhd,hdr->bhr",
                  shape=f"B={b}, H={h}, R={r}, D={d}",
                  latency_ms=t_ms, tflops=flops / t_s / 1e12,
                  bandwidth_gbps=gbps)
    _report_bench(
        equation='FP8 "bhd,hdr->bhr"',
        note=f"FP8 e4m3 · {_bench_engine_note()} · 对 d 求和, h 为 batch",
        shape_line=f"x_fp8[{b},{h},{d}]+sx[{b},{h}] · y_fp8[{h},{d},{r}]+sy[{h},{r}] "
                   f"-> z[{b},{h},{r}]  (M={b} N={r} K={d}, batch=H={h})",
        t_ms=t_ms, flops=flops, gbps=gbps)


def run_fp8_bhd_bhr_hdr_bench() -> None:
    # x[b,h,d] · y[b,h,r] -> z[h,d,r]（对 b 求和, h=batch）; 内层 GEMM M=d N=r K=b
    b, h, d, r = 1024, 8, 2048, 2048
    x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
    x_fp8, x_scale = cast_fp8_bhd_scale_hd(x)  # scale [h, d]
    y_fp8, y_scale = cast_fp8_bhr_scale_hr(y)  # scale [h, r]
    z = torch.empty((h, d, r), device=DEVICE, dtype=torch.bfloat16)

    t_ms = bench_ms(lambda: fp8_einsum("bhd,bhr->hdr", (x_fp8, x_scale), (y_fp8, y_scale), z,
                                       recipe=(1, 1, 128)))
    t_s = t_ms * 1e-3
    flops = 2 * b * h * d * r
    gbps = count_bytes(x_fp8, x_scale, y_fp8, y_scale, z) / t_s / 1e9
    record_result(test_type="performance", equation="FP8 bhd,bhr->hdr",
                  shape=f"B={b}, H={h}, R={r}, D={d}",
                  latency_ms=t_ms, tflops=flops / t_s / 1e12,
                  bandwidth_gbps=gbps)
    _report_bench(
        equation='FP8 "bhd,bhr->hdr"',
        note=f"FP8 e4m3 · {_bench_engine_note()} · 对 b 求和, h 为 batch",
        shape_line=f"x_fp8[{b},{h},{d}]+sx[{h},{d}] · y_fp8[{b},{h},{r}]+sy[{h},{r}] "
                   f"-> z[{h},{d},{r}]  (M={d} N={r} K={b}, batch=H={h})",
        t_ms=t_ms, flops=flops, gbps=gbps)


def run_all_benches() -> None:
    run_bmk_bnk_mn_bench()
    run_bhr_hdr_bhd_bench()
    run_fp8_bhr_hdr_bhd_bench()
    run_fp8_bhd_hdr_bhr_bench()
    run_fp8_bhd_bhr_hdr_bench()


if __name__ == "__main__":
    import sys

    # 用法: python op_test/test_einsum.py [all|accuracy|bench]
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    torch.manual_seed(0)
    random.seed(0)

    print("deepgemm einsum tests\n")
    failures = []
    if mode in ("all", "accuracy"):
        accuracy_tests = (
            test_bmk_bnk_mn_accuracy,
            test_bhr_hdr_bhd_accuracy,
            test_bhd_hdr_bhr_accuracy,
            test_fp8_bhr_hdr_bhd_accuracy,
            test_fp8_bhd_hdr_bhr_accuracy,
            test_fp8_bhd_bhr_hdr_accuracy,
        )
        for test in accuracy_tests:
            try:
                test()
                print(f"{test.__name__} OK")
            except AssertionError as error:
                failures.append(f"{test.__name__}: {error}")

    if mode in ("all", "bench"):
        print("---- benchmark ----")
        run_all_benches()

    if failures:
        raise AssertionError("\n".join(failures))
