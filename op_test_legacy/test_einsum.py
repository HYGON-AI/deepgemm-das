# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# python3 op_test/test_einsum.py
# python3 -m pytest op_test/test_einsum.py -v
"""Einsum 测试：BF16 走 bmm，FP8 走 fp8_gemm。"""
from __future__ import annotations

import random
import time
from typing import Optional, Tuple

import pytest
import torch

from deepgemm import einsum, fp8_einsum

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

DEVICE = "cuda"


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
    """y: [h, d, r] fp8; scale [h, r] for fp8_einsum bhd,hdr->bhr (GEMM n=r)."""
    h, d, r = y.shape
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    q = torch.empty((h, d, r), device=y.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((h, r), device=y.device, dtype=torch.float32)
    for hi in range(h):
        for ri in range(r):
            col = y[hi, :, ri]
            scale = col.float().abs().max().clamp(min=1e-4) / fp8_info.max
            s[hi, ri] = scale
            q[hi, :, ri] = (col.float() / scale).clamp(
                min=fp8_info.min, max=fp8_info.max
            ).to(torch.float8_e4m3fn)
    return q, s


def cast_fp8_bhd_scale_hd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [b, h, d] fp8; scale [h, d] for fp8_einsum bhd,bhr->hdr (GEMM m=d)."""
    b, h, d = x.shape
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    q = torch.empty((b, h, d), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((h, d), device=x.device, dtype=torch.float32)
    for hi in range(h):
        for di in range(d):
            col = x[:, hi, di]
            scale = col.float().abs().max().clamp(min=1e-4) / fp8_info.max
            s[hi, di] = scale
            q[:, hi, di] = (col.float() / scale).clamp(
                min=fp8_info.min, max=fp8_info.max
            ).to(torch.float8_e4m3fn)
    return q, s


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
    """x: [b, h, r] fp8; scale [h, r] for fp8_einsum bhd,bhr->hdr (GEMM n=r)."""
    b, h, r = x.shape
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    q = torch.empty((b, h, r), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((h, r), device=x.device, dtype=torch.float32)
    for hi in range(h):
        for ri in range(r):
            col = x[:, hi, ri]
            scale = col.float().abs().max().clamp(min=1e-4) / fp8_info.max
            s[hi, ri] = scale
            q[:, hi, ri] = (col.float() / scale).clamp(
                min=fp8_info.min, max=fp8_info.max
            ).to(torch.float8_e4m3fn)
    return q, s


def bench_ms(fn, warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if HAS_TRITON:
        result = triton.testing.do_bench(fn, return_mode="mean")
        if isinstance(result, (list, tuple)):
            result = result[0]
        return float(result) * 1e3
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat * 1e3


# ---------------------------------------------------------------------------
# BF16 einsum
# ---------------------------------------------------------------------------

def test_bmk_bnk_mn_accuracy() -> None:
    for s in (129, 512):
        for m, n, k in [(128, 384, 128), (256, 256, 256)]:
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
                assert diff < atol, (
                    f"s={s}, m={m}, n={n}, k={k}, dtype={dtype}, max_diff={diff:.6g}, atol={atol}"
                )


def test_bhr_hdr_bhd_accuracy() -> None:
    for h, r, d in [(128, 512, 128), (8, 1024, 512)]:
        for b in (4, 32, 128):
            x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            ref_z = ref_bhr_hdr_bhd(x, y)
            z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            einsum("bhr,hdr->bhd", x, y, z)
            diff = calc_diff(z, ref_z)
            assert diff < 1e-5, (
                f"b={b}, h={h}, r={r}, d={d}, max_diff={diff:.6g}"
            )


def test_bhd_hdr_bhr_accuracy() -> None:
    for h, r, d in [(128, 512, 128), (8, 1024, 512)]:
        for b in (4, 32, 128):
            x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            ref_z = ref_bhd_hdr_bhr(x, y)
            z = torch.empty((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            einsum("bhd,hdr->bhr", x, y, z)
            diff = calc_diff(z, ref_z)
            assert diff < 1e-5, (
                f"b={b}, h={h}, r={r}, d={d}, max_diff={diff:.6g}"
            )


# ---------------------------------------------------------------------------
# FP8 einsum（hipBLASLt fp8_gemm 路径，atol 与 test_fp8_gemm 一致）
# ---------------------------------------------------------------------------

def test_fp8_bhr_hdr_bhd_accuracy() -> None:
    for h, r, d in [(8, 1024, 512)]:
        for b in (4, 32, 128):
            x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            x_fp8, x_scale = cast_fp8_bhr(x)
            y_fp8, y_scale = cast_fp8_hdr(y)  # bhr,hdr->bhd: y scale [h, d]
            ref_z = ref_fp8_bhr_hdr_bhd(x_fp8, x_scale, y_fp8, y_scale)
            z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)

            fp8_einsum("bhr,hdr->bhd", (x_fp8, x_scale), (y_fp8, y_scale), z)
            assert calc_diff(z, ref_z) < 1.0, f"FP8 bhr,hdr->bhd b={b}, h={h}, r={r}, d={d}"


def test_fp8_bhd_hdr_bhr_accuracy() -> None:
    for h, r, d in [(8, 1024, 512)]:
        for b in (4, 32, 128):
            x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
            x_fp8, x_scale = cast_fp8_bhd(x)
            y_fp8, y_scale = cast_fp8_hdr_scale_hr(y)
            ref_z = ref_fp8_bhd_hdr_bhr(x_fp8, x_scale, y_fp8, y_scale)
            z = torch.empty((b, h, r), device=DEVICE, dtype=torch.bfloat16)

            fp8_einsum("bhd,hdr->bhr", (x_fp8, x_scale), (y_fp8, y_scale), z)
            assert calc_diff(z, ref_z) < 1.0, f"FP8 bhd,hdr->bhr b={b}, h={h}, r={r}, d={d}"


def test_fp8_bhd_bhr_hdr_accuracy() -> None:
    for h, r, d in [(8, 1024, 512)]:
        for b in (128, 512):
            x = torch.randn((b, h, d), device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
            x_fp8, x_scale = cast_fp8_bhd_scale_hd(x)
            y_fp8, y_scale = cast_fp8_bhr_scale_hr(y)
            ref_z = ref_fp8_bhd_bhr_hdr(x_fp8, x_scale, y_fp8, y_scale)
            # BF16 输出、beta=0：float32 累加 (c=z) 在 hipBLASLt 上会产出 NaN
            z = torch.empty((h, d, r), device=DEVICE, dtype=torch.bfloat16)

            fp8_einsum("bhd,bhr->hdr", (x_fp8, x_scale), (y_fp8, y_scale), z, recipe=(1, 1, 128))
            assert calc_diff(z, ref_z) < 1.0, f"FP8 bhd,bhr->hdr b={b}, h={h}, r={r}, d={d}"


# ---------------------------------------------------------------------------
# 可选性能打印（__main__）
# ---------------------------------------------------------------------------

def run_bmk_bnk_mn_bench() -> None:
    print('Bench "bmk,bnk->mn":')
    s, m, n, k = 4096, 256, 256, 256
    a = torch.randn((s, m, k), dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn((s, n, k), dtype=torch.bfloat16, device=DEVICE)
    d = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
    t_ms = bench_ms(lambda: einsum("bmk,bnk->mn", a, b, d))
    t_s = t_ms * 1e-3
    print(
        f"  {t_ms:.0f} us | {2 * s * m * n * k / t_s / 1e12:.1f} TFLOPS | "
        f"{count_bytes(a, b, d) / t_s / 1e9:.1f} GB/s"
    )


def run_bhr_hdr_bhd_bench() -> None:
    print('Bench "bhr,hdr->bhd":')
    b, h, r, d = 4096, 8, 4096, 1024
    x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
    z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)
    t_ms = bench_ms(lambda: einsum("bhr,hdr->bhd", x, y, z))
    t_s = t_ms * 1e-3
    print(
        f"  b={b} h={h} r={r} d={d}: {t_ms:.0f} us | "
        f"{2 * b * h * r * d / t_s / 1e12:.1f} TFLOPS | "
        f"{count_bytes(x, y, z) / t_s / 1e9:.1f} GB/s"
    )


def run_fp8_bhr_hdr_bhd_bench() -> None:
    print('Bench FP8 "bhr,hdr->bhd":')
    b, h, r, d = 4096, 8, 4096, 1024
    x = torch.randn((b, h, r), device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=DEVICE, dtype=torch.bfloat16)
    ref_z = torch.einsum("bhr,hdr->bhd", x, y)
    x_fp8, x_scale = cast_fp8_bhr(x)
    y_fp8, y_scale = cast_fp8_hdr(y)  # scale [h, d]
    z = torch.empty((b, h, d), device=DEVICE, dtype=torch.bfloat16)
    fp8_einsum("bhr,hdr->bhd", (x_fp8, x_scale), (y_fp8, y_scale), z)
    print(f"  max diff vs bf16 ref: {calc_diff(z, ref_z):.4f}")
    t_ms = bench_ms(lambda: fp8_einsum("bhr,hdr->bhd", (x_fp8, x_scale), (y_fp8, y_scale), z))
    t_s = t_ms * 1e-3
    print(
        f"  {t_ms:.0f} us | {2 * b * h * r * d / t_s / 1e12:.1f} TFLOPS | "
        f"{count_bytes(x_fp8, x_scale, y_fp8, y_scale, z) / t_s / 1e9:.1f} GB/s"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    print("deepgemm einsum tests\n")

    test_bmk_bnk_mn_accuracy()
    print("bmk,bnk->mn accuracy OK")

    test_bhr_hdr_bhd_accuracy()
    print("bhr,hdr->bhd accuracy OK")

    test_bhd_hdr_bhr_accuracy()
    print("bhd,hdr->bhr accuracy OK")

    test_fp8_bhr_hdr_bhd_accuracy()
    print("FP8 bhr,hdr->bhd accuracy OK")

    test_fp8_bhd_hdr_bhr_accuracy()
    print("FP8 bhd,hdr->bhr accuracy OK")

    test_fp8_bhd_bhr_hdr_accuracy()
    print("FP8 bhd,bhr->hdr accuracy OK")

    print()
    run_bmk_bnk_mn_bench()
    run_bhr_hdr_bhd_bench()
    run_fp8_bhr_hdr_bhd_bench()
