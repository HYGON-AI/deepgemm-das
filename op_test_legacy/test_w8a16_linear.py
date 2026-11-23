# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from triton_kernel.w8a16_gemm import w8a16_gemm as triton_w8a16_gemm  # noqa: E402

import deepgemm  # noqa: E402


SHAPES = [
    (1, 5120, 2048),
    (5, 5120, 2048),
    (16, 5120, 2048),
    (128, 5120, 2048),
    (2048, 5120, 2048),
    (1, 2048, 4096),
    (16, 2048, 4096),
    (128, 2048, 4096),
    (1536, 2048, 4096),
]

PERF_M = [1, 2, 4, 8, 16, 32, 96, 128, 256, 1024, 2048, 4096, 8192]
PERF_NK = [(5120, 2048), (2048, 4096)]


def make_inputs(m, n, k, dtype=torch.float16):
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    w = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="cuda")
    scale = (torch.randn(n, dtype=torch.float16, device="cuda").abs() + 0.5) * 0.01
    return x, w, scale


def golden_fp32(x, w, scale, bias=None):
    out = F.linear(x.float(), w.float() * scale.view(-1, 1).float())
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def check_close(name, out, ref, atol, rtol):
    max_diff = (out.float() - ref.float()).abs().max().item()
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    print(f"{name}: passed, max_abs_diff={max_diff:.6f}")


def do_bench(fn, warmup=50, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=0, rep=rep)


def run_correctness():
    print("=" * 80)
    print("Correctness: deepgemm HIPC w8a16_linear")
    print("=" * 80)
    for dtype in (torch.float16, torch.bfloat16):
        atol = 2.0 if dtype is torch.bfloat16 else 0.5
        for m, n, k in SHAPES:
            x, w, scale = make_inputs(m, n, k, dtype)
            out = deepgemm.w8a16_linear(x, w, scale)
            ref = golden_fp32(x, w, scale)
            check_close(f"dtype={dtype} M={m} N={n} K={k} vs golden", out, ref, atol, 0.02)

            tri = triton_w8a16_gemm(x, w, scale)
            check_close(f"dtype={dtype} M={m} N={n} K={k} vs triton", out, tri, atol, 0.02)

    x, w, scale = make_inputs(16, 5120, 2048, torch.bfloat16)
    scale_bf16 = scale.to(torch.bfloat16)
    out = deepgemm.w8a16_linear(x, w, scale_bf16)
    ref = golden_fp32(x, w, scale_bf16)
    check_close("bf16 scale input", out, ref, 2.0, 0.02)

    x, w, scale = make_inputs(2 * 16, 5120, 2048, torch.float16)
    x3 = x.view(2, 16, 2048)
    out3 = deepgemm.w8a16_linear(x3, w, scale)
    ref3 = golden_fp32(x, w, scale).view(2, 16, 5120)
    check_close("3D input", out3, ref3, 0.5, 0.02)

    x, w, scale = make_inputs(16, 2048, 4096, torch.float16)
    bias = torch.randn(2048, dtype=torch.float16, device="cuda")
    out = deepgemm.w8a16_linear(x, w, scale, bias=bias)
    ref = golden_fp32(x, w, scale, bias=bias)
    check_close("bias", out, ref, 0.5, 0.02)

    x, w, scale = make_inputs(1, 5120, 2048, torch.float16)
    cfg = deepgemm.get_w8a16_linear_config(1, 5120, 2048)
    cfg["WEIGHT_LAYOUT"] = 1
    try:
        deepgemm.w8a16_linear(x, deepgemm.w8a16_marlin_weight(w), scale, config=cfg)
        raise AssertionError("decode should reject WEIGHT_LAYOUT=1")
    except ValueError as exc:
        assert "decode" in str(exc)

    x, w, scale = make_inputs(256, 5120, 2048, torch.float16)
    deepgemm.clear_w8a16_dequant_cache()
    deepgemm.w8a16_linear(x, w, scale)
    cached_out = deepgemm.w8a16_linear(x, w, scale)
    tri = triton_w8a16_gemm(x, w, scale)
    check_close("dequant cache hit", cached_out, tri, 0.5, 0.02)
    w[0, 0] = w[0, 0] + 1
    scale[0] = scale[0] * 1.1
    refreshed_out = deepgemm.w8a16_linear(x, w, scale)
    refreshed_tri = triton_w8a16_gemm(x, w, scale)
    check_close("dequant cache invalidation", refreshed_out, refreshed_tri, 0.5, 0.02)

    for m, n, k in ((32, 2048, 4096), (128, 5120, 2048)):
        x, w, scale = make_inputs(m, n, k, torch.float16)
        w_marlin = deepgemm.w8a16_marlin_weight(w)
        assert w_marlin.shape == w.shape
        assert w_marlin.dtype == torch.int8
        cfg = deepgemm.get_w8a16_linear_config(m, n, k)
        cfg["WEIGHT_LAYOUT"] = 1
        out = deepgemm.w8a16_linear(x, w_marlin, scale, config=cfg)
        ref = golden_fp32(x, w, scale)
        check_close(f"marlin layout M={m} N={n} K={k}", out, ref, 0.5, 0.02)


def run_perf():
    print("=" * 80)
    print("Performance: HIPC raw/marlin vs Triton")
    print("=" * 80)
    print(
        f"{'M':>6s} {'N':>6s} {'K':>6s} {'path':>8s} {'layout':>8s} "
        f"{'hipc_us':>10s} {'triton_us':>10s} {'speedup':>9s}"
    )
    print("-" * 78)
    for n, k in PERF_NK:
        for m in PERF_M:
            x, w, scale = make_inputs(m, n, k, torch.float16)
            cfg = deepgemm.get_w8a16_linear_config(m, n, k)
            kernel_type = int(cfg.get("KERNEL_TYPE", 0))
            path = {0: "decode", 1: "prefill", 2: "gemm"}.get(kernel_type, f"kt{kernel_type}")
            triton_ms = do_bench(lambda: triton_w8a16_gemm(x, w, scale))
            hipc_ms = do_bench(lambda: deepgemm.w8a16_linear(x, w, scale))
            hipc_us = hipc_ms * 1000.0
            triton_us = triton_ms * 1000.0
            speedup = triton_us / hipc_us
            print(
                f"{m:6d} {n:6d} {k:6d} {path:>8s} {'raw':>8s} "
                f"{hipc_us:9.2f}u {triton_us:9.2f}u {speedup:8.3f}x"
            )
            if kernel_type != 0:
                w_marlin = deepgemm.w8a16_marlin_weight(w)
                cfg_marlin = dict(cfg)
                cfg_marlin["WEIGHT_LAYOUT"] = 1
                marlin_path = "prefill" if kernel_type == 2 else path
                hipc_ms = do_bench(lambda: deepgemm.w8a16_linear(x, w_marlin, scale, config=cfg_marlin))
                hipc_us = hipc_ms * 1000.0
                speedup = triton_us / hipc_us
                print(
                    f"{m:6d} {n:6d} {k:6d} {marlin_path:>8s} {'marlin':>8s} "
                    f"{hipc_us:9.2f}u {triton_us:9.2f}u {speedup:8.3f}x"
                )
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true", help="run the full HIPC-vs-Triton perf sweep")
    args = parser.parse_args()
    run_correctness()
    if args.perf:
        run_perf()


if __name__ == "__main__":
    main()
