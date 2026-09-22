import argparse

import torch
import triton
import triton.language as tl

import deepgemm


# PERF_M = [1, 2, 4, 8, 16, 32, 128, 256, 512, 1024]
# PERF_NK = [(12288, 2048)]
# PERF_NK = [(9216, 2048), (2048, 4096), (12288, 2048)]
PERF_M = list(range(1, 33))
PERF_NK = [(12288, 2560), (2560, 4096), (18432, 2560), (2560, 9216), (10240, 2560)]
SHAPES = [(m, n, k) for n, k in PERF_NK for m in PERF_M]


@triton.jit
def _triton_w8a16_ref_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k_idxs[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, tl.trans(w.to(tl.float16)))

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc *= scale[None, :].to(tl.float32)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_w8a16_ref(x, w, scale):
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"x must be fp16/bf16, got {x.dtype}")
    if w.dtype != torch.int8:
        raise TypeError(f"w must be int8, got {w.dtype}")

    orig_shape = x.shape
    out_dtype = x.dtype
    x_2d = x.contiguous().view(-1, x.shape[-1])
    if x_2d.dtype == torch.bfloat16:
        x_2d = x_2d.to(torch.float16)
    w = w.contiguous()
    scale = scale.contiguous().view(w.shape[0])

    M, K = x_2d.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    block_m = 16 if M <= 16 else 32
    block_n = 64
    block_k = 256
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _triton_w8a16_ref_kernel[grid](
        x_2d,
        w,
        scale,
        out,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    return out.to(out_dtype).view(*orig_shape[:-1], N)


def make_inputs(m, n, k, dtype=torch.float16):
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    w = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="cuda")
    scale = (torch.randn(n, dtype=torch.float16, device="cuda").abs() + 0.5) * 0.01
    return x, w, scale


def check_close(name, out, ref, atol=0.5, rtol=0.02):
    max_diff = (out.float() - ref.float()).abs().max().item()
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    print(f"{name}: passed, max_abs_diff={max_diff:.6f}")


def _is_type2_raw_config(cfg):
    return int(cfg.get("KERNEL_TYPE", 0)) == 2 and int(cfg.get("WEIGHT_LAYOUT", 0)) == 0


def do_bench(fn, cfg=None, warmup=50, rep=200, cache_mode="cold"):
    if cache_mode == "cold":
        def cold_fn():
            if cfg is not None and _is_type2_raw_config(cfg):
                deepgemm.clear_w8a16_dequant_cache()
            return fn()

        for _ in range(warmup):
            cold_fn()
        torch.cuda.synchronize()
        # This Triton version clears L2 before each measured run.  The wrapper
        # above also clears deepgemm's dequant cache for KERNEL_TYPE=2.
        return triton.testing.do_bench(cold_fn, warmup=0, rep=rep)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def config_tag(cfg):
    return (
        f"kt{int(cfg.get('KERNEL_TYPE', 0))}:"
        f"KW={int(cfg.get('NUM_K_WARPS', 0))},"
        f"NPW={int(cfg.get('N_PER_WARP', 0))},"
        f"KPW={int(cfg.get('K_PER_WARP', 0))},"
        f"NW={int(cfg.get('NUM_N_WARPS', 0))},"
        f"SK={int(cfg.get('SPLITK', 0))},"
        f"BM={int(cfg.get('BLOCK_M', 0))},"
        f"WL={int(cfg.get('WEIGHT_LAYOUT', 0))}"
    )


def run_correctness():
    print("=" * 80)
    print("Correctness: deepgemm.w8a16_linear vs embedded Triton reference")
    print("=" * 80)
    for m, n, k in SHAPES:
        x, w, scale = make_inputs(m, n, k, torch.float16)
        out = deepgemm.w8a16_linear(x, w, scale)
        ref = triton_w8a16_ref(x, w, scale)
        check_close(f"M={m} N={n} K={k}", out, ref)


def run_perf(cache_mode="cold"):
    print("=" * 80)
    print(f"Performance: deepgemm.w8a16_linear ({cache_mode} cache)")
    print("=" * 80)
    print(f"{'M':>6s} {'N':>6s} {'K':>6s} {'path':>8s} {'us':>10s}  config")
    print("-" * 96)
    for n, k in PERF_NK:
        for m in PERF_M:
            x, w, scale = make_inputs(m, n, k, torch.float16)
            cfg = deepgemm.get_w8a16_linear_config(m, n, k)
            kernel_type = int(cfg.get("KERNEL_TYPE", 0))
            path = {0: "decode", 1: "prefill", 2: "gemm"}.get(kernel_type, f"kt{kernel_type}")
            us = do_bench(lambda: deepgemm.w8a16_linear(x, w, scale), cfg=cfg, cache_mode=cache_mode) * 1000.0
            print(f"{m:6d} {n:6d} {k:6d} {path:>8s} {us:9.2f}u  {config_tag(cfg)}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true", help="run performance test")
    parser.add_argument("--correctness-only", action="store_true", help="run correctness only")
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "hot"),
        default="cold",
        help="performance cache mode; cold clears L2 and KERNEL_TYPE=2 dequant cache, default: cold",
    )
    args = parser.parse_args()

    if args.perf:
        run_perf(args.cache_mode)
    elif args.correctness_only:
        run_correctness()
    else:
        run_correctness()
        run_perf(args.cache_mode)


if __name__ == "__main__":
    main()
