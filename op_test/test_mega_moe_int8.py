# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Test script for HCU gfx938 MegaMoE INT8 W8A8.

INT8 (W8A8-INT8) version of test_mega_moe_fp8.py:
  - Activations and weights are signed int8 (symmetric quantization).
  - Activation scale: per-token, scale = amax / 127.0
  - Weight scale: per-channel, scale = amax / 127.0
  - DeepEP low_latency_dispatch uses quant_type=1 (symmetric INT8).
  - GEMM: output = (q_a @ w.T) * a_scale * b_scale  (int32 accumulator)
  - Output: BF16

Reference paths were removed together with the legacy triton helpers;
framework-mode reference testing is FP8-only for now.  This script supports
fused-kernel benchmarking only (DG_SKIP_REF=1, the default).

Fused path (unified entry, auto LL/HT by capacity):
  - deepgemm.mega.fp8_mega_moe(is_int8=True)
"""

import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import deepgemm.mega as megamoe

# INT8 reference helpers were removed; only fused benchmarking is supported.
SKIP_REF = os.environ.get("DG_SKIP_REF", "1") == "1"
# Fused kernel mode threshold: at or above this, use HT fused kernel;
# below, use low-latency (LL) fused kernel.
FUSED_MODE_TOKEN_THRESHOLD = int(os.environ.get("FUSED_MODE_TOKEN_THRESHOLD", "256"))


def quantize_int8_per_channel_weight(w: torch.Tensor, eps: float = 1e-10):
    """Per-output-channel symmetric INT8 quantization for weights.

    Args:
      w: [E, N, K] BF16
    Returns:
      wq: [E, N, K] int8
      scale: [E, N] float32 (amax / 127.0)
    """
    amax = w.abs().float().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = amax / 127.0
    wq = torch.round(w.float() / scale).clamp(-127.0, 127.0).to(torch.int8)
    return wq, scale.squeeze(-1).contiguous()


# The fused path graphs by default -- a single self-contained kernel launch.
BENCH_FUSED_CUDAGRAPH = os.environ.get("BENCH_FUSED_CUDAGRAPH", "1") == "1"
# Graph objects must stay alive as long as their captured output tensors are
# read (the outputs live in the graph's private memory pool).
_BENCH_GRAPHS = []


def bench_gpu(fn, warmup=10, iters=50, *args, use_graph=False, **kwargs):
    for _ in range(warmup):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()

    graph = None
    if use_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            _BENCH_GRAPHS.append(graph)
        except Exception as e:
            print(f"bench_gpu: CUDA graph capture failed ({e}); "
                  f"falling back to eager", flush=True)
            graph = None
            torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        if graph is not None:
            graph.replay()
        else:
            out = fn(*args, **kwargs)
        ends[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times.sort()
    return {
        "median": times[iters // 2],
        "p95": times[int(iters * 0.95)],
        "min": times[0],
        "mean": sum(times) / iters,
    }, out


# ============================================================================
# Fused kernel wrappers
# ============================================================================
def run_fused(
    x_local: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights,                   # (w6, sf) tuple from transform_weights_for_mega_moe
    l2_weights,                   # (w6, sf) tuple
    y: torch.Tensor,
    sym_buffer: megamoe.SymmBuffer,
):
    """Unified fused entry: C++ picks the LL/HT config by capacity."""
    num_tokens = x_local.shape[0]
    megamoe.mega_moe_preprocess(
        x_local, topk_idx, topk_weights, sym_buffer, num_tokens, is_int8=True
    )
    megamoe.fp8_mega_moe(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        is_int8=True,
    )
    return y


# ============================================================================
# test: main per-rank entry point
# ============================================================================
def test(
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    tokens: int,
    hidden: int,
    intermediate_hidden: int,
    async_mode: bool,
    debug_mode: bool,
    print_recv_counts: bool,
):
    def init(rank_idx: int):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        print(f"[rank {rank_idx}] >>> calling dist.init_process_group(nccl, rank_idx={rank_idx}, world_size={num_ranks})", flush=True)
        dist.init_process_group("nccl", rank=rank_idx, world_size=num_ranks)
        print(f"[rank {rank_idx}] <<< dist.init_process_group done", flush=True)
        torch.cuda.set_device(rank_idx)
        print(f"[rank {rank_idx}] <<< cuda.set_device({rank_idx}) done, current={torch.cuda.current_device()}", flush=True)
        group = dist.group.WORLD
        print(f"[rank {rank_idx}] <<< got WORLD group", flush=True)
        return group

    group = init(rank_idx)

    print(
        f"[rank {rank_idx}] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')} "
        f"device_id={torch.cuda.current_device()} "
        f"device_name={torch.cuda.get_device_name(rank_idx)}",
        flush=True,
    )

    tokens_local = tokens // num_ranks

    # Generate input (BF16)
    print(f"[rank {rank_idx}] >>> generating input_bf16 [{tokens_local}x{hidden}] on dev={torch.cuda.current_device()}", flush=True)
    input_fp32 = torch.rand(
        tokens_local, hidden, dtype=torch.float32, device=torch.cuda.current_device()
    )
    if debug_mode:
        input_fp32 = torch.ones(
            tokens_local, hidden, device=torch.cuda.current_device(), dtype=torch.float32
        )
    input_bf16 = input_fp32.to(torch.bfloat16)
    del input_fp32
    torch.cuda.empty_cache()
    print(f"[rank {rank_idx}] <<< input_bf16 ready, shape={input_bf16.shape}", flush=True)

    # Generate score -- topk
    print(f"[rank {rank_idx}] >>> generating score/topk", flush=True)
    score_local = torch.rand(
        tokens_local, num_experts, device=torch.cuda.current_device(), dtype=torch.float32
    )
    score_local = torch.softmax(score_local, dim=-1, dtype=torch.float32)
    topk_weights, topk_idx = torch.topk(score_local, num_topk)
    del score_local
    torch.cuda.empty_cache()
    topk_idx = topk_idx.to(torch.int64)
    topk_weights = topk_weights.to(torch.float32)
    print(f"[rank {rank_idx}] <<< topk ready, ids={topk_idx.shape} weights={topk_weights.shape}", flush=True)

    num_local_experts = num_experts // num_ranks

    # Weights (BF16, then quantize to INT8)
    # l1 (gate/up): [num_local_experts, 2*intermediate_hidden, hidden]
    print(f"[rank {rank_idx}] >>> generating l1_weights_bf16 [{num_local_experts}x{2*intermediate_hidden}x{hidden}]", flush=True)
    l1_weights_bf16 = torch.rand(
        num_local_experts,
        2 * intermediate_hidden,
        hidden,
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16,
    )
    if debug_mode:
        l1_weights_bf16 = torch.ones(
            num_local_experts,
            2 * intermediate_hidden,
            hidden,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

    # l2 (down): [num_local_experts, hidden, intermediate_hidden]
    print(f"[rank {rank_idx}] >>> generating l2_weights_bf16 [{num_local_experts}x{hidden}x{intermediate_hidden}]", flush=True)
    l2_weights_bf16 = torch.rand(
        num_local_experts,
        hidden,
        intermediate_hidden,
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16,
    )
    if debug_mode:
        l2_weights_bf16 = torch.ones(
            num_local_experts,
            hidden,
            intermediate_hidden,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

    # --- INT8 quantize weights (per-channel) ---
    print(f"[rank {rank_idx}] >>> INT8 quantizing weights", flush=True)
    l1_weights_int8, l1_weights_sf = quantize_int8_per_channel_weight(l1_weights_bf16)
    l2_weights_int8, l2_weights_sf = quantize_int8_per_channel_weight(l2_weights_bf16)
    print(f"[rank {rank_idx}] <<< INT8 weights ready, l1={l1_weights_int8.shape} l2={l2_weights_int8.shape}", flush=True)

    # --- Prepare MegaMoE weights: gate/up interleave + W6 Marlin pack ---
    # transform_weights_for_mega_moe returns (w6, sf) tuples for the fused
    # path.
    print(f"[rank {rank_idx}] >>> preparing MegaMoE weights (transform_weights_for_mega_moe)", flush=True)
    transformed_l1_weights, transformed_l2_weights = \
        megamoe.transform_weights_for_mega_moe(
            (l1_weights_int8, l1_weights_sf),
            (l2_weights_int8, l2_weights_sf))
    print(f"[rank {rank_idx}] <<< MegaMoE weights ready", flush=True)

    del l1_weights_bf16, l2_weights_bf16
    torch.cuda.empty_cache()

    # =====================================================================
    # 1. Golden reference path
    # =====================================================================

    y = torch.zeros(
        tokens_local, hidden, device=torch.cuda.current_device(), dtype=torch.bfloat16
    )

    x_ref = input_bf16.contiguous()

    if SKIP_REF:
        # The INT8 reference path relied on removed triton helpers; run the
        # fused kernel only and report raw timings.
        combine_ref = None
        ref_source = "skipped"
        t_ref_mean = t_ref_median = float("nan")

    torch.cuda.empty_cache()

    # =====================================================================
    # 2. Fused kernel
    # =====================================================================
    num_max_tokens_per_rank = tokens_local
    sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    print(f"[rank {rank_idx}] SymmBuffer created", flush=True)

    use_ht_fused = tokens_local >= FUSED_MODE_TOKEN_THRESHOLD
    print(f"[rank {rank_idx}] Fused kernel mode: {'HT' if use_ht_fused else 'LL'} (tokens_local={tokens_local}, threshold={FUSED_MODE_TOKEN_THRESHOLD})", flush=True)

    # Unified fused entry: (w6, sf) weight tuples, one Marlin set for both
    # LL and HT.
    t_mega, out = bench_gpu(
        run_fused,
        2, 10,
        x_ref, topk_idx, topk_weights,
        transformed_l1_weights, transformed_l2_weights,
        y, sym_buffer,
        use_graph=BENCH_FUSED_CUDAGRAPH,
    )
    y = out
    del x_ref
    torch.cuda.empty_cache()
    dist.barrier()
    t_mega_mean = t_mega["mean"]
    t_mega_median = t_mega["median"]
    print(f"---------------- fused_mega_moe_int8 total (including GPU execution): {t_mega}ms, rank_idx:{rank_idx}")

    if combine_ref is None:
        # Fused-only run: no reference to compare against.
        print(f"test rank {rank_idx} async mode {async_mode} PASS (INT8, fused-only, DG_SKIP_REF=1)")
    else:
        speedup = t_ref_median / t_mega_median if t_mega_median > 0 else 0.0
        print(f"---------------- speedup (ref_median={ref_source} / fused_median): {t_ref_median:.3f} / {t_mega_median:.3f} = {speedup:.2f}x, rank_idx:{rank_idx}")

        if rank_idx == 0:
            torch.set_printoptions(profile="full")

        # =====================================================================
        # 3. Compare golden reference vs fused kernel
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"[rank {rank_idx}] === DETAILED COMPARISON: combine_ref ({ref_source}) vs y (fused INT8) ===")
        print(f"{'='*70}")
        print(f"  combine_ref shape={combine_ref.shape} dtype={combine_ref.dtype}")
        print(f"  y  shape={y.shape} dtype={y.dtype}")

        ref_min, ref_max = combine_ref.min().item(), combine_ref.max().item()
        fus_min, fus_max = y.min().item(), y.max().item()
        ref_mean, fus_mean = combine_ref.float().mean().item(), y.float().mean().item()
        print(f"  combine_ref: min={ref_min:.6f} max={ref_max:.6f} mean={ref_mean:.6f}")
        print(f"  y : min={fus_min:.6f} max={fus_max:.6f} mean={fus_mean:.6f}")

        diff = (combine_ref.float() - y.float()).abs()
        abs_diff = diff
        rel_diff = diff / (combine_ref.float().abs() + 1e-6)
        print(f"  abs_diff:   max={abs_diff.max().item():.6f} mean={abs_diff.mean().item():.6f} median={abs_diff.median().item():.6f}")
        print(f"  rel_diff:   max={rel_diff.max().item():.6f} mean={rel_diff.mean().item():.6f} median={rel_diff.median().item():.6f}")

        for atol in [1e-3, 1e-2, 1e-1, 0.5, 1.0]:
            n_pass = (abs_diff <= atol).sum().item()
            n_total = abs_diff.numel()
            print(f"  atol={atol:.0e}: {n_pass}/{n_total} pass ({100.0*n_pass/n_total:.2f}%)")

        nan_fus = torch.isnan(y).sum().item()
        inf_fus = torch.isinf(y).sum().item()
        nan_ref = torch.isnan(combine_ref).sum().item()
        inf_ref = torch.isinf(combine_ref).sum().item()
        print(f"  NaN: ref={nan_ref} fus={nan_fus}  |  Inf: ref={inf_ref} fus={inf_fus}")
        print(f"{'='*70}\n")

        # INT8 has larger numerical error than FP8; use looser tolerance
        passed = torch.allclose(combine_ref, y, atol=1.0, rtol=3e-1)
        if passed:
            print(f"test rank {rank_idx} async mode {async_mode} PASS (INT8)")
        else:
            print(f"test rank {rank_idx} async mode {async_mode} FAIL -- mismatch details above")
        assert passed, f"rank {rank_idx}: MEGA MOE INT8 output mismatch (golden vs fused)"

    sym_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    # Default: 4 cards (GPU 0-3), 512 tokens per rank
    # Set HIP_VISIBLE_DEVICES=0,1,2,3 before running, e.g.:
    #   HIP_VISIBLE_DEVICES=0,1,2,3 python test_mega_moe_int8.py 4 512
    num_ranks = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    local_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    print_recv_counts = "--print-recv-counts" in sys.argv[3:]

    num_experts = num_ranks * 48
    num_topk = 6
    hidden = 7168
    intermediate_hidden = 3072
    mp.spawn(
        test,
        args=(
            num_ranks,
            num_experts,
            num_topk,
            num_ranks * local_tokens,
            hidden,
            intermediate_hidden,
            True,   # async_mode
            False,  # debug_mode
            print_recv_counts,
        ),
        nprocs=num_ranks,
        join=True,
    )
