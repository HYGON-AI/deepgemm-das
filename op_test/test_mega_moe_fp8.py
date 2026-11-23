# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Test script for HCU gfx938 MegaMoE FP8 W8A8.

Test structure is fully aligned with mega-moe/V2/tests/test_mega_moe12.py:
  - Golden reference path (framework mode only): run_ref_ll_framework uses
    deep_ep.Buffer.low_latency_dispatch + deepgemm masked GEMM
    + lightop SiLU/quant + deep_ep.Buffer.low_latency_combine;
    run_ref_ht_framework uses deep_ep.Buffer normal dispatch
    + lightop ep_scatter + contiguous GEMM + lightop ep_gather
    + deep_ep.Buffer.combine --combine_ref
  - Fused kernel path (replaced): V2's mega_buffer.low_latency_mega_moe_channelwise
    is replaced by deepgemm.mega.mega_moe.fp8_mega_moe (mega_moe_fp8_kernel)
  - Comparison: |combine_ref - y| <= atol + atol_frac*amax|ref| + rtol*|ref|
    elementwise, plus a no-NaN/Inf check on both outputs.
    DG_INIT_MODE selects the data distribution: "randn" (default, scaled
    randn*0.05 + softmax topk; atol=0.00355 + 0.03*amax + 0.02*|ref|) or "nv"
    (NV-style unit-variance randn + raw topk scores; 0.02*amax + 0.02*|ref|).
"""

import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import deep_ep
from deepgemm import m_grouped_fp8_gemm_nt_contiguous, marlin_fp8_contiguous_weight
import deepgemm.mega as megamoe

# NOTE: the fused kernel picks LL/HT per rank inside the C++ launcher
# (expected_tokens_per_expert vs DG_MEGA_HT_EXPECTED/--THRESHOLD, default
# 13.0 in expected units).  There is no python-side fused threshold anymore.
# DeepEP normal buffer size and config (aligned with HCU baseline)
# Supernode EP16: DeepEP normal-mode NVL buffer is mapped once per peer, so
# 2 GiB x 15 peers can exhaust the mapping space on a 16-card node. Allow an
# explicit smaller cap while keeping the 8-card default unchanged.
DEEPEP_BUFFER_BYTES = int(float(os.environ.get("DG_DEEPEP_BUFFER_BYTES", "2.0e9")))
DEEPEP_CONFIG = (24, 8, 256)
DEEPEP_EXPERT_ALIGNMENT = 256
ACTIVATION_CLAMP = 10.0

# ---------------------------------------------------------------------------
# Framework-aligned reference path (SGLang deepep, --deepep-mode auto).
# The framework resolves the mode per *batch type* (extend -> NORMAL,
# decode/verify -> LOW_LATENCY), not per a token threshold.  DG_REF_MODE
# replays that selection for captured vectors:
#   "framework" (default) -> auto-style: tokens <= DG_DEEPEP_LL_CAP -> LL,
#                            else normal
#   "ll"        -> force low-latency ref (errors if tokens > DG_DEEPEP_LL_CAP)
#   "normal"    -> force normal (high-throughput) ref
# DG_DEEPEP_LL_CAP mirrors SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
# (server default 128; the V4.1 launcher leaves the override commented out).
# ---------------------------------------------------------------------------
DG_REF_MODE = os.environ.get("DG_REF_MODE", "framework").strip().lower()
DG_DEEPEP_LL_CAP = int(os.environ.get("DG_DEEPEP_LL_CAP", "128"))
# /home/proj_dpsk-v4/configs/deepep_IntraConfig.json used by the V4.1 launcher.
FRAMEWORK_DEEPEP_DISPATCH_CONFIG = dict(
    num_sms=48,
    num_max_nvl_chunked_send_tokens=6,
    num_max_nvl_chunked_recv_tokens=256,
    num_max_rdma_chunked_send_tokens=6,
    num_max_rdma_chunked_recv_tokens=128,
)
FRAMEWORK_DEEPEP_COMBINE_CONFIG = dict(
    num_sms=48,
    num_max_nvl_chunked_send_tokens=4,
    num_max_nvl_chunked_recv_tokens=256,
    num_max_rdma_chunked_send_tokens=6,
    num_max_rdma_chunked_recv_tokens=128,
)
# AUTO mode: num_qps_per_rank = max(config num_sms, num_experts / group_size)
FRAMEWORK_DEEPEP_QPS = int(os.environ.get("DG_DEEPEP_QPS", "48"))

# Verbose / debug controls (all default off for performance testing)
# Set VERBOSE=1 env var to enable stage-level debug prints.
# Set DEBUG_ENABLED=1 env var to enable per-stage tensor tracing.
VERBOSE = os.environ.get("VERBOSE", "0") == "1"
DEBUG_ENABLED = os.environ.get("DEBUG_ENABLED", "0") == "1"

# Benchmark controls
#   BENCH_WARMUP / BENCH_ITERS: warmup and timed iteration counts.
#   BENCH_ITER_BARRIER=1 (default): dist.barrier() before each timed iteration so
#     cross-rank drift does not accumulate into the measured dispatch wait.
#   BENCH_REF_CUDAGRAPH=1: capture the non-fused ref path into a CUDA graph and
#     replay it (removes per-iteration launch overhead from ref timing). Falls
#     back to eager automatically if capture fails.
#   BENCH_FUSED_CUDAGRAPH=1 (default): capture the fused mega_moe call into a
#     CUDA graph and replay it -- same reason as the ref graph. The normal
#     (DeepEP all_to_all) ref path cannot be captured, so it stays eager;
#     the fused path is a single self-contained kernel launch and captures
#     cleanly (falls back to eager if capture fails).
BENCH_WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
BENCH_ITERS = int(os.environ.get("BENCH_ITERS", "20"))
BENCH_ITER_BARRIER = os.environ.get("BENCH_ITER_BARRIER", "1") == "1"
# Ref CUDA graph defaults: the LL-masked ref graphs (NV times its baseline
# under cudagraph too); the normal-contiguous ref cannot be captured on ROCm
# (DeepEP all_to_all is not capturable), so it stays eager.
# The fused path graphs by default (BENCH_FUSED_CUDAGRAPH) -- it is a single
# self-contained kernel launch; NV times it via bench_kineto instead.
BENCH_REF_CUDAGRAPH_LL = os.environ.get("BENCH_REF_CUDAGRAPH_LL", "1") == "1"
BENCH_REF_CUDAGRAPH_NORMAL = os.environ.get("BENCH_REF_CUDAGRAPH_NORMAL", "0") == "1"
BENCH_FUSED_CUDAGRAPH = os.environ.get("BENCH_FUSED_CUDAGRAPH", "1") == "1"
# Repeat the fused-vs-ref comparison this many extra times with the same inputs
# (catches intermittent/nondeterministic failures). 1 = single check.
CORRECTNESS_ITERS = int(os.environ.get("CORRECTNESS_ITERS", "1"))
# BENCH_KINETO=1 (default): additionally time the fused kernel the same way the
# NV test does (deep_gemm.testing.bench_kineto): torch.profiler CUDA events,
# 8GB L2 flush + cuda._sleep + rank barrier before each iteration. Reports pure
# per-kernel GPU time, excluding launch gaps. Falls back silently if the
# profiler is unavailable on this build.
BENCH_KINETO = os.environ.get("BENCH_KINETO", "1") == "1"
SKIP_REF = os.environ.get("SKIP_REF", "0") == "1"  # skip DeepEP ref path (debug)
DEBUG_EXPERT = int(os.environ.get("DEBUG_EXPERT", "0"))  # which expert to trace
DEBUG_RANK = int(os.environ.get("DEBUG_RANK", "0"))     # only print on this rank
DEBUG_TOPN = int(os.environ.get("DEBUG_TOPN", "5"))     # print first N tokens
DEBUG_NELEM = int(os.environ.get("DEBUG_NELEM", "8"))  # print first N elements per token
_debug_printed = {"ll": False, "normal": False}


def _dbg_tensor(name, t, rank_idx, expert=None, offset=0, tag="ll"):
    """Print first DEBUG_TOPN rows x DEBUG_NELEM cols of a tensor for debug."""
    global _debug_printed
    if not DEBUG_ENABLED or rank_idx != DEBUG_RANK:
        return
    if _debug_printed.get(tag, False):
        return
    if t is None:
        print(f"[DBG {name}] None", flush=True)
        return
    if expert is not None and t.dim() == 3:
        # 3D [E, M, H] --extract expert slice
        t = t[expert]
    if t.dim() >= 2:
        n = min(DEBUG_TOPN, t.shape[0])
        vals = t[:n, :DEBUG_NELEM].float().tolist()
    else:
        n = min(DEBUG_TOPN, t.shape[0])
        vals = t[:n].float().tolist()
    print(f"[DBG {name}] shape={list(t.shape)} first{n}x{DEBUG_NELEM if t.dim()>=2 else 1}={vals}", flush=True)


def _dbg_done(tag):
    global _debug_printed
    _debug_printed[tag] = True


# ============================================================================
# Hang watchdog (DG_HANG_WATCHDOG_S): every N seconds dump python stacks +
# workspace/signal counters + dg_hang_records into rank{R}.{i}.txt.
#
# Snapshot path: ctypes.string_at directly on the device VAs — host reads via
# BAR and does NOT depend on the GPU queue, so it still produces a snapshot
# while the fused kernel is wedged spinning (hipMemcpyAsync on a side stream
# can fall back to a shader copy that queues behind the wedged kernel and
# blocks forever; it is kept only as a fallback).
# ============================================================================
def _start_hang_watchdog(sym_buffer, rank_idx, num_ranks, num_experts, max_tokens, num_topk):
    import threading
    import faulthandler
    import time
    import ctypes
    period = float(os.environ["DG_HANG_WATCHDOG_S"])
    dump_dir = os.environ.get("DG_WATCHDOG_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "log", "hang_watchdog")
    os.makedirs(dump_dir, exist_ok=True)

    def _align(v, a):
        return (v + a - 1) // a * a

    # Copy only the counter prefixes — the full local_buffer (300+MB) is too
    # slow to snapshot while the fused kernel saturates memory bandwidth.
    nr = num_ranks
    nepr_ = num_experts // nr
    _max_recv = nr * max_tokens
    _pool_tokens = _align(_max_recv * min(num_topk, nepr_) + nepr_ * 191, 192)
    _npb = _pool_tokens // 16
    _ipc_need = _align(nr * 8, 16) + _align(nr * 8, 16) + _align(4, 16) \
        + num_experts * 8 + nepr_ * 8 + num_experts * 8  # recv_count+recv_sum+send_count
    _loc_need = 16 + _align(_align(_npb, 2), 4) * 4 \
        + _npb * 8 + _npb * 4 + _npb * 16 * 12 + 9216  # +metadata+phase+warp arrays
    _ipc_h = torch.empty(_ipc_need, dtype=torch.uint8, pin_memory=True)
    _loc_h = torch.empty(_loc_need, dtype=torch.uint8, pin_memory=True)
    _sig_bytes = max(num_ranks, 4) * 16
    _sig_h = torch.empty(_sig_bytes, dtype=torch.uint8, pin_memory=True)
    _hr_bytes = 1024 * 4
    _hr_h = torch.empty(_hr_bytes, dtype=torch.uint8, pin_memory=True)
    _hip = ctypes.CDLL("libamdhip64.so")
    _wd_side = torch.cuda.Stream()
    try:
        import deepgemm.mega._C as mega_C
        _hr_ptr = mega_C.get_hang_record_ptr()
    except Exception:
        _hr_ptr = 0

    _SITE = {1: 'subset_barrier', 2: 'rank_barrier', 3: 'grid_sync_leader',
             4: 'fetch_recv', 5: 'L1wait', 6: 'L2wait', 7: 'SiLUwait',
             8: 'math_warp_barrier', 9: 'pull_send_count'}

    def _snap(dst_t, dev_ptr, nbytes):
        """Host-direct read of a device buffer; side-stream memcpy fallback."""
        try:
            dst_t.copy_(torch.frombuffer(bytearray(
                ctypes.string_at(dev_ptr, nbytes)), dtype=torch.uint8))
            return "string_at"
        except Exception:
            _hip.hipMemcpyAsync(
                ctypes.c_void_p(dst_t.data_ptr()),
                ctypes.c_void_p(dev_ptr),
                ctypes.c_size_t(nbytes), ctypes.c_int(2),
                ctypes.c_void_p(_wd_side.cuda_stream))
            _wd_side.synchronize()
            return "memcpy"

    def _wd():
        i = 0
        while True:
            time.sleep(period)
            stag = str(i)
            i += 1
            try:
                with open(os.path.join(dump_dir, f"rank_idx{rank_idx}.{stag}.stack.txt"), "w") as f:
                    faulthandler.dump_traceback(f)
            except Exception:
                pass
            try:
                ipc = _ipc_h
                loc = _loc_h
                sig_host = _sig_h
                how = [
                    _snap(ipc, sym_buffer.handle.buffer_ptr, _ipc_need),
                    _snap(loc, sym_buffer.handle.local_buffer_ptr, _loc_need),
                    _snap(sig_host, sym_buffer.handle.signal_buffer_ptr, _sig_bytes),
                ]
                if _hr_ptr:
                    how.append(_snap(_hr_h, _hr_ptr, _hr_bytes))

                ipc_header = _align(nr * 8, 16) + _align(nr * 8, 16) + _align(4, 16)
                ws = ipc[ipc_header:]
                recv_count = ws[: num_experts * 8].view(torch.int64)
                nepr = num_experts // nr
                recv_sum = ws[num_experts * 8: num_experts * 8 + nepr * 8].view(torch.int64)
                # send_count moved into the IPC workspace (peer pull source).
                send = ws[num_experts * 8 + nepr * 8:
                          num_experts * 8 + nepr * 8 + num_experts * 8].view(torch.int64)
                pool_tokens = _align(nr * max_tokens * min(num_topk, nepr)
                                     + nepr_ * 191, 192)
                npb = pool_tokens // 16
                gsync = loc[:16].view(torch.int32)
                l1a_off = 16
                l1a_len = _align(_align(npb, 2), 4)
                l1a = loc[l1a_off: l1a_off + l1a_len * 4].view(torch.int32)
                l2m_off = l1a_off + l1a_len * 4
                l2m = loc[l2m_off: l2m_off + npb * 8].view(torch.int64)
                qd_off = l2m_off + npb * 8
                qd = loc[qd_off: qd_off + npb * 4].view(torch.int32)
                meta_end = qd_off + npb * 4 + npb * 16 * 12
                ph_off = _align(meta_end, 16)
                ph = loc[ph_off: ph_off + 512].view(torch.int32)
                phd = loc[ph_off + 512: ph_off + 1024].view(torch.int32)
                wp = loc[ph_off + 1024: ph_off + 1024 + 2048 * 4].view(torch.int32)
                sig = sig_host[: nr * 16].view(torch.int32).view(4, nr)

                L = []
                L.append(f"rank_idx={rank_idx} snap={stag} via={how} npb={npb}")
                for g in range(3):
                    v = int(gsync[g])
                    L.append(f"grid_sync[{g}]: count={v & 0xffff} epoch={(v >> 16) & 0xffff}")
                L.append(f"send_count hi: min={int((send >> 32).min())} "
                         f"max={int((send >> 32).max())} "
                         f"lo_sum={int((send & 0xffffffff).sum())}")
                L.append(f"recv_sum hi: min={int((recv_sum >> 32).min())} "
                         f"max={int((recv_sum >> 32).max())} "
                         f"lo_sum={int((recv_sum & 0xffffffff).sum())}")
                rcv = recv_count.view(nr, nepr)
                L.append(f"recv_count per_src={[int((rcv[r] & 0xffffffff).sum()) for r in range(nr)]}")
                for t in range(4):
                    L.append(f"signal tag{t}: {[int(v) for v in sig[t]]}")
                L.append(f"l1_arrival: nonzero={int((l1a != 0).sum())} "
                         f"partial={[int(v) for v in l1a if int(v) != 0][:20]}")
                L.append(f"l2_mask: nonzero={int((l2m != 0).sum())} full={int((l2m == -1).sum())}")
                L.append(f"quant_done: nonzero={int((qd != 0).sum())} done2={int((qd == 2).sum())}")
                import collections
                hist = collections.Counter(int(v) for v in ph[:72])
                L.append(f"phase_hist: {dict(sorted(hist.items()))}")
                stuck = [(int(b), int(ph[b]), int(phd[b])) for b in range(72)
                         if int(ph[b]) != 50]
                L.append(f"stuck_blocks(<50): {stuck[:80]}")
                wp_v = wp.view(128, 16)
                for b in range(4):
                    L.append(f"warp_prog blk{b}: {[int(v) for v in wp_v[b]]}")

                # dg_hang_records: [magic 'HANG', site, tid, a, b, c, clk, seq]
                # per block; probe area at u32 index >= 800 (P1 peer-pointer
                # snapshot + P2 broadcast-dst mask).
                if _hr_ptr:
                    hr = _hr_h.view(torch.int32)
                    seq = int(hr[834])
                    L.append(f"hang_seq={seq}")
                    recs = []
                    for b in range(128):
                        r = hr[b * 8: b * 8 + 8]
                        if int(r[0]) == 0x48414E47 and int(r[7]) == seq:
                            recs.append(
                                f"blk{b} {_SITE.get(int(r[1]), int(r[1]))} "
                                f"tid={int(r[2])} a={int(r[3])} b={int(r[4])} "
                                f"c={int(r[5])} clk={int(r[6])}")
                    L.append(f"hang_records({len(recs)}): {recs[:80]}")
                    if (int(hr[832]) & 0xffff0000) == 0x500E0000:
                        L.append(f"P1 tag={hex(int(hr[832]))} "
                                 f"bcast_dst_mask={bin(int(hr[833]))} "
                                 f"seq@{int(hr[834])}")
                        pbp = hr[800: 800 + nr * 2].view(torch.int64)
                        psp = hr[816: 816 + nr * 2].view(torch.int64)
                        L.append(f"peer_buf_ptrs={[hex(int(v)) for v in pbp]}")
                        L.append(f"peer_sig_ptrs={[hex(int(v)) for v in psp]}")
                with open(os.path.join(dump_dir, f"rank_idx{rank_idx}.{stag}.txt"), "w") as f:
                    f.write("\n".join(L) + "\n")
            except Exception as e:
                try:
                    with open(os.path.join(dump_dir, f"rank_idx{rank_idx}.{stag}.err.txt"), "w") as f:
                        f.write(repr(e))
                except Exception:
                    pass
        # Do NOT os._exit here: mp.spawn join() would SIGTERM sibling ranks
        # mid-dump. The outer `timeout` kills everything.

    threading.Thread(target=_wd, daemon=True).start()


# ============================================================================
# bench_gpu: identical to V2 test_mega_moe12.py
# ============================================================================
# Graph objects must stay alive as long as their captured output tensors are
# read (the outputs live in the graph's private memory pool).
_BENCH_GRAPHS = []


def _uneven_all_gather(tensor, group=None):
    world_size = dist.get_world_size(group)
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size, group=group)
    all_sizes = [int(s.item()) for s in all_sizes]
    max_size = max(all_sizes)
    if tensor.shape[0] < max_size:
        padding = torch.zeros(
            (max_size - tensor.shape[0],) + tuple(tensor.shape[1:]),
            dtype=tensor.dtype, device=tensor.device)
        padded = torch.cat([tensor, padding], dim=0)
    else:
        padded = tensor.contiguous()
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    return torch.cat(
        [gathered[i].narrow(0, 0, all_sizes[i]) for i in range(world_size)], dim=0)


def bench_gpu(fn, *args, warmup=BENCH_WARMUP, iters=BENCH_ITERS,
              barrier_fn=None, use_graph=False, **kwargs):
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
        if barrier_fn is not None:
            barrier_fn()
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
# bench_kineto: same methodology as NV deep_gemm.testing.bench_kineto --
#   profiler-measured per-kernel GPU time, L2 flushed before every iteration,
#   torch.cuda._sleep + optional rank barrier to absorb launch overhead.
# ============================================================================
def bench_kineto(fn, kernel_names, num_tests: int = 30,
                 barrier=None, flush_l2: bool = True):
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)

    # By default, flush L2 with an excessive 8 GB memset (same as NV)
    flush_l2_size = int(8e9 // 4)

    # Warmup once (autotuning kernels may print on first call)
    fn()

    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule, acc_events=True) as profiler:
        for _ in range(2):
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
                if barrier is not None:
                    # Large sleep kernel + barrier to eliminate unbalanced CPU
                    # launch overhead (same as NV)
                    if hasattr(torch.cuda, "_sleep"):
                        torch.cuda._sleep(int(2e7))  # ~10ms
                    barrier()
                fn()
            torch.cuda.synchronize()
            profiler.step()

    # Parse the profiling table
    prof_lines = profiler.key_averages().table(
        sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names

    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num if total_num > 0 else 0)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


# ============================================================================
# run_ref_ll_framework: low-latency reference aligned with the SGLang framework
#   (_DeepEPDispatcherImplLowLatency + forward_groupgemm_w8a8_fp8_masked):
#     low_latency_dispatch(cap=DG_DEEPEP_LL_CAP, quant_type=2, fp8_round_scale=F)
#     -> deepgemm.m_grouped_fp8_gemm_nt_masked (cu=4*CU, expected_m per framework)
#     -> lightop fuse_silu_mul_fp8_quant_ep(tokens_per_expert=masked_m, limit)
#     -> deepgemm.m_grouped_fp8_gemm_nt_masked
#     -> low_latency_combine(zero_copy=False, return_recv_hook=False)
# ============================================================================
def run_ref_ll_framework(
    num_ranks: int,
    num_experts: int,
    tokens_local: int,
    x_local: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_weights_sf: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_weights_sf: torch.Tensor,
    async_mode: bool,
    buffer,
    rank_idx: int = 0,
    ll_cap: int = None,
):
    from deepgemm import m_grouped_fp8_gemm_nt_masked
    from lightop.activation import fuse_silu_mul_fp8_quant_ep

    if ll_cap is None:
        ll_cap = DG_DEEPEP_LL_CAP
    if tokens_local > ll_cap:
        raise RuntimeError(
            f"LL capacity exceeded: tokens={tokens_local} > cap={ll_cap} "
            "(framework raises DeepepDispatcherTokenOverflow)"
        )
    num_topk = topk_idx.shape[1]

    # --- dispatch_a (framework: cap is a fixed config value, not per-call tokens) ---
    dispatch_out, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x_local,
        topk_idx,
        topk_weights,
        ll_cap,
        num_experts,
        2,      # quant_type=2: FP8 payload + per-token scales
        False,  # fp8_round_scale
        0,
        async_mode,
        False,  # return_recv_hook
    )
    if async_mode:
        event.current_stream_wait()

    (ll_input_fp8, ll_input_scale) = dispatch_out
    num_experts_ll, m_ll, k_ll = ll_input_fp8.size()

    # --- expected_m exactly as forward_groupgemm_w8a8_fp8_masked ---
    # min(m, (tokens * group_size * topk + num_experts) // num_experts)
    expected_m_ll = min(
        m_ll,
        (tokens_local * num_ranks * num_topk + num_experts) // num_experts,
    )

    # --- L1 masked GEMM (deepgemm entry, cu=None -> 4 * physical CU) ---
    n_ll = l1_weights.numel() // (l1_weights.size(0) * l1_weights.size(1) * 64)
    gateup_output = torch.empty(
        (num_experts_ll, m_ll, n_ll),
        device=ll_input_fp8.device,
        dtype=torch.bfloat16,
    )
    m_grouped_fp8_gemm_nt_masked(
        (ll_input_fp8, ll_input_scale),
        (l1_weights, l1_weights_sf),
        gateup_output,
        recv_count,
        expected_m_ll,
    )

    # --- SiLU + Mul + FP8 quant via lightop (framework HCU op) ---
    q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant_ep(
        input=gateup_output,
        fp8type=0,
        tokens_per_expert=recv_count,
        limit=ACTIVATION_CLAMP,
    )

    # --- L2 masked GEMM ---
    n2_ll = l2_weights.numel() // (l2_weights.size(0) * l2_weights.size(1) * 64)
    down_output = torch.empty(
        (num_experts_ll, m_ll, n2_ll),
        device=q_a2_all.device,
        dtype=torch.bfloat16,
    )
    m_grouped_fp8_gemm_nt_masked(
        (q_a2_all, q_a2_scale),
        (l2_weights, l2_weights_sf),
        down_output,
        recv_count,
        expected_m_ll,
    )

    # --- combine_a (framework kwargs) ---
    combined_ref, event, hook = buffer.low_latency_combine(
        down_output,
        topk_idx,
        topk_weights,
        handle,
        zero_copy=False,
        async_finish=async_mode,
        return_recv_hook=False,
    )
    if async_mode:
        event.current_stream_wait()

    return combined_ref


# ============================================================================
# run_ref_ht_framework: normal-dispatch reference aligned with the SGLang
#   framework (_DeepEPDispatcherImplNormal + forward_groupgemm_w8a8_fp8_contiguous):
#     lightop per_token_quant_fp8 -> get_dispatch_layout -> buffer.dispatch
#       (framework Config num_sms=48, expert_alignment=256)
#     -> lightop ep_scatter (plain byte scatter, no weighting)
#     -> m_grouped_fp8_gemm_nt_contiguous x2 (lightop fuse_silu_mul_fp8_quant
#        between them)
#     -> lightop ep_gather (applies topk weights) -> buffer.combine
#   Falls back to the triton ep_scatter/ep_gather pair when lightop.moe is
#   unavailable -- both sides apply topk weights exactly once either way.
# ============================================================================
def run_ref_ht_framework(
    num_ranks: int,
    num_experts: int,
    tokens_local: int,
    x_local: torch.Tensor,           # bf16 [tokens_local, hidden]
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_weights_sf: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_weights_sf: torch.Tensor,
    async_mode: bool,
    buffer: deep_ep.Buffer,
    rank_idx: int = 0,
    hidden: int = 7168,
    intermediate_hidden: int = 3072,
    print_recv_counts: bool = False,
):
    from lightop.quant import per_token_quant_fp8
    from lightop.activation import fuse_silu_mul_fp8_quant
    try:
        from lightop import moe as lightop_moe
    except Exception:
        lightop_moe = None

    num_local_experts = num_experts // num_ranks
    num_topk = topk_idx.shape[1]
    device = x_local.device

    # --- dispatch_a (normal): framework quantizes with lightop ---
    x_fp8, x_scale = per_token_quant_fp8(x_local)
    topk_idx = topk_idx.to(torch.int64)

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
        previous_event=None,
        async_finish=async_mode,
        allocate_on_comm_stream=False,
    )

    dispatch_result = buffer.dispatch(
        x=(x_fp8, x_scale),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=deep_ep.Config(**FRAMEWORK_DEEPEP_DISPATCH_CONFIG),
        previous_event=previous_event,
        allocate_on_comm_stream=False,
        expert_alignment=DEEPEP_EXPERT_ALIGNMENT,
        async_finish=async_mode,
    )
    if len(dispatch_result) == 6:
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, event = dispatch_result
    elif len(dispatch_result) == 7:
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, _, handle, event = dispatch_result
    else:
        raise RuntimeError(f"unexpected dispatch return arity: {len(dispatch_result)}")
    if event is not None and hasattr(event, "current_stream_wait") and getattr(event, "event", None) is not None:
        event.current_stream_wait()

    if VERBOSE and print_recv_counts and rank_idx == 0:
        print(f"[NORMAL-FW] recv_counts: {recv_counts}", flush=True)

    # --- DeepEPNormalDispatchOutput unpack ---
    if isinstance(recv_x, tuple):
        recv_x_fp8, recv_x_scale = recv_x
    else:
        r_amax = recv_x.abs().float().amax(dim=1, keepdim=True).clamp(min=1e-4)
        recv_x_scale = (r_amax / 448.0).view(-1, 1).contiguous()
        recv_x_fp8 = (recv_x / (r_amax / 448.0)).clamp(-448, 448).to(torch.float8_e4m3fn)
    recv_x_fp8 = recv_x_fp8.contiguous()
    # per-token fp8 scale is [num_recv] or [num_recv, 1]; keep the [*, 1]
    # layout lightop ep_scatter writes into input_tensor[1].
    recv_x_scale = recv_x_scale.float().view(-1, 1).contiguous()
    recv_topk_idx = recv_topk_idx.contiguous()
    recv_topk_weights = recv_topk_weights.float().contiguous()
    # framework gets num_recv_tokens_per_expert as a host list and copies it
    # to GPU int32 for ep_scatter.
    counts_gpu = torch.as_tensor(recv_counts, dtype=torch.int32).to(device)

    counts_list = counts_gpu.cpu().tolist()
    all_tokens = int(sum(counts_list))
    counts_are_aligned = all(c % DEEPEP_EXPERT_ALIGNMENT == 0 for c in counts_list)
    if all_tokens <= 0:
        out = torch.zeros(
            (tokens_local, hidden), dtype=torch.bfloat16, device=device
        )
        combine_result = buffer.combine(
            out, handle=handle,
            config=deep_ep.Config(**FRAMEWORK_DEEPEP_COMBINE_CONFIG),
            async_finish=async_mode,
        )
        combine_event = combine_result[-1]
        if combine_event is not None and hasattr(combine_event, "current_stream_wait"):
            combine_event.current_stream_wait()
        return combine_result[0]

    num_recv = recv_x_fp8.shape[0]
    topk_ids_i64 = recv_topk_idx.to(torch.int64).contiguous()

    # --- ep_scatter: framework prefers lightop (plain byte scatter) ---
    input_tensor = torch.empty(
        (all_tokens, hidden), dtype=recv_x_fp8.dtype, device=device
    )
    input_scale = torch.empty(
        (all_tokens, 1), dtype=torch.float32, device=device
    )
    use_lightop_scatter = (
        lightop_moe is not None
        and hasattr(lightop_moe, "ep_scatter")
        and counts_are_aligned
    )
    output_index = torch.full(
        (num_recv, num_topk), -1, dtype=torch.int32, device=device
    )
    if use_lightop_scatter:
        m_indices = torch.full(
            (all_tokens,), -1, dtype=torch.int32, device=device
        )
        lightop_moe.ep_scatter(
            recv_x_fp8,
            recv_x_scale,
            topk_ids_i64,
            None,
            counts_gpu,
            input_tensor,
            input_scale,
            m_indices,
            output_index,
            num_local_experts,
            DEEPEP_EXPERT_ALIGNMENT,
        )
    else:
        raise RuntimeError(
            "lightop.moe.ep_scatter is required for the framework ref path")

    # --- L1 contiguous GEMM ---
    n1 = 2 * intermediate_hidden
    l1_out = torch.empty(all_tokens, n1, dtype=torch.bfloat16, device=device)
    m_grouped_fp8_gemm_nt_contiguous(
        (input_tensor, input_scale.squeeze(-1)),
        (l1_weights, l1_weights_sf),
        l1_out,
        m_indices,
    )

    # --- SiLU + Mul + FP8 quant via lightop ---
    q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant(
        input=l1_out,
        fp8type=0,
        limit=ACTIVATION_CLAMP,
    )

    # --- L2 contiguous GEMM ---
    l2_out = torch.empty(all_tokens, hidden, dtype=torch.bfloat16, device=device)
    m_grouped_fp8_gemm_nt_contiguous(
        (q_a2_all, q_a2_scale.squeeze(-1)),
        (l2_weights, l2_weights_sf),
        l2_out,
        m_indices,
    )

    # --- ep_gather: lightop applies topk weights ---
    gather_out = torch.empty(num_recv, hidden, dtype=torch.bfloat16, device=device)
    lightop_moe.ep_gather(
        l2_out,
        topk_ids_i64,
        recv_topk_weights,
        output_index,
        None,
        gather_out,
    )

    # --- combine ---
    combine_result = buffer.combine(
        gather_out,
        handle=handle,
        config=deep_ep.Config(**FRAMEWORK_DEEPEP_COMBINE_CONFIG),
        async_finish=async_mode,
    )
    combined_ref = combine_result[0]
    combine_event = combine_result[-1]
    if combine_event is not None and hasattr(combine_event, "current_stream_wait") and getattr(combine_event, "event", None) is not None:
        combine_event.current_stream_wait()

    return combined_ref


# DG_MEGA_CFG_ALLGATHER=1: all-gather each rank's actual token count before the
# Note: DG_MEGA_CFG_ALLGATHER=1 makes deepgemm.mega.fp8_mega_moe gather
# the actual per-rank token counts before each fused launch, so the kernel's
# per-expert density heuristic uses the true cross-rank total instead of the
# capacity upper bound.


# ============================================================================
# run_fused: replace V2's mega_buffer.low_latency_mega_moe_channelwise
#   with deepgemm.mega.mega_moe.fp8_mega_moe (mega_moe_fp8_kernel)
# ============================================================================
def run_fused(
    x_local: torch.Tensor,        # bf16 [tokens_local, hidden]
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights,                   # (w6, sf) tuple from transform_weights_for_mega_moe
    l2_weights,                   # (w6, sf) tuple
    y: torch.Tensor,
    sym_buffer: megamoe.SymmBuffer,
):
    """Call the unified fused mega_moe kernel via deepgemm.mega.fp8_mega_moe.

    One entry for all token counts: the C++ dispatcher selects LL or HT per
    rank from expected_tokens_per_expert (DG_MEGA_HT_EXPECTED/--THRESHOLD,
    default 13.0 expected units).  Both consume the same Marlin w6 weights.
    """
    # --- Fused pre-quantize + copy: bf16 -> FP8 channelwise quant + topk copy ---
    num_tokens = x_local.shape[0]
    megamoe.mega_moe_preprocess(
        x_local, topk_idx, topk_weights, sym_buffer, num_tokens
    )

    # Launch unified fused kernel -- (w6, sf) weight tuples, auto LL/HT.
    # Pass the same activation clamp the ref kernels apply -- otherwise the
    # fused side runs unclamped (default inf) while the ref truncates at
    # ACTIVATION_CLAMP, producing a systematic one-sided output bias.
    megamoe.fp8_mega_moe(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        activation_clamp=ACTIVATION_CLAMP,
        fast_math=True,
    )
    return y


# Phase profiling dump for DG_MEGA_PROF=1.
# Slot map (per block, written by thread0, cycles via clock64):
#   0 t_init | dsp: 1 t_count 2 t_meta 3 t_bcast 4 t_pull_start 5 t_pull_done
#   11 pull_tokens | epi: 1 t_crossrole 2 t_l1_done 3 l1_wait 4 l1_gemm
#   5 l1_tiles | all: 6 l2_wait 7 l2_gemm 8 l2_fill 9 scatter 10 l2_tiles
#   12 t_l2_done 13 t_barrier2 14 t_cleanup 15 t_end
def dump_mega_prof(prof: torch.Tensor, rank_idx: int, kernel_ms: float,
                   n_disp_sms: int = 4):
    import numpy as np
    nslot = 32 if prof.numel() % 32 == 0 else 16
    p = prof.view(-1, nslot).numpy().astype(np.float64)
    nb = p.shape[0]
    span = p[:, 15] - p[:, 0]
    med_span = np.median(span[span > 0]) if (span > 0).any() else 1.0
    us = kernel_ms * 1e3 / med_span  # us per cycle (calibrated on median block)
    print(f"[rank {rank_idx}] === mega_moe phase profile ({nb} blocks, "
          f"calib {1.0/us/1e3:.3f} GHz) ===", flush=True)
    epi = np.arange(nb) >= n_disp_sms
    dsp = ~epi

    def fmt_rows(idx, label):
        for b in np.nonzero(idx)[0]:
            r = p[b]
            print(f"  blk{b:02d} {label}: span={r[15]-r[0]:>12.0f} "
                  f"cyc({(r[15]-r[0])*us:8.1f}us) "
                  f"l1w={r[3]*us:7.1f} l1g={r[4]*us:7.1f} l1t={int(r[5]):4d} "
                  f"l2w={r[6]*us:7.1f} l2g={r[7]*us:7.1f} fill={r[8]*us:6.1f} "
                  f"sct={r[9]*us:7.1f} l2t={int(r[10]):4d}", flush=True)

    # Aggregate stats
    def agg(name, col, mask):
        v = p[mask, col]
        if v.size and v.max() > 0:
            print(f"  {name}: mean={v.mean()*us:9.1f}us max={v.max()*us:9.1f}us",
                  flush=True)

    fmt_rows(dsp, "dsp")
    for b in np.nonzero(dsp)[0]:
        r = p[b]
        print(f"    blk{b:02d} dsp seg: count={((r[1]-r[0])*us):7.1f}us "
              f"meta={((r[2]-r[1])*us):7.1f}us "
              f"bcast={((r[3]-r[2])*us):7.1f}us "
              f"sync={((r[4]-r[3])*us):7.1f}us "
              f"pull={((r[5]-r[4])*us):8.1f}us tokens(warp0)={int(r[11])}",
              flush=True)
        if nslot == 32 and r[16] > 0:
            print(f"      sync sub: gsync1={((r[16]-r[3])*us):7.1f}us "
                  f"rankbar={((r[17]-r[16])*us):7.1f}us "
                  f"gsync2={((r[4]-r[17])*us):7.1f}us", flush=True)
    agg("epi l1_wait", 3, epi)
    agg("epi l1_gemm", 4, epi)
    agg("all l2_wait", 6, np.ones(nb, bool))
    agg("all l2_gemm", 7, np.ones(nb, bool))
    agg("all l2_fill", 8, np.ones(nb, bool))
    agg("all scatter", 9, np.ones(nb, bool))
    # Wall-clock milestones (relative to earliest t_init)
    base = p[:, 0].min()
    def ms_ts(col, mask):
        v = p[mask, col] - base
        return (v.min()*us, np.median(v)*us, v.max()*us)
    if epi.any():
        print("  epi t_crossrole (min/med/max us): %.1f/%.1f/%.1f" %
              ms_ts(1, epi), flush=True)
        print("  epi t_l1_done  (min/med/max us): %.1f/%.1f/%.1f" %
              ms_ts(2, epi), flush=True)
    print("  all t_l2_done  (min/med/max us): %.1f/%.1f/%.1f" %
          ms_ts(12, np.ones(nb, bool)), flush=True)
    print("  all t_bar2     (min/med/max us): %.1f/%.1f/%.1f" %
          ms_ts(13, np.ones(nb, bool)), flush=True)
    print("  all t_cleanup  (min/med/max us): %.1f/%.1f/%.1f" %
          ms_ts(14, np.ones(nb, bool)), flush=True)
    print("  all t_end      (min/med/max us): %.1f/%.1f/%.1f" %
          ms_ts(15, np.ones(nb, bool)), flush=True)


#   1. Golden reference run (framework ref --combine_ref)
#   2. Fused kernel run (run_fused --y) --replaced
#   3. Compare combine_ref vs y
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
    rank_offset: int = 0,
):
    # Supernode: mp.spawn hands us the node-local rank (0..local_nproc-1),
    # which is also the local HIP device id. Everything below -- process-group
    # rank, SymmBuffer peer index, fp8_mega_moe(rank_idx=...), token counts --
    # uses the GLOBAL rank, so rebind rank_idx and keep local_device apart.
    local_device = rank_idx
    rank_idx = rank_offset + rank_idx

    # init dist env
    def init(rank_idx: int):
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        print(f"[rank {rank_idx}] >>> calling dist.init_process_group(nccl, rank_idx={rank_idx}, world_size={num_ranks})", flush=True)
        dist.init_process_group("nccl", rank=rank_idx, world_size=num_ranks)
        print(f"[rank {rank_idx}] <<< dist.init_process_group done", flush=True)
        torch.cuda.set_device(local_device)
        print(f"[rank {rank_idx}] <<< cuda.set_device({local_device}) done, current={torch.cuda.current_device()}", flush=True)
        group = dist.group.WORLD
        print(f"[rank {rank_idx}] <<< got WORLD group", flush=True)
        return group

    group = init(rank_idx)

    print(
        f"[rank {rank_idx}] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')} "
        f"device_id={torch.cuda.current_device()} "
        f"device_name={torch.cuda.get_device_name(local_device)}",
        flush=True,
    )

    alloc_memory_size = 0
    tokens_local = tokens // num_ranks
    # Imbalanced per-rank token counts: DG_TOKEN_COUNTS="512,16,16,16,16,16,16,16"
    # (one entry per rank). max_tokens is a capacity/config parameter that must
    # be identical on every rank: DG_MAX_TOKENS sets it explicitly, otherwise
    # it defaults to the max of the (identical) DG_TOKEN_COUNTS list --no
    # allgather needed since every rank reads the same env config.
    _token_counts_env = os.environ.get("DG_TOKEN_COUNTS", "").strip()
    if _token_counts_env:
        _token_counts = [int(v) for v in _token_counts_env.split(",")]
        assert len(_token_counts) == num_ranks, (
            f"DG_TOKEN_COUNTS has {len(_token_counts)} entries, expected {num_ranks}"
        )
        tokens_local = _token_counts[rank_idx]
        max_tokens = int(os.environ.get("DG_MAX_TOKENS", "0")) or max(_token_counts)
        assert max_tokens >= max(_token_counts), (
            f"DG_MAX_TOKENS={max_tokens} < max(DG_TOKEN_COUNTS)={max(_token_counts)}"
        )
    else:
        max_tokens = int(os.environ.get("DG_MAX_TOKENS", "0")) or tokens_local
    # Capacity must be >= 1 for buffer allocation even when all ranks have 0 tokens.
    max_tokens = max(1, max_tokens)
    # Mode selection uses the actual per-call token load (cross-rank max of
    # DG_TOKEN_COUNTS, identical on every rank -- no allgather needed), not
    # the DG_MAX_TOKENS capacity.  This mirrors the framework, which picks
    # LL/HT per call from the real token count.
    tokens_mode_sel = max(_token_counts) if _token_counts_env else tokens_local
    print(f"[rank {rank_idx}] tokens_local={tokens_local} max_tokens={max_tokens}", flush=True)

    # Init mode (DG_INIT_MODE): "randn" , (randn*0.05 + softmax topk weights); 
    # "nv" aligns with the NV DeepGEMM test (unit-variance randn bf16, raw topk scores as
    # weights).  Legacy DG_RANDN_INIT=0 maps to "nv".
    _init_mode = os.environ.get("DG_INIT_MODE", "").strip().lower()
    if not _init_mode:
        _init_mode = "randn" if os.environ.get("DG_RANDN_INIT", "1") == "1" else "nv"
    NV_INIT = _init_mode == "nv"
    RANDN_INIT = not NV_INIT
    _seed = int(os.environ.get("DG_SEED", "1234")) + rank_idx
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    _input_scale = float(os.environ.get("DG_INPUT_SCALE", "0.05"))
    _weight_scale = float(os.environ.get("DG_WEIGHT_SCALE", "0.05"))
    print(f"[rank {rank_idx}] seed={_seed} init_mode={_init_mode}", flush=True)

    # Generate input
    x_sf_local = None
    if RANDN_INIT:
        print(f"[rank {rank_idx}] >>> generating input x_bf16(randn*{_input_scale}) [{tokens_local}x{hidden}] (randn init)", flush=True)
        x_bf16 = torch.randn(
            tokens_local, hidden, dtype=torch.bfloat16,
            device=torch.cuda.current_device()
        ) * _input_scale
        # Per-token FP8 quantization: scale = amax.clamp(1e-4)/448.
        _amax = x_bf16.abs().float().amax(dim=1, keepdim=True).clamp(min=1.0e-4)
        x_sf_local = (_amax / 448.0).contiguous()
        input_fp8 = (x_bf16.float() / x_sf_local).to(torch.float8_e4m3fn).contiguous()
        del x_bf16, _amax
        torch.cuda.empty_cache()
    else:
        # NV init: unit-variance randn bf16 (no scale), same per-token
        # amax/448 quantization as the randn path.
        print(f"[rank {rank_idx}] >>> generating input x_bf16(randn) [{tokens_local}x{hidden}] (nv init)", flush=True)
        x_bf16 = torch.randn(
            tokens_local, hidden, dtype=torch.bfloat16,
            device=torch.cuda.current_device()
        )
        _amax = x_bf16.abs().float().amax(dim=1, keepdim=True).clamp(min=1.0e-4)
        x_sf_local = (_amax / 448.0).contiguous()
        input_fp8 = (x_bf16.float() / x_sf_local).to(torch.float8_e4m3fn).contiguous()
        del x_bf16, _amax
        torch.cuda.empty_cache()
    print(f"[rank {rank_idx}] <<< input_fp8 ready, shape={input_fp8.shape}", flush=True)

    element_size = input_fp8.element_size()
    num_elements = input_fp8.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    # Weights
    num_local_experts = num_experts // num_ranks

    if x_sf_local is None:
        print(f"[rank {rank_idx}] >>> generating x_sf_local", flush=True)
        x_sf_local = torch.rand(
            tokens_local, 1, device=torch.cuda.current_device(), dtype=torch.float32
        )
        if debug_mode:
            x_sf_local = torch.ones(
                tokens_local, 1, device=torch.cuda.current_device(), dtype=torch.float32
            )

    element_size = x_sf_local.element_size()
    num_elements = x_sf_local.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    # l1 (gate/up): [num_local_experts, 2*intermediate_hidden, hidden]
    if RANDN_INIT:
        # randn init: randn bf16 * weight_scale -> per-channel amax/448 FP8 quantize
        # (cast_grouped_weight_to_fp8_channelwise).
        print(f"[rank {rank_idx}] >>> generating l1_weights(randn*{_weight_scale}) (randn init)", flush=True)
        l1_weights_bf16 = torch.randn(
            num_local_experts,
            2 * intermediate_hidden,
            hidden,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        ) * _weight_scale
        l1_weights_sf = (
            l1_weights_bf16.abs().float().amax(dim=-1).clamp(min=1.0e-4) / 448.0
        ).contiguous()
        l1_weights = (
            l1_weights_bf16.float() / l1_weights_sf.unsqueeze(-1)
        ).to(torch.float8_e4m3fn).contiguous()
        del l1_weights_bf16
    else:
        # NV init: unit-variance randn bf16, per-channel amax/448 quantization.
        print(f"[rank {rank_idx}] >>> generating l1_weights(randn) (nv init)", flush=True)
        l1_weights_bf16 = torch.randn(
            num_local_experts,
            2 * intermediate_hidden,
            hidden,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        l1_weights_sf = (
            l1_weights_bf16.abs().float().amax(dim=-1).clamp(min=1.0e-4) / 448.0
        ).contiguous()
        l1_weights = (
            l1_weights_bf16.float() / l1_weights_sf.unsqueeze(-1)
        ).to(torch.float8_e4m3fn).contiguous()
        del l1_weights_bf16
        if debug_mode:
            l1_weights = torch.ones(
                num_local_experts,
                2 * intermediate_hidden,
                hidden,
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            ).to(torch.float8_e4m3fn)
    torch.cuda.empty_cache()
    print(f"[rank {rank_idx}] <<< l1_weights_fp8 ready, shape={l1_weights.shape}", flush=True)

    # Keep non-interleaved 3D copies: the reference path expects separated
    # gate/up halves for its SiLU kernel, while the fused-side gate/up
    # interleave + Marlin pack happens inside transform_weights_for_mega_moe.
    l1_weights_fp8_ref = l1_weights.contiguous()
    l1_weights_fp8 = l1_weights.contiguous()

    element_size = l1_weights.element_size()
    num_elements = l1_weights.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    if debug_mode and not RANDN_INIT:
        l1_weights_sf = torch.ones(
            num_local_experts,
            2 * intermediate_hidden,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

    # Keep a non-interleaved scale copy for the reference path; the
    # fused-side gate/up scale interleave is done by
    # transform_weights_for_mega_moe.
    l1_weights_sf_ref = l1_weights_sf.contiguous()

    element_size = l1_weights_sf.element_size()
    num_elements = l1_weights_sf.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    # l2 (down): [num_local_experts, hidden, intermediate_hidden]
    if RANDN_INIT:
        print(f"[rank {rank_idx}] >>> generating l2_weights(randn*{_weight_scale}) (randn init)", flush=True)
        l2_weights_bf16 = torch.randn(
            num_local_experts,
            hidden,
            intermediate_hidden,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        ) * _weight_scale
        l2_weights_sf = (
            l2_weights_bf16.abs().float().amax(dim=-1).clamp(min=1.0e-4) / 448.0
        ).contiguous()
        l2_weights = (
            l2_weights_bf16.float() / l2_weights_sf.unsqueeze(-1)
        ).to(torch.float8_e4m3fn).contiguous()
        del l2_weights_bf16
    else:
        # NV init: unit-variance randn bf16, per-channel amax/448 quantization.
        print(f"[rank {rank_idx}] >>> generating l2_weights(randn) (nv init)", flush=True)
        l2_weights_bf16 = torch.randn(
            num_local_experts,
            hidden,
            intermediate_hidden,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        l2_weights_sf = (
            l2_weights_bf16.abs().float().amax(dim=-1).clamp(min=1.0e-4) / 448.0
        ).contiguous()
        l2_weights = (
            l2_weights_bf16.float() / l2_weights_sf.unsqueeze(-1)
        ).to(torch.float8_e4m3fn).contiguous()
        del l2_weights_bf16
        if debug_mode:
            l2_weights = torch.ones(
                num_local_experts,
                hidden,
                intermediate_hidden,
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            ).to(torch.float8_e4m3fn)
    torch.cuda.empty_cache()
    print(f"[rank {rank_idx}] <<< l2_weights_fp8 ready, shape={l2_weights.shape}", flush=True)

    # Keep a 3D copy for the fused kernel (which expects [E, hidden, ih])
    l2_weights_fp8 = l2_weights.contiguous()

    # l2 has no gate/up interleave; the masked-GEMM ref path reuses the same
    # Marlin w6 tensor as the fused path (transformed_l2_weights[0]).

    element_size = l2_weights.element_size()
    num_elements = l2_weights.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    if debug_mode and not RANDN_INIT:
        l2_weights_sf = torch.ones(
            num_local_experts,
            hidden,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

    element_size = l2_weights_sf.element_size()
    num_elements = l2_weights_sf.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    # Generate score --topk (after weights, matching the reference test's
    # generation order: x -> l1 -> l2 -> scores, so a fixed seed reproduces
    # the same RNG stream).
    print(f"[rank {rank_idx}] >>> generating score/topk", flush=True)
    # DG_HOT_EXPERTS=N: restrict routing to the first N experts of each rank
    # so each hot expert receives ~tokens*topk/N rows (stress multi-tile-per-
    # expert GEMM). Hot set is spread across ranks: [r*epr, r*epr+N).
    hot_e = int(os.environ.get("DG_HOT_EXPERTS", "0"))
    if hot_e > 0:
        _epr = num_experts // num_ranks
        _keep = torch.zeros(num_experts, dtype=torch.bool, device=torch.cuda.current_device())
        for _r in range(num_ranks):
            _keep[_r * _epr:_r * _epr + min(hot_e, _epr)] = True
    if RANDN_INIT:
        # randn init: randn scores -> topk on raw scores -> softmax over topk
        # (weights sum to 1 across the selected slots).
        score_local = torch.randn(
            tokens_local, num_experts, device=torch.cuda.current_device(), dtype=torch.float32
        )
        if hot_e > 0:
            score_local[:, ~_keep] = -1e30
        topk_weights, topk_idx = torch.topk(score_local, num_topk, dim=-1, largest=True, sorted=False)
        topk_weights = torch.softmax(topk_weights.float(), dim=-1)
    else:
        # NV init: randn scores -> topk; raw scores used as topk weights
        # (no softmax normalization).
        score_local = torch.randn(
            tokens_local, num_experts, device=torch.cuda.current_device(), dtype=torch.float32
        )
        if hot_e > 0:
            score_local[:, ~_keep] = -1e30
        topk_weights, topk_idx = torch.topk(score_local, num_topk, dim=-1, largest=True, sorted=False)
    del score_local
    torch.cuda.empty_cache()
    topk_idx = topk_idx.to(torch.int64)
    topk_weights = topk_weights.to(torch.float32)
    # DG_HOT_RANK="<r1,r2,...>": force ALL tokens' topk onto the listed ranks'
    # experts by REGENERATING topk_idx directly (not masking scores to -1e30).
    # This produces a true single-rank hot-expert load imbalance (every rank
    # sends its traffic to the same owning rank). Combined with DG_HOT_EXPERTS=N,
    # restrict to the first N experts of those ranks.
    _hot_ranks_env = os.environ.get("DG_HOT_RANK", "").strip()
    if _hot_ranks_env:
        _epr = num_experts // num_ranks
        _he = int(os.environ.get("DG_HOT_EXPERTS", "0"))
        _allowed = []
        for _r in [int(v) for v in _hot_ranks_env.split(",") if v.strip() != ""]:
            _nn = min(_he, _epr) if _he > 0 else _epr
            _allowed.extend(range(_r * _epr, _r * _epr + _nn))
        _allowed = torch.tensor(_allowed, dtype=torch.int64,
                                device=torch.cuda.current_device())
        _A = int(_allowed.numel())
        assert _A >= num_topk, f"DG_HOT_RANK allows {_A} experts < topk {num_topk}"
        # per-token selection of num_topk DISTINCT experts (no replacement)
        _cols = torch.rand(tokens_local, _A, device=torch.cuda.current_device()) \
                    .topk(num_topk, dim=-1).indices
        topk_idx = _allowed[_cols].contiguous().to(torch.int64)
        topk_weights = torch.softmax(
            torch.randn(tokens_local, num_topk, device=torch.cuda.current_device()),
            dim=-1).float()
        print(f"[rank {rank_idx}] DG_HOT_RANK={_hot_ranks_env}: forced all topk onto "
              f"{_A} experts [{int(_allowed.min())},{int(_allowed.max())}] "
              f"(actual idx {int(topk_idx.min())}-{int(topk_idx.max())})", flush=True)
    # DG_HOT_MULT=<m>: make ONE rank receive m times the balanced-average
    # amount of ROUTED ROWS. The unit is one (token, topk) slot = one pool row
    # = one cross-rank transfer: if two topk slots of the same token land on
    # two local experts of one rank, that is TWO rows and TWO transfers (no
    # dedup). Under uniform routing each rank receives total_slots/num_ranks
    # rows, so we place a fraction m/num_ranks of ALL slots on the hot rank
    # and spread the rest uniformly over the other ranks.
    _hot_mult = float(os.environ.get("DG_HOT_MULT", "0"))
    if _hot_mult > 0:
        _epr = num_experts // num_ranks
        _hot_r = int(os.environ.get("DG_HOT_RECV_RANK", "0"))
        assert 0 <= _hot_r < num_ranks, f"DG_HOT_RECV_RANK={_hot_r} out of range"
        _p_slot = min(1.0, _hot_mult / num_ranks)
        _dev = torch.cuda.current_device()
        # Distinct candidate experts per token from the hot block and from the
        # other blocks. The two sets are disjoint, so every token still has
        # num_topk distinct experts no matter which slots are marked hot.
        _hot_cand = torch.rand(tokens_local, _epr, device=_dev) \
                        .topk(num_topk, dim=-1).indices + _hot_r * _epr
        _cold_cand = torch.rand(tokens_local, num_experts - _epr, device=_dev) \
                         .topk(num_topk, dim=-1).indices
        _cold_cand = _cold_cand + (_cold_cand >= _hot_r * _epr).long() * _epr
        _use_hot = torch.rand(tokens_local, num_topk, device=_dev) < _p_slot
        topk_idx = torch.where(_use_hot, _hot_cand, _cold_cand) \
                       .contiguous().to(torch.int64)
        topk_weights = torch.softmax(
            torch.randn(tokens_local, num_topk, device=_dev), dim=-1).float()
        _local_share = ((topk_idx // _epr) == _hot_r).float().mean().item()
        print(f"[rank {rank_idx}] DG_HOT_MULT={_hot_mult:g} hot_rank={_hot_r} "
              f"target_slot_frac={_p_slot:.4f} (avg={1.0 / num_ranks:.4f}) "
              f"local_slot_mult={_local_share * num_ranks:.4f}x", flush=True)
        # Ground truth of the RECEIVED amount: sum the (token,topk) rows
        # destined for each rank over all ranks. This includes duplicates,
        # i.e. exactly what dispatch/pull has to transfer.
        _dst = (topk_idx.clamp(min=0) // _epr).flatten()
        _rows = torch.bincount(_dst, minlength=num_ranks).float()
        dist.all_reduce(_rows, op=dist.ReduceOp.SUM)
        _avg = _rows.sum() / num_ranks
        if rank_idx == 0:
            print(f"[HOT_RECV_VERIFY] rows_per_rank="
                  f"{[int(v) for v in _rows.tolist()]} avg={_avg:.0f} "
                  f"hot_rank={_hot_r} hot_rows={_rows[_hot_r]:.0f} "
                  f"MULT={(_rows[_hot_r] / _avg).item():.4f}x "
                  f"(target {_hot_mult:g}x)", flush=True)
    # DG_TOPK1=1: route each token to only its first expert (rest = -1/0),
    # isolating a single-expert path for precision debugging.
    if os.environ.get("DG_TOPK1", "0") == "1":
        topk_idx[:, 1:] = -1
        topk_weights[:, 1:] = 0.0
        print(f"[rank {rank_idx}] DG_TOPK1: only k=0 routed "
              f"(expert={topk_idx[:,0].tolist()})", flush=True)
    print(f"[rank {rank_idx}] <<< topk ready, ids={topk_idx.shape} weights={topk_weights.shape}", flush=True)

    # NV-style routed row count: all-gather every rank's topk_idx, keep only
    # entries owned by this rank's local experts, then count them. This counts
    # duplicate (token, topk) rows but excludes DeepEP alignment padding.
    num_experts_per_rank = num_experts // num_ranks
    gathered_topk_idx = _uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[
        (gathered_topk_idx < rank_idx * num_experts_per_rank)
        | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
    ] = -1
    num_recv_tokens = int((gathered_topk_idx != -1).sum().item())
    del gathered_topk_idx
    print(f"[rank {rank_idx}] NV-style num_recv_tokens={num_recv_tokens} "
          f"(local expert range [{rank_idx * num_experts_per_rank}, "
          f"{(rank_idx + 1) * num_experts_per_rank}))", flush=True)

    element_size = topk_idx.element_size()
    num_elements = topk_idx.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    element_size = topk_weights.element_size()
    num_elements = topk_weights.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    # =====================================================================
    # 1. Golden reference path
    #    - tokens < 512:  low_latency mode (original V2 reference)
    #    - tokens >= 512: normal (high-throughput) mode as primary reference.
    #      At exactly 512, also run low_latency for cross-validation
    #      (LL 512 as golden to validate normal-mode migration correctness).
    # =====================================================================
    # Reference mode: LL ref when max_tokens <= threshold, HT (normal) ref
    # otherwise --keys on the cross-rank max so all ranks pick the same path.
    # DeepEP low_latency cannot allocate its RocSHMEM heap for large token
    # counts (OOM observed at 512: hint=5.33GB > ~4GB heap limit), so fall back
    # to the normal ref as golden when the LL buffer would not fit.
    # RDMA hint and LL dispatch layout are both keyed on the actual per-call
    # token count (tokens_mode_sel), matching the framework's real behavior.
    #
    # DG_REF_MODE (framework-aligned ref): "ll" / "normal" / "framework".
    # "framework" replays deepep-mode auto: the launcher uses the fixed
    # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK (default 128) as the LL
    # capacity, so captured decode vectors run LL and extend vectors run
    # normal.  The decode-side envs (LL layered opt / HDP flush disable) are
    # set the same way the service launcher does.
    if DG_REF_MODE == "ll":
        use_normal_mode = False
    elif DG_REF_MODE == "normal":
        use_normal_mode = True
    elif DG_REF_MODE == "framework":
        use_normal_mode = tokens_mode_sel > DG_DEEPEP_LL_CAP
    else:
        raise ValueError(
            f"unsupported DG_REF_MODE={DG_REF_MODE}; "
            "only ll/normal/framework are supported")

    if use_normal_mode:
        num_rdma_hint = 0
    else:
        # Decode-side DeepEP flags, set before any Buffer/hint use --
        # mirrors the launcher (decode only) and changes the RDMA hint.
        os.environ.setdefault("DEEPEP_ENABLE_LL_LAYERED_OPT", "1")
        os.environ.setdefault("ROCSHMEM_DISABLE_HDP_FLUSH", "1")
        num_rdma_hint = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            DG_DEEPEP_LL_CAP, hidden, num_ranks, num_experts
        )
    if rank_idx == 0:
        print(
            f"[REF-FW] DG_REF_MODE={DG_REF_MODE} -> "
            f"{'normal' if use_normal_mode else 'low_latency'} "
            f"(ll_cap={DG_DEEPEP_LL_CAP})",
            flush=True,
        )

    y = torch.zeros(
        tokens_local, hidden, device=torch.cuda.current_device(), dtype=torch.bfloat16
    )

    element_size = y.element_size()
    num_elements = y.numel()
    alloc_memory_size = element_size * num_elements + alloc_memory_size

    x_ref = input_fp8.to(torch.float32) * x_sf_local
    x_ref = x_ref.to(torch.bfloat16)

    # NV-style weight transform: gate/up interleave (granularity 8) +
    # Marlin w6 pack; returns (w6, sf) tuples consumed by fp8_mega_moe.
    print(f"[rank {rank_idx}] >>> preparing MegaMoE weights (transform_weights_for_mega_moe)", flush=True)
    transformed_l1_weights, transformed_l2_weights = \
        megamoe.transform_weights_for_mega_moe(
            (l1_weights_fp8, l1_weights_sf),
            (l2_weights_fp8, l2_weights_sf))
    # Reference paths keep their own non-interleaved Marlin l1 weights.
    # gfx938 marlin_fp8_contiguous_weight now returns the pair32 pack that
    # the upstream masked/contiguous pair32 GEMMs consume.
    l1_weights_ref = marlin_fp8_contiguous_weight(l1_weights_fp8_ref)
    l2_weights_ref = marlin_fp8_contiguous_weight(l2_weights_fp8)
    l2_weights = transformed_l2_weights[0]
    print(f"[rank {rank_idx}] <<< MegaMoE weights ready", flush=True)

    if SKIP_REF:
        # Skip the DeepEP reference path entirely (hang-isolation experiments).
        combine_ref = None
        ref_source = "skipped"
        t_ref = {"mean": 1.0, "median": 1.0, "min": 1.0}
        t_ref_mean = t_ref_median = t_ref_min = 1.0
    elif use_normal_mode:
        # --- Normal (high-throughput) mode: primary reference ---
        # Framework AUTO mode allocates one buffer for both dispatch modes
        # with num_qps_per_rank=max(config num_sms, experts/ranks)=48.
        _normal_qps = FRAMEWORK_DEEPEP_QPS
        print(f"[rank {rank_idx}] >>> creating deep_ep.Buffer (normal/high-throughput mode, num_qps={_normal_qps})", flush=True)
        normal_buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=DEEPEP_BUFFER_BYTES,
            num_rdma_bytes=0,
            low_latency_mode=False,
            num_qps_per_rank=_normal_qps,
            explicitly_destroy=True,
        )
        print(f"[rank {rank_idx}] <<< deep_ep.Buffer (normal) created", flush=True)
        alloc_memory_size = alloc_memory_size + DEEPEP_BUFFER_BYTES

        t_normal, combine_ref_normal = bench_gpu(
            run_ref_ht_framework,
            num_ranks, num_experts, tokens_local,
            x_ref, topk_idx, topk_weights,
            l1_weights_ref, l1_weights_sf_ref, l2_weights_ref, l2_weights_sf,
            async_mode, normal_buffer, rank_idx,
            hidden, intermediate_hidden, print_recv_counts,
            barrier_fn=dist.barrier if BENCH_ITER_BARRIER else None,
            use_graph=BENCH_REF_CUDAGRAPH_NORMAL,
        )

        normal_buffer.destroy()
        torch.cuda.empty_cache()
        dist.barrier()
        t_ref_mean = t_normal["mean"]
        t_ref_median = t_normal["median"]
        t_ref = t_normal
        print(f"---------------- DeepEP Normal + GEMM + SILU + GEMM ref total: {t_normal}ms, rank_idx:{rank_idx}")

        combine_ref = combine_ref_normal
        ref_source = "normal(framework)"

        memory_gb = alloc_memory_size / (1024 ** 3)
        print(f"------------------------------memory_gb is: {memory_gb}")
    else:
        # --- Low-latency mode only (tokens < threshold) ---
        print(f"[rank {rank_idx}] >>> calling get_low_latency_rdma_size_hint", flush=True)
        num_rdma_bytes = num_rdma_hint
        print(f"[rank {rank_idx}]----------num_rdma_bytes GB is {num_rdma_bytes / (1024 ** 3)}", flush=True)
        print(f"[rank {rank_idx}] >>> creating deep_ep.Buffer (low_latency, num_qps={num_local_experts})", flush=True)
        buffer = deep_ep.Buffer(
            group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_local_experts,
            explicitly_destroy=True,
        )
        print(f"[rank {rank_idx}] <<< deep_ep.Buffer created", flush=True)
        print(f"[rank {rank_idx}] deep_ep Buffer created", flush=True)

        alloc_memory_size = alloc_memory_size + num_rdma_bytes
        memory_gb = alloc_memory_size / (1024 ** 3)
        print(f"------------------------------memory_gb is: {memory_gb}")

        t_ref, combine_ref = bench_gpu(
            run_ref_ll_framework,
            num_ranks, num_experts, tokens_local,
            x_ref, topk_idx, topk_weights,
            l1_weights_ref, l1_weights_sf_ref, l2_weights_ref, l2_weights_sf,
            async_mode, buffer, rank_idx,
            ll_cap=DG_DEEPEP_LL_CAP,
            barrier_fn=dist.barrier if BENCH_ITER_BARRIER else None,
            use_graph=BENCH_REF_CUDAGRAPH_LL,
        )
        ref_source = "low_latency(framework)"

        buffer.destroy()
        torch.cuda.empty_cache()
        dist.barrier()
        t_ref_mean = t_ref["mean"]
        t_ref_median = t_ref["median"]
        print(f"---------------- DeepEP + GEMM + SILU + GEMM ref total (including GPU execution): {t_ref}ms, rank_idx:{rank_idx}")

    torch.cuda.empty_cache()

    # =====================================================================
    # 2. Fused kernel: mega_moe_fp8_kernel via deepgemm.mega.fp8_mega_moe
    #    (replaces V2's mega_buffer.low_latency_mega_moe_channelwise)
    # =====================================================================
    num_max_tokens_per_rank = max_tokens
    sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    print(f"[rank {rank_idx}] SymmBuffer created", flush=True)

    # Unified fused entry: one call, C++ picks LL/HT per rank from the
    # expected-tokens-per-expert heuristic -- there is no single "mode" to
    # report here (ranks may legitimately differ).  The real per-rank choice
    # can be dumped with DG_MEGA_CFG_LOG=1 / DG_MEGA_CFG_VERBOSE=1.
    print(f"[rank {rank_idx}] Fused kernel: mode=auto (per-rank LL/HT selected in C++), tokens_local={tokens_local}", flush=True)

    if os.environ.get("DG_HANG_WATCHDOG_S"):
        _start_hang_watchdog(sym_buffer, rank_idx, num_ranks, num_experts,
                             max_tokens, num_topk)

    # Fused path consumes (w6, sf) weight tuples from
    # transform_weights_for_mega_moe -- one Marlin set for both LL and HT.
    fused_args = (
        x_ref, topk_idx, topk_weights,
        transformed_l1_weights, transformed_l2_weights,
        y, sym_buffer,
    )
    t_mega, out = bench_gpu(
        run_fused,
        *fused_args,
        barrier_fn=dist.barrier if BENCH_ITER_BARRIER else None,
        use_graph=BENCH_FUSED_CUDAGRAPH,
    )
    y = out

    # NV-style kineto kernel-only timing (mega kernel + preprocess separately)
    t_kernel = None
    if BENCH_KINETO:
        try:
            t_kernel, t_pre = bench_kineto(
                lambda: run_fused(*fused_args),
                ('mega_moe_fp8_kernel', 'mega_moe_preprocess'),
                num_tests=BENCH_ITERS,
                barrier=dist.barrier if BENCH_ITER_BARRIER else None,
            )
            print(f"---------------- fused_mega_moe kineto: mega_kernel={t_kernel*1e3:.3f}ms "
                  f"preprocess={t_pre*1e3:.3f}ms sum={(t_kernel+t_pre)*1e3:.3f}ms, rank_idx:{rank_idx}")
        except Exception as e:
            print(f"[rank {rank_idx}] bench_kineto unavailable ({e}); skipping", flush=True)

    # Phase profiling dump (DG_MEGA_PROF=1)
    if os.environ.get("DG_MEGA_PROF"):
        try:
            import deepgemm.mega._C as mega_C
            prof_buf = mega_C.mega_moe_prof_fetch()
            if prof_buf.numel():
                if os.environ.get("DG_MEGA_NDISP_SMS"):
                    nd = int(os.environ["DG_MEGA_NDISP_SMS"])
                else:
                    # Mirror the C++ choice: exact when DG_MEGA_CFG_LOG=1
                    # wrote megamoe_cfg_rankN.log, else fall back to 4.
                    nd = 4
                    cfg_log = f"megamoe_cfg_rank{rank_idx}.log"
                    if os.path.exists(cfg_log):
                        try:
                            with open(cfg_log) as f:
                                content = f.read()
                            pos = content.rfind("ndisp=")
                            if pos >= 0:
                                nd = int(content[pos + 6:].split()[0])
                        except (OSError, ValueError):
                            pass
                dump_mega_prof(prof_buf, rank_idx, t_mega["median"], nd)
        except Exception as e:
            print(f"[rank {rank_idx}] prof fetch failed: {e}", flush=True)
    del x_ref
    torch.cuda.empty_cache()
    dist.barrier()
    t_mega_mean = t_mega["mean"]
    t_mega_median = t_mega["median"]
    print(f"---------------- fused_mega_moe total (including GPU execution): {t_mega}ms, rank_idx:{rank_idx}")

    # DG_TEST_DUMP=<dir>: save this rank's fused output so two builds (or two
    # kernel variants run with the same DG_SEED) can be compared bit-exactly.
    _dump_dir = os.environ.get("DG_TEST_DUMP", "").strip()
    if _dump_dir:
        try:
            os.makedirs(_dump_dir, exist_ok=True)
            torch.save({"rank": rank_idx, "y": out.detach().cpu()},
                       os.path.join(_dump_dir, f"y_rank{rank_idx}.pt"))
            print(f"[rank {rank_idx}] dumped fused output -> {_dump_dir}", flush=True)
        except Exception as e:
            print(f"[rank {rank_idx}] dump failed: {e}", flush=True)

    # Speedup comparison (median latency, more stable than mean against outliers)
    speedup = t_ref_median / t_mega_median if t_mega_median > 0 else 0.0
    t_ref_min = t_ref["min"]
    t_mega_min = t_mega["min"]
    speedup_min = t_ref_min / t_mega_min if t_mega_min > 0 else 0.0
    print(f"---------------- speedup (ref_median={ref_source} / fused_median): {t_ref_median:.3f} / {t_mega_median:.3f} = {speedup:.2f}x, rank_idx:{rank_idx}")
    print(f"---------------- speedup (ref_min={ref_source} / fused_min): {t_ref_min:.3f} / {t_mega_min:.3f} = {speedup_min:.2f}x, rank_idx:{rank_idx}")

    # NV-style effective FP8 GEMM throughput:
    #   num_recv_tokens = gathered topk entries owned by this rank's experts
    #   FLOPs           = 2 * num_recv_tokens * H * I * 3   (gate + up + down)
    #   TFLOPS          = FLOPs / fused wall median (default)
    gemm_flops = 2.0 * num_recv_tokens * hidden * intermediate_hidden * 3.0

    def _nv_tflops(ms):
        return gemm_flops / (ms * 1e-3) / 1e12 if ms > 0 else 0.0

    print(f"---------------- NV-style GEMM flops: {gemm_flops / 1e12:.4f} TFLOP/rank "
          f"(num_recv_tokens={num_recv_tokens}, H={hidden}, I={intermediate_hidden})")
    print(f"---------------- NV-style fused TFLOPS (wall_median={t_mega_median:.3f} ms): "
          f"{_nv_tflops(t_mega_median):.2f}, rank_idx:{rank_idx}")
    if t_kernel is not None and t_kernel > 0:
        print(f"---------------- NV-style fused TFLOPS (kineto_mega_kernel={t_kernel * 1e3:.3f} ms): "
              f"{_nv_tflops(t_kernel * 1e3):.2f}, rank_idx:{rank_idx}")
    if not SKIP_REF:
        print(f"---------------- same-FLOPs ref TFLOPS (wall_median={t_ref_median:.3f} ms): "
              f"{_nv_tflops(t_ref_median):.2f}, rank_idx:{rank_idx}")

    # --- Extra correctness iterations with the same inputs ---
    # Catches intermittent/nondeterministic failures that a single check misses.
    for ci in range(max(0, CORRECTNESS_ITERS - 1)):
        out_ci = run_fused(*fused_args)
        torch.cuda.synchronize()
        ci_finite = bool(torch.isfinite(out_ci.float()).all().item())
        ci_ok = ci_finite and (
            combine_ref is None or torch.allclose(
                combine_ref, out_ci, atol=3.55e-3, rtol=2e-1
            )
        )
        print(f"[rank {rank_idx}] correctness iter {ci + 2}/{CORRECTNESS_ITERS}: "
              f"{'PASS' if ci_ok else 'FAIL'}", flush=True)
        if not ci_ok:
            print(f"[rank {rank_idx}] CHECK_FAILED (non-fatal): correctness iter {ci + 2}")

    if rank_idx == 0:
        torch.set_printoptions(profile="full")

    # Check non-1.0 elements
    non_one = (y != 1.0).sum().item()
    print(f"y: {non_one} elements != 1.0 (out of {y.numel()})")

    rows_with_non_one = 0
    for i in range(y.shape[0]):
        if (y[i] != 1.0).sum() > 0:
            rows_with_non_one += 1
    print(f"y: {rows_with_non_one} rows with any non-1.0 values")

    # =====================================================================
    # 3. Compare golden reference vs fused kernel --detailed diagnostics
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"[rank {rank_idx}] === DETAILED COMPARISON: combine_ref ({ref_source}) vs y (fused) ===")
    print(f"{'='*70}")
    print(f"  combine_ref shape={None if combine_ref is None else combine_ref.shape} "
          f"dtype={None if combine_ref is None else combine_ref.dtype}")
    print(f"  y  shape={y.shape} dtype={y.dtype}")

    if combine_ref is None:
        fused_ok = bool(torch.isfinite(y.float()).all().item()) if y.numel() else True
        print(f"  (SKIP_REF --fused completed; finite={fused_ok}; "
              f"{'PASS' if fused_ok else 'FAIL'})")
        print(f"test rank {rank_idx} async mode {async_mode} "
              f"{'PASS' if fused_ok else 'FAIL'}")
        sym_buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        return

    # 0 local tokens: outputs are empty --trivially PASS (guard all reductions below)
    if combine_ref.numel() == 0:
        passed = y.numel() == 0
        print(f"  (0 local tokens --empty outputs; {'PASS' if passed else 'FAIL'})")
        if passed:
            print(f"test rank {rank_idx} async mode {async_mode} PASS")
        else:
            print(f"test rank {rank_idx} async mode {async_mode} FAIL --unexpected content on empty output")
        if not passed:
            print(f"[rank {rank_idx}] CHECK_FAILED (non-fatal): unexpected content on empty output")
        sym_buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        return

    # Basic stats
    ref_min, ref_max = combine_ref.min().item(), combine_ref.max().item()
    fus_min, fus_max = y.min().item(), y.max().item()
    ref_mean, fus_mean = combine_ref.float().mean().item(), y.float().mean().item()
    print(f"  combine_ref: min={ref_min:.6f} max={ref_max:.6f} mean={ref_mean:.6f}")
    print(f"  y : min={fus_min:.6f} max={fus_max:.6f} mean={fus_mean:.6f}")

    # Element-wise diff
    diff = (combine_ref.float() - y.float()).abs()
    abs_diff = diff
    rel_diff = diff / (combine_ref.float().abs() + 1e-6)
    print(f"  abs_diff:   max={abs_diff.max().item():.6f} mean={abs_diff.mean().item():.6f} median={abs_diff.median().item():.6f}")
    print(f"  rel_diff:   max={rel_diff.max().item():.6f} mean={rel_diff.mean().item():.6f} median={rel_diff.median().item():.6f}")

    # How many elements pass various tolerances
    for atol in [1e-3, 1e-2, 1e-1, 0.5, 1.0]:
        n_pass = (abs_diff <= atol).sum().item()
        n_total = abs_diff.numel()
        print(f"  atol={atol:.0e}: {n_pass}/{n_total} pass ({100.0*n_pass/n_total:.2f}%)")

    # Per-row max abs diff
    per_row_max = abs_diff.max(dim=1).values
    per_row_mean = abs_diff.mean(dim=1)
    print(f"\n  Per-row max abs_diff: max={per_row_max.max().item():.6f} mean={per_row_max.mean().item():.6f}")
    worst_rows = per_row_max.argsort(descending=True)
    print(f"  Top-10 worst rows (by max abs_diff):")
    for idx in range(min(10, worst_rows.shape[0])):
        r = worst_rows[idx].item()
        print(f"    row {r:4d}: max_abs_diff={per_row_max[r].item():.6f} "
              f"mean_abs_diff={per_row_mean[r].item():.6f} "
              f"ref[min={combine_ref[r].min().item():.4f} max={combine_ref[r].max().item():.4f}] "
              f"fus[min={y[r].min().item():.4f} max={y[r].max().item():.4f}]")

    # Show first few elements of worst row
    worst_row = worst_rows[0].item()
    n_show = min(16, combine_ref.shape[1])
    print(f"\n  Worst row {worst_row} --first {n_show} elements:")
    print(f"    ref: {combine_ref[worst_row, :n_show].tolist()}")
    print(f"    fus: {y[worst_row, :n_show].tolist()}")
    print(f"    dif: {[(combine_ref[worst_row, j].float() - y[worst_row, j].float()).item() for j in range(n_show)]}")

    # Check for zero rows in fused output (potential missed tokens)
    zero_rows_fus = (y.abs().sum(dim=1) == 0).sum().item()
    zero_rows_ref = (combine_ref.abs().sum(dim=1) == 0).sum().item()
    print(f"\n  Zero rows: ref={zero_rows_ref} fus={zero_rows_fus}")

    # Check for NaN/Inf
    nan_fus = torch.isnan(y).sum().item()
    inf_fus = torch.isinf(y).sum().item()
    nan_ref = torch.isnan(combine_ref).sum().item()
    inf_ref = torch.isinf(combine_ref).sum().item()
    print(f"  NaN: ref={nan_ref} fus={nan_fus}  |  Inf: ref={inf_ref} fus={inf_fus}")

    # Ratio comparison (fused / ref) to detect scale issues
    ratio = y.float() / (combine_ref.float() + 1e-6)
    valid_ratio = ratio[combine_ref.float().abs() > 1e-3]
    if valid_ratio.numel() > 0:
        print(f"  ratio (fus/ref where |ref|>1e-3): min={valid_ratio.min().item():.4f} "
              f"max={valid_ratio.max().item():.4f} mean={valid_ratio.mean().item():.4f}")
    print(f"{'='*70}\n")

    # Pass criterion (init-mode aware): every element must satisfy
    #   |fus - ref| <= atol + atol_frac * amax|ref| + rtol * |ref|
    # plus no non-finite values on either side.  The fused path quantizes the
    # intermediate to fp8 per-128-group while the ref keeps per-token scales,
    # so the residual noise floor scales with the output amplitude -- the
    # atol_frac term tracks it via the global ref amplitude.
    #   randn init (outputs ~+-0.1, noise tail ~4% of amax):
    #     atol=0.00355 + 0.03*amax + 0.02*|ref| (~0.008 effective, ~2x margin
    #     over the observed 0.005 tail; structural bugs produce 0.03+ errors)
    #   nv init (unit-variance randn, outputs ~+-1e5, noise ~0.9% of amax,
    #     i.e. a few bf16 ulps): 0.02*amax + 0.02*|ref| (~2.4x margin)
    if NV_INIT:
        check_atol = float(os.environ.get("DG_TEST_ATOL", "0"))
        check_atol_frac = float(os.environ.get("DG_TEST_ATOL_FRAC", "0.02"))
        check_rtol = float(os.environ.get("DG_TEST_RTOL", "0.02"))
    else:
        check_atol = float(os.environ.get("DG_TEST_ATOL", "0.00355"))
        check_atol_frac = float(os.environ.get("DG_TEST_ATOL_FRAC", "0.03"))
        check_rtol = float(os.environ.get("DG_TEST_RTOL", "0.02"))
    ref_amax = combine_ref.float().abs().max().item()
    max_abs_diff = abs_diff.max().item()
    tol = check_atol + check_atol_frac * ref_amax + check_rtol * combine_ref.float().abs()
    tol_excess = abs_diff - tol
    n_tol_fail = (tol_excess > 0).sum().item()
    finite_ok = nan_fus == 0 and inf_fus == 0 and nan_ref == 0 and inf_ref == 0
    print(f"  check: max_abs={max_abs_diff:.6f} "
          f"tol={check_atol:g}+{check_atol_frac:g}*amax({ref_amax:.4f})+{check_rtol:g}*|ref| "
          f"| tol_fail={n_tol_fail} | finite={finite_ok}")
    passed = (n_tol_fail == 0) and finite_ok
    if passed:
        print(f"test rank {rank_idx} async mode {async_mode} PASS")
    else:
        print(f"test rank {rank_idx} async mode {async_mode} FAIL --mismatch details above")
    if not passed:
        worst_excess = tol_excess.max().item()
        print(f"[rank {rank_idx}] CHECK_FAILED (non-fatal): max_abs={max_abs_diff:.6f} "
              f"tol_fail={n_tol_fail} worst_excess={worst_excess:.6f}")

    sym_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    # Default: 4 cards (GPU 0-3), 512 tokens per rank
    # Set HIP_VISIBLE_DEVICES=0,1,2,3 before running, e.g.:
    #   HIP_VISIBLE_DEVICES=0,1,2,3 python test_mega_moe_fp8.py 4 512
    num_ranks = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    local_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    print_recv_counts = "--print-recv-counts" in sys.argv[3:]

    # Supernode launch: one launcher per node, each owning local_nproc HCUs.
    # Set MASTER_ADDR / MASTER_PORT / DG_MEGA_NODE_RANK per node; num_ranks
    # above is the GLOBAL world size (e.g. 32 for two 16-card nodes).
    nnodes = int(os.environ.get("DG_MEGA_NNODES", "1"))
    node_rank = int(os.environ.get("DG_MEGA_NODE_RANK", "0"))
    assert num_ranks % nnodes == 0, (
        f"num_ranks={num_ranks} must be divisible by DG_MEGA_NNODES={nnodes}"
    )
    local_nproc = num_ranks // nnodes
    rank_offset = node_rank * local_nproc

    # Default shape = the deployed V4.1 MoE (48 experts/rank), so a plain
    # `python3 op_test/test_mega_moe_fp8.py 8 4096` runs the production config
    # without any environment variable.
    num_experts = num_ranks * 48
    num_topk = 6
    hidden = 5120
    intermediate_hidden = 2304
    # Shape override: DG_TEST_SHAPE="hidden,intermediate_hidden,num_topk,num_experts"
    _shape_env = os.environ.get("DG_TEST_SHAPE", "").strip()
    if _shape_env:
        hidden, intermediate_hidden, num_topk, num_experts = [
            int(v) for v in _shape_env.split(",")
        ]
        assert num_experts % num_ranks == 0
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
            rank_offset,
        ),
        nprocs=local_nproc,
        join=True,
    )
