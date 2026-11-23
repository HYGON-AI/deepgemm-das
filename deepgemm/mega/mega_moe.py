# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Python interface for HCU gfx938 MegaMoE (FP8 W8A8).

This module provides:
  - SymmBuffer: symmetric buffer management (HIP IPC or HIP fabric)
  - get_symm_buffer_for_mega_moe: allocate + exchange peer handles
  - transform_fp8_weights_for_mega_moe: channel-wise FP8 quantization + permute
  - fp8_mega_moe: fused MegaMoE launch

The implementation follows the SM90 MegaMoE single-kernel flow, with:
  - V1 GEMM (gemm_channelwise_stage) for MFMA computation
  - V1 silu_and_mul_quant_channelwise for SwiGLU + per-token quantization
  - DeepEP warp_reduce for 64-lane reductions
"""

import os
import torch
import torch.distributed as dist
from types import SimpleNamespace
from typing import Optional, Tuple

try:
    import deepgemm.mega._C as _C
    _MEGA_EXT_AVAILABLE = True
except ImportError:
    _C = None
    _MEGA_EXT_AVAILABLE = False


def _mega_arch_supported() -> bool:
    if not _is_hip_backend() or not torch.cuda.is_available():
        return False
    arch = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()),
        "gcnArchName", "") or ""
    return arch.split(":")[0].strip() == "gfx938"


def _require_mega_support() -> None:
    if not _MEGA_EXT_AVAILABLE or not _mega_arch_supported():
        raise RuntimeError(
            "deepgemm.mega (MegaMoE) is not supported on this platform: "
            "the fused kernels are only built for HCU gfx938."
        )


# DG_MEGA_CFG_ALLGATHER=1: before each fused launch, all-gather the actual
# per-rank token counts so the kernel's per-expert density heuristic uses
# the true cross-rank total instead of the capacity upper bound.  Costs
# ~0.25-0.3 ms per call; opt-in, off by default.
_MEGA_CFG_ALLGATHER = os.environ.get("DG_MEGA_CFG_ALLGATHER", "0") == "1"


def _gather_tokens_all_ranks(sym_buffer: "SymmBuffer", num_tokens: int) -> int:
    """All-gather actual token counts across ranks; returns the global sum."""
    local = torch.tensor([num_tokens], dtype=torch.int64, device="cuda")
    out = torch.zeros(sym_buffer.group.size(), dtype=torch.int64,
                      device=local.device)
    dist.all_gather_into_tensor(out, local, group=sym_buffer.group)
    return int(out.sum().item())


def _is_hip_backend() -> bool:
    return torch.version.hip is not None


def _align(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# Peer-memory transport for the symmetric buffers.
#
# HIP IPC handles only reach GPUs of the same node. On an HCU supernode the
# ranks are spread over several hosts, so the buffers must instead be exported
# through DTK HIP fabric handles (hipHsaExtRpcMemoryCreate / AttachDevice).
# Transport is an explicit, orthogonal choice: the EP/rank count never
# switches the mode, so existing single-node runs keep the exact IPC path.
# ---------------------------------------------------------------------------
_PEER_MEMORY_IPC = "ipc"
_PEER_MEMORY_FABRIC = "fabric"
_PEER_MEMORY_ENV = "DG_MEGA_PEER_MEMORY"
# Accept the HCU-named alias used by supernode launch scripts.
_PEER_MEMORY_ENV_COMPAT = "MEGAMOE_HCU_PEER_MEMORY"


def _peer_memory_mode(override: Optional[str] = None) -> str:
    value = override
    if value is None:
        value = os.environ.get(_PEER_MEMORY_ENV) or os.environ.get(_PEER_MEMORY_ENV_COMPAT)
    if value is None or value.strip().lower() in ("", "0", "false", "off", "no", "ipc"):
        return _PEER_MEMORY_IPC
    if value.strip().lower() in ("1", "true", "on", "yes", "rpc", "fabric", "mnvl"):
        return _PEER_MEMORY_FABRIC
    raise ValueError(
        f"{_PEER_MEMORY_ENV} must be unset/ipc or one of "
        "rpc/fabric/mnvl/1/true/on/yes"
    )


def _require_fabric_support() -> None:
    """Fail loudly if the installed extension predates HIP fabric support."""
    if not hasattr(_C, "allocate_hip_fabric_buffer"):
        raise RuntimeError(
            f"{_PEER_MEMORY_ENV}=rpc requires a MegaMoE extension with HIP "
            "fabric peer-memory support. Rebuild and reinstall the wheel; "
            "the installed _C extension has no fabric buffer bindings."
        )


class SymmBuffer:
    """Symmetric buffer for MegaMoE (HIP IPC or HIP fabric peer memory).

    Allocates a shared buffer across ranks, exchanges handles via
    dist.all_gather_object, and installs peer pointers. The buffer layout
    follows SM90 SM90Workspace + buffer slices (with per-token SF).

    HIP IPC is the default; DG_MEGA_PEER_MEMORY=rpc selects HIP fabric
    handles for supernode runs. Both transports use uncached device memory.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
        cuda_graph_max_tokens_per_rank: Optional[int] = None,
        peer_memory_mode: Optional[str] = None,
    ):
        _require_mega_support()
        if not _is_hip_backend():
            raise RuntimeError("deepgemm.mega is the HIP/HCU MegaMoE package")
        peer_memory_mode = _peer_memory_mode(peer_memory_mode)
        if peer_memory_mode == _PEER_MEMORY_FABRIC:
            _require_fabric_support()
        use_fabric = peer_memory_mode == _PEER_MEMORY_FABRIC

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.cuda_graph_max_tokens_per_rank = int(
            cuda_graph_max_tokens_per_rank
            if cuda_graph_max_tokens_per_rank is not None
            else num_max_tokens_per_rank
        )

        # 1. Compute buffer sizes
        #    ipc_bytes: fine-grained (no L2 cache) for P2P regions
        #    local_bytes: regular hipMalloc (with L2 cache) for GEMM staging
        ipc_bytes, local_bytes, slice_fn = _C.get_symm_buffer_size_for_mega_moe(
            group.size(),
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch,
            activation,
        )

        print(f"[rank {group.rank()}] SymmBuffer {peer_memory_mode}: "
              f"{ipc_bytes / 1024 / 1024:.2f} MB "
              f"({ipc_bytes} bytes), Local: {local_bytes / 1024 / 1024:.2f} MB "
              f"({local_bytes} bytes), num_ranks={group.size()} num_experts={num_experts} "
              f"num_max_tokens={num_max_tokens_per_rank} num_topk={num_topk} "
              f"hidden={hidden} intermediate_hidden={intermediate_hidden}", flush=True)

        # 2. Allocate the symmetric buffer. Fabric/RPC handles travel through
        #    the same all_gather_object as IPC handles; only the C++ code that
        #    creates/attaches them differs.
        if use_fabric:
            self.buffer, buffer_ptr, local_handle = (
                _C.allocate_hip_fabric_buffer(ipc_bytes)
            )
            open_handles = _C.open_hip_fabric_handles
        else:
            self.buffer, buffer_ptr, local_handle = _C.allocate_hip_ipc_buffer(ipc_bytes)
            open_handles = _C.open_hip_ipc_handles

        # 2b. Allocate local buffer (regular hipMalloc, with L2 cache — for GEMM)
        self.local_buffer, local_buffer_ptr = _C.allocate_hip_local_buffer(local_bytes)

        # 3. All-gather handles across ALL ranks, including the other nodes
        handles = [None] * group.size()
        dist.all_gather_object(handles, local_handle, group)
        buffer_ptrs = open_handles(handles, group.rank())
        buffer_ptrs[group.rank()] = buffer_ptr

        # 4. Signal buffer for the rank barrier: it must be remote-writable too,
        #    so it follows the same transport as the data buffer.
        #    Need 4 independent barrier slots (dispatch pre-pull, epilogue
        #    pre-combine, post-cleanup, spare) × num_ranks int slots
        signal_num_bytes = _align(max(group.size(), 4) * 4 * 4, 128)
        if use_fabric:
            signal_ptr, signal_handle = (
                _C.allocate_hip_fabric_signal_buffer(signal_num_bytes)
            )
        else:
            signal_ptr, signal_handle = (
                _C.allocate_hip_ipc_signal_buffer(signal_num_bytes)
            )
        signal_handles = [None] * group.size()
        dist.all_gather_object(signal_handles, signal_handle, group)
        signal_ptrs = open_handles(signal_handles, group.rank())
        signal_ptrs[group.rank()] = signal_ptr

        # 5. Install peer pointers
        self.handle = SimpleNamespace(
            buffer_ptrs=buffer_ptrs,
            buffer_ptr=buffer_ptr,
            local_buffer_ptr=local_buffer_ptr,
            signal_ptrs=signal_ptrs,
            signal_buffer_ptr=signal_ptr,
            peer_memory_mode=peer_memory_mode,
        )

        self.buffer.zero_()
        self.local_buffer.zero_()
        _C.set_mega_moe_peer_ptrs(self.buffer, buffer_ptrs, signal_ptrs)
        torch.cuda.synchronize()
        self.group.barrier()

        # 6. Slice input buffers (from both IPC and local buffers)
        slices = slice_fn(self.buffer, self.local_buffer)
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
        ) = slices[:8]

    def destroy(self):
        if self.handle is not None:
            torch.cuda.synchronize()
            close_handles = (
                _C.close_hip_fabric_handles
                if getattr(self.handle, "peer_memory_mode", _PEER_MEMORY_IPC)
                == _PEER_MEMORY_FABRIC
                else _C.close_hip_ipc_handles
            )
            remote_ptrs = [
                ptr if rank != self.group.rank() else 0
                for rank, ptr in enumerate(self.handle.buffer_ptrs)
            ]
            remote_signal_ptrs = [
                ptr if rank != self.group.rank() else 0
                for rank, ptr in enumerate(self.handle.signal_ptrs)
            ]
            close_handles(remote_ptrs)
            close_handles(remote_signal_ptrs)
            # Both transports allocate the local buffer with
            # hipExtMallocWithFlags and wrap it with torch::from_blob, so the
            # tensor does NOT own the memory: free it explicitly with hipFree.
            _C.free_hip_ipc_buffer(self.handle.buffer_ptr)
            _C.free_hip_ipc_signal_buffer(self.handle.signal_buffer_ptr)
            # Free local GEMM buffer (allocated via hipMalloc, with L2 cache)
            _C.free_hip_local_buffer(self.handle.local_buffer_ptr)
        self.handle = None
        self.buffer = None
        self.local_buffer = None
        self.group = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
    peer_memory_mode: Optional[str] = None,
    **kwargs,
) -> SymmBuffer:
    """Allocate a SymmBuffer for MegaMoE.

    peer_memory_mode: None (read DG_MEGA_PEER_MEMORY), "ipc" for same-node
    HIP IPC, or "rpc"/"fabric" for supernode HIP fabric buffers.
    """
    return SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch=use_fp8_dispatch,
        activation=activation,
        peer_memory_mode=peer_memory_mode,
        **kwargs,
    )


def transform_fp8_weights_for_mega_moe(
    l1_weights: torch.Tensor,  # [num_experts_per_rank, 2*ih, hidden] BF16
    l2_weights: torch.Tensor,  # [num_experts_per_rank, hidden, ih] BF16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Channel-wise FP8 quantization for MegaMoE weights.

    Returns:
      l1_w_fp8:  [num_experts_per_rank, 2*ih, hidden] FP8 E4M3 (gate-up interleaved)
      l1_w_sf:   [num_experts_per_rank, 2*ih] FP32 (per-channel scale, interleaved)
      l2_w_fp8:  [num_experts_per_rank, hidden, ih] FP8 E4M3
      l2_w_sf:   [num_experts_per_rank, hidden] FP32 (per-channel scale)
    """
    def quantize_channelwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # w: [E, N, K] BF16
        # Per-channel scale: amax over K, one scale per (E, N)
        amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)  # [E, N, 1]
        scale = amax / 448.0  # [E, N, 1]
        w_fp8 = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return w_fp8, scale.squeeze(-1).contiguous()

    # Interleave gate/up in N dimension with granularity 8 before quantization.
    # This enables per-tile SwiGLU in the L1 GEMM epilogue: each BLOCK_N tile
    # contains equal gate/up columns, so SwiGLU can be computed immediately
    # after the L1 GEMM without waiting for all N-blocks.
    l1_weights = interleave_gate_up_weights(l1_weights)

    l1_w_fp8, l1_w_sf = quantize_channelwise(l1_weights)
    l2_w_fp8, l2_w_sf = quantize_channelwise(l2_weights)
    return l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf


def transform_int8_weights_for_mega_moe(
    l1_weights: torch.Tensor,  # [num_experts_per_rank, 2*ih, hidden] BF16
    l2_weights: torch.Tensor,  # [num_experts_per_rank, hidden, ih] BF16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Channel-wise INT8 quantization for MegaMoE weights (W8A8-INT8).

    Uses symmetric per-channel quantization with scale = amax / 127.0,
    matching DeepEP's _quantize_low_latency_layered quant_type=1 convention.

    Returns:
      l1_w_int8: [num_experts_per_rank, 2*ih, hidden] INT8
      l1_w_sf:   [num_experts_per_rank, 2*ih] FP32 (per-channel scale)
      l2_w_int8: [num_experts_per_rank, hidden, ih] INT8
      l2_w_sf:   [num_experts_per_rank, hidden] FP32 (per-channel scale)
    """
    def quantize_channelwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # w: [E, N, K] BF16
        # Per-channel scale: amax over K, one scale per (E, N)
        amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)  # [E, N, 1]
        scale = amax / 127.0  # [E, N, 1]
        w_int8 = torch.round(w / scale).clamp(-127.0, 127.0).to(torch.int8)
        return w_int8, scale.squeeze(-1).contiguous()

    l1_w_int8, l1_w_sf = quantize_channelwise(l1_weights)
    l2_w_int8, l2_w_sf = quantize_channelwise(l2_weights)
    return l1_w_int8, l1_w_sf, l2_w_int8, l2_w_sf


def interleave_gate_up_weights(
    l1_weights: torch.Tensor,  # [E, 2*ih, K] BF16
) -> torch.Tensor:
    """Interleave gate/up rows with granularity 8 in the N (output) dimension.

    Input layout (separated halves):
      rows [0, ih)       = gate weights
      rows [ih, 2*ih)    = up weights

    Output layout (interleaved, granularity 8):
      rows [0, 8)        = gate[0:8]
      rows [8, 16)       = up[0:8]
      rows [16, 24)      = gate[8:16]
      rows [24, 32)      = up[8:16]
      ...

    This matches NV SM100's granularity-8 interleaving, enabling per-tile
    SwiGLU fusion in the L1 GEMM epilogue.
    """
    E, N, K = l1_weights.shape
    assert N % 2 == 0, f"N must be even, got {N}"
    ih = N // 2
    assert ih % 8 == 0, f"intermediate_hidden must be divisible by 8, got {ih}"
    gate = l1_weights[:, :ih, :]  # [E, ih, K]
    up = l1_weights[:, ih:, :]    # [E, ih, K]
    # Reshape to groups of 8, stack gate/up, then interleave
    gate = gate.reshape(E, ih // 8, 8, K)
    up = up.reshape(E, ih // 8, 8, K)
    # Stack: [E, ih//8, 2, 8, K] → [E, ih//8, 16, K] → [E, 2*ih, K]
    interleaved = torch.stack([gate, up], dim=2)  # [E, ih//8, 2, 8, K]
    interleaved = interleaved.reshape(E, 2 * ih, K)
    return interleaved.contiguous()


def interleave_gate_up_scales(
    l1_scales: torch.Tensor,  # [E, 2*ih] FP32
) -> torch.Tensor:
    """Interleave gate/up per-channel scales with granularity 8.

    Must match the weight interleaving from interleave_gate_up_weights.
    """
    return interleave_gate_up_weights(l1_scales.unsqueeze(-1)).squeeze(-1)


def transform_weights_for_mega_moe(
    l1_weights,
    l2_weights,
) -> Tuple:
    """Transform quantized weights into the MegaMoE kernel layout.

    NV-aligned weight preparation: accepts quantized weight tensors
    [E, N, K] (FP8 e4m3fn or INT8), each either bare or as a
    ``(weight, scale)`` tuple, and returns the packed Marlin w6 tensors
    consumed by ``fp8_mega_moe``.

      - L1 (gate/up): interleave gate/up with granularity 8 (weight and
        per-channel scale), then Marlin w6 pack.
      - L2 (down):    Marlin w6 pack only.

    Returns:
      ``((l1_w6, l1_sf), (l2_w6, l2_sf))`` for tuple inputs, or
      ``(l1_w6, l2_w6)`` for bare-tensor inputs.
    """
    l1_sf = l2_sf = None
    if isinstance(l1_weights, tuple):
        l1_weights, l1_sf = l1_weights
    if isinstance(l2_weights, tuple):
        l2_weights, l2_sf = l2_weights

    is_int8 = l1_weights.dtype == torch.int8
    if is_int8:
        from deepgemm import marlin_i8_contiguous_weight as marlin_pack
    else:
        # Upstream gfx938 switched marlin_fp8_*_weight to the pair32 pack;
        # the MegaMoE kernels still consume the Marlin w6 layout.
        from deepgemm.m_group_gemm import pack_int8_weight_w6_ll as marlin_pack

    # L1: interleave gate/up (granularity 8) so each BLOCK_N tile holds
    # equal gate/up columns -> per-tile SwiGLU in the L1 epilogue.
    l1_w6 = marlin_pack(interleave_gate_up_weights(l1_weights).contiguous().clone())
    if l1_sf is not None:
        l1_sf = interleave_gate_up_scales(l1_sf)
    l2_w6 = marlin_pack(l2_weights.contiguous().clone())

    if l1_sf is None and l2_sf is None:
        return l1_w6, l2_w6
    return (l1_w6, l1_sf), (l2_w6, l2_sf)


# DG_MEGA_TOPK_STATS=<dir>: per-call topk send histogram capture.
# Each fp8_mega_moe call appends (num_tokens, num_experts, bincount) rows;
# flushed every 512 calls to <dir>/topk_rank<r>_<seq>.pt. Summing the same
# call index across rank files yields the true per-expert recv counts.
_MEGA_TOPK_STATS_DIR = os.environ.get("DG_MEGA_TOPK_STATS", "")
_topk_stats_rows = []
_topk_stats_seq = 0


def _flush_topk_stats(sym_buffer: "SymmBuffer") -> None:
    global _topk_stats_seq
    if not _topk_stats_rows:
        return
    os.makedirs(_MEGA_TOPK_STATS_DIR, exist_ok=True)
    path = os.path.join(
        _MEGA_TOPK_STATS_DIR,
        f"topk_rank{sym_buffer.group.rank()}_{_topk_stats_seq}.pt",
    )
    torch.save(
        [(n, ne, c.cpu().to(torch.int32)) for n, ne, c in _topk_stats_rows],
        path,
    )
    _topk_stats_rows.clear()
    _topk_stats_seq += 1


import atexit as _atexit

_TOPK_STATS_RANK = [0]


def _flush_topk_stats_at_exit() -> None:
    if _topk_stats_rows:
        os.makedirs(_MEGA_TOPK_STATS_DIR, exist_ok=True)
        path = os.path.join(
            _MEGA_TOPK_STATS_DIR,
            f"topk_rank{_TOPK_STATS_RANK[0]}_{_topk_stats_seq}.pt",
        )
        torch.save(
            [(n, ne, c.cpu().to(torch.int32)) for n, ne, c in _topk_stats_rows],
            path,
        )


if _MEGA_TOPK_STATS_DIR:
    _atexit.register(_flush_topk_stats_at_exit)


def fp8_mega_moe(
    y: torch.Tensor,                    # [num_tokens, hidden] BF16 output
    l1_weights,                         # (w6, sf) tuple or bare w6 tensor
    l2_weights,                         # (w6, sf) tuple or bare w6 tensor
    sym_buffer: SymmBuffer,
    activation_clamp: float = float("inf"),
    fast_math: bool = True,
    is_int8: bool = False,
    l1_weights_sf: Optional[torch.Tensor] = None,
    l2_weights_sf: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    num_topk: Optional[int] = None,
    tokens_all_ranks: int = -1,
):
    """Launch fused FP8/INT8 MegaMoE (unified LL/HT entry).

    NV-style single entry for all token counts: the C++ launcher picks
    the GEMM family and block-M from the expected tokens per expert --
    expected >= DG_MEGA_HT_EXPECTED (default 24, equivalent to the old
    DG_MEGA_HT_THRESHOLD=192 capacity gate on the 8-rank shape) selects
    the HT family (BM64/BM128), below selects LL (BM16/32).  Selection
    is receiver-local: ranks may pick different families/tiles under
    imbalanced loads.  Both families consume the same Marlin w6 weight
    layout [E, K/64, N/16, 4, 16, 16] produced by
    ``transform_weights_for_mega_moe`` (a flat 3D view of the same
    storage is also accepted).

    Config-selection token source (for expected tokens/expert):
      - tokens_all_ranks >= 0: exact cross-rank token total (explicit).
      - DG_MEGA_CFG_ALLGATHER=1: gathered automatically inside this call.
      - default: this rank's actual num_tokens (same as NV).
      - DG_MEGA_CFG_ACTUAL=0: num_max_tokens_per_rank (capacity upper bound).

    Args:
      y:               output [num_tokens, hidden] BF16
      l1_weights:      gate/up projection -- (w6, sf) tuple from
                       transform_weights_for_mega_moe, or a bare w6
                       [E, K/64, N/16, 4, 16, 16] tensor (then pass
                       l1_weights_sf explicitly)
      l2_weights:      down projection -- same convention as l1_weights
      sym_buffer:      SymmBuffer (HIP IPC symmetric buffer); num_experts
                       and num_topk are taken from it unless overridden
      activation_clamp: clamp value for SwiGLU (inf = no clamp)
      fast_math:       use fast math (expf, fast_rcp)
      is_int8:         if True, use W8A8-INT8 (int8 MMA + int8 quant);
                       if False, use W8A8-FP8 (fp8 MMA + fp8 quant)
      l1_weights_sf:   per-channel scale [E_per_rank, 2*ih] FP32
                       (only needed with bare-tensor weights)
      l2_weights_sf:   per-channel scale [E_per_rank, hidden] FP32
      num_experts:     total experts across all ranks
                       (default: sym_buffer.num_experts)
      num_topk:        top-k value (default: sym_buffer.num_topk)
    """
    if isinstance(l1_weights, tuple):
        l1_weights, l1_weights_sf = l1_weights
    if isinstance(l2_weights, tuple):
        l2_weights, l2_weights_sf = l2_weights
    if num_experts is None:
        num_experts = sym_buffer.num_experts
    if num_topk is None:
        num_topk = sym_buffer.num_topk
    assert l1_weights_sf is not None and l2_weights_sf is not None, \
        "weight scales required: pass (w, sf) tuples or *_sf kwargs"
    if tokens_all_ranks < 0 and _MEGA_CFG_ALLGATHER:
        tokens_all_ranks = _gather_tokens_all_ranks(
            sym_buffer, y.size(0))
    if _MEGA_TOPK_STATS_DIR:
        _TOPK_STATS_RANK[0] = sym_buffer.group.rank()
        idx = sym_buffer.topk_idx[: y.size(0)].reshape(-1)
        counts = torch.bincount(idx[idx >= 0], minlength=num_experts)
        _topk_stats_rows.append((y.size(0), num_experts, counts))
        if len(_topk_stats_rows) >= 512:
            _flush_topk_stats(sym_buffer)
    _C.fp8_mega_moe(
        y,
        l1_weights,
        l2_weights,
        l1_weights_sf,
        l2_weights_sf,
        sym_buffer.buffer,
        sym_buffer.local_buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.handle.signal_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        num_experts,
        num_topk,
        activation_clamp,
        fast_math,
        is_int8,
        tokens_all_ranks,
    )


def mega_moe_preprocess(
    x_bf16: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    sym_buffer: "SymmBuffer",
    num_tokens: int,
    is_int8: bool = False,
):
    """Fused bf16 -> FP8/INT8 channelwise quantization + topk copy into sym buffer.

    Replaces the separate PyTorch quantization + copy_ calls with a single
    HIP kernel (mega_moe_preprocess_kernel) that:
      1. Computes per-row amax and scale (amax / 448.0 for FP8, amax / 127.0 for INT8)
      2. Quantizes bf16 input to FP8 e4m3fn or signed INT8
      3. Copies topk_idx and topk_weights into sym buffer
      4. Writes quantized data, scale, topk into sym_buffer directly

    Args:
      x_bf16:       input [num_tokens, hidden] BF16
      topk_idx:     [num_tokens, topk] int64 or int32
      topk_weights: [num_tokens, topk] float32 or bfloat16
      sym_buffer:   SymmBuffer (HIP IPC symmetric buffer)
      num_tokens:   number of tokens to process
      is_int8:      if True, quantize to signed INT8 (W8A8-INT8);
                    if False, quantize to FP8 E4M3 (W8A8-FP8)
    """
    rows = int(num_tokens)
    _C.mega_moe_preprocess(
        x_bf16,
        topk_idx,
        topk_weights,
        sym_buffer.x,
        sym_buffer.x_sf,
        sym_buffer.topk_idx,
        sym_buffer.topk_weights,
        rows,
        is_int8,
    )
