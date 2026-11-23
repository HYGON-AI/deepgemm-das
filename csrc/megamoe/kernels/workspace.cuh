// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// Workspace metadata layout for HCU MegaMoE.
// Ported from SM90 layout/mega_moe.cuh:SM90Workspace (lines 210-318).
//
// The workspace is split into two regions:
//   - IPC region (fine-grained sym_buffer, no L2 cache): fields accessed by
//     remote ranks via sym.map — expert_send_count and src_token_topk_idx.
//     expert_send_count is written only by the local rank but remote-READ by
//     peer dispatch blocks after the before-pull rank barrier (pull protocol —
//     no remote writes on the count path).
//   - Local region (regular hipMalloc, L2-cached): fields only touched by the
//     local GPU — grid sync counters, expert_recv_count, expert_recv_count_sum,
//     l1_arrival_count, l2_arrival_mask, l2_quant_done, token_src_metadata.
//     Local fields use agent-scope atomics and benefit from L2 caching.
//     NOTE: expert_recv_count/_sum moved here because the pull protocol left
//     them with no remote access at all (no sym.map in the repo); keeping them
//     in the uncached IPC region only cost 64 blocks a system-scope spin.

#include <cstdint>
#include <hip/hip_runtime.h>
#include "params.h"

namespace deep_gemm::mega::hcu {

// kNumExpertsPerRank/kNumRanks != 0 folds all topology math (strides, expert
// count addressing) to constant expressions. The default <0,0> instantiation
// keeps the runtime path for host-side sizing in buffer_layout.cuh and for
// kernels launched with an unspecialized topology.
template <uint32_t kNumExpertsPerRank = 0, uint32_t kNumRanks = 0>
struct HCUMegaMoeWorkspace {
    void* base;        // IPC workspace base (cross-rank fields)
    void* local_base;  // local sync workspace base (hipMalloc, L2-cached)
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;
    // num_max_pool_tokens removed — only used during construction, compute on-the-fly
    uint32_t num_max_pool_blocks;
    // L1 ring-buffer capacity: when the full pool would exceed the 16K-token
    // equivalent, the pull -> L1 GEMM staging area is capped at this size and
    // reused as a ring (producer waits for the consumer's per-block release).
    // Equal to num_max_pool_blocks * kMinCandidateBlockM when not capped.
    uint32_t num_l1_pool_tokens;
    uint32_t num_l1_pool_blocks;
    bool has_l1_ring;

    // Fold to constants when topology is specialized; read the runtime fields
    // in the default <0,0> instantiation.
    __host__ __device__ uint32_t epr_v() const {
        if constexpr (kNumExpertsPerRank != 0) return kNumExpertsPerRank;
        else return num_experts_per_rank;
    }
    __host__ __device__ uint32_t nr_v() const {
        if constexpr (kNumRanks != 0) return kNumRanks;
        else return num_ranks;
    }
    __host__ __device__ uint32_t ne_v() const {
        if constexpr (kNumExpertsPerRank != 0 && kNumRanks != 0)
            return kNumExpertsPerRank * kNumRanks;
        else return num_experts;
    }

    // 16 bytes for grid sync counters: 3 barriers x 1 x uint32_t
    // (packed count+epoch word), padded to 16 for u64 alignment.
    static constexpr uint64_t kNumGridSyncBytes = 16;
    static constexpr uint32_t kNumMaxGridSyncCounters = 3;

    __host__ __device__
    HCUMegaMoeWorkspace(void* base,
                        void* local_base,
                        const uint32_t num_ranks,
                        const uint32_t num_experts,
                        const uint32_t num_max_tokens_per_rank,
                        const uint32_t num_topk,
                        const uint32_t num_sms = kNumSMs,
                        // Ring kill-switch. Host sizing callers pass the
                        // DG_MEGA_DISABLE_RING env; kernel instances pass
                        // kUseRingBuffer so both sides always agree on the
                        // local-buffer layout.
                        const bool l1_ring_enabled = true)
        : base(base), local_base(local_base),
          num_ranks(num_ranks), num_experts(num_experts),
          num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        const uint32_t num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
        // Activation is pinned to the fixed-SM bound (identical on host and
        // every kernel instantiation); the capacity itself may differ per SM
        // count, which is safe: a kernel never sizes its ring above the cap
        // the host used for allocation (kernel kNumSMs <= host num_sms).
        has_l1_ring = l1_ring_enabled &&
                      num_max_pool_tokens > get_l1_ring_activation_pool_tokens();
        num_l1_pool_tokens = has_l1_ring
                                 ? constexpr_min(num_max_pool_tokens,
                                                 get_l1_ring_pool_tokens(num_sms))
                                 : num_max_pool_tokens;
        num_l1_pool_blocks = num_l1_pool_tokens / kMinCandidateBlockM;
    }

    // IPC region layout (cross-rank; accessed via sym.map by remote ranks):
    //   [expert_send_count:      num_experts x u64]
    //   [src_token_topk_idx:     nepr * nr * num_max_recv_tokens_per_expert x int]
    __host__ __device__
    uint64_t get_num_bytes() const {
        uint64_t n = 0;
        n += static_cast<uint64_t>(ne_v()) * sizeof(uint64_t);                 // expert send count
        n += static_cast<uint64_t>(epr_v()) * nr_v() *
             num_max_recv_tokens_per_expert * sizeof(int);                      // src_token_topk_idx
        return align_u64(n, 16);
    }

    // Bytes of the rank-local recv-count tables (recv_count + recv_count_sum),
    // laid out right after the grid-sync counters in the local region.
    __host__ __device__
    uint64_t get_recv_count_bytes() const {
        return align_u64((static_cast<uint64_t>(ne_v()) + epr_v()) * sizeof(uint64_t), 16);
    }

    // Local region layout (single-rank only; regular L2-cached memory):
    //   [grid sync counters: 16B]
    //   [expert_recv_count:      num_ranks * num_experts_per_rank x u64]
    //   [expert_recv_count_sum:  num_experts_per_rank x u64]
    //   [l1_arrival_count:   align(align32(l1pb,2),4) x u32]  — ring-sized
    //   [l1_ready_gen:       same x u32]  — ring only
    //   [l1_cons_units:      same x u32]  — ring only
    //   [l1_gate_gen:        same x u32]  — ring only
    //   [l2_arrival_mask:    npb x u64]
    //   [l2_quant_done:      npb x i32]
    //   [token_src_metadata: num_max_pool_tokens x TokenSrcMetadata]
#if DG_HANG_DIAG
    //   [block phase:        kMaxPhaseSMs x u32]  — per-block progress marker
    //   [block phase detail: kMaxPhaseSMs x u32]  — pool_block_idx / token_idx
    //   [warp progress:      kMaxPhaseSMs*16 x u32] — per-warp pull token_idx
    // (diag-only: allocated and written under DG_MEGA_HANG_PRINT/DG_MEGAMOE_DBUG)
    static constexpr uint32_t kMaxPhaseSMs = 128;
    static constexpr uint32_t kMaxPhaseWarps = 128 * 16;
#endif
    __host__ __device__
    uint64_t get_local_num_bytes() const {
        const uint32_t num_max_pool_tokens = num_max_pool_blocks * kMinCandidateBlockM;
        const uint64_t l1_sync_words =
            align_u64(align_u32(num_l1_pool_blocks, 2u), 4);
        uint64_t n = 0;
        n += kNumGridSyncBytes;                                                // grid sync counters
        n += get_recv_count_bytes();                                            // recv count + sum
        n += l1_sync_words * sizeof(uint32_t);                                 // l1_arrival_count
        n += l1_sync_words * (has_l1_ring ? 3 : 0) * sizeof(uint32_t);         // ready/cons/gate
        n += static_cast<uint64_t>(num_max_pool_blocks) * sizeof(uint64_t);    // l2_arrival_mask
        n += static_cast<uint64_t>(num_max_pool_blocks) * sizeof(int32_t);     // l2_quant_done
        n += static_cast<uint64_t>(num_max_pool_tokens) * sizeof(TokenSrcMetadata); // token_src_metadata
#if DG_HANG_DIAG
        n += kMaxPhaseSMs * sizeof(uint32_t);                                  // block phase
        n += kMaxPhaseSMs * sizeof(uint32_t);                                  // block phase detail
        n += kMaxPhaseWarps * sizeof(uint32_t);                                // per-warp progress
#endif
        return align_u64(n, 16);
    }

    __host__ __device__
    void* get_end_ptr() const {
        return static_cast<uint8_t*>(base) + get_num_bytes();
    }

    // Grid sync counters: each barrier uses 1 x uint32_t packed
    // count+epoch word (low 16 bits = arrivals, high 16 bits = epoch).
    // Local region offset 0.
    template <uint32_t kIndex = 0>
    __device__ uint32_t* get_grid_sync_count_ptr() const {
        static_assert(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(local_base) + kIndex;
    }

    // L1 arrival count: [align(num_l1_pool_blocks, 2)] uint32_t — local region.
    // In ring mode this counts tokens of the in-flight generation; the last
    // consumer of a generation resets it before publishing the gate.
    __device__ uint32_t* get_l1_arrival_count_ptr(const uint32_t pool_block_idx = 0) const {
        return reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(local_base) + kNumGridSyncBytes +
            get_recv_count_bytes()) + pool_block_idx;
    }

    // Ring sync arrays (allocated only when has_l1_ring):
    //   ready_gen  — last fully produced generation (producer -> consumer)
    //   cons_units — cumulative consumed L1 tiles (consumer bookkeeping)
    //   gate_gen   — last freed generation incl. the arrival reset
    //                (consumer -> producer)
    __device__ uint32_t* get_l1_ready_gen_ptr(const uint32_t ring_block = 0) const {
        return get_l1_arrival_count_ptr(align_u32(num_l1_pool_blocks, 2u)) + ring_block;
    }
    __device__ uint32_t* get_l1_cons_units_ptr(const uint32_t ring_block = 0) const {
        return get_l1_ready_gen_ptr(align_u32(num_l1_pool_blocks, 2u)) + ring_block;
    }
    __device__ uint32_t* get_l1_gate_gen_ptr(const uint32_t ring_block = 0) const {
        return get_l1_cons_units_ptr(align_u32(num_l1_pool_blocks, 2u)) + ring_block;
    }

    // L2 arrival mask: [num_max_pool_blocks] uint64_t — local region.
    __device__ uint64_t* get_l2_arrival_mask_ptr(const uint32_t pool_block_idx = 0) const {
        const auto l1_slots = align_u32(num_l1_pool_blocks, 2u);
        const auto base_ptr = get_l1_arrival_count_ptr(
            l1_slots * (has_l1_ring ? 4 : 1));
        return reinterpret_cast<uint64_t*>(base_ptr) + pool_block_idx;
    }

    // L2 quant done flag: [num_max_pool_blocks] int32_t — local region.
    // Used to coordinate SwiGLU+quant: 0=not started, 1=in progress, 2=done
    __device__ int32_t* get_l2_quant_done_ptr(const uint32_t pool_block_idx = 0) const {
        const auto base_ptr = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<int32_t*>(base_ptr) + pool_block_idx;
    }

    // Token source metadata: [num_max_pool_tokens] TokenSrcMetadata — local region.
    __device__ TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t pool_token_idx = 0) const {
        const auto base_ptr = reinterpret_cast<TokenSrcMetadata*>(
            get_l2_quant_done_ptr(num_max_pool_blocks));
        return base_ptr + pool_token_idx;
    }

#if DG_HANG_DIAG
    // Per-block progress markers (debug/hang diagnosis): [kMaxPhaseSMs] u32.
    // Written by thread 0 of each block at stage boundaries; read from host
    // while the kernel is spinning to pinpoint the stuck stage.
    __device__ uint32_t* get_block_phase_ptr(const uint32_t sm_idx = 0) const {
        const uint32_t num_max_pool_tokens = num_max_pool_blocks * kMinCandidateBlockM;
        const auto base_ptr = reinterpret_cast<uint32_t*>(
            get_token_src_metadata_ptr(num_max_pool_tokens));
        return base_ptr + sm_idx;
    }
    __device__ uint32_t* get_block_phase_detail_ptr(const uint32_t sm_idx = 0) const {
        return get_block_phase_ptr(kMaxPhaseSMs) + sm_idx;
    }
    // Per-warp progress: [sm_idx][warp_idx] — each warp's lane0 writes its
    // current token_idx during pull so stuck warps are individually visible.
    __device__ uint32_t* get_warp_progress_ptr(const uint32_t sm_idx, const uint32_t warp_idx) const {
        return get_block_phase_ptr(kMaxPhaseSMs + kMaxPhaseSMs) + sm_idx * 16 + warp_idx;
    }
#endif

    // Expert recv count: [num_ranks, num_experts_per_rank] uint64_t — local
    // region right after the grid-sync counters.  Rank-local only (the pull
    // protocol removed the remote count push), so it is L2-cached and all its
    // atomics use agent scope.
    __device__ uint64_t* get_expert_recv_count_ptr(
        const uint32_t rank_idx = 0, const uint32_t expert_idx = 0) const {
        return reinterpret_cast<uint64_t*>(
            static_cast<uint8_t*>(local_base) + kNumGridSyncBytes) +
            rank_idx * epr_v() + expert_idx;
    }

    // Expert recv count sum: [num_experts_per_rank] uint64_t — local region.
    __device__ uint64_t* get_expert_recv_count_sum_ptr(const uint32_t expert_idx = 0) const {
        return get_expert_recv_count_ptr(nr_v(), expert_idx);
    }

    // Expert send count: [num_experts] uint64_t — IPC region offset 0, written
    // only by the local rank during dispatch (per-SM atomicAdd of
    // (1<<32)|count) and remote-READ by peers after the before-pull barrier.
    // Cleared by the owner after the post-cleanup barrier, once every peer's
    // pull has completed.
    __device__ uint64_t* get_expert_send_count_ptr(const uint32_t expert_idx = 0) const {
        return static_cast<uint64_t*>(base) + expert_idx;
    }

    // Source token topk idx: [num_experts_per_rank, num_ranks, num_max_recv_tokens_per_expert] int — IPC.
    __device__ uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t expert_idx = 0, const uint32_t rank_idx = 0, const uint32_t token_idx = 0) const {
        const auto base_ptr = reinterpret_cast<uint32_t*>(
            get_expert_send_count_ptr(ne_v()));
        return base_ptr +
            expert_idx * (nr_v() * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }
};

} // namespace deep_gemm::mega::hcu
