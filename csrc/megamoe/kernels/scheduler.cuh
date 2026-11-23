// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// MegaMoE scheduler for HCU gfx938 (64-lane wavefront).
// Ported from SM90 scheduler/mega_moe.cuh, with 32-lane intrinsics
// replaced by DeepEP-style 64-lane warp_reduce.

#include <cstdint>
#include <hip/hip_runtime.h>
#include "params.h"
#include "workspace.cuh"
#include "warp_reduce.cuh"

namespace deep_gemm::mega::hcu {

enum class BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumSMs,
          uint32_t kNumExpertsPerRank = 0, uint32_t kNumRanks = 0,
          uint32_t L1_BLOCK_N = 0>
struct MegaMoEScheduler {
    // L1 and L2 may use different tile widths: the W12-P L1 pass is 768
    // columns wide while the L2 groupwise GEMM keeps BLOCK_N.  L1_BLOCK_N == 0
    // keeps both at BLOCK_N (production default).
    static constexpr uint32_t kL1BlockN = (L1_BLOCK_N != 0) ? L1_BLOCK_N : BLOCK_N;
    // Compile-time constants from shapes
    // L1 tiles are counted with ceil as well: a 768-wide L1 pass leaves a
    // narrower trailing block when 2*ih does not divide by it
    // (hy4: 4096 = 5x768 + 256, run on 8 math warps by the kernel).  Only the
    // count and the arrival-mask width depend on this -- n_block_idx is passed
    // through as the column tile index, never multiplied by kL1BlockN here.
    static constexpr uint32_t kNumL1BlockNs =
        (L1_SHAPE_N + kL1BlockN - 1) / kL1BlockN;
    static constexpr uint32_t kL1TailWidth = L1_SHAPE_N % kL1BlockN;
    // L2 tiles are counted with ceil: a 768-wide tile does not divide 5120
    // (6x768 + 512), so the kernel maps the last tile back to
    // L2_SHAPE_N - BLOCK_N and lets it overlap its predecessor. Exact
    // multiples keep their old count.
    static constexpr uint32_t kNumL2BlockNs =
        (L2_SHAPE_N + BLOCK_N - 1) / BLOCK_N;
    static constexpr uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K;
    static constexpr uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K;

    static_assert(kL1TailWidth == 0 || kL1TailWidth % (8 * 16) == 0,
                  "L1 tail width must be zero or tile 8 math warps x 16 columns");
    static_assert(L2_SHAPE_N >= BLOCK_N, "Invalid shape");
    static_assert(kNumL2BlockNs * BLOCK_N - L2_SHAPE_N < BLOCK_N,
                  "L2 tail overlap must stay below one tile");
    static_assert(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    static_assert(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");

    // Runtime topology — passed as constructor args and kept in registers to
    // avoid LDS reads from the workspace object in the scheduling hot path.

    // Per-lane expert slot count: folds to ceil(kNumExpertsPerRank/kWarpSize)
    // when topology is specialized (48 -> 1), else stays at the runtime max.
    // Bounds the stored_expert_counts register array.
    static constexpr uint32_t kExpertsPerLane =
        (kNumExpertsPerRank != 0)
            ? (kNumExpertsPerRank + kWarpSize - 1) / kWarpSize
            : static_cast<uint32_t>(kMaxExpertsPerLane);

    const HCUMegaMoeWorkspace<kNumExpertsPerRank, kNumRanks>& workspace;
    const uint32_t num_experts_per_rank_;
    const uint32_t num_ranks_;

    // Per-lane cached token counts kept in registers.  Every access is
    // inside a #pragma unroll loop with a compile-time index, so the
    // array stays in registers (never demoted to local memory).
    // stored_expert_counts[i] holds expert (i * 64 + lane_id)'s count.
    uint32_t stored_expert_counts[kExpertsPerLane];

    BlockPhase next_phase = BlockPhase::Linear1;
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    __device__ MegaMoEScheduler(const HCUMegaMoeWorkspace<kNumExpertsPerRank, kNumRanks>& ws,
                                 uint32_t experts_per_rank,
                                 uint32_t ranks)
        : workspace(ws), num_experts_per_rank_(experts_per_rank), num_ranks_(ranks) {
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; ++i)
            stored_expert_counts[i] = 0;
        block_idx = blockIdx.x;
    }

    // Accessor for per-lane expert count — replaces stored_num_tokens_per_expert[i]
    __device__ uint32_t get_stored_expert_count(uint32_t i) const {
        return stored_expert_counts[i];
    }
    __device__ void set_stored_expert_count(uint32_t i, uint32_t val) {
        stored_expert_counts[i] = val;
    }

    // Copy the per-lane count cache from another scheduler instantiation —
    // used when the L1 scheduler already fetched it for this warp, avoiding
    // a second GMEM fetch pass.
    template <typename Other>
    __device__ void copy_expert_counts_from(const Other& src) {
        const uint32_t epl = get_num_experts_per_lane();
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; ++i) {
            if (i >= epl) break;
            stored_expert_counts[i] = src.stored_expert_counts[i];
        }
    }

    // Accessors fold to compile-time constants when topology is specialized.
    __device__ uint32_t get_num_experts_per_rank() const {
        return (kNumExpertsPerRank != 0) ? kNumExpertsPerRank : num_experts_per_rank_;
    }
    __device__ uint32_t get_num_ranks() const {
        return (kNumRanks != 0) ? kNumRanks : num_ranks_;
    }
    __device__ uint32_t get_num_experts_per_lane() const {
        return (get_num_experts_per_rank() + kWarpSize - 1) / kWarpSize;
    }

    __device__ uint32_t get_wave_expert_end_idx() const {
        // Single wave: all experts processed in one wave
        return get_num_experts_per_rank();
    }

    __device__ uint32_t get_num_tokens(const uint32_t expert_idx) const {
        uint32_t valid_value = 0;
        const uint32_t epl = get_num_experts_per_lane();
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; ++i) {
            if (i >= epl) break;
            if (expert_idx == i * kWarpSize + lane_id())
                valid_value = get_stored_expert_count(i);
        }
        // Broadcast from the lane that holds the value.  readfirstlane keeps
        // the result in the scalar (SGPR) domain so the carried scheduler
        // state below does not occupy VGPRs across the GEMM inner loop.
        return broadcast_lane0(shfl_sync(valid_value, expert_idx % kWarpSize));
    }

    __device__ uint32_t get_pool_block_offset(const uint32_t expert_idx) {
        uint32_t num_blocks = 0;
        const uint32_t epl = get_num_experts_per_lane();
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; ++i) {
            if (i >= epl) break;
            if (i * kWarpSize + lane_id() < expert_idx)
                num_blocks += ceil_div_u32(get_stored_expert_count(i), BLOCK_M);
        }
        return broadcast_lane0(warp_reduce_sum(num_blocks));
    }

    __device__ void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    __device__ void set_expert_idx(const uint32_t expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    __device__ uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    __device__ uint32_t get_current_num_m_blocks() const {
        return ceil_div_u32(current_num_tokens, BLOCK_M);
    }

    __device__ uint32_t get_valid_m() const {
        return constexpr_min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
    }

    __device__ bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            // Skip experts with 0 tokens (avoids div-by-zero and phantom pool blocks)
            if (num_m_blocks == 0) {
                advance_expert_idx();
                continue;
            }
            // (n,m) iteration order: n = block_idx / num_m_blocks, m = block_idx % num_m_blocks
            // This makes consecutive blocks share the same n (B matrix), improving L2 cache reuse.
            n_block_idx = block_idx / num_m_blocks;
            if (n_block_idx < kNumL1BlockNs)
                return true;
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    __device__ bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            // Skip experts with 0 tokens
            if (num_m_blocks == 0) {
                advance_expert_idx();
                continue;
            }
            // (n,m) iteration order: n = block_idx / num_m_blocks, m = block_idx % num_m_blocks
            n_block_idx = block_idx / num_m_blocks;
            if (n_block_idx < kNumL2BlockNs) {
                return true;
            }
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    __device__ void get_next_block(BlockPhase& phase, uint32_t& expert_idx,
                                    uint32_t& m_blk, uint32_t& n_blk) {
        while (true) {
            if (current_local_expert_idx >= get_num_experts_per_rank())
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    m_block_idx = block_idx - n_block_idx * get_current_num_m_blocks();
                    block_idx += kNumSMs;
                    phase = BlockPhase::Linear1;
                    expert_idx = current_local_expert_idx;
                    m_blk = m_block_idx;
                    n_blk = n_block_idx;
                    return;
                } else {
                    next_phase = BlockPhase::Linear2;
                    // Align down to wave boundary (single wave: align to 0)
                    set_expert_idx(0);
                }
            } else {
                if (fetch_next_l2_block()) {
                    m_block_idx = block_idx - n_block_idx * get_current_num_m_blocks();
                    block_idx += kNumSMs;
                    phase = BlockPhase::Linear2;
                    expert_idx = current_local_expert_idx;
                    m_blk = m_block_idx;
                    n_blk = n_block_idx;
                    return;
                } else {
                    next_phase = BlockPhase::Linear1;
                }
            }
        }
        phase = BlockPhase::None;
        expert_idx = 0;
        m_blk = 0;
        n_blk = 0;
    }

    __device__ void fetch_expert_recv_count() {
        const uint32_t epr = get_num_experts_per_rank();
        const uint32_t epl = get_num_experts_per_lane();
        const uint32_t nr = get_num_ranks();
        // Each lane caches experts at indices (i * 64 + lane_id)
        #pragma unroll
        for (uint32_t i = 0; i < kExpertsPerLane; ++i) {
            if (i >= epl) break;
            const auto expert_idx = i * kWarpSize + lane_id();
            uint64_t value = 0;
            if (expert_idx < epr) {
#if DG_HANG_DIAG
                const auto t0 = clock64();
                bool printed = false;
#endif
                do {
                    // recv_count_sum lives in the L2-cached local region now
                    // (rank-local only), so agent scope is sufficient -- the
                    // system-scope spin was paying IPC latency per probe.
                    value = __hip_atomic_load(
                        reinterpret_cast<volatile const uint64_t*>(
                            workspace.get_expert_recv_count_sum_ptr(expert_idx)),
                        __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#if DG_HANG_DIAG
                    if (!printed && clock64() - t0 > DG_HANG_TIMEOUT_CYCLES) {
                        DG_HANG_PRINTF_IF(lane_id() == 0, "MegaMoE spin timeout [fetch_recv]: expert=%u exp=%u got=%llu blk=%d\n",
expert_idx, nr,
(unsigned long long)(value >> 32), (int)blockIdx.x);
                        DG_HANG_MARK(4, expert_idx, nr,
                                     static_cast<uint32_t>(value >> 32));
                        printed = true;
                    }
#endif
                    __builtin_amdgcn_s_sleep(1);
                } while (static_cast<uint32_t>(value >> 32) != nr);
            }
            set_stored_expert_count(i, static_cast<uint32_t>(value));
        }
        syncwarp();
        // Print expert recv counts for debugging (only block 0, lane 0)
#if DG_MEGAMOE_DBUG
        if (lane_id() == 0 && blockIdx.x == 0) {
            for (uint32_t e = 0; e < get_num_experts_per_rank(); ++e) {
                uint32_t cnt = shfl_sync(get_stored_expert_count(0), e % kWarpSize);
                DG_DBUG_PRINTF("[DBUG] SCHED blk=%d expert[%u] recv_count=%u\n",
                       blockIdx.x, e, cnt);
            }
        }
#endif
    }

    template <typename Func>
    __device__ void for_each_block(Func&& func) {
        fetch_expert_recv_count();
        set_expert_idx(0);
        while (true) {
            BlockPhase phase;
            uint32_t expert_idx, m_blk, n_blk;
            get_next_block(phase, expert_idx, m_blk, n_blk);
            if (phase == BlockPhase::None)
                break;
            func(phase, expert_idx,
                 phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_blk, n_blk);
        }
    }

    // L1-only iteration: used by epilogue blocks before dispatch blocks join.
    template <typename Func>
    __device__ void for_each_l1_block(Func&& func) {
        fetch_expert_recv_count();
        set_expert_idx(0);
        next_phase = BlockPhase::Linear1;
        while (true) {
            // Only fetch L1 blocks — do NOT fall through to L2
            if (current_local_expert_idx >= get_num_experts_per_rank())
                break;
            if (!fetch_next_l1_block())
                break;
            m_block_idx = block_idx - n_block_idx * get_current_num_m_blocks();
            block_idx += kNumSMs;
            func(current_local_expert_idx, kNumL1BlockKs, m_block_idx, n_block_idx);
        }
    }

    // L2-only iteration: used by ALL blocks (epilogue + dispatch) after L1.
    // Reset block_idx to blockIdx.x so L2 work is distributed uniformly across
    // ALL blocks (epilogue blocks had block_idx advanced during L1; dispatch
    // blocks kept their original blockIdx.x). Without this reset, epilogue
    // blocks would skip L2 tiles because block_idx is too large.
    template <typename Func>
    __device__ void for_each_l2_block(Func&& func) {
        // L2 starts from expert 0 (L1 already done by epilogue blocks)
        block_idx = blockIdx.x;  // reset for uniform L2 distribution
        set_expert_idx(0);
        next_phase = BlockPhase::Linear2;
        while (true) {
            // Only fetch L2 blocks — do NOT fall through to L1
            if (current_local_expert_idx >= get_num_experts_per_rank())
                break;
            if (!fetch_next_l2_block())
                break;
            m_block_idx = block_idx - n_block_idx * get_current_num_m_blocks();
            block_idx += kNumSMs;
            func(current_local_expert_idx, kNumL2BlockKs, m_block_idx, n_block_idx);
        }
    }

#if 0
    // Readiness-aware L2 iteration: defer-and-revisit to eliminate
    // head-of-line blocking. Disabled: 8-card benchmarks showed no
    // measurable benefit over for_each_l2_block (pool blocks become
    // ready roughly in order, so the extra probe pass is pure
    // overhead). Kept for future follow-up.
    //
    // Two passes over the block's static kNumSMs-strided tile
    // assignment:
    //   pass 0 — probe each tile's pool block; run it if its L1 arrivals
    //            are complete, otherwise mark it deferred in a bitmap and
    //            skip it (no spinning);
    //   pass 1 — revisit only the deferred tiles, calling func directly so
    //            it blocks on the arrival mask exactly as the original
    //            scheduler did.
    // Ownership is unchanged — every tile is executed exactly once by its
    // statically assigned block; no cross-block claim atomics and no
    // rescan traffic. Readiness probes run on thread 0 and are broadcast
    // via shared memory: per-lane atomic loads could disagree across lanes
    // and diverge the block at the __syncthreads inside func.
    // Tiles beyond kMaxDeferred fall back to the original blocking wait.
    // Callback signature is identical to for_each_l2_block.
    template <typename Func>
    __device__ void for_each_ready_l2_block(Func&& func) {
        constexpr uint32_t kMaxDeferred = 512;
        __shared__ uint32_t s_deferred[kMaxDeferred / 32];
        __shared__ uint32_t s_ready;
        const uint64_t expected_mask = (kNumL1BlockNs >= 64)
            ? ~0ull
            : ((1ull << kNumL1BlockNs) - 1ull);
        const uint32_t epr = get_num_experts_per_rank();

        for (uint32_t i = threadIdx.x; i < kMaxDeferred / 32; i += blockDim.x)
            s_deferred[i] = 0;
        __syncthreads();

        for (uint32_t pass = 0; pass < 2; ++pass) {
            uint32_t tile_ord = 0;
            block_idx = blockIdx.x;
            set_expert_idx(0);
            next_phase = BlockPhase::Linear2;
            while (true) {
                if (current_local_expert_idx >= epr)
                    break;
                if (!fetch_next_l2_block())
                    break;
                const uint32_t my_ord = tile_ord++;
                m_block_idx = block_idx - n_block_idx * get_current_num_m_blocks();
                block_idx += kNumSMs;
                const bool in_bitmap = (my_ord < kMaxDeferred);

                if (pass == 1) {
                    // Second pass: only deferred tiles, blocking inside func.
                    if (!in_bitmap ||
                        !((s_deferred[my_ord >> 5] >> (my_ord & 31)) & 1u))
                        continue;
                    func(current_local_expert_idx, kNumL2BlockKs,
                         m_block_idx, n_block_idx);
                    continue;
                }

                const uint32_t pb = current_pool_block_offset + m_block_idx;
                if (threadIdx.x == 0) {
                    int32_t qd = __hip_atomic_load(
                        workspace.get_l2_quant_done_ptr(pb),
                        __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                    uint32_t r = (qd == 2) ? 1u : 0u;
                    if (r == 0) {
                        const uint64_t mask = __hip_atomic_load(
                            workspace.get_l2_arrival_mask_ptr(pb),
                            __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                        r = (mask == expected_mask) ? 1u : 0u;
                    }
                    s_ready = r;
                }
                __syncthreads();
                const uint32_t ready_flag = s_ready;
                __syncthreads();  // s_ready must be fully read before reuse

                if (ready_flag == 0 && in_bitmap) {
                    if (threadIdx.x == 0)
                        s_deferred[my_ord >> 5] |= 1u << (my_ord & 31);
                    continue;
                }
                // Ready, or overflow tile that cannot be deferred.
                func(current_local_expert_idx, kNumL2BlockKs,
                     m_block_idx, n_block_idx);
            }
            __syncthreads();  // bitmap writes visible before pass 1 reads
        }
    }
#endif
};

} // namespace deep_gemm::mega::hcu
