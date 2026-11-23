// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// Fused MegaMoE kernel for HCU gfx938 --UNIFIED SPLIT-BLOCK architecture.
// Single kernel covering both the LL (low-latency) and HT (high-throughput)
// paths; the GEMM family is selected by the instantiated block-M:
//   kBlockMTile <= 48  -> LL channelwise/groupwise GEMM (raw 3D weights)
//   kBlockMTile >= 64  -> HT gemm_ht_tile GEMM (Marlin 6D weights)
// HT uses high-throughput GEMM (gemm_ht_tile) with:
//   - BM=128, BN=256, BK=64 (larger M tile for better compute density)
//   - A matrix loaded via matrix_load_64x16_b8_t_lds (hardware LDS write)
//   - B matrix uses w6_offset indexing (Marlin permuted 6D weights)
//   - 8 warps with kNFragments=2 per warp
//   - Double-buffered A (LDS) and B (registers)
//   - Output written directly to GMEM (no SMEM staging)
//
// Block split by role:
//   - Blocks 0..kNumDispatchSMs-1:  DISPATCH blocks (all 16 warps do dispatch)
//   - Blocks kNumDispatchSMs..63:   EPILOGUE blocks (all 16 warps do GEMM/SiLU/scatter)
//
// Benefits over the original fused-block design:
//   1. Role-internal sync uses __syncthreads (hardware barrier) instead of
//      SubsetBarrier (software spin-loop in SMEM).
//   2. GEMM uses all 16 warps (kNumMathWarps=16) instead of 8, doubling
//      compute parallelism per epilogue block.
//   3. Cross-role sync uses grid_sync (atomic counter) instead of
//      sync_dispatch_epilogue (SubsetBarrier within a block).
//
// Sync structure:
//
// DISPATCH blocks (0..kNumDispatchSMs-1, all 16 warps):
//   1. __syncthreads                     --after count tokens per expert
//   2. __syncthreads                     --after stake out SM offsets
//   3. grid_sync(dispatch)               --after write src_token_topk_idx
//      __syncthreads                     --grid_sync is thread0-only; hold all
//                                          threads until the sync completes
//   4. __syncthreads                     --after broadcast recv_count
//   5. grid_sync(dispatch)               --before rank_barrier
//   6. rank_barrier                      --cross-rank
//   7. grid_sync(all)                    --cross-role: dispatch ready to pull
//      __syncthreads
//   8. pull loop
//   9. __syncthreads                     --after pull (all dispatch warps done)
//
// EPILOGUE blocks (kNumDispatchSMs..63, all 16 warps):
//   1. grid_sync(all)                    --cross-role: wait for dispatch broadcast
//   2. for_each_block:
//      Linear1: wait L1 arrival ->L1 GEMM (16 warps) ->SwiGLU+quant ->signal L2 arrival
//      Linear2: wait L2 arrival ->L2 GEMM (16 warps) ->epilogue scatter
//   3. __syncthreads                     --after for_each_block
//
// ALL blocks (0..63):
//   4. grid_sync(all)                    --all blocks finished
//   5. rank_barrier                      --cross-rank
//   6. cleanup workspace
//   7. rank_barrier                      --post-cleanup cross-rank
//   8. combine (all warps participate)

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include "params.h"
#include "workspace.cuh"
#include "buffer_layout.cuh"
#include "comm.cuh"
#include "warp_reduce.cuh"
#include "scheduler.cuh"
#include "gemm_device.cuh"
#include "gemm_device_ht.cuh"
#include "activation_device.cuh"

namespace deep_gemm::mega::hcu {

#if DG_MEGAMOE_DBUG
// Global debug counters for scatter tracking (max 128 tokens * 6 topk)
__device__ uint32_t g_scatter_total[128 * 6];
__device__ uint32_t g_scatter_zero_total[128 * 6];
#endif

// Grid sync counter indices for split-block architecture (shared with split kernel)
#ifndef DG_MEGA_SPLIT_GRID_SYNC_DEFINED
#define DG_MEGA_SPLIT_GRID_SYNC_DEFINED
constexpr uint32_t kSplitDispatchGridSyncIndex   = 0;  // dispatch blocks only
constexpr uint32_t kSplitCrossRoleGridSyncIndex  = 1;  // all blocks (cross-role)
constexpr uint32_t kSplitCombineGridSyncIndex    = 2;  // all blocks (pre-rank_barrier)

// Cross-rank barrier tags
constexpr uint32_t kSplitBeforePullBarrierTag    = 1;
constexpr uint32_t kSplitBeforeCombineBarrierTag = 2;
constexpr uint32_t kSplitAfterCleanupBarrierTag  = 3;
#endif

template <uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kBlockMTile = 128,
          uint32_t kBlockKTile = 64,
          uint32_t kNumSMs = 64,
          uint32_t kNumDispatchSMs = 8,
          uint32_t kNumTotalWarps = 16,
          uint32_t kNumMathWarps = 8,
          uint32_t kL1PipeStage = 2,
          uint32_t kL2PipeStage = 2,
          uint32_t kBlockNTile = 256,
          bool kIsInt8 = false,
          uint32_t kNumExpertsPerRank = 0,
          uint32_t kNumRanks = 0,
          bool kUseRingBuffer = false,
          uint32_t kL1BlockNTile = 0,
          uint32_t kNumMathWarpsL2Arg = 0,
          uint32_t kHTGroupSizeArg = 0,
          bool kL1TailAllWarps = false,
          uint32_t kL1WideTilesArg = 0>
__global__ __launch_bounds__(kNumTotalWarps * kWarpSize, 1)
void mega_moe_fp8_kernel_unified(
    hip_bfloat16* y,
    const char* l1_weights,
    const char* l2_weights,
    const float* l1_weights_scale,
    const float* l2_weights_scale,
    uint8_t* sym_buffer,                // IPC sym_buffer base (fine-grained, no L2 cache)
    uint8_t* local_buffer,              // Local buffer base (regular hipMalloc, with L2 cache)
    int** signal_buffers,
    int rank_idx,
    int num_experts_per_rank,
    int num_ranks,
    int num_tokens,
    int num_max_tokens_per_rank,
    float activation_clamp,
    bool fast_math,
    uint64_t* prof)
{
#if !defined(__gfx938__)
    // MegaMoE device code is gfx938-only. Under multi-arch offload builds
    // the other targets compile this empty stub instead of the HCU
    // builtins used below.
    return;
#else
    // =====================================================================
    // Constants & identification
    // =====================================================================
    constexpr uint32_t kNumThreads       = kNumTotalWarps * kWarpSize;  // 1024
    constexpr uint32_t kNumEpilogueSMs   = kNumSMs - kNumDispatchSMs;
    constexpr uint32_t kWarpSizeLocal    = kWarpSize;

    static_assert(kNumMathWarps <= kNumTotalWarps, "kNumMathWarps must be <= kNumTotalWarps");
    static_assert(kNumMathWarpsL2Arg == 0 || kNumMathWarpsL2Arg <= kNumTotalWarps,
                  "L2 math warps must not exceed the launched warp count");
    static_assert(kNumDispatchSMs < kNumSMs, "kNumDispatchSMs must be < kNumSMs");
    // Only the 1024-thread launch (16 warps) is capped at 192 VGPRs; the
    // 768-thread layouts (V4.1 W12-P / L2-W12X) have a 256-VGPR budget and
    // must keep their historical register layout.  Passed to the HT GEMMs.
    constexpr bool kRegOptTileKernel = (kNumTotalWarps >= 16);
    // The per-phase tile/math-warp divisibility checks live below, after
    // kL1BlockN / kNumMathWarpsL2 are derived (they may differ between L1
    // and L2 under W12-P).

    // Topology: 0 = runtime fallback. When specialized (e.g. 48x8) every use
    // below folds to a constant expression: dispatch div/mod becomes
    // mul-by-magic, per-lane loop bounds fold, dead iterations are removed.
    const uint32_t epr = (kNumExpertsPerRank != 0)
        ? kNumExpertsPerRank : static_cast<uint32_t>(num_experts_per_rank);
    const uint32_t nr  = (kNumRanks != 0)
        ? kNumRanks : static_cast<uint32_t>(num_ranks);
    const uint32_t ne  = epr * nr;   // num_experts
    const uint32_t epl = (epr + kWarpSize - 1) / kWarpSize;   // experts per lane
    const uint32_t rpl = (nr + kWarpSize - 1) / kWarpSize;    // ranks per lane

    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = thread_idx / kWarpSizeLocal;
    const uint32_t lane       = thread_idx % kWarpSizeLocal;

    // Phase profiling (runtime-gated by non-null prof): thread 0 of each block
    // records clock64() timestamps / accumulated cycles into prof[block*16+slot].
    // Slots:  0 t_init | dsp: 1 t_count 2 t_meta 3 t_bcast 4 t_pull_start
    //         5 t_pull_done 11 pull_tokens | epi: 1 t_crossrole 2 t_l1_done
    //         3 l1_wait_cyc 4 l1_gemm_cyc 5 l1_tiles |
    //         all: 6 l2_wait_cyc 7 l2_gemm_cyc 8 l2_fill_cyc 9 scatter_cyc
    //         10 l2_tiles 12 t_l2_done 13 t_barrier2 14 t_cleanup 15 t_end
#ifdef DG_MEGA_PROF
    constexpr uint32_t kProfSlots = 32;
#define DG_PROF_TS(slot)                                                        \
    do { if (prof != nullptr && thread_idx == 0)                                \
        prof[sm_idx * kProfSlots + (slot)] = clock64(); } while (0)
#define DG_PROF_ACC(acc)                                                        \
    do { if (prof != nullptr && thread_idx == 0)                                \
        acc += clock64() - prof_t0; } while (0)
#define DG_PROF_T0() const uint64_t prof_t0 = (prof != nullptr) ? clock64() : 0
#else
#define DG_PROF_TS(slot)  do {} while (0)
#define DG_PROF_ACC(acc)  do {} while (0)
#define DG_PROF_T0()
#endif

// Per-block progress markers (same convention as the LL split kernel):
// thread0 publishes a phase code per block; host reads it while spinning.
// Compiled in under DG_MEGA_HANG_PRINT or DG_MEGAMOE_DBUG.
#if DG_HANG_DIAG
#define DG_PHASE(code, detail)                                                  \
    do { if (thread_idx == 0) {                                                 \
        __hip_atomic_store(workspace.get_block_phase_ptr(sm_idx),               \
            static_cast<uint32_t>(code), __ATOMIC_RELAXED,                      \
            __HIP_MEMORY_SCOPE_AGENT);                                          \
        __hip_atomic_store(workspace.get_block_phase_detail_ptr(sm_idx),        \
            static_cast<uint32_t>(detail), __ATOMIC_RELAXED,                    \
            __HIP_MEMORY_SCOPE_AGENT);                                          \
    } } while (0)
#else
#define DG_PHASE(code, detail) do {} while (0)
#endif

    const bool is_dispatch_block = sm_idx < kNumDispatchSMs;
    const uint32_t dispatch_sm_idx = sm_idx;                          // 0..kNumDispatchSMs-1
    const uint32_t epilogue_sm_idx = sm_idx - kNumDispatchSMs;        // 0..kNumEpilogueSMs-1

    // In epilogue blocks, all warps participate in GEMM (math_warp_offset=0)
    const uint32_t math_warp_offset = 0;
    const uint32_t epilogue_warp_idx = warp_idx;  // 0..15

    // GEMM tile constants
    constexpr uint32_t BLOCK_M = kBlockMTile;
    constexpr uint32_t BLOCK_N = kBlockNTile;     // 256 or 512
    constexpr uint32_t BLOCK_K = kBlockKTile;     // 64 or 128
    constexpr uint32_t L1_SHAPE_N = 2 * kIntermediateHidden;
    constexpr uint32_t L1_SHAPE_K = kHidden;
    constexpr uint32_t L2_SHAPE_N = kHidden;
    constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
    // L1 tile width may differ from the L2 tile width: the W12-P L1 pass runs
    // 12 warps on a 768-column window (kNFragments = 768/(12*16) = 4, so every
    // wave keeps the proven acc[8][4] shape) while the L2 groupwise GEMM keeps
    // BLOCK_N.  kL1BlockNTile == 0 keeps both at BLOCK_N (production default).
    constexpr uint32_t kL1BlockN = (kL1BlockNTile != 0) ? kL1BlockNTile : kBlockNTile;
    // Math warps that run the trailing L1 block.  The default keeps the
    // historical 8 (a narrow tail leaves the other warps mirroring barriers
    // only); kL1TailAllWarps runs all kNumMathWarps with the non-uniform
    // fragment split in gemm_ht_tile_bm128_bk64_silu_quant, which removes the
    // 2x A-LDS-per-column cost that makes the tail slow under skew.
    constexpr uint32_t kL1TailWarps = kL1TailAllWarps ? kNumMathWarps : 8u;
    // L2 math warps: the phases may need different counts (W12-P runs all 12
    // warps in L1 and the proven 8 + 4 mirror warps in L2).
    constexpr uint32_t kNumMathWarpsL2 =
        (kNumMathWarpsL2Arg != 0) ? kNumMathWarpsL2Arg : kNumMathWarps;
    // L1 n-blocks are ceil'd so a 768-wide L1 pass can cover a 2*ih that it
    // does not divide (hy4: 4096 = 5x768 + 256).  The trailing block is
    // kL1TailN columns wide and runs on 8 math warps; kL1HasTail == false
    // keeps the old exact tiling (kNumL1BlockNs == L1_SHAPE_N / kL1BlockN and
    // kL1TailN == kL1BlockN).
    // Mixed L1 tiling (DG_MEGA_HT_L1W12_MIX): instead of 5x768 + one 256-wide
    // tail, cover 2*ih with 4x768 + 2x512 (hy4: 3072 + 1024 = 4096).  That
    // keeps the A-reread count at 6 (vs 8 for the all-512 baseline) while every
    // tile is full width: the narrow tiles are the exact standard BN512/8-warp
    // GEMM, so the 2x A-LDS-per-column and the 2-wave occupancy of the 256-wide
    // tail disappear.  4*768 = 3072 = 6*512 is a whole number of narrow tiles,
    // so each segment can keep its own tile_n * BLOCK_N column offset -- no
    // absolute-column parameter is needed.  Out cols 4*384 + 2*256 = 2048 = ih,
    // and groups 4*3 + 2*2 = 16 = ih/128.
    constexpr bool kL1Mixed = (kL1WideTilesArg != 0);
    constexpr uint32_t kNumWideL1Tiles = kL1WideTilesArg;
    constexpr uint32_t kNarrowL1BlockN = 512;
    constexpr uint32_t kNarrowL1TileBase =
        kNumWideL1Tiles * kL1BlockN / kNarrowL1BlockN;
    constexpr uint32_t kNumNarrowL1Tiles =
        (L1_SHAPE_N - kNumWideL1Tiles * kL1BlockN) / kNarrowL1BlockN;
    constexpr uint32_t kNumL1BlockNs = kL1Mixed
        ? (kNumWideL1Tiles + kNumNarrowL1Tiles)
        : ((L1_SHAPE_N + kL1BlockN - 1) / kL1BlockN);
    constexpr uint32_t kL1TailN = L1_SHAPE_N - (kNumL1BlockNs - 1) * kL1BlockN;
    constexpr bool kL1HasTail = !kL1Mixed && (kL1TailN != kL1BlockN);
    // Column tile index the epilogue uses for the tail block: it makes both
    // n_block_start (= kL1TailTileN * kL1TailN) and out_base
    // (= kL1TailTileN * kL1TailN / 2) land on the trailing columns.
    constexpr uint32_t kL1TailTileN =
        kL1HasTail ? ((kNumL1BlockNs - 1) * kL1BlockN / kL1TailN) : 0;
    // L2 n-blocks are ceil'd: a 768-wide L2 tile does not divide 5120
    // (6 x 768 + 512), so the last tile is shifted back to
    // L2_SHAPE_N - BLOCK_N and overlaps its predecessor.  Both tiles then
    // compute the shared columns from identical inputs, i.e. they write
    // identical bytes -- the duplicated scatter writes are benign.
    constexpr uint32_t kNumL2BlockNs =
        (L2_SHAPE_N + BLOCK_N - 1) / BLOCK_N;
    constexpr uint32_t kL2TailOverlap =
        kNumL2BlockNs * BLOCK_N - L2_SHAPE_N;  // 0 when it divides exactly

    // L2-W12X (BN768 + 12 math warps): exact coverage instead of the overlap
    // above.  Wide tiles ((n_blk < kNumL2WideTiles) x 768 cols) run all 12
    // warps.  If L2_SHAPE_N leaves a half-width remainder the last tile keeps
    // the BN512 / 8 math + 4 mirror layout (V4.1: 5120 = 6x768 + 512 -> six
    // wide tiles plus one 512-wide tail).  When it divides exactly there is no
    // tail at all (hy4 / minimax-m3.1: 6144 = 8x768) and kL2HasTail drops the
    // second GEMM/scatter body at compile time, so that shape carries a single
    // 12-wave body -- no duplicated live ranges to spill.
    // Same shape test as megamoe/patches/minimax_m31_support_20261104.patch.
    constexpr uint32_t kL2WideN = 768;
    constexpr bool kUseL2W12X =
        (BLOCK_N == kL2WideN) && (BLOCK_M == 128) && (BLOCK_K == 64) &&
        (kNumMathWarpsL2 == 12) && (kNumTotalWarps >= 12) &&
        (kL2TailOverlap == 0 ? (L2_SHAPE_N % kL2WideN == 0)
                             : (kL2TailOverlap == 256));
    constexpr uint32_t kNumL2WideTiles = kUseL2W12X ? (L2_SHAPE_N / kL2WideN) : 0;
    constexpr uint32_t kL2TailN =
        L2_SHAPE_N - kNumL2WideTiles * kL2WideN;  // 512 (V4.1) or 0 (exact)
    constexpr bool kL2HasTail = kUseL2W12X && (kL2TailN != 0);
    static_assert(!kUseL2W12X || kL2TailN == 0 || kL2TailN == 512,
                  "L2-W12X allows no tail tile or exactly one 512-wide tail");

    // Fused L1 (BM=128/BK=64): the L1 GEMM epilogue performs SwiGLU +
    // group-wise quantization and writes l2_token/l2_sf directly --no
    // per-block L1 scratch, no separate SwiGLU/quant pass. Works for both
    // FP8 and INT8 (int8 uses i8 MMA + int32 rescale in the groupwise L2).
    // Both tile shapes use the fused L1 epilogue (SwiGLU + group-wise quant
    // inside the GEMM) and the group-wise L2 GEMM -- no scratch roundtrip.
    // LL instantiations (BM<=48) use the channelwise/groupwise GEMM family
    // selected by kLLGemm; HT instantiations use gemm_ht_tile.
    constexpr bool kLLGemm = (BLOCK_M <= 48);
    static_assert(kLLGemm ||
                  (BLOCK_M % 16 == 0 && BLOCK_M >= 64 && BLOCK_K == 64) ||
                  (BLOCK_M == 64 && BLOCK_K == 128),
                  "unified HT requires BM%16==0>=64/BK=64 or BM=64/BK=128");
    static_assert(!kLLGemm || (BLOCK_N == 256 && BLOCK_K == 64),
                  "unified LL requires BN=256/BK=64");
    // HT group-quant granularity equals one fused-GEMM call's output width:
    // every L1 call produces exactly one scale group, and the L2 groupwise
    // GEMM consumes the same granularity.
    // Group-quant granularity: one fused L1 call's SwiGLU output width by
    // default.  Overridable because a 768-wide L1 tile yields 384 channels,
    // which does not divide ih = 2048 (hy4) -- there the caller pins 128 and
    // a single tile emits several groups (kGS/kWarpsPerGroup in the epilogue,
    // 768/2 = 3 x 128; the 256-wide tail emits one).
    constexpr uint32_t kHTGroupSize =
        (kHTGroupSizeArg != 0) ? kHTGroupSizeArg : (kL1BlockN / 2);
    // The L2 boundary-ratio table overlays smem_silu_result
    // (kIntermediateHidden floats); with kHTGroupSize = 128 this is exactly
    // BLOCK_M * ih/128 == ih for BM128, i.e. full -- guard the equality.
    static_assert(kLLGemm || (BLOCK_M > 128) ||
                  (BLOCK_M * (L2_SHAPE_K / kHTGroupSize) <= kIntermediateHidden),
                  "L2 boundary-ratio table must fit the silu_result SMEM region");
    constexpr uint32_t kL2Groups = L2_SHAPE_K / kHTGroupSize;
    static_assert(kL1BlockN % (kNumMathWarps * 16) == 0,
                  "L1 tile width must divide evenly by the L1 math-warp count * 16");
    static_assert(kBlockNTile % (kNumMathWarpsL2 * 16) == 0,
                  "L2 tile width must divide evenly by kNumMathWarpsL2 * 16");

    // Divisibility invariants: every tile/group count below is computed with
    // integer division and silently truncates if the shape does not tile
    // evenly -- fail at compile time instead.
    static_assert(!kL1HasTail || (kL1BlockN > kL1TailN),
                  "L1 tail must be narrower than the main tile");
    static_assert(!kL1HasTail || (kL1TailN % 16 == 0),
                  "L1 tail width must tile whole 16-column fragments");
    // The historical layout hands every one of its 8 warps an equal share; the
    // all-warp variant allows a non-uniform split instead.
    static_assert(!kL1HasTail || kL1TailAllWarps || (kL1TailN % (8 * 16) == 0),
                  "8-warp L1 tail must tile 8 math warps x 16 columns");
    // Every math warp that runs the tail must own at least one 16-column
    // fragment, otherwise the non-uniform split has nothing to give it.
    static_assert(!kL1HasTail || (kL1TailN / 16 >= kL1TailWarps),
                  "L1 tail must give every participating math warp a fragment");
    // kL1TailTileN is an integer division, so a tail that does not divide the
    // preceding columns would silently place the whole tail tile at the wrong
    // offset (2*ih = 4352: 5*768/512 truncates to 7 -> 3584 instead of 3840)
    // and quantize the wrong columns without any error.  Only exact placements
    // are allowed.
    static_assert(!kL1HasTail || ((kNumL1BlockNs - 1) * kL1BlockN) % kL1TailN == 0,
                  "L1 tail tile index must be exact (kL1TailTileN truncates otherwise)");
    // Mixed tiling must partition 2*ih exactly, with the wide prefix landing
    // on a whole narrow tile (otherwise the two segments' tile_n* BLOCK_N
    // column offsets would leave a hole or overlap).
    static_assert(!kL1Mixed ||
                  (kNumWideL1Tiles * kL1BlockN) % kNarrowL1BlockN == 0,
                  "mixed L1 tiling: wide prefix must be a whole number of narrow tiles");
    static_assert(!kL1Mixed ||
                  (kNumWideL1Tiles * kL1BlockN +
                   kNumNarrowL1Tiles * kNarrowL1BlockN) == L1_SHAPE_N,
                  "mixed L1 tiling must cover 2*ih exactly");
    static_assert(!kL1Mixed ||
                  (kNumWideL1Tiles * (kL1BlockN / 2) +
                   kNumNarrowL1Tiles * (kNarrowL1BlockN / 2)) == L2_SHAPE_K,
                  "mixed L1 tiling must emit exactly ih SwiGLU columns");
    static_assert(!kL1Mixed ||
                  ((kL1BlockN / 2) % kHTGroupSize == 0 &&
                   (kNarrowL1BlockN / 2) % kHTGroupSize == 0),
                  "mixed L1 tiling: both tile widths must emit whole quant groups");
    static_assert(L2_SHAPE_N >= BLOCK_N, "hidden must be at least one L2 tile wide");
    static_assert(kL2TailOverlap < BLOCK_N, "L2 tail overlap must stay below one tile");
    static_assert(L1_SHAPE_K % BLOCK_K == 0, "hidden must divide evenly by BLOCK_K");
    static_assert(L2_SHAPE_K % BLOCK_K == 0, "intermediate_hidden must divide evenly by BLOCK_K");
    static_assert(L2_SHAPE_K % kHTGroupSize == 0,
                  "intermediate_hidden must divide evenly by the L1 group size (BLOCK_N/2)");
    static_assert(kIntermediateHidden % kGroupQuantSize == 0,
                  "intermediate_hidden must divide evenly by kGroupQuantSize");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of the 16-row MMA tile");
    static_assert(BLOCK_N % kWarpSizeLocal == 0,
                  "BLOCK_N must cover a whole number of lanes in the l2_token scatter");
    static_assert(kHidden % 16 == 0, "hidden must tile the 16-byte pull vector");
    static_assert((kHidden * sizeof(hip_bfloat16) / 16) % kWarpSizeLocal == 0,
                  "combine: bf16 hidden must cover a whole number of int4 per lane");

    // Token topology
    // kNumTokensPerWarp must be <=kWarpSize / kNumTopk to avoid skipping tokens.
    // With BLOCK_M=128 and kNumTopk=6, BLOCK_M/kNumTopk=21 > 64/6=10, causing gaps.
    // Use kWarpSize/kNumTopk=10 as the dispatch token step (independent of GEMM BLOCK_M).
    // LL keeps its per-BM step (BLOCK_M/kNumTopk) to match its dispatch cadence.
    constexpr uint32_t kNumTokensPerWarp =
        kLLGemm ? BLOCK_M / kNumTopk : kWarpSize / kNumTopk;
    constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;

    // =====================================================================
    // Workspace & sym buffer (same pattern as original kernel)
    // =====================================================================
    uint8_t* workspace_base = get_workspace_base(sym_buffer, nr);
    using WorkspaceT = HCUMegaMoeWorkspace<kNumExpertsPerRank, kNumRanks>;
    __shared__ __align__(16) uint8_t smem_workspace_storage[sizeof(WorkspaceT)];
    WorkspaceT& workspace = *reinterpret_cast<WorkspaceT*>(smem_workspace_storage);
    if (thread_idx == 0) {
        new (&workspace) WorkspaceT(workspace_base, local_buffer, nr,
                                    ne, num_max_tokens_per_rank, kNumTopk,
                                    kNumSMs, kUseRingBuffer);
    }

    __shared__ __align__(16) uint8_t smem_slices_storage[sizeof(BufferSlices)];
    __shared__ __align__(16) uint8_t smem_sym_storage[sizeof(SymBuffer)];
    BufferSlices& slices = *reinterpret_cast<BufferSlices*>(smem_slices_storage);
    SymBuffer& sym = *reinterpret_cast<SymBuffer*>(smem_sym_storage);
    if (thread_idx == 0) {
        slices = slice_sym_buffers(workspace_base, local_buffer, nr,
                                    ne, num_max_tokens_per_rank, kNumTopk,
                                    kHidden, kIntermediateHidden, kNumSMs,
                                    kUseRingBuffer);
        sym.peer_buffer_ptrs = get_peer_buffer_ptrs(sym_buffer);
        sym.rank_idx = rank_idx;
        sym.num_ranks = nr;
    }
    // Scatter address fast path: precompute per-rank peer_base deltas into LDS
    // so the per-16B-row scatter needs one ds_read instead of two uncached IPC
    // dereferences.  The runtime-fallback <0,0> instantiation (used when e.g.
    // EPLB-redundant num_experts/rank does not match any compiled e48/e16 row)
    // has no compile-time kNumRanks, so size its table by the global
    // MEGA_MAX_RANKS (covers every runtime nr <= kMaxRanks) instead
    // of a 1-slot array -- otherwise a runtime dst_rank overflows LDS and reads
    // a garbage delta, producing an invalid remote address / VMFault.
    // Initialized BEFORE the barrier below so sym + deltas become visible with
    // ONE syncthreads.  dst_rank is per-lane => LDS gather, not readlane.
    constexpr uint32_t kPeerDeltaSlots =
        kNumRanks > 0 ? kNumRanks : MEGA_MAX_RANKS;
    __shared__ __align__(8) int64_t smem_peer_delta[kPeerDeltaSlots];
    if (thread_idx < static_cast<uint32_t>(nr)) {
        const auto* peer_ptrs = get_peer_buffer_ptrs(sym_buffer);
        smem_peer_delta[thread_idx] =
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(peer_ptrs[thread_idx])) -
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(peer_ptrs[rank_idx]));
    }
    __syncthreads();  // workspace/slices/sym init AND smem_peer_delta visible
    DG_PHASE(1, rank_idx);
    auto sym_map_dst = [&](void* local_ptr, uint32_t dst_rank) -> void* {
        return static_cast<uint8_t*>(local_ptr) + smem_peer_delta[dst_rank];
    };


#if DG_HANG_DIAG
    // P1 probe: snapshot the peer pointer tables exactly as this kernel sees
    // them.  If a sender's peer_buffer_ptrs[dst] read resolves stale/wrong,
    // every remote write of that (src,dst) pair lands at a dead address.
    if (blockIdx.x == 0 && thread_idx == 0) {
        dg_hang_launch_seq += 1;
        volatile uint64_t* pbp =
            reinterpret_cast<volatile uint64_t*>(dg_hang_records + 800);
        volatile uint64_t* psp =
            reinterpret_cast<volatile uint64_t*>(dg_hang_records + 816);
        for (uint32_t i = 0; i < nr; ++i) {
            pbp[i] = reinterpret_cast<uint64_t>(sym.peer_buffer_ptrs[i]);
            psp[i] = reinterpret_cast<uint64_t>(signal_buffers[i]);
        }
        dg_hang_records[832] = 0x500E0000u | static_cast<uint32_t>(rank_idx);
        dg_hang_records[834] = dg_hang_launch_seq;
    }
#endif

    // =====================================================================
    // Shared memory layout
    //   LL: Dynamic [expert_count][smem_l2_out]; Static [math_amax]
    //       [l2_scale_table]
    //   HT: Dynamic: [expert_count(num_experts)]
    //       Static: [a_lds_ht(stages*BM*BK)] [silu_result] [warp_amax]
    // =====================================================================
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // A matrix LDS buffer for HT GEMM: stages × BLOCK_M × BLOCK_K.
    // BM256 uses a dedicated 2-stage pipeline (32KB); BM<=192 keeps 3.
    // BM128 and BM64: 4 stages fit (32KB) and buy one extra tile of
    // A-load slack; BM144/160 and the 16-warp (HT_LB=1024) variants have
    // no SMEM headroom, keep 3. Stub for LL instantiations (no a_lds).
    constexpr uint32_t kAStagesHT =
        kLLGemm ? 1u
        : (BLOCK_M >= 256) ? 2u
        : (((BLOCK_M == 128 || BLOCK_M == 64) && kNumTotalWarps <= 12)
               ? 4u : 3u);
    __shared__ __align__(16) char a_lds_ht[kLLGemm ? 16 : kAStagesHT * BLOCK_M * BLOCK_K];

    // Per-warp expert counts for the scheduler live in registers now
    // (MegaMoEScheduler::stored_expert_counts); no SMEM table needed.

    // BM256 unified scratch (8KB): the dispatch-phase expert_count table
    // (dispatch blocks only) aliases the L1 row-amax atomic-max table
    // ([BLOCK_M] entries = 1KB, epilogue blocks only) -- the two uses are
    // on disjoint blocks, so they cannot race.
    __shared__ uint32_t smem_count_amax[(BLOCK_M >= 256) ? 2048 : 1];

    // SiLU intermediate result buffer (static SMEM, epilogue blocks only)
    // Fused L1: repurposed as the per-tile L2 boundary-ratio table
    // [BLOCK_M x kL2Groups] floats. Sized for the larger of the two uses
    // (BM=128: 128*24*4 = 12KB = 2*3072*2; BM>128 needs more).
    // BM256 reads boundary ratios straight from GMEM instead, so this is
    // only a stub there.
    constexpr uint32_t kSiluResultElems =
        kLLGemm ? 16u :
        (BLOCK_M >= 256) ? 16u :
        ((BLOCK_M * kL2Groups * 2 > 2 * kIntermediateHidden)
            ? BLOCK_M * kL2Groups * 2 : 2 * kIntermediateHidden);
    __shared__ __align__(16) hip_bfloat16 smem_silu_result[kSiluResultElems];

    // Per-expert token counters for the dispatch pull loop.  Live only
    // during the dispatch phase, so the table shares smem_silu_result
    // storage (dead until the L1 amax / L2 boundary-table phases) instead
    // of dynamic SMEM -- frees num_experts*4 bytes so NTW16 fits in 64KB.
    // BM256 uses smem_count_amax instead (sized for up to 2048 experts).
    // LL keeps the original dynamic-SMEM layout. Falls back to the dynamic
    // region if num_experts ever exceeds the static capacity (host mirrors
    // this predicate for smem_bytes).
    uint32_t* const smem_expert_count =
        kLLGemm ? reinterpret_cast<uint32_t*>(smem_buffer)
        : (BLOCK_M >= 256)
            ? ((ne <= 2048u)
                    ? smem_count_amax
                    : reinterpret_cast<uint32_t*>(smem_buffer))
            : ((ne * sizeof(uint32_t) <= sizeof(smem_silu_result))
                    ? reinterpret_cast<uint32_t*>(smem_silu_result)
                    : reinterpret_cast<uint32_t*>(smem_buffer));

    // LL-only dynamic scratch: L2 GEMM output staging [BLOCK_M, BLOCK_N] BF16.
    // Unused under HT instantiations.
    const uint32_t smem_expert_count_bytes = ne * sizeof(uint32_t);
    const uint32_t smem_expert_count_aligned = align_u32(smem_expert_count_bytes, 256);
    hip_bfloat16* const smem_l2_out = reinterpret_cast<hip_bfloat16*>(
        smem_buffer + smem_expert_count_aligned);

    // LL-only L2 boundary-ratio table ([BLOCK_M, kL2Groups] with stride +1 to
    // spread LDS banks) -- refilled per tile before the groupwise GEMM.
    // Stub under HT instantiations (HT reuses smem_silu_result instead).
    __shared__ float smem_l2_scale_table[
        kLLGemm ? kBlockMTile * (kIntermediateHidden / kGroupQuantSize + 1) : 1];

    // Row-amax partials for the fused L1 SwiGLU+quant epilogue (FP8 path).
    // For BLOCK_M > 128 this scratch is overlaid on smem_silu_result: the
    // boundary-ratio table is only written after the whole L1 phase (guarded
    // by the __syncthreads at the L1/L2 boundary), so the region is dead
    // while the L1 epilogues run. Saves kNumMathWarps*BLOCK_M*4 bytes.
    // BM256 uses smem_count_amax as a [BLOCK_M] atomic-max table (1KB).
    // amax lives on the (dead) a_lds region whenever it is overlaid --
    // either because NMATH>8 needs 8KB (NMATH16) or because BM128 runs
    // the 4-stage pipeline and the extra 4KB would overflow the 64KB
    // budget.  The device-side WAR barrier before the writes is gated by
    // the same predicate.
    // LL instantiations always use a dedicated storage table
    // (kNumMathWarps*BLOCK_M floats) as in the original split kernel.
    constexpr bool kAmaxOnALds =
        (!kLLGemm) && (BLOCK_M <= 128) && (kNumMathWarps > 8 || kAStagesHT > 3);
    __shared__ __align__(16) float smem_math_amax_storage[
        (!kLLGemm && (BLOCK_M > 128 || kAmaxOnALds)) ? 1 : kNumMathWarps * BLOCK_M];
    float* const smem_math_amax =
        kLLGemm ? smem_math_amax_storage
        : (BLOCK_M >= 256)
            ? reinterpret_cast<float*>(smem_count_amax)
            : ((BLOCK_M > 128)
                    ? reinterpret_cast<float*>(smem_silu_result)
                    : kAmaxOnALds
                            ? reinterpret_cast<float*>(a_lds_ht)
                            : smem_math_amax_storage);

    // Subset barrier state for HT GEMM (only kNumMathWarps warps participate)
    __shared__ uint32_t smem_bar_count;
    __shared__ uint32_t smem_bar_sense;

    // Debug: count scatter writes per (token, topk) slot
    // Only for small token counts to avoid flooding
#if DG_MEGAMOE_DBUG
    __shared__ uint32_t smem_scatter_counts[128 * 6];  // max 128 tokens * 6 topk
    __shared__ uint32_t smem_scatter_zeros[128 * 6];   // count zero scatter values
#endif

    // =====================================================================
    // Zero SMEM and workspace
    // =====================================================================
    for (uint32_t i = thread_idx; i < ne; i += kNumThreads)
        smem_expert_count[i] = 0;

    // Zero scatter debug counts
#if DG_MEGAMOE_DBUG
    for (uint32_t i = thread_idx; i < 128u * kNumTopk; i += kNumThreads) {
        smem_scatter_counts[i] = 0;
        smem_scatter_zeros[i] = 0;
    }
    if (blockIdx.x == 0) {
        for (uint32_t i = thread_idx; i < 128u * kNumTopk; i += kNumThreads) {
            g_scatter_total[i] = 0;
            g_scatter_zero_total[i] = 0;
        }
    }
#endif
    __syncthreads();
    DG_PROF_TS(0);  // t_init

    // =====================================================================
    // Scheduler --epilogue blocks only (kNumEpilogueSMs stride for L1)
    // =====================================================================
    using Scheduler = MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumEpilogueSMs,    // kNumSMs: epilogue block stride
        kNumExpertsPerRank, kNumRanks, kL1BlockN>;
    Scheduler scheduler(workspace, epr, nr);
    if (!is_dispatch_block) {
        scheduler.block_idx = epilogue_sm_idx;
    }

    // L2 scheduler: ALL blocks participate (kNumSMs stride)
    // Dispatch blocks join L2 after completing dispatch.
    using L2Scheduler = MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumSMs,            // kNumSMs: ALL blocks for L2
        kNumExpertsPerRank, kNumRanks, kL1BlockN>;
    L2Scheduler l2_scheduler(workspace, epr, nr);
    l2_scheduler.block_idx = sm_idx;  // all blocks use their global SM index

    // =====================================================================
    // DISPATCH BLOCKS: all 16 warps do dispatch work
    // =====================================================================
    if (is_dispatch_block) {
        DG_DBUG_PRINTF_IF(thread_idx == 0 && dispatch_sm_idx == 0,
                          "[DBUG] DSP-SPLIT r=%d entered dispatch, dsms=%u\n",
                          rank_idx, kNumDispatchSMs);

        auto read_topk_idx = [&](auto process) {
            #pragma unroll
            for (uint32_t i = (dispatch_sm_idx * kNumTotalWarps + warp_idx) * kNumTokensPerWarp;
                 i < static_cast<uint32_t>(num_tokens);
                 i += kNumDispatchSMs * kNumTotalWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane / kNumTopk) < static_cast<uint32_t>(num_tokens) && lane < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        *(slices.input_topk_idx_buffer.get_data_buffer<int64_t>(i) + lane));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane, expert_idx);
                }
                // __syncwarp();
            }
        };

        // #Count tokens per expert
        read_topk_idx([&](uint32_t token_topk_idx, int expert_idx) {
            __hip_atomic_fetch_add(smem_expert_count + expert_idx, 1u,
                                   __ATOMIC_RELAXED, __memory_scope_work_group);
        });
        // __syncwarp();
        __syncthreads();  // replaced sync_dispatch("D#1")
        DG_PROF_TS(1);  // dsp: t_count_done
        DG_PHASE(2, 0);

        // #Stake out per-expert SM offsets
        for (uint32_t i = thread_idx; i < ne; i += kNumThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                __hip_atomic_fetch_add(reinterpret_cast<volatile unsigned long long*>(
                    workspace.get_expert_send_count_ptr(i)), send_value,
                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        }
        __syncthreads();  // replaced sync_dispatch("D#2")

        // #Write source token-topk indices to remote ranks
        read_topk_idx([&](uint32_t token_topk_idx, int expert_idx) {
            const auto dst_rank_idx = expert_idx / epr;
            const auto dst_local_expert_idx = expert_idx % epr;
            const auto dst_slot = __hip_atomic_fetch_add(smem_expert_count + expert_idx, 1u,
                                   __ATOMIC_RELAXED, __memory_scope_work_group);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                dst_local_expert_idx, sym.rank_idx, dst_slot);
            *static_cast<uint32_t*>(sym.map(const_cast<uint32_t*>(dst_ptr), dst_rank_idx)) = token_topk_idx;
        });

        // #grid_sync (dispatch blocks only).
        // Every block drains its OWN remote writes BEFORE the arrival.  The
        // grid_sync below is agent-scope, so it does not order block 1..N-1's
        // system-scope stores; without the pre-arrival fence block 0 could
        // signal rank_barrier while a sibling's src_token_topk_idx pushes are
        // still in flight and a peer would read stale slots.  Cost is zero:
        // the fences were already concurrent, and now they overlap the
        // grid_sync wait instead of running after it.
        __syncthreads();
        __threadfence_system();
        grid_sync(workspace.template get_grid_sync_count_ptr<kSplitDispatchGridSyncIndex>(),
                  kNumDispatchSMs);
        // grid_sync only blocks thread0; barrier the whole block so non-zero
        // threads can't race ahead and read partially-staked send_count below.
        __syncthreads();
        DG_PROF_TS(2);  // dsp: t_meta_done
        DG_PHASE(3, 0);

        // (removed): recv counts are no longer PUSHED to peer ranks. Peers
        // pull them from this rank's expert_send_count after the before-pull
        // barrier (see below), so the count path carries no remote writes —
        // a dropped remote store/atomic can no longer hang fetch_recv.
        DG_PROF_TS(3);  // dsp: t_bcast_done
        DG_PHASE(4, 0);

        // The grid_sync above (end of the broadcast section) already fences
        // all dispatch blocks' send_count stakes + remote pushes.
        // -> sm0 rank_barrier -> grid_sync (dispatch) -> pull counts ->
        // grid_sync (all).
        // Make send_count (SYSTEM-scope atomics above) and the src_token_topk_idx
        // remote pushes visible at system scope before the barrier releases
        // peers to remote-read them.
        __threadfence_system();
        // One rank_barrier per rank: only dispatch block 0 signals/waits.
        // (All-blocks rank_barrier would pile kNumDispatchSMs arrivals onto
        // every peer slot -- 36x tag traffic and an all-blocks-of-all-ranks
        // barrier, much stronger than intended and a known hang amplifier.)
        if (dispatch_sm_idx == 0) {
            rank_barrier(signal_buffers, rank_idx, nr, kSplitBeforePullBarrierTag);
        }
        // Hold the other dispatch blocks until block0's cross-rank barrier
        // completes, then barrier the whole block so no thread reads
        // pre-barrier peer send_count state.
        grid_sync(workspace.template get_grid_sync_count_ptr<kSplitDispatchGridSyncIndex>(),
                  kNumDispatchSMs);
        __syncthreads();

        // Pull expert counts: remote-read each source rank's
        // expert_send_count slice for this rank (final after the barrier —
        // hi32 != 0 once the source rank's dispatch SMs started contributing;
        // completeness is guaranteed by the rank_barrier ordering, NOT by the
        // marker value, so peer ranks may launch different dispatch counts)
        // and locally rebuild this rank's recv_count rows + recv_sum. Entries
        // are re-read until the marker lands, so a transiently-stale remote
        // read heals instead of hanging fetch_recv; a permanently-zeroed
        // source surfaces as a loud spin timeout (DG_HANG_MARK site 9).
        for (uint32_t src = dispatch_sm_idx;
             src < nr; src += kNumDispatchSMs) {
            const auto* src_send = reinterpret_cast<volatile const uint64_t*>(
                sym.map(const_cast<uint64_t*>(workspace.get_expert_send_count_ptr(
                            static_cast<uint32_t>(sym.rank_idx) *
                            epr)),
                        static_cast<int>(src)));
            for (uint32_t e = thread_idx;
                 e < epr;
                 e += kNumThreads) {
                uint64_t ent;
#if DG_HANG_DIAG
                const auto t0 = clock64();
                bool printed = false;
#endif
                while (true) {
                    ent = __hip_atomic_load(src_send + e, __ATOMIC_ACQUIRE,
                                            __HIP_MEMORY_SCOPE_SYSTEM);
                    if (static_cast<uint32_t>(ent >> 32) != 0)
                        break;
#if DG_HANG_DIAG
                    if (!printed && clock64() - t0 > DG_HANG_TIMEOUT_CYCLES) {
                        DG_HANG_PRINTF("MegaMoE pull send_count timeout: "
                                       "rank=%d src=%u local_e=%u hi=%u\n",
                                       rank_idx, src, e,
                                       static_cast<uint32_t>(ent >> 32));
                        DG_HANG_MARK(9, static_cast<int>(e),
                                     static_cast<int>(src),
                                     static_cast<int>(ent >> 32));
                        printed = true;
                    }
#endif
                    __builtin_amdgcn_s_sleep(1);
                }
                __hip_atomic_store(
                    reinterpret_cast<volatile uint64_t*>(
                        workspace.get_expert_recv_count_ptr(src, e)),
                    ent & 0xffffffffull, __ATOMIC_RELEASE,
                    __HIP_MEMORY_SCOPE_AGENT);
                // Add one arrival marker per source rank (not the peer's
                // own hi32 contribution count) so fetch_expert_recv_count
                // can expect exactly num_ranks markers regardless of how
                // many dispatch SMs each peer launched with.
                // Rank-local now (L2-cached local region) -> agent scope, with
                // RELEASE so the recv_count store above is visible to the
                // ACQUIRE spin in fetch_expert_recv_count.
                __hip_atomic_fetch_add(
                    reinterpret_cast<unsigned long long*>(
                        workspace.get_expert_recv_count_sum_ptr(e)),
                    (1ull << 32) | (ent & 0xffffffffull),
                    __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        __syncthreads();

        // Cross-role sync: dispatch blocks signal "ready to pull" to epilogue blocks
        grid_sync(workspace.template get_grid_sync_count_ptr<kSplitCrossRoleGridSyncIndex>(), kNumSMs);
        __syncthreads();
        DG_PROF_TS(4);  // dsp: t_pull_start
        DG_PHASE(5, 0);

        // #Token / SF pull loop
        scheduler.fetch_expert_recv_count();

        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kMaxRanksPerLane] = {};
        uint32_t stored_rank_cum[kMaxRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumDispatchSMs * kNumTotalWarps;
#ifdef DG_MEGA_PROF
        uint32_t prof_pull_tokens = 0;
#endif
        for (uint32_t token_idx = dispatch_sm_idx * kNumTotalWarps + warp_idx; ;
             token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= static_cast<int>(epr))
                    break;
                expert_pool_block_offset += ceil_div_u32(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= static_cast<int>(epr))
                break;
#ifdef DG_MEGA_PROF
            prof_pull_tokens += (prof != nullptr && lane == 0) ? 1 : 0;
#endif

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kMaxRanksPerLane; ++i) {
                    if (i >= rpl) break;
                    const uint32_t j = i * kWarpSize + lane;
                    stored_rank_count[i] = j < nr ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
                // Rank-major order: precompute exclusive prefix sums of per-rank
                // token counts so each token's (rank, idx_in_rank) is an O(1)
                // ballot/clz lookup instead of the per-token round-robin scan.
                uint32_t carry = 0;
                #pragma unroll
                for (uint32_t i = 0; i < kMaxRanksPerLane; ++i) {
                    if (i >= rpl) { stored_rank_cum[i] = 0; break; }
                    const uint32_t c = stored_rank_count[i];
                    uint32_t incl = c;
                    #pragma unroll
                    for (uint32_t d = 1; d < kWarpSize; d <<= 1) {
                        const uint32_t v = shfl_sync(incl, lane >= d ? lane - d : lane);
                        if (lane >= d) incl += v;
                    }
                    stored_rank_cum[i] = carry + incl - c;
                    carry += shfl_sync(incl, kWarpSize - 1);
                }
            }

            // Rank-major pool order: token t in this expert comes from the rank
            // whose exclusive count prefix is the largest <= t.
            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t current_rank_in_expert_idx = 0;
            uint32_t token_idx_in_rank = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kMaxRanksPerLane; ++i) {
                if (i >= rpl) break;
                const bool pred = (i * kWarpSize + lane) < nr
                    && stored_rank_cum[i] <= token_idx_in_expert;
                const uint64_t mask = __ballot(pred);
                if (mask != 0) {
                    const uint32_t hi = kWarpSize - 1 - __builtin_clzll(mask);
                    current_rank_in_expert_idx = i * kWarpSize + hi;
                    token_idx_in_rank = token_idx_in_expert - shfl_sync(stored_rank_cum[i], hi);
                }
            }

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = expert_pool_block_offset + token_idx_in_expert / BLOCK_M;

            // Ring-buffer mapping (pull -> L1 GEMM staging). The L1 buffers
            // hold at most num_l1_pool_tokens slots; a pool block reuses the
            // physical slot of the block ring_blocks earlier once the L1
            // consumers released it.
            uint32_t ring_block_idx = pool_block_idx;
            uint32_t ring_gen = 0;
            uint32_t l1_token_idx = pool_token_idx;
            uint32_t ring_expected = 0;
            if constexpr (kUseRingBuffer) {
                const uint32_t ring_blocks = workspace.num_l1_pool_tokens / BLOCK_M;
                ring_block_idx = pool_block_idx % ring_blocks;
                ring_gen = pool_block_idx / ring_blocks;
                l1_token_idx = ring_block_idx * BLOCK_M +
                               (token_idx_in_expert % BLOCK_M);
                const uint32_t block_start_in_expert =
                    token_idx_in_expert - (token_idx_in_expert % BLOCK_M);
                ring_expected = constexpr_min(
                    BLOCK_M,
                    expert_end_idx - expert_start_idx - block_start_in_expert);
            }

            // P2P copy: nontemporal load/store to bypass L1 cache
            // Use vec<int32_t, 4> (GCC vector type) instead of int4 (HIP_vector_type)
            // because __builtin_nontemporal_load/store only supports integer/float/vector types.
            using vec4i = __attribute__((__vector_size__(16))) int32_t;
            const uint8_t* remote_token_ptr = static_cast<const uint8_t*>(
                sym.map(slices.input_token_buffer.get_data_buffer(src_token_idx), current_rank_in_expert_idx));
            uint8_t* local_l1_ptr = slices.l1_token_buffer.get_data_buffer<uint8_t>(l1_token_idx);
            constexpr uint32_t kNumInt4 = kHidden / 16;

            // SF remote pointer (topk weight is not staged into L1 —
            // combine reads input_topk_weights_buffer directly)
            const float* remote_sf_ptr = static_cast<const float*>(
                sym.map(slices.input_sf_buffer.get_data_buffer<float>(src_token_idx), current_rank_in_expert_idx));

            // Software-pipeline the remote reads: issue lane0's scalar loads
            // plus all of this lane's token-body loads back-to-back, then
            // drain with stores.  The rolled load->store loop serialized each
            // 16B chunk (and each scalar) on the full XGMI round-trip
            // (~13us/token); batching amortizes it to ~one round-trip.
            float sf_val = 0.0f;
            if (lane == 0) {
                sf_val = __builtin_nontemporal_load(remote_sf_ptr);
            }
            constexpr uint32_t kChunksPerLane =
                (kNumInt4 + kWarpSizeLocal - 1) / kWarpSizeLocal;  // 7
            vec4i tmp[kChunksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kChunksPerLane; ++i) {
                const uint32_t idx = lane + i * kWarpSizeLocal;
                const auto* src =
                    reinterpret_cast<const vec4i*>(remote_token_ptr) + idx;
                if (idx < kNumInt4)
                    tmp[i] = __builtin_nontemporal_load(src);
            }
            if constexpr (kUseRingBuffer) {
                // Generation 0 slots start free; later generations wait for
                // the gate published by the last L1 consumer of the previous
                // generation (which also resets the arrival counter). The
                // wait sits AFTER the remote loads so the atomic read
                // overlaps the XGMI round-trip; only the local stores below
                // must be ordered behind the gate.
                if (ring_gen != 0) {
                    if (lane == 0) {
                        const uint32_t* gate_ptr =
                            workspace.get_l1_gate_gen_ptr(ring_block_idx);
#if DG_HANG_DIAG
                        const auto t0 = clock64();
                        bool printed = false;
#endif
                        while (__hip_atomic_load(
                                   gate_ptr, __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_AGENT) < ring_gen) {
#if DG_HANG_DIAG
                            if (!printed && clock64() - t0 > DG_HANG_TIMEOUT_CYCLES) {
                                DG_HANG_PRINTF("MegaMoE spin timeout [RingGate]: gen>=%u got=%u blk=%d rb=%u\n",
                                       ring_gen, *gate_ptr, (int)blockIdx.x, ring_block_idx);
                                DG_HANG_MARK(5, ring_gen, *gate_ptr, ring_block_idx);
                                printed = true;
                            }
#endif
                            __builtin_amdgcn_s_sleep(1);
                        }
                    }
                    __syncwarp();
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < kChunksPerLane; ++i) {
                const uint32_t idx = lane + i * kWarpSizeLocal;
                auto* dst = reinterpret_cast<vec4i*>(local_l1_ptr) + idx;
                if (idx < kNumInt4)
                    __builtin_nontemporal_store(tmp[i], dst);
            }
            if (lane == 0) {
                __builtin_nontemporal_store(sf_val, slices.l1_sf_buffer.get_data_buffer<float>(l1_token_idx));
            }
            // __syncwarp();

            // Write token_src_metadata + l1_arrival_count (store-release, system scope)
            if (lane == 0) {
                auto* meta = workspace.get_token_src_metadata_ptr(pool_token_idx);
                meta->rank_idx = current_rank_in_expert_idx;
                meta->token_idx = src_token_idx;
                meta->topk_idx = src_topk_idx;
                if constexpr (kUseRingBuffer) {
                    // RELEASE add publishes this token's staging writes.  Only
                    // the last-arriving token of the generation additionally
                    // needs an acquire fence so its release-store of ready_gen
                    // transitively publishes every peer warp's slot data —
                    // keeping acquire off the per-token critical path lets the
                    // next iteration's remote loads issue without waiting.
                    const uint32_t prev = __hip_atomic_fetch_add(
                        workspace.get_l1_arrival_count_ptr(ring_block_idx), 1,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                    if (prev + 1 == ring_expected) {
                        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
                        __hip_atomic_store(
                            workspace.get_l1_ready_gen_ptr(ring_block_idx),
                            ring_gen + 1, __ATOMIC_RELEASE,
                            __HIP_MEMORY_SCOPE_AGENT);
                    }
                } else {
                    __hip_atomic_fetch_add(
                        workspace.get_l1_arrival_count_ptr(pool_block_idx), 1,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                }
            }
            // __syncwarp();
        }

        // __syncthreads();  // all dispatch warps done -- redundant, 635 syncs before L2 GEMM
        // Dispatch blocks fall through to L2 section
        DG_PROF_TS(5);  // dsp: t_pull_done
        DG_PHASE(6, 0);
#ifdef DG_MEGA_PROF
        if (prof != nullptr && thread_idx == 0)
            prof[sm_idx * kProfSlots + 11] = prof_pull_tokens;
#endif
    }
    // =====================================================================
    // EPILOGUE BLOCKS: wait for dispatch, then do L1 GEMM
    // =====================================================================
    if (!is_dispatch_block) {
        // Cross-role sync: wait for dispatch blocks to finish broadcast + rank_barrier
        grid_sync(workspace.template get_grid_sync_count_ptr<kSplitCrossRoleGridSyncIndex>(), kNumSMs);
        __syncthreads();  // grid_sync is thread0-only; hold all threads until done
        DG_PROF_TS(1);  // epi: t_crossrole_done
        DG_PHASE(10, 0);

        // Initialize subset barrier state for HT GEMM
        if (thread_idx == 0) {
            smem_bar_count = 0;
            smem_bar_sense = 0;
        }
        __syncthreads();

#ifdef DG_MEGA_PROF
        uint64_t prof_l1_wait = 0, prof_l1_gemm = 0;
        uint32_t prof_l1_tiles = 0;
#endif

        // L1 GEMM only (epilogue blocks). L2 will be done by ALL blocks.
        scheduler.for_each_l1_block([&](uint32_t expert_idx,
                                         uint32_t num_k_blocks,
                                         uint32_t m_blk, uint32_t n_blk) {
            const uint32_t valid_m = scheduler.get_valid_m();
            const uint32_t pool_block_idx = broadcast_lane0(
                scheduler.get_current_pool_block_offset() + m_blk);
            const uint32_t m_idx = broadcast_lane0(pool_block_idx * BLOCK_M);

            // Ring-buffer mapping: the L1 staging read uses the physical slot
            // (ring_block_idx * BLOCK_M); L2 writes and metadata keep the full
            // pool index (m_idx / pool_block_idx).
            uint32_t ring_block_idx = pool_block_idx;
            uint32_t ring_gen = 0;
            uint32_t l1_m_idx = m_idx;
            if constexpr (kUseRingBuffer) {
                const uint32_t ring_blocks =
                    workspace.num_l1_pool_tokens / BLOCK_M;
                ring_block_idx = broadcast_lane0(pool_block_idx % ring_blocks);
                ring_gen = broadcast_lane0(pool_block_idx / ring_blocks);
                l1_m_idx = ring_block_idx * BLOCK_M;
            }

            // ============ L1 GEMM ============
#ifdef DG_MEGA_PROF
            const uint64_t prof_t0 = (prof != nullptr) ? clock64() : 0;
#endif
            if (epilogue_warp_idx == 0 && lane == 0) {
                if constexpr (kUseRingBuffer) {
                    if (valid_m != 0) {
                        const uint32_t* ready_ptr =
                            workspace.get_l1_ready_gen_ptr(ring_block_idx);
                        const uint32_t expected_gen = ring_gen + 1;
#if DG_HANG_DIAG
                        const auto t0 = clock64();
                        bool printed = false;
#endif
                        while (__hip_atomic_load(
                                   ready_ptr, __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_AGENT) < expected_gen) {
#if DG_HANG_DIAG
                            if (!printed && clock64() - t0 > DG_HANG_TIMEOUT_CYCLES) {
                                DG_HANG_PRINTF("MegaMoE spin timeout [L1wait-ring]: gen>=%u got=%u blk=%d rb=%u\n",
                                       expected_gen, *ready_ptr, (int)blockIdx.x, ring_block_idx);
                                DG_HANG_MARK(5, expected_gen, *ready_ptr, ring_block_idx);
                                printed = true;
                            }
#endif
                            __builtin_amdgcn_s_sleep(1);
                        }
                    }
                } else {
                    volatile const uint32_t* vptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const uint32_t expected = valid_m;
#if DG_HANG_DIAG
                    const auto t0 = clock64();
                    bool printed = false;
#endif
                    while (__hip_atomic_load(vptr, __ATOMIC_ACQUIRE,
                                              __HIP_MEMORY_SCOPE_AGENT) != expected) {
#if DG_HANG_DIAG
                        if (!printed && clock64() - t0 > DG_HANG_TIMEOUT_CYCLES) {
                            DG_HANG_PRINTF("MegaMoE spin timeout [L1wait]: exp=%u got=%u blk=%d pb=%u\n",
                                   expected, *vptr, (int)blockIdx.x, pool_block_idx);
                            DG_HANG_MARK(5, expected, *vptr, pool_block_idx);
                            printed = true;
                        }
#endif
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
            DG_PROF_ACC(prof_l1_wait);
            DG_PHASE(11, pool_block_idx);

            if (valid_m == 0) {
                if constexpr (kUseRingBuffer) {
                    // Nothing was produced for this slot generation; still
                    // release it so the next generation's producer can pass.
                    if (epilogue_warp_idx == 0 && lane == 0) {
                        const uint32_t prev = __hip_atomic_fetch_add(
                            workspace.get_l1_cons_units_ptr(ring_block_idx), 1,
                            __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                        if (prev + 1 == (ring_gen + 1) * kNumL1BlockNs) {
                            __hip_atomic_store(
                                workspace.get_l1_arrival_count_ptr(ring_block_idx),
                                0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                            __hip_atomic_store(
                                workspace.get_l1_gate_gen_ptr(ring_block_idx),
                                ring_gen + 1, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_AGENT);
                        }
                    }
                }
                __syncthreads();
                return;
            }
#ifdef DG_MEGA_PROF
            const uint64_t prof_t1 = (prof != nullptr) ? clock64() : 0;
#endif

            // L1 GEMM setup. The fused epilogue performs SwiGLU + group-wise
            // quant and writes l2_token/l2_sf directly -- no per-block L1
            // scratch buffer.
            const char* matrixA = reinterpret_cast<const char*>(
                slices.l1_token_buffer.get_data_buffer(l1_m_idx));
            const float* matrixA_scale = slices.l1_sf_buffer.get_data_buffer<float>(l1_m_idx);

            if constexpr (kLLGemm) {
                // ---- LL GEMM: channelwise tile over raw 3D weights ----
                constexpr uint32_t kLdElementsPerBlockL1 =
                    (128 / 8) * kWarpSizeLocal * kNumMathWarps *
                    (BLOCK_N / (kNumMathWarps * 16));
                const char* matrixB = reinterpret_cast<const char*>(
                    l1_weights + expert_idx * L1_SHAPE_N * L1_SHAPE_K
                    + n_blk * kLdElementsPerBlockL1);
                const float* matrixB_scale = l1_weights_scale + expert_idx * L1_SHAPE_N
                    + n_blk * BLOCK_N;

                // L1PS >= 8 encodes the warp-K variant (masked-GEMM style
                // warp_k=2 split); the inner pipeline depth is L1PS - 8.
                constexpr bool kL1WarpK = (kL1PipeStage >= 8);
                constexpr int kL1PipeEff = kL1WarpK ? (kL1PipeStage - 8) : kL1PipeStage;
                if (epilogue_warp_idx < kNumMathWarps) {
                    // Fused: GEMM epilogue does SwiGLU + group-wise quant (FP8/INT8)
                    // and writes l2_token/l2_sf directly from accumulators — no SMEM
                    // or global roundtrip. Contains exactly one s_barrier (mirrored
                    // below by non-math warps); the warp-K variant contains four.
                    if constexpr (kL1WarpK) {
                        gemm_channelwise_tile_silu_quant_wk<L1_SHAPE_N, L1_SHAPE_K,
                                               BLOCK_M, BLOCK_N, BLOCK_K,
                                               16, kNumMathWarps, kL1PipeEff, 2, kIsInt8>(
                            matrixA, matrixB, matrixA_scale, matrixB_scale,
                            0, valid_m,
                            scheduler.get_num_tokens(expert_idx),
                            math_warp_offset,
                            slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                            slices.l2_sf_buffer.get_data_buffer<float>(0),
                            m_idx, n_blk,
                            kIntermediateHidden / kGroupQuantSize,
                            kIntermediateHidden,
                            smem_math_amax,
                            reinterpret_cast<float*>(smem_l2_out),
                            activation_clamp, fast_math);
                    } else {
                        gemm_channelwise_tile_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                               BLOCK_M, BLOCK_N, BLOCK_K,
                                               16, kNumMathWarps, kL1PipeStage, kIsInt8>(
                            matrixA, matrixB, matrixA_scale, matrixB_scale,
                            0, valid_m,
                            scheduler.get_num_tokens(expert_idx),
                            math_warp_offset,
                            slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                            slices.l2_sf_buffer.get_data_buffer<float>(0),
                            m_idx, n_blk,
                            kIntermediateHidden / kGroupQuantSize,
                            kIntermediateHidden,
                            smem_math_amax,
                            activation_clamp, fast_math,
                            reinterpret_cast<hip_bfloat16*>(
                                slices.l2_output_buffer.get_data_buffer(0)));
                    }
                } else {
                    // Mirror the fused GEMM's s_barrier; the warp-K variant has
                    // kL1WkBarriers total (2*(N_REPEAT/2) reduction + 1 amax).
                    asm volatile("s_barrier\n\t" ::: "memory");
                    if constexpr (kL1WarpK) {
                        constexpr int kL1WkBarriers =
                            BLOCK_N / ((kNumMathWarps / 2) * 16);
                        #pragma unroll
                        for (int b = 1; b < kL1WkBarriers; ++b) {
                            asm volatile("s_barrier\n\t" ::: "memory");
                        }
                    }
                }
            } else {
                // ---- HT GEMM: gemm_ht_tile over Marlin-permuted weights ----
                const int expert_b_base_l1 = broadcast_lane0(
                    expert_idx * L1_SHAPE_N * L1_SHAPE_K);
                const float* matrixB_scale = l1_weights_scale +
                    broadcast_lane0(expert_idx * L1_SHAPE_N);

#if DG_MEGAMOE_DBUG
                if (epilogue_warp_idx == 0 && lane == 0 && sm_idx == kNumDispatchSMs) {
                    DG_DBUG_PRINTF("L1HT blk=%d exp=%u m_idx=%u n_blk=%u valid_m=%u A=%p Bbase=%d M=%u\n",
                        (int)blockIdx.x, expert_idx, m_idx, n_blk, valid_m,
                        (void*)matrixA, expert_b_base_l1,
                        scheduler.get_num_tokens(expert_idx));
                }
#endif

                // L1 tile role: the trailing block (only when 2*ih does not
                // divide by the L1 tile width) is kL1TailN wide and runs on 8
                // math warps, the other kNumMathWarps-8 warps mirror its
                // s_barriers.  Both variants carry the same barrier count
                // (kL1GemmStageBars + kEpiRounds + kAmaxOnALds, where
                // kAmaxOnALds is true for both via kAStagesHT == 4), so the
                // mirror loop below stays untouched.
                const bool l1_tail = kL1HasTail && (n_blk == kNumL1BlockNs - 1);
                // Mixed tiling runs the narrow 512-wide tiles with the standard
                // 8-warp GEMM; the remaining warps only mirror its barriers.
                const uint32_t l1_math_warps = kL1Mixed
                    ? ((n_blk < kNumWideL1Tiles) ? kNumMathWarps : 8u)
                    : (l1_tail ? kL1TailWarps : kNumMathWarps);
                if (epilogue_warp_idx < l1_math_warps) {
                    constexpr int kL1Groups = kIntermediateHidden / kHTGroupSize;
                    if constexpr (BLOCK_M == 256) {
                        // Dedicated BM256 path: 8 math warps, each covering
                        // all 256 rows x 32 cols (acc[16][2]).
                        gemm_ht_tile_bm256_bk64_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                 BLOCK_N, BLOCK_K, kIsInt8>(
                            matrixA, l1_weights, expert_b_base_l1,
                            matrixA_scale, matrixB_scale,
                            0, n_blk, valid_m,
                            scheduler.get_num_tokens(expert_idx),
                            math_warp_offset,
                            slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                            slices.l2_sf_buffer.get_data_buffer<float>(0),
                            m_idx, kL1Groups, kIntermediateHidden,
                            smem_math_amax, a_lds_ht,
                            &smem_bar_count, &smem_bar_sense,
                            activation_clamp, fast_math);
                    } else if constexpr (BLOCK_M == 64 && BLOCK_K == 128) {
                        // BM64/BK128 fused path: same SwiGLU + group-wise quant
                        // epilogue; 4 math warps each produce 32 of the call's 128
                        // output columns (one quant group per call).
                        gemm_ht_tile_bm64_bk128_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                 BLOCK_M, BLOCK_N, BLOCK_K,
                                 kNumMathWarps, kIsInt8, true, kAStagesHT>(
                            matrixA, l1_weights, expert_b_base_l1,
                            matrixA_scale, matrixB_scale,
                            0, n_blk, valid_m,
                            scheduler.get_num_tokens(expert_idx),
                            math_warp_offset,
                            slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                            slices.l2_sf_buffer.get_data_buffer<float>(0),
                            m_idx, kL1Groups, kIntermediateHidden,
                            smem_math_amax, a_lds_ht,
                            &smem_bar_count, &smem_bar_sense,
                            activation_clamp, fast_math);
                    } else {
                        // BK64 family: BM128 (and BM64-V2, 4 M-subtiles).
                        if constexpr (kL1Mixed) {
                            if (n_blk < kNumWideL1Tiles) {
                                // Wide 768-column tile on all 12 math warps;
                                // tile_n = n_blk gives columns n_blk*768.
                                gemm_ht_tile_bm128_bk64_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                         BLOCK_M, kL1BlockN, BLOCK_K,
                                         kNumMathWarps, kIsInt8, true, kAStagesHT, kHTGroupSize, false, kRegOptTileKernel>(
                                    matrixA, l1_weights, expert_b_base_l1,
                                    matrixA_scale, matrixB_scale,
                                    0, n_blk, valid_m,
                                    scheduler.get_num_tokens(expert_idx),
                                    math_warp_offset,
                                    slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                                    slices.l2_sf_buffer.get_data_buffer<float>(0),
                                    m_idx, kL1Groups, kIntermediateHidden,
                                    smem_math_amax, a_lds_ht,
                                    &smem_bar_count, &smem_bar_sense,
                                    activation_clamp, fast_math);
                            } else {
                                // Narrow 512-column tile on 8 math warps.  The
                                // wide prefix is exactly kNarrowL1TileBase
                                // narrow tiles, so this tile_n lands on the
                                // trailing columns without an absolute offset.
                                gemm_ht_tile_bm128_bk64_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                         BLOCK_M, kNarrowL1BlockN, BLOCK_K,
                                         8, kIsInt8, true, kAStagesHT, kHTGroupSize, true, kRegOptTileKernel>(
                                    matrixA, l1_weights, expert_b_base_l1,
                                    matrixA_scale, matrixB_scale,
                                    0, n_blk, valid_m,
                                    scheduler.get_num_tokens(expert_idx),
                                    math_warp_offset,
                                    slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                                    slices.l2_sf_buffer.get_data_buffer<float>(0),
                                    m_idx, kL1Groups, kIntermediateHidden,
                                    smem_math_amax, a_lds_ht,
                                    &smem_bar_count, &smem_bar_sense,
                                    activation_clamp, fast_math,
                                    static_cast<int32_t>(kNumWideL1Tiles * kL1BlockN +
                                        (n_blk - kNumWideL1Tiles) * kNarrowL1BlockN));
                            }
                        } else if constexpr (kL1HasTail) {
                            if (l1_tail) {
                                // Trailing kL1TailN-wide block on kL1TailWarps
                                // maths warps.  kL1TailTileN keeps both the B
                                // column offset (tile_n * kL1TailN) and the
                                // epilogue's out_base on the trailing columns.
                                gemm_ht_tile_bm128_bk64_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                         BLOCK_M, kL1TailN, BLOCK_K,
                                         (kL1TailAllWarps
                                              ? static_cast<int>(kNumMathWarps)
                                              : 8),
                                         kIsInt8, true, kAStagesHT, kHTGroupSize, false, kRegOptTileKernel>(
                                    matrixA, l1_weights, expert_b_base_l1,
                                    matrixA_scale, matrixB_scale,
                                    0, kL1TailTileN, valid_m,
                                    scheduler.get_num_tokens(expert_idx),
                                    math_warp_offset,
                                    slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                                    slices.l2_sf_buffer.get_data_buffer<float>(0),
                                    m_idx, kL1Groups, kIntermediateHidden,
                                    smem_math_amax, a_lds_ht,
                                    &smem_bar_count, &smem_bar_sense,
                                    activation_clamp, fast_math);
                            } else {
                                gemm_ht_tile_bm128_bk64_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                         BLOCK_M, kL1BlockN, BLOCK_K,
                                         kNumMathWarps, kIsInt8, true, kAStagesHT, kHTGroupSize, false, kRegOptTileKernel>(
                                    matrixA, l1_weights, expert_b_base_l1,
                                    matrixA_scale, matrixB_scale,
                                    0, n_blk, valid_m,
                                    scheduler.get_num_tokens(expert_idx),
                                    math_warp_offset,
                                    slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                                    slices.l2_sf_buffer.get_data_buffer<float>(0),
                                    m_idx, kL1Groups, kIntermediateHidden,
                                    smem_math_amax, a_lds_ht,
                                    &smem_bar_count, &smem_bar_sense,
                                    activation_clamp, fast_math);
                            }
                        } else {
                            // No L1 tail (kL1HasTail == false, the production
                            // shape): instantiate the exact pre-L1W12
                            // specialization -- omitting kGroupSizeArg keeps
                            // the same template arguments the baseline used
                            // before the 12-wave work, so its codegen is not
                            // perturbed by an extra parameter (kGS then falls
                            // back to BLOCK_N/2 == kHTGroupSize anyway).
                            gemm_ht_tile_bm128_bk64_silu_quant<L1_SHAPE_N, L1_SHAPE_K,
                                     BLOCK_M, kL1BlockN, BLOCK_K,
                                     kNumMathWarps, kIsInt8, true, kAStagesHT, kHTGroupSize,
                                         false, kRegOptTileKernel>(
                                matrixA, l1_weights, expert_b_base_l1,
                                matrixA_scale, matrixB_scale,
                                0, n_blk, valid_m,
                                scheduler.get_num_tokens(expert_idx),
                                math_warp_offset,
                                slices.l2_token_buffer.get_data_buffer<uint8_t>(0),
                                slices.l2_sf_buffer.get_data_buffer<float>(0),
                                m_idx, kL1Groups, kIntermediateHidden,
                                smem_math_amax, a_lds_ht,
                                &smem_bar_count, &smem_bar_sense,
                                activation_clamp, fast_math);
                        }
                    }
#ifdef DG_MEGA_W12P_DBG
                    // Compare the L1 intermediate between the baseline and
                    // W12-P runs: first 8 fp8 bytes + this tile's group scale.
                    if (sm_idx == 0 && m_idx == 0 &&
                        epilogue_warp_idx == 0 && lane == 0) {
                        const uint8_t* lt =
                            slices.l2_token_buffer.get_data_buffer<uint8_t>(0);
                        const float* sfb =
                            slices.l2_sf_buffer.get_data_buffer<float>(0);
                        const int c0 = n_blk * (kL1BlockN / 2);
                        printf("[W12PDBG] exp=%u n_blk=%u nb=%u g=%u valid_m=%u "
                               "col%u..%u=%d,%d,%d,%d,%d,%d,%d,%d sf=%.7g | "
                               "sfrow=%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g\n",
                               expert_idx, n_blk, kNumL1BlockNs, kL2Groups,
                               valid_m, c0, c0 + 7,
                               (int)lt[c0 + 0], (int)lt[c0 + 1],
                               (int)lt[c0 + 2], (int)lt[c0 + 3],
                               (int)lt[c0 + 4], (int)lt[c0 + 5],
                               (int)lt[c0 + 6], (int)lt[c0 + 7],
                               sfb[n_blk],
                               sfb[0], sfb[1], sfb[2], sfb[3], sfb[4],
                               sfb[5], sfb[6], sfb[7], sfb[8]);
                    }
#endif
                } else {
                    // The GEMM uses the all-wave hardware s_barrier; non-math
                    // warps must execute the same number of barrier calls.
                    // K / BLOCK_K barriers per call (one per half-iteration,
                    // epilogue drains included).  The fused variant adds one
                    // more (row-amax reduction).
                    constexpr int kL1GemmStageBars = L1_SHAPE_K / BLOCK_K;
                    // The fused epilogue runs its amax->quant sequence in
                    // two row-half rounds when BLOCK_M/16 > 4 (BM128), so it
                    // executes one row-amax barrier per round.  A further +1
                    // when the amax table overlays a_lds (the epilogue first
                    // drains pending LDS stage reads with an extra barrier).
                    constexpr int kEpiRounds = (BLOCK_M / 16 > 4) ? 2 : 1;
                    constexpr int kL1BarriersPerCall =
                        kL1GemmStageBars + kEpiRounds + (kAmaxOnALds ? 1 : 0);
                    for (int i = 0; i < kL1BarriersPerCall; ++i)
                        asm volatile("s_barrier\n\t" ::: "memory");
                }
            }  // kLLGemm L1 family

            // Signal L1 arrival (store-release, agent scope).
            // s_barrier only synchronizes wave arrival -- it does NOT drain
            // outstanding VMEM -- so the per-thread fence (each warp flushes
            // its own l2_token/l2_sf stores) must come BEFORE the barrier,
            // otherwise the L2 consumer on another SM can pass the mask wait
            // while the last tiles' stores are still in flight and read a
            // half-written intermediate (observed as a scale-group-0-only
            // result: deterministic at 4096/8192, invisible when debug prints
            // slowed the producer down).
            __threadfence();
            __syncthreads();
#ifdef DG_MEGA_PROF
            if (prof != nullptr && thread_idx == 0) {
                prof_l1_gemm += clock64() - prof_t1;
                ++prof_l1_tiles;
            }
#endif
            if (epilogue_warp_idx == 0 && lane == 0) {
                __hip_atomic_fetch_or(
                    reinterpret_cast<uint64_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                    (1ull << n_blk), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                if constexpr (kUseRingBuffer) {
                    // All kNumL1BlockNs consumers of this slot generation are
                    // now done: the last one resets the arrival counter and
                    // publishes the gate so the producer may write gen+1.
                    const uint32_t prev = __hip_atomic_fetch_add(
                        workspace.get_l1_cons_units_ptr(ring_block_idx), 1,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                    if (prev + 1 == (ring_gen + 1) * kNumL1BlockNs) {
                        __hip_atomic_store(
                            workspace.get_l1_arrival_count_ptr(ring_block_idx),
                            0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                        __hip_atomic_store(
                            workspace.get_l1_gate_gen_ptr(ring_block_idx),
                            ring_gen + 1, __ATOMIC_RELEASE,
                            __HIP_MEMORY_SCOPE_AGENT);
                    }
                }
            }
        });
        __syncthreads();  // L1 done for this epilogue block
        DG_PROF_TS(2);  // epi: t_l1_done
        DG_PHASE(12, 0);
#ifdef DG_MEGA_PROF
        if (prof != nullptr && thread_idx == 0) {
            prof[sm_idx * kProfSlots + 3] = prof_l1_wait;
            prof[sm_idx * kProfSlots + 4] = prof_l1_gemm;
            prof[sm_idx * kProfSlots + 5] = prof_l1_tiles;
        }
#endif
    }

    // =====================================================================
    // ALL BLOCKS: L2 GEMM (epilogue + dispatch blocks)
    // Epilogue blocks finished L1, dispatch blocks finished dispatch.
    // Both participate in L2 using kNumSMs stride.
    // =====================================================================
    {
        // Dispatch blocks need to fetch expert recv counts for L2 scheduling.
        // Epilogue blocks already fetched them via for_each_l1_block on
        // `scheduler` — copy its per-lane register cache (no memory traffic).
        if (is_dispatch_block) {
            l2_scheduler.fetch_expert_recv_count();
        } else {
            l2_scheduler.copy_expert_counts_from(scheduler);
        }

        // Re-initialize subset barrier state for L2 GEMM
        if (thread_idx == 0) {
            smem_bar_count = 0;
            smem_bar_sense = 0;
        }
        __syncthreads();

#ifdef DG_MEGA_PROF
        uint64_t prof_l2_wait = 0, prof_l2_gemm = 0, prof_l2_fill = 0, prof_scatter = 0;
        uint32_t prof_l2_tiles = 0;
#endif

        // Scatter rows of one [BLOCK_M x BLOCK_N] tile slice from the
        // l2_output scratch to the remote combine buffer.  All epilogue warps
        // drive the loop after the GEMM (sequential, as before).
        // The chunk grid is flat ((row, chunk) pairs handed out lane-major) so
        // any tile width works: 32 lanes/row for BN256, 64 for BN512, 96 for
        // BN768.  dst_col_base is the tile's ABSOLUTE column in the output
        // row, which keeps the destination index correct when the last tile is
        // shifted back to overlap its predecessor.
        // BM256 writes a compact [256 x BLOCK_N] tile (row stride BLOCK_N,
        // no per-n_blk column offset); other configs keep the
        // [BLOCK_M x L2_SHAPE_N] strided layout.
        constexpr uint32_t kL2ScatterSrcStride = (BLOCK_M == 256) ? BLOCK_N : L2_SHAPE_N;
        auto l2_scatter_rows = [&](auto chunks_per_row_tag,
                                   const hip_bfloat16* scratch,
                                   const uint32_t m_base,
                                   const uint32_t vm,
                                   const uint32_t src_row_stride,
                                   const uint32_t src_col_off,
                                   const uint32_t dst_col_base,
                                   const uint32_t warp_first,
                                   const uint32_t warp_step) {
            constexpr uint32_t kColsPerLaneI4 = 8;  // 8 bf16 = 16 bytes = int4
            // Compile-time chunk count so the row/column split stays shifts:
            // 64 chunks/row = 512 columns (1 KB per warp, one destination),
            // 32 = 256 columns (512 B per row, two rows per warp).
            constexpr uint32_t kChunksPerRow = decltype(chunks_per_row_tag)::value;
            static_assert(kChunksPerRow * kColsPerLaneI4 <= BLOCK_N,
                          "int4 scatter segment must fit inside the tile row");
            const uint32_t num_chunks = vm * kChunksPerRow;
            for (uint32_t i = warp_first * kWarpSizeLocal + lane;
                 i < num_chunks;
                 i += warp_step * kWarpSizeLocal) {
                const uint32_t my_row = i / kChunksPerRow;
                const uint32_t my_col_lane = i % kChunksPerRow;
                {
                    using vec4i = __attribute__((__vector_size__(16))) int32_t;
                    const auto* gmem_ptr = reinterpret_cast<const vec4i*>(
                        scratch + my_row * src_row_stride + src_col_off
                        + my_col_lane * kColsPerLaneI4);
                    vec4i packed = *gmem_ptr;

                    const auto& meta = *workspace.get_token_src_metadata_ptr(m_base + my_row);
                    const uint32_t dst_rank_idx = meta.rank_idx;
                    const uint32_t dst_token_idx = meta.token_idx;
                    const uint32_t dst_topk_idx = meta.topk_idx;

                    void* dst_ptr = slices.combine_token_buffer.get_data_buffer(dst_token_idx, dst_topk_idx);
                    auto* dst = reinterpret_cast<vec4i*>(sym_map_dst(dst_ptr, dst_rank_idx));
                    dst[dst_col_base / kColsPerLaneI4 + my_col_lane] = packed;

#if DG_MEGAMOE_DBUG
                    if (my_col_lane == 0 && dst_token_idx < 128u && dst_topk_idx < kNumTopk) {
                        atomicAdd(&smem_scatter_counts[dst_token_idx * kNumTopk + dst_topk_idx], 1u);
                        atomicAdd(&g_scatter_total[dst_token_idx * kNumTopk + dst_topk_idx], 1u);
                        if (packed[0] == 0 && packed[1] == 0 && packed[2] == 0 && packed[3] == 0) {
                            atomicAdd(&smem_scatter_zeros[dst_token_idx * kNumTopk + dst_topk_idx], 1u);
                            atomicAdd(&g_scatter_zero_total[dst_token_idx * kNumTopk + dst_topk_idx], 1u);
                        }
                    }
#endif
                }
            }
        };

        // L2 GEMM: SiLU+quant + L2 GEMM + scatter
        l2_scheduler.for_each_l2_block([&](uint32_t expert_idx,
                                            uint32_t num_k_blocks,
                                            uint32_t m_blk, uint32_t n_blk) {
            const uint32_t valid_m = l2_scheduler.get_valid_m();
            const uint32_t pool_block_idx = broadcast_lane0(
                l2_scheduler.get_current_pool_block_offset() + m_blk);
            const uint32_t m_idx = broadcast_lane0(pool_block_idx * BLOCK_M);
            // Absolute column base of this L2 tile.  When BLOCK_N does not
            // divide L2_SHAPE_N the last tile is shifted back to
            // L2_SHAPE_N - BLOCK_N and overlaps its predecessor (identical
            // inputs -> identical outputs, so the duplicate writes in the
            // overlap are benign).
            // Absolute column base of this L2 tile.
            //   L2-W12X: wide tiles at n_blk*768, the single tail tile at
            //            6*768 = 4608 with width 512 (exact coverage).
            //   otherwise: ceil'd tiles with the last one shifted back to
            //            L2_SHAPE_N - BLOCK_N (overlaps its predecessor;
            //            identical inputs -> identical duplicate bytes).
            [[maybe_unused]] const uint32_t n_col_base = kUseL2W12X
                ? ((n_blk < kNumL2WideTiles) ? (n_blk * kL2WideN)
                                             : (kNumL2WideTiles * kL2WideN))
                : ((kL2TailOverlap != 0 &&
                    n_blk * BLOCK_N > L2_SHAPE_N - BLOCK_N)
                       ? (L2_SHAPE_N - BLOCK_N)
                       : n_blk * BLOCK_N);
            const bool l2_wide_tile = kUseL2W12X && (n_blk < kNumL2WideTiles);

            // ============ L2 GEMM ============
            // l2_token/l2_sf were written by the fused L1 epilogues; blocks
            // only wait for the pool's L1 arrival mask.
#ifdef DG_MEGA_PROF
            const uint64_t prof_t0 = (prof != nullptr) ? clock64() : 0;
#endif
            if (epilogue_warp_idx == 0 && lane == 0) {
                volatile const uint64_t* vptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                const uint64_t expected = (kNumL1BlockNs >= 64) ?
                    ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
#if DG_HANG_DIAG
                const auto t0 = clock64();
                bool printed = false;
#endif
                while (__hip_atomic_load(vptr, __ATOMIC_ACQUIRE,
                                          __HIP_MEMORY_SCOPE_AGENT) != expected) {
#if DG_HANG_DIAG
                    if (!printed && clock64() - t0 > DG_HANG_TIMEOUT_CYCLES) {
                        DG_HANG_PRINTF("MegaMoE spin timeout [L2wait]: exp=%llu got=%llu blk=%d pb=%u\n",
                               (unsigned long long)expected, (unsigned long long)*vptr,
                               (int)blockIdx.x, pool_block_idx);
                        DG_HANG_MARK(6, (uint32_t)expected, (uint32_t)*vptr, pool_block_idx);
                        printed = true;
                    }
#endif
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            __syncthreads();
            DG_PROF_ACC(prof_l2_wait);
            DG_PHASE(20, pool_block_idx);

            if (valid_m == 0) {
                __syncthreads();
                return;
            }
#ifdef DG_MEGA_PROF
            const uint64_t prof_t1 = (prof != nullptr) ? clock64() : 0;
#endif

            // Pre-fill this tile's [BLOCK_M, kL2Groups] boundary-ratio
            // table into SMEM: r_g = s_{g-1}/s_g for g >= 1, with slot 0
            // holding s_last = s_{G-1} for the epilogue.  The groupwise L2
            // GEMM then needs one LDS read per row per boundary -- no frcp,
            // no persistent scale registers, no GMEM loads.  Invalid rows
            // are padded with 1.0f so ratios are 1.
            // HT scale table overlays smem_silu_result (dead under LL).
            float* const smem_l2_scale =
                reinterpret_cast<float*>(&smem_silu_result[0]);
            if constexpr (kLLGemm) {
                // LL layout: dedicated table with stride kL2Groups+1 to
                // spread LDS banks.
                const float* matrixA_scale_ll =
                    slices.l2_sf_buffer.get_data_buffer<float>(m_idx);
                constexpr uint32_t kL2ScaleStride = kL2Groups + 1;
                for (uint32_t i = thread_idx; i < BLOCK_M * kL2Groups;
                     i += kNumThreads) {
                    const uint32_t row = i / kL2Groups;
                    const uint32_t g = i % kL2Groups;
                    float v = 1.0f;
                    if (row < valid_m) {
                        const float s_g = matrixA_scale_ll[row * kL2Groups + g];
                        v = (g == 0)
                            ? matrixA_scale_ll[row * kL2Groups + (kL2Groups - 1)]
                            : matrixA_scale_ll[row * kL2Groups + g - 1]
                                * __frcp_rn(s_g);
                    }
                    smem_l2_scale_table[row * kL2ScaleStride + g] = v;
                }
                __syncthreads();
            } else if constexpr (BLOCK_M < 256) {
                // HT layout: [BLOCK_M x kL2Groups] flat table reusing
                // smem_silu_result storage.
                const float* sf_row_base =
                    slices.l2_sf_buffer.get_data_buffer<float>(0) +
                    static_cast<int64_t>(m_idx) * kL2Groups;
                for (uint32_t i = thread_idx; i < BLOCK_M * kL2Groups;
                     i += kNumThreads) {
                    const uint32_t row = i / kL2Groups;
                    const uint32_t g = i % kL2Groups;
                    float v = 1.0f;
                    if (row < valid_m) {
                        const float* srow = sf_row_base + row * kL2Groups;
                        v = (g == 0) ? srow[kL2Groups - 1]
                                     : srow[g - 1] / srow[g];
                    }
                    smem_l2_scale[i] = v;
                }
                __syncthreads();
#ifdef DG_MEGA_W12P_DBG
                if (n_blk == 0 && thread_idx == 0 && sm_idx == 0 && m_idx == 0) {
                    printf("[W12PL2] exp=%u g=%u vm=%u "
                           "r=%.7g,%.7g,%.7g,%.7g,%.7g,%.7g "
                           "raw=%.7g,%.7g,%.7g,%.7g,%.7g,%.7g\n",
                           expert_idx, kL2Groups, valid_m,
                           smem_l2_scale[0], smem_l2_scale[1], smem_l2_scale[2],
                           smem_l2_scale[3], smem_l2_scale[4], smem_l2_scale[5],
                           sf_src[0], sf_src[1], sf_src[2],
                           sf_src[3], sf_src[4], sf_src[5]);
                }
#endif
            }
#ifdef DG_MEGA_PROF
            if (prof != nullptr && thread_idx == 0)
                prof_l2_fill += clock64() - prof_t1;
            const uint64_t prof_t2 = (prof != nullptr) ? clock64() : 0;
#endif

            // L2 GEMM setup
            const char* matrixA = reinterpret_cast<const char*>(
                slices.l2_token_buffer.get_data_buffer(m_idx));
            // Per-block scratch: HT GEMM writes this tile then scatter
            // consumes it before the next tile, so a block-private slot
            // suffices.  Unused under LL (stages output in smem_l2_out).
            hip_bfloat16* const l2_output = reinterpret_cast<hip_bfloat16*>(
                slices.l2_output_buffer.get_data_buffer(sm_idx));

            if constexpr (kLLGemm) {
                // ---- LL GEMM: groupwise tile over raw 3D weights, output
                // staged in SMEM (smem_l2_out) ----
                constexpr uint32_t kLdElementsPerBlockL2 =
                    (128 / 8) * kWarpSizeLocal * kNumMathWarps *
                    (BLOCK_N / (kNumMathWarps * 16));
                const char* matrixB = reinterpret_cast<const char*>(
                    l2_weights + expert_idx * L2_SHAPE_N * L2_SHAPE_K
                    + n_blk * kLdElementsPerBlockL2);
                const float* matrixB_scale = l2_weights_scale + expert_idx * L2_SHAPE_N
                    + n_blk * BLOCK_N;
                if (epilogue_warp_idx < kNumMathWarps) {
                    gemm_groupwise_tile<L2_SHAPE_N, L2_SHAPE_K, kGroupQuantSize,
                                         BLOCK_M, BLOCK_N, BLOCK_K,
                                         16, kNumMathWarps, kL2PipeStage, kIsInt8>(
                        matrixA, matrixB, matrixB_scale,
                        0, 0, valid_m, smem_l2_out,
                        l2_scheduler.get_num_tokens(expert_idx),
                        math_warp_offset,
                        BLOCK_N,
                        smem_l2_scale_table);
                }
            } else {
                // ---- HT GEMM: Marlin permuted weights, direct GMEM output ----
                const int expert_b_base_l2 = broadcast_lane0(
                    expert_idx * L2_SHAPE_N * L2_SHAPE_K);
                // A scales: per-token group-wise [kL2Groups] floats written by
                // the fused L1 epilogue.
                const float* matrixA_scale =
                    slices.l2_sf_buffer.get_data_buffer<float>(0) +
                    m_idx * kL2Groups;
                const float* matrixB_scale = l2_weights_scale +
                    broadcast_lane0(expert_idx * L2_SHAPE_N);

#if DG_MEGAMOE_DBUG
                if (epilogue_warp_idx == 0 && lane == 0 && sm_idx == 0) {
                    DG_DBUG_PRINTF("L2HT blk=%d exp=%u m_idx=%u n_blk=%u valid_m=%u A=%p Bbase=%d C=%p M=%u\n",
                        (int)blockIdx.x, expert_idx, m_idx, n_blk, valid_m,
                        (void*)matrixA, expert_b_base_l2, (void*)l2_output,
                        l2_scheduler.get_num_tokens(expert_idx));
                }
#endif

                // L2-W12X keeps its historical split (wide tiles 12 math
                // warps, the narrow tail 8 math + 4 mirror).  Every other
                // layout must enter the L2 GEMM with exactly the warp count
                // the GEMM is instantiated for: hardcoding 8 here left warps
                // 8..15 out of a 16-warp L2 tile, so their output columns
                // were never computed (silently wrong results).
                const int l2_math_warps = kUseL2W12X
                    ? (l2_wide_tile ? 12 : 8)
                    : static_cast<int>(kNumMathWarpsL2);
                if (epilogue_warp_idx < l2_math_warps) {
                    if constexpr (BLOCK_M == 256) {
                        // Dedicated BM256 path: compact [256x256] tile output,
                        // boundary ratios read straight from l2_sf in GMEM.
                        gemm_ht_tile_bm256_bk64_groupwise<L2_SHAPE_N, L2_SHAPE_K,
                                 kHTGroupSize, BLOCK_N, BLOCK_K, kIsInt8>(
                            matrixA, l2_weights, expert_b_base_l2,
                            matrixA_scale, matrixB_scale,
                            0, n_blk, valid_m, l2_output,
                            l2_scheduler.get_num_tokens(expert_idx),
                            math_warp_offset,
                            256,
                            a_lds_ht,
                            &smem_bar_count, &smem_bar_sense);
                    } else if constexpr (BLOCK_M == 64 && BLOCK_K == 128) {
                        // BM64/BK128 group-wise L2: with GS=128 every K-tile is a
                        // group boundary (rescale per tile).
                        gemm_ht_tile_bm64_bk128_groupwise<L2_SHAPE_N, L2_SHAPE_K,
                                 kHTGroupSize, BLOCK_M, BLOCK_N, BLOCK_K,
                                 kNumMathWarpsL2, kIsInt8, true, kAStagesHT>(
                            matrixA, l2_weights, expert_b_base_l2,
                            matrixA_scale, matrixB_scale,
                            0, n_blk, valid_m, l2_output,
                            l2_scheduler.get_num_tokens(expert_idx),
                            math_warp_offset,
                            L2_SHAPE_N,
                            smem_l2_scale,
                            a_lds_ht,
                            &smem_bar_count, &smem_bar_sense);
                    } else if constexpr (kUseL2W12X) {
                        // L2-W12X: 768-wide tiles with 12 math warps, plus the
                        // single 512-wide tail tile on the proven 8-warp layout.
                        if (l2_wide_tile || !kL2HasTail) {
                            gemm_ht_tile_bm128_bk64_groupwise<L2_SHAPE_N, L2_SHAPE_K,
                                     kHTGroupSize, BLOCK_M, kL2WideN, BLOCK_K,
                                     12, kIsInt8, true, kAStagesHT, kRegOptTileKernel>(
                                matrixA, l2_weights, expert_b_base_l2,
                                matrixA_scale, matrixB_scale,
                                0, n_blk, valid_m, l2_output,
                                l2_scheduler.get_num_tokens(expert_idx),
                                math_warp_offset,
                                L2_SHAPE_N,
                                smem_l2_scale,
                                a_lds_ht,
                                &smem_bar_count, &smem_bar_sense,
                                static_cast<int32_t>(n_col_base));
                        } else if constexpr (kL2HasTail) {
                            gemm_ht_tile_bm128_bk64_groupwise<L2_SHAPE_N, L2_SHAPE_K,
                                     kHTGroupSize, BLOCK_M, kL2TailN, BLOCK_K,
                                     8, kIsInt8, true, kAStagesHT, kRegOptTileKernel>(
                                matrixA, l2_weights, expert_b_base_l2,
                                matrixA_scale, matrixB_scale,
                                0, n_blk, valid_m, l2_output,
                                l2_scheduler.get_num_tokens(expert_idx),
                                math_warp_offset,
                                L2_SHAPE_N,
                                smem_l2_scale,
                                a_lds_ht,
                                &smem_bar_count, &smem_bar_sense,
                                static_cast<int32_t>(n_col_base));
                        }
                    } else {
                        // BK64 family: BM128 (and BM64-V2, 4 M-subtiles).
                        gemm_ht_tile_bm128_bk64_groupwise<L2_SHAPE_N, L2_SHAPE_K,
                                 kHTGroupSize, BLOCK_M, BLOCK_N, BLOCK_K,
                                 kNumMathWarpsL2, kIsInt8, true, kAStagesHT, kRegOptTileKernel>(
                            matrixA, l2_weights, expert_b_base_l2,
                            matrixA_scale, matrixB_scale,
                            0, n_blk, valid_m, l2_output,
                            l2_scheduler.get_num_tokens(expert_idx),
                            math_warp_offset,
                            L2_SHAPE_N,
                            smem_l2_scale,
                            a_lds_ht,
                            &smem_bar_count, &smem_bar_sense,
                            static_cast<int32_t>(n_col_base));
                    }
                } else {
                    // Same s_barrier mirror for the L2 GEMM stage barriers.
                    constexpr int kL2GemmStageBars = L2_SHAPE_K / BLOCK_K;
                    for (int i = 0; i < kL2GemmStageBars; ++i)
                        asm volatile("s_barrier\n\t" ::: "memory");
                }
            }  // kLLGemm L2 family

            __syncthreads();
#ifdef DG_MEGA_PROF
            if (prof != nullptr && thread_idx == 0)
                prof_l2_gemm += clock64() - prof_t2;
            const uint64_t prof_t3 = (prof != nullptr) ? clock64() : 0;
#endif

            // L2 Epilogue: scatter to combine buffer
            if constexpr (kLLGemm) {
                // LL: rolled int2 scatter from the SMEM tile (smem_l2_out).
                constexpr uint32_t kColsPerLane = BLOCK_N / kWarpSizeLocal;
                constexpr uint32_t kScatterVecBytes = kColsPerLane * sizeof(hip_bfloat16);
                static_assert(kScatterVecBytes == sizeof(int2), "kColsPerLane must give int2-sized scatter");

                for (uint32_t row_in_block = epilogue_warp_idx;
                     row_in_block < valid_m;
                     row_in_block += kNumTotalWarps) {
                    const auto* smem_ptr = reinterpret_cast<const int2*>(
                        smem_l2_out + row_in_block * BLOCK_N + lane * kColsPerLane);
                    int2 packed = smem_ptr[0];

                    const auto& meta = *workspace.get_token_src_metadata_ptr(m_idx + row_in_block);
                    const uint32_t dst_rank_idx = meta.rank_idx;
                    const uint32_t dst_token_idx = meta.token_idx;
                    const uint32_t dst_topk_idx = meta.topk_idx;

                    void* dst_ptr = slices.combine_token_buffer.get_data_buffer(dst_token_idx, dst_topk_idx);
                    auto* dst = reinterpret_cast<int2*>(sym_map_dst(dst_ptr, dst_rank_idx));
                    dst[n_blk * kWarpSizeLocal + lane] = packed;
                }
            } else {
                // HT: int4 scatter from l2_output_buffer (GMEM).
                // L2-W12X wide tiles are covered by two contiguous segments --
                // [0,512) at 64 lanes/row and [512,768) at 32 lanes/row -- so
                // the remote write pattern is the same 1KB/512B per-destination
                // burst BN512 had (no 96-lane flat grid, no divide by 96).
                // Every other tile keeps the single full-width segment.
                if constexpr (kUseL2W12X) {
                    if (l2_wide_tile || !kL2HasTail) {
                        // Flat chunk grid over the 768-wide tile (96
                        // chunks/row).  Measured better than the two-segment
                        // split: one remote transaction per (row, chunk) and
                        // one metadata read per chunk beat two passes with
                        // differing burst sizes.
                        l2_scatter_rows(std::integral_constant<uint32_t, kL2WideN / 8>{},
                                        l2_output, m_idx, valid_m,
                                        kL2ScatterSrcStride,
                                        n_col_base, n_col_base,
                                        epilogue_warp_idx, kNumTotalWarps);
                    }
                    if constexpr (kL2HasTail) {
                        // The narrow tail tile is kL2TailN columns wide.
                        if (!l2_wide_tile) {
                            l2_scatter_rows(std::integral_constant<uint32_t, kL2TailN / 8>{},
                                            l2_output, m_idx, valid_m,
                                            kL2ScatterSrcStride,
                                            n_col_base, n_col_base,
                                            epilogue_warp_idx, kNumTotalWarps);
                        }
                    }
                } else {
                    l2_scatter_rows(std::integral_constant<uint32_t, BLOCK_N / 8>{},
                                    l2_output, m_idx, valid_m,
                                    kL2ScatterSrcStride,
                                    (BLOCK_M == 256) ? 0u : n_col_base, n_col_base,
                                    epilogue_warp_idx, kNumTotalWarps);
                }
            }
            // __syncwarp();  -- redundant, 884 syncs after L2 loop
#ifdef DG_MEGA_PROF
            if (prof != nullptr && thread_idx == 0) {
                prof_scatter += clock64() - prof_t3;
                ++prof_l2_tiles;
            }
#endif
        });

        __syncthreads();  // all blocks done with L2
        DG_PROF_TS(12);  // t_l2_done
        DG_PHASE(21, 0);
#ifdef DG_MEGA_PROF
        if (prof != nullptr && thread_idx == 0) {
            prof[sm_idx * kProfSlots + 6]  = prof_l2_wait;
            prof[sm_idx * kProfSlots + 7]  = prof_l2_gemm;
            prof[sm_idx * kProfSlots + 8]  = prof_l2_fill;
            prof[sm_idx * kProfSlots + 10] = prof_l2_tiles;
        }
        if (prof != nullptr && thread_idx == 0) {
            prof[sm_idx * kProfSlots + 9] = prof_scatter;
        }
#endif
    }

#if DG_MEGAMOE_DBUG
    // Print scatter counts for first block --shows which (token, topk) slots
    // were written and how many N-blocks contributed to each.
    if (sm_idx == 0 && thread_idx == 0) {
        for (uint32_t t = 0; t < num_tokens && t < 128u; ++t) {
            for (uint32_t k = 0; k < kNumTopk; ++k) {
                const uint32_t c = smem_scatter_counts[t * kNumTopk + k];
                const uint32_t z = smem_scatter_zeros[t * kNumTopk + k];
                if (c == 0) {
                    DG_DBUG_PRINTF("SCATTER_MISS tok=%u k=%u cnt=0\n", t, k);
                } else if (c != kNumL2BlockNs) {
                    DG_DBUG_PRINTF("SCATTER_PARTIAL tok=%u k=%u cnt=%u zeros=%u exp=%u\n",
                                   t, k, c, z, kNumL2BlockNs);
                } else if (z > 0) {
                    DG_DBUG_PRINTF("SCATTER_ZEROS tok=%u k=%u cnt=%u zeros=%u\n",
                                   t, k, c, z);
                }
            }
        }
    }
    __syncthreads();

    // Print GLOBAL scatter counts (across ALL blocks) --only from block 0
    if (sm_idx == 0 && thread_idx == 0) {
        for (uint32_t t = 0; t < num_tokens && t < 128u; ++t) {
            for (uint32_t k = 0; k < kNumTopk; ++k) {
                const uint32_t gt = g_scatter_total[t * kNumTopk + k];
                const uint32_t gz = g_scatter_zero_total[t * kNumTopk + k];
                if (gt != kNumL2BlockNs) {
                    DG_DBUG_PRINTF("GSCATTER tok=%u k=%u total=%u zeros=%u exp=%u\n",
                                   t, k, gt, gz, kNumL2BlockNs);
                }
            }
        }
    }
    __syncthreads();
#endif

    // =====================================================================
    // ALL BLOCKS: grid_sync ->cleanup ->rank_barrier ->combine
    // =====================================================================
    // Grid sync: ensure ALL blocks on this rank finished
    grid_sync(
        workspace.template get_grid_sync_count_ptr<kSplitCombineGridSyncIndex>(), kNumSMs);

    // NOTE: no pre-cleanup rank_barrier -- cleanup only touches counters whose
    // remote writes are already gated by fetch_expert_recv_count (marker==32
    // implies every rank's contributions landed), and remote scatter writes
    // for combine are ordered by the post-cleanup rank_barrier below.
    __syncthreads();
    DG_PROF_TS(13);  // t_combine_barrier_done
    DG_PHASE(30, 0);

    // Cleanup workspace
    {
        const uint32_t max_cleanup_iters = ceil_div_u32(
            epr, kNumSMs);
        uint32_t cleanup_pool_block_offset = 0;
        {
            // Pass 1: read recv_sum -> clear recv_count rows + per-pool-block
            // l1/l2 counters. recv_sum itself is NOT zeroed here: warps read it
            // concurrently, so a warp0 store could make later warps see 0 and
            // skip clearing (only bites when an expert receives >1024 tokens).
            for (uint32_t iter = 0, i = sm_idx; iter < max_cleanup_iters; ++iter, i += kNumSMs) {
                if (i < epr) {
                    const auto num_recv_tokens = static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_sum_ptr(i));
                    const auto num_recv_m_blocks = ceil_div_u32(num_recv_tokens, BLOCK_M);
                    cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                    for (uint32_t j = thread_idx; j < nr; j += kNumThreads)
                        *workspace.get_expert_recv_count_ptr(j, i) = 0;

                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumThreads) {
                        const uint32_t pb = cleanup_pool_block_offset + j;
                        if constexpr (!kUseRingBuffer)
                            *workspace.get_l1_arrival_count_ptr(pb) = 0;
                        *workspace.get_l2_arrival_mask_ptr(pb) = 0;
                    }
                    // Zero l2_quant_done
                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumThreads) {
                        const uint32_t pb = cleanup_pool_block_offset + j;
                        *workspace.get_l2_quant_done_ptr(pb) = 0;
                    }
                }
            }
            if constexpr (kUseRingBuffer) {
                // Ring sync arrays are slot-indexed (not pool-block-indexed):
                // clear the four contiguous u32 arrays once. Every block does
                // this redundantly — all writes are 0, so it is benign.
                const uint32_t ring_words =
                    align_u32(workspace.num_l1_pool_blocks, 2u) * 4;
                for (uint32_t j = thread_idx; j < ring_words; j += kNumThreads)
                    *workspace.get_l1_arrival_count_ptr(j) = 0;
            }
            // Pass 2: now that every warp has consumed recv_sum, zero it.
            __syncthreads();
            for (uint32_t iter = 0, i = sm_idx; iter < max_cleanup_iters; ++iter, i += kNumSMs) {
                if (i < epr && warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                }
            }
        }
    }
    __syncthreads();
    DG_PROF_TS(14);  // t_cleanup_done
    DG_PHASE(31, 0);

    // Post-cleanup cross-rank barrier: prevents an early exit (e.g. 0 local
    // tokens -> empty combine) from relaunching into a peer's still-running
    // cleanup, which would erase freshly written recv counts and deadlock.
    // The pre-cleanup grid_sync above already implies all blocks finished
    // their remote-write work; cleanup counters are rank-local only, so the
    // barrier needs no additional all-block sync before it.
    // One rank_barrier per rank: only sm0 signals/waits (see dispatch-side
    // note -- all-blocks rank_barrier is a 36x-stronger barrier).
    if (sm_idx == 0) {
        rank_barrier(signal_buffers, rank_idx, nr, kSplitAfterCleanupBarrierTag);
    }
    // Hold all blocks until sm0's cross-rank barrier completes.
    grid_sync(workspace.template get_grid_sync_count_ptr<kSplitCombineGridSyncIndex>(),
              kNumSMs);
    __syncthreads();
    // Clear send_count only AFTER the post-cleanup barrier: peers remote-read
    // it between the before-pull barrier and this barrier, so zeroing earlier
    // could hand a straggler's pull empty rows and deadlock its fetch_recv.
    if (sm_idx == 0) {
        for (uint32_t i = thread_idx; i < ne; i += kNumThreads)
            *workspace.get_expert_send_count_ptr(i) = 0;
        __syncthreads();
    }
    DG_PHASE(32, 0);

    // Combine: y[token] = sum_k(topk_weights[k] * combine[k, token, :])
    {
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(hip_bfloat16);
        constexpr uint32_t kNumChunkUint4 = kNumHiddenBytes / 16;
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / kWarpSizeLocal;
        constexpr uint32_t kNumBF16PerUint4 = 16 / sizeof(hip_bfloat16);

        for (int token_idx = sm_idx * kNumTotalWarps + static_cast<int>(warp_idx);
             token_idx < num_tokens;
             token_idx += kNumSMs * static_cast<int>(kNumTotalWarps)) {

            // Hoist per-token top-k weights out of the inner loop (LL already does this)
            float weights[kNumTopk];
            #pragma unroll
            for (uint32_t k = 0; k < kNumTopk; ++k) {
                weights[k] = slices.input_topk_weights_buffer.get_data_buffer<float>(token_idx)[k];
            }

            auto* dst_base = reinterpret_cast<int4*>(y + token_idx * kHidden);

            for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
                float acc[kNumBF16PerUint4] = {};

                for (uint32_t k = 0; k < kNumTopk; ++k) {
                    const float weight = weights[k];
                    const auto* src = reinterpret_cast<const int4*>(
                        slices.combine_token_buffer.get_data_buffer(token_idx, k));
                    int4 val = src[j * kWarpSizeLocal + lane];
                    auto* bf16_vals = reinterpret_cast<const hip_bfloat16*>(&val);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumBF16PerUint4; ++l) {
                        acc[l] += weight * static_cast<float>(bf16_vals[l]);
                    }
                }

                int4 out;
                auto* out_bf16 = reinterpret_cast<hip_bfloat16*>(&out);
                #pragma unroll
                for (uint32_t l = 0; l < kNumBF16PerUint4; ++l) {
                    out_bf16[l] = hip_bfloat16(acc[l]);
                }
                dst_base[j * kWarpSizeLocal + lane] = out;
            }
        }
    }
    DG_PROF_TS(15);  // t_end
    DG_PHASE(50, 0);

#undef DG_PROF_TS
#undef DG_PROF_ACC
#undef DG_PROF_T0
#undef DG_PHASE
#endif // !defined(__gfx938__)
}

} // namespace deep_gemm::mega::hcu
