// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

// ============================================================================
// Debug logging macros with levels.
// Define DG_MEGAMOE_DBUG=1 at compile time to enable diagnostics.
// The debug LEVEL is controlled at RUNTIME via the DG_DBUG_LEVEL env var
// (read by host, copied to a __constant__ symbol):
//   0 → no output
//   1 → TRACE only (stage value tracing for correctness debug)
//   2 → full debug (scheduling, hang detection, all metadata)
// When DG_MEGAMOE_DBUG=0, all printf calls compile to no-ops (release mode).
// ============================================================================
#ifndef DG_MEGAMOE_DBUG
#define DG_MEGAMOE_DBUG 0
#endif

#if DG_MEGAMOE_DBUG
// Runtime debug level — a single __managed__ variable visible from both
// host and device. Set from DG_DBUG_LEVEL env var before kernel launch.
#include <cstdio>
__managed__ int dg_dbug_level = 2;

#define DG_TRACE_PRINTF(...)          do { if (dg_dbug_level >= 1) printf(__VA_ARGS__); } while(0)
#define DG_DBUG_PRINTF(...)           do { if (dg_dbug_level >= 2) printf(__VA_ARGS__); } while(0)
#define DG_DBUG_PRINTF_HOST(...)      do { if (dg_dbug_level >= 2) printf(__VA_ARGS__); } while(0)
// Condition-gated variants: the entire if-block compiles away in release mode
#define DG_TRACE_PRINTF_IF(cond, ...) do { if (cond) { if (dg_dbug_level >= 1) printf(__VA_ARGS__); } } while(0)
#define DG_DBUG_PRINTF_IF(cond, ...)  do { if (cond) { if (dg_dbug_level >= 2) printf(__VA_ARGS__); } } while(0)
#else
#define DG_TRACE_PRINTF(...)          ((void)0)
#define DG_DBUG_PRINTF(...)           ((void)0)
#define DG_DBUG_PRINTF_HOST(...)      ((void)0)
#define DG_TRACE_PRINTF_IF(cond, ...) ((void)0)
#define DG_DBUG_PRINTF_IF(cond, ...)  ((void)0)
#endif

// ============================================================================
// Hang-site diagnostics — enabled by build flag DG_MEGA_HANG_PRINT=1
// (or implicitly by DG_MEGAMOE_DBUG=1).  Compiles in ONLY the spin/timeout
// prints plus hang-record writes; none of the heavier debug instrumentation.
//
// On the first timeout a spinning site writes an 8-u32 record into
// dg_hang_records[blockIdx.x*8].  The host fetches the symbol's device VA
// once via _C.get_hang_record_ptr() and afterwards reads it with plain
// ctypes.string_at — no GPU queue involved, so the record is reachable even
// while the kernel is wedged (unlike printf, whose buffer may never flush).
//
// Record layout per block (u32):
//   [0] magic 'HANG'  [1] site id  [2] threadIdx.x
//   [3..5] site args (exp/got/epoch/…)  [6] clock64 lo32  [7] reserved
// Probe area (u32 index >= 800): peer-pointer snapshots + broadcast mask.
// Site ids: 1=subset_barrier 2=rank_barrier 3=grid_sync_leader
//           4=fetch_recv 5=L1wait 6=L2wait 7=SiLUwait 8=math_warp_barrier
// ============================================================================
#ifndef DG_MEGA_HANG_PRINT
#define DG_MEGA_HANG_PRINT 0
#endif

// Always defined (4KB) so the host can fetch its address in any build.
// dg_hang_launch_seq is bumped once per fused-kernel launch (block0/thread0);
// records carry it in word [7] so the watchdog can filter out stale marks
// left over from earlier launches (the array is never re-zeroed on device).
__device__ uint32_t dg_hang_records[128 * 8];
__device__ uint32_t dg_hang_launch_seq = 0;

#if DG_MEGA_HANG_PRINT || DG_MEGAMOE_DBUG
#include <cstdio>
#define DG_HANG_DIAG 1
#define DG_HANG_PRINTF(...)          do { printf(__VA_ARGS__); } while (0)
#define DG_HANG_PRINTF_IF(cond, ...) do { if (cond) printf(__VA_ARGS__); } while (0)
#define DG_HANG_MARK(site, a, b, c)                                           \
    do {                                                                      \
        volatile uint32_t* hr_ = dg_hang_records + blockIdx.x * 8;              \
        hr_[0] = 0x48414E47u;                                                 \
        hr_[1] = static_cast<uint32_t>(site);                                 \
        hr_[2] = static_cast<uint32_t>(threadIdx.x);                          \
        hr_[3] = static_cast<uint32_t>(a);                                    \
        hr_[4] = static_cast<uint32_t>(b);                                    \
        hr_[5] = static_cast<uint32_t>(c);                                    \
        hr_[6] = static_cast<uint32_t>(clock64() & 0xffffffffu);              \
        hr_[7] = dg_hang_launch_seq;                                          \
        __threadfence_system();                                               \
    } while (0)
#else
#define DG_HANG_DIAG 0
#define DG_HANG_PRINTF(...)          ((void)0)
#define DG_HANG_PRINTF_IF(cond, ...) ((void)0)
#define DG_HANG_MARK(site, a, b, c)  ((void)0)
#endif

// Unified spin-timeout for every hang site below (clock64 cycles, ~30 s at
// 2 GHz). Only referenced inside DG_HANG_DIAG-guarded timeout checks; the
// guards compile the clock64 compare out entirely in normal builds.
#define DG_HANG_TIMEOUT_CYCLES 60000000000ll

namespace deep_gemm::mega::hcu {

// Per-token source metadata for combine write-back (matches SM90 layout)
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

// Runtime parameters passed to the fused MegaMoE kernel.
// Mirrors the SM90 host-side parameter set, adapted for HCU gfx938.
struct MegaMoeParams {
    // Output
    hip_bfloat16* y;                  // [num_tokens, hidden]

    // Weights (FP8, channel-wise per-channel scale)
    const uint8_t* l1_weights;        // [num_experts_per_rank, 2*intermediate_hidden, hidden]
    const uint8_t* l2_weights;        // [num_experts_per_rank, hidden, intermediate_hidden]
    const float* l1_weights_scale;    // [num_experts_per_rank, 2*intermediate_hidden]
    const float* l2_weights_scale;    // [num_experts_per_rank, hidden]

    // Symmetric buffer (HIP IPC) and signal buffers
    uint8_t* sym_buffer;              // local sym_buffer base (IPC mapped)
    int** signal_buffers;             // peer signal ptrs (rank barrier)

    // Topology
    int rank_idx;
    int num_ranks;
    int num_experts;                  // total experts across all ranks
    int num_experts_per_rank;
    int num_max_tokens_per_rank;
    int num_topk;
    int num_tokens;                   // runtime token count for this rank

    // Shapes
    int hidden;
    int intermediate_hidden;

    // Activation
    float activation_clamp;
    bool fast_math;

    // Optional cumulative stats
    const int* cumulative_local_expert_recv_stats;  // may be nullptr
};

// BLOCK_M values actually reachable by the launcher:
//   LL: 16 / 32 / 48   (DG_MEGA_BLOCK_M, mega_moe.cu)
//   HT: 64 / 128       (DG_MEGA_HT_BM)
// 96 / 192 were inherited from the SM90 candidate list but never dispatched.
static constexpr int kNumCandidateBlockMs = 5;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {16, 32, 48, 64, 128};
// NOTE: the env-only HT BM144/160/256 variants need a larger max (>= the tile)
// and a matching kLCMCandidateBlockM; they are commented out of dispatch by
// default.  With 128 here, DG_MEGA_HT_BM=160 trips the "must divide the
// pool/ring token capacity" check in mega_moe.cu on purpose.
static constexpr int kMaxCandidateBlockM = 128;
static constexpr int kMinCandidateBlockM = 16;
// LCM of the candidate BM list above = 384 (16/32/48/64/128 all divide it).
// This must be the true LCM: the pool capacity is aligned to it and the
// launcher asserts every candidate BLOCK_M divides that capacity.  It used to
// read 192, which is NOT a multiple of 128, so the check only passed by
// accident of the V4.1 rounding; other topk/expert ratios are not that lucky
// and aborted with "HT BLOCK_M=128 must divide the pool/ring token capacity".
// Same fix as megamoe/patches/minimax_m31_support_20261104.patch.
static constexpr int kLCMCandidateBlockM = 384;

// HCU warp size
static constexpr int kWarpSize = 64;

// MFMA tile constants (from V1 gemm_channelwise_stage)
static constexpr int kMfmaM = 16;
static constexpr int kMfmaN = 16;
static constexpr int kMfmaK = 32;  // packed FP8: 32 bytes = 32 FP8 elements

// V1 GEMM config
static constexpr int kBlockM = 32;
static constexpr int kBlockN = 256;
static constexpr int kBlockK = 64;
static constexpr int kNumWarps = 4;
static constexpr int kWarpN = 16;
static constexpr int kNRepeat = kBlockN / (kNumWarps * kWarpN);  // = 4
static constexpr int kPipeStage = 2;

// Grid size: one block per CU. kNumSMs is a template parameter used in
// scheduler stride and launch grid. Runtime detection selects between
// kNumSMs=64 and kNumSMs=72 instantiations (BW1100 has 72 CUs).
// This global is only used as a default fallback for buffer sizing; all
// launch paths pass the runtime-detected CU count explicitly.
static constexpr int kNumSMs = 64;     // default fallback (64-CU devices)

// HT L2 GEMM output scratch: per-block slot size in rows.
// The HT kernel writes one [BLOCK_M x BLOCK_N] tile to l2_output_buffer and
// immediately scatters it to the remote combine buffer, so a per-block
// scratch suffices (no per-pool-block allocation needed). Sized for the
// largest HT BLOCK_M (128); smaller BLOCK_M variants still fit the slot.
static constexpr int kHTL2ScratchBlockM = 128;

// HT L1 GEMM output: the L1 epilogue fuses per-tile SwiGLU + group-wise
// quant and writes l2_token/l2_sf directly — no separate L1 output or
// post-SwiGLU scratch buffers are needed.

// Max bounds for runtime-sized arrays in scheduler/kernel.
// kNumExpertsPerRank and kNumRanks are runtime params (not template params),
// so we use these upper bounds for fixed-size per-lane arrays.
static constexpr int kMaxExpertsPerRank = 256;
static constexpr int kMaxExpertsPerLane  = (kMaxExpertsPerRank + kWarpSize - 1) / kWarpSize;  // 4
static constexpr int kMaxRanks           = 128;
static constexpr int kMaxRanksPerLane    = (kMaxRanks + kWarpSize - 1) / kWarpSize;            // 2

// Scatter-address fast path: LDS peer-delta table size for the runtime-fallback
// (<0,0>) kernel instantiation, which has no compile-time kNumRanks.  It must
// cover every runtime rank count (nr <= kMaxRanks); otherwise indexing with a
// runtime dst_rank overflows LDS and faults with an invalid remote address.
// Override at build time only if a deployment ever exceeds this rank count.
#ifndef MEGA_MAX_RANKS
#define MEGA_MAX_RANKS 128
#endif

// Helper math functions
__host__ __device__ static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

__host__ __device__ static inline uint32_t align_u32(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__host__ __device__ static inline uint64_t align_u64(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__host__ __device__ static inline uint32_t constexpr_min(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

// Group-wise quantization group size for SwiGLU output (post-L1 GEMM).
// Each group of kGroupQuantSize elements in the K dimension of L2 GEMM
// (i.e., the intermediate_hidden dimension) gets its own FP8 scale.
// This allows each L1 tile to independently quantize its SwiGLU output
// without waiting for all N-blocks to arrive.
// Must be divisible by BLOCK_K (64) and divide kIntermediateHidden (3072).
static constexpr int kGroupQuantSize = 128;

// Pool capacity (from SM90 layout/mega_moe.cuh:17-25)
__host__ __device__ static inline uint32_t get_num_max_pool_tokens(
    uint32_t num_ranks, uint32_t num_max_tokens_per_rank,
    uint32_t num_topk, uint32_t num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = constexpr_min(num_topk, num_experts_per_rank);
    return align_u32(
        num_max_recv_tokens * num_max_experts_per_token +
            num_experts_per_rank * (static_cast<uint32_t>(kMaxCandidateBlockM) - 1),
        static_cast<uint32_t>(kLCMCandidateBlockM));
}

// L1 ring-buffer sizing for the pull -> L1 GEMM staging buffers
// (l1_token/l1_sf). The ring capacity is the live pool-block
// bound, mirroring NV SM100 `get_num_max_live_pool_blocks`
// (scheduler/mega_moe.cuh, apis/mega.hpp): at most num_sms pool blocks are
// consumed in flight (one tile per epilogue block) plus one wave of pull-warp
// gate stalls; we provision extra waves of lookahead on top. There is no
// L1/L2 interleave frontier term because our pull is gate-bound and L2 runs
// after L1, not interleaved.
//
// Activation: the ring exists only to cap memory, so it engages exactly when
// the pool would exceed this capacity (below that, the full staging area is
// smaller than the cap anyway and allocation stays ring-free and sync-free).
// The capacity is aligned so every candidate BLOCK_M divides it (384 is a
// multiple of 16/32/48/64/128).
static constexpr uint32_t kL1RingAlign = 384;
static constexpr uint32_t kL1RingLookaheadWaves = 3;
__host__ __device__ static inline uint32_t get_l1_ring_pool_tokens(
    uint32_t num_sms) {
    const uint32_t num_live_pool_blocks = num_sms * (kL1RingLookaheadWaves + 2);
    return align_u32(
        num_live_pool_blocks * static_cast<uint32_t>(kMaxCandidateBlockM),
        kL1RingAlign);
}
// Activation bound used for the host launch predicate and
// HCUMegaMoeWorkspace::has_l1_ring. It MUST be identical on both sides, so it
// is pinned to the fixed kNumSMs constant rather than the runtime/template SM
// count (a kernel launched with fewer SMs must not see a larger ring than the
// host allocated anyway, but the on/off decision itself must never diverge).
__host__ __device__ static inline uint32_t get_l1_ring_activation_pool_tokens() {
    return get_l1_ring_pool_tokens(kNumSMs);
}

} // namespace deep_gemm::mega::hcu
