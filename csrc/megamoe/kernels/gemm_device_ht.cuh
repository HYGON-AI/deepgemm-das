// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// High-throughput GEMM device function for HCU gfx938 MFMA FP8.
// Adapted from m_grouped_fp8_gemm_nt_contiguous BM128 kernel.
//
// Key differences from gemm_channelwise_tile (low-latency version):
//   - A matrix loaded via matrix_load_64x16_b8_t_lds (hardware LDS write)
//   - B matrix uses w6_offset indexing (Marlin permuted 6D weights)
//   - BM=128, BN=256, BK=64 (larger M tile, fewer K iterations)
//   - 8 warps with kNFragments=2 (vs 4 warps with kNFragments=4)
//   - Double-buffered A (LDS) and B (registers)
//   - Output written directly to GMEM (no SMEM staging)

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "params.h"
#include "warp_reduce.cuh"       // broadcast_lane0
#include "activation_device.cuh"  // swiglu_nv

namespace deep_gemm::mega::hcu {

// W16 register-pressure relief for the HT tiles of the 1024-thread launch
// (W16 L1 1024/16w and both W16 L2 layouts), whose k-loop live set otherwise
// exactly exhausts the 192-VGPR budget.  Callers pass kRegOptArg only from
// kernels with kNumTotalWarps >= 16, so 768-thread layouts (V4.1 W12-P /
// L2-W12X, 256 VGPR) keep their historical register layout untouched.
//   bit 0 = 3 A-frag rotation slots instead of 4 (saves 8 dword)
//   bit 1 = 4 fused-epilogue row rounds instead of 2 (needs a kernel without
//           mirror warps, i.e. the 16-warp W16 layout)
// Both are on by default; set DG_MEGA_W16_REGOPT=0 (or another bit mask) at
// build time to A/B against the historical layout.
#if defined(DG_MEGA_W16_REGOPT)
constexpr int kRegOptMask = DG_MEGA_W16_REGOPT;
#else
constexpr int kRegOptMask = 3;
#endif

// Vector types
template <typename scalar_t, size_t len>
using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

using vec4_u32 = vec<uint32_t, 4>;
using vec4_i32 = vec<int32_t, 4>;
using vec2_i32 = vec<int32_t, 2>;
using vec8_i8  = vec<int8_t, 8>;
using vec4_f32 = vec<float, 4>;

union AOperandHT {
    vec8_i8 fp8x8[4];
    vec4_i32 i32x4[2];
};

union BOperandHT {
    vec8_i8 fp8x8[2];
    vec2_i32 i32x2[2];
    vec4_i32 i32x4;
};

// ---------------------------------------------------------------------------
// Helper functions (ported from bm128_bn256_bk64_nloop4.inc.cuh)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t pack_bf16x2_ht(float lo, float hi) {
    const hip_bfloat16 lo_bf16(lo);
    const hip_bfloat16 hi_bf16(hi);
    const uint16_t lo_raw = *reinterpret_cast<const uint16_t*>(&lo_bf16);
    const uint16_t hi_raw = *reinterpret_cast<const uint16_t*>(&hi_bf16);
    return static_cast<uint32_t>(lo_raw) |
           (static_cast<uint32_t>(hi_raw) << 16);
}

__device__ __forceinline__ uint64_t pack_bf16x4_ht(float v0, float v1,
                                                    float v2, float v3) {
    const uint32_t lo = pack_bf16x2_ht(v0, v1);
    const uint32_t hi = pack_bf16x2_ht(v2, v3);
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

template <typename T>
__device__ __forceinline__ vec4_u32 make_matrix_resource_ht(const T* ptr,
                                                             int stride) {
    vec4_u32 rsrc;
    *reinterpret_cast<uint64_t*>(&rsrc) =
        reinterpret_cast<uint64_t>(ptr) & 0xffffffffffffULL;
    rsrc[2] = stride;
    rsrc[3] = 0;
    return rsrc;
}

__forceinline__ __device__ void buffer_load_i32x4_ht(const char* base,
                                                      vec4_i32& value,
                                                      int byte_offset) {
    __builtin_amdgcn_sched_barrier(0);
    using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    intx4 resource;
    *reinterpret_cast<uint64_t*>(&resource) = reinterpret_cast<uint64_t>(base);
    resource[1] += 0x40400000;
    resource[2] = 0x80000000;
    resource[3] = 0x00020000;
    asm volatile(
        "s_nop 2 \n\t"
        "buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n"
        : "=v"(value), "+v"(byte_offset), "+s"(resource));
}

// B buffer descriptor hoisted out of the K loop (A1): buffer_load_i32x4_ht
// rebuilds all four SGPRs on every call because its descriptor operand is "+s"
// (inout), costing ~5 SALU plus an s_nop 2 and two sched_barriers per load.
// The base never changes inside the K loop, so build it once in the prologue.
using ht_rsrc_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

__device__ __forceinline__ ht_rsrc_t make_buffer_resource_ht(const void* base) {
    ht_rsrc_t rsrc;
    *reinterpret_cast<uint64_t*>(&rsrc) = reinterpret_cast<uint64_t>(base);
    rsrc[1] += 0x40400000;
    rsrc[2] = 0x80000000;
    rsrc[3] = 0x00020000;
    return rsrc;
}

__forceinline__ __device__ void buffer_load_i32x4_rsrc_ht(
    ht_rsrc_t& rsrc, vec4_i32& value, int byte_offset) {
    // The original buffer_load_i32x4_ht needed `sched_barrier`
    // + `s_nop 2` to cover the SALU->VMEM hazard of rebuilding the descriptor
    // inside the call, and a plain `asm volatile` carries every dependency
    // without a "memory" clobber.  With the descriptor built once in the
    // prologue neither the barrier nor the nop is needed; dropping both
    // measured neutral-to-faster than keeping them (7.487 vs 7.529 fused_min).
    asm volatile("buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n"
                 : "=v"(value), "+v"(byte_offset), "+s"(rsrc));
}

__device__ __forceinline__ void wait_vmcnt_zero_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt4_zero_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(4)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt5_zero_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(5)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt7_zero_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(7)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt2_zero_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(2)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

template <int N>
__device__ __forceinline__ void wait_vmcnt_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(%0)\n\t" :: "n"(N) : "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// Sense-reversing subset barrier for kNumMathWarps warps.
// Only the first lane of each participating warp touches the counter.
// All other lanes in the wavefront are in lockstep, so no intra-warp sync needed.
// p_count and p_sense must be in shared memory, initialized to {0, 0} before
// the first call.  The sense auto-flips on each barrier, so successive calls
// work without re-initialization as long as all participating warps call the
// barrier the same number of times.
__device__ __forceinline__ void math_warp_barrier_ht(
    volatile uint32_t* p_count, volatile uint32_t* p_sense, uint32_t num_warps) {
    const uint32_t lane = threadIdx.x & (kWarpSize - 1);
    if (lane == 0) {
        const uint32_t my_sense = *p_sense;
        const uint32_t old = __hip_atomic_fetch_add(
            (uint32_t*)p_count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (old == num_warps - 1) {
            *p_count = 0u;
            __threadfence_block();
            *p_sense = my_sense ^ 1u;
        } else {
#if DG_HANG_DIAG
            const auto t0 = clock64();
            bool printed = false;
#endif
            while (*p_sense == my_sense) {
#if DG_HANG_DIAG
                if (!printed && clock64() - t0 > DG_HANG_TIMEOUT_CYCLES) {
                    DG_HANG_PRINTF("MegaMoE hang [mwbar]: blk=%d warp=%d sense=%u\n",
                           (int)blockIdx.x, (int)(threadIdx.x / kWarpSize), my_sense);
                    DG_HANG_MARK(8, my_sense, *p_count, 0);
                    printed = true;
                }
#endif
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
}

// Combined: wait for all VM ops + subset barrier (replaces s_barrier version)
// s_barrier implicitly waits for vmcnt(0) and lgkmcnt(0); our subset barrier
// must explicitly do the same, otherwise matrix_load data may not be in LDS
// when we start reading it.
__device__ __forceinline__ void wait_vmcnt4_subset_barrier_ht(
    volatile uint32_t* p_count, volatile uint32_t* p_sense, uint32_t num_warps) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
    math_warp_barrier_ht(p_count, p_sense, num_warps);
}

// Hardware all-wave barrier (s_barrier) stage sync. Cheaper than the LDS
// sense-reversal subset barrier, but requires EVERY warp in the block to
// execute it 鈥?callers must mirror the call count in the non-math-warp
// else branch (each gemm_ht_tile_* call contains exactly K / BLOCK_K
// stage barriers). Non-math warps mirror with raw s_barrier only.
__device__ __forceinline__ void wait_vmcnt0_all_barrier_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// Same, but the explicit vmcnt(N) leaves the N most-recent VMEM ops in
// flight across the barrier -- s_barrier only waits for wave arrival, it
// does NOT drain the wait counters.  This is the wait_vmcntN_barrier
// pattern from m_grouped_*_gemm_nt_contiguous: the next tile's A
// matrix_load + B buffer_loads keep fetching while all warps sync, which
// is what gives the multi-stage pipeline real prefetch depth.
template <int N>
__device__ __forceinline__ void wait_vmcnt_all_barrier_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier\n\t"
                 :: "n"(N) : "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// Same, but drains LDS ops instead of VMEM ops.  Use when the barrier
// publishes SMEM data to other warps (e.g. the row-amax reduction table):
// writers ds_write before the barrier, readers consume it afterwards.
__device__ __forceinline__ void wait_lgkmcnt0_all_barrier_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_lgkmcnt_zero_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// lgkmcnt(1): allow the most-recent LDS op to stay in flight -- used by
// the 3-slot A-frag rotation so the m+1 prefetch needn't land before the
// current row's MMAs issue.
__device__ __forceinline__ void wait_lgkmcnt_one_ht() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(1)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// VMEM-wait + math-warp subset barrier with an in-flight allowance.
// Unlike s_barrier, the subset barrier is a pure LDS spin and does NOT
// drain this warp's outstanding VMEM ops, so s_waitcnt vmcnt(N) can leave
// the next tile's A matrix_load + B buffer_loads in flight across the
// stage boundary -- this is what gives the 3-stage pipeline real prefetch
// depth.  Use N=0 for the last tile (nothing left to overlap).
template <int N>
__device__ __forceinline__ void wait_vmcnt_subset_barrier_ht(
    volatile uint32_t* p_count, volatile uint32_t* p_sense, uint32_t num_warps) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(%0)\n\t" :: "n"(N) : "memory");
    __builtin_amdgcn_sched_barrier(0);
    math_warp_barrier_ht(p_count, p_sense, num_warps);
}

// lgkmcnt(0) + subset barrier: publishes SMEM writes (e.g. row-amax
// partials) to the math warps when the all-wave s_barrier is not used.
__device__ __forceinline__ void wait_lgkmcnt0_subset_barrier_ht(
    volatile uint32_t* p_count, volatile uint32_t* p_sense, uint32_t num_warps) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
    math_warp_barrier_ht(p_count, p_sense, num_warps);
}

#define MATRIX_LOAD_64X16_B8_T_LDS_HT(LDSADDR, SRSRC)                         \
  do {                                                                         \
    int soffset = (LDSADDR) + 0x80000000;                                      \
    __builtin_amdgcn_sched_barrier(0);                                         \
    asm volatile("s_nop 4\n\t"                                                \
                 "matrix_load_64x16_b8 %0, %1, moffset:0 t lds\n\t"           \
                 :: "s"(SRSRC), "s"(soffset)                                 \
                 : "memory");                                                 \
    __builtin_amdgcn_sched_barrier(0);                                         \
  } while (0)

#define MATRIX_LOAD_128X16_B8_T_LDS_HT(LDSADDR, SRSRC)                        \
  do {                                                                         \
    int soffset = (LDSADDR) + 0x80000000;                                      \
    asm volatile("s_nop 4\n\t"                                                \
                 "matrix_load_128x16_b8 %0, %1, moffset:0 t lds\n\t"          \
                 :: "s"(SRSRC), "s"(soffset)                                 \
                 : "memory");                                                 \
  } while (0)

#define DS_READ_A_M16_FROM_LDS_HT(STAGE_BASE, M_LOOP, A_REG)                   \
  do {                                                                         \
    const int a_lds_load_base = (M_LOOP) * kALdsTileBytes;                     \
    const int a_lds_load_offset =                                              \
        a_lds_load_base + a_lane_row_pair * kALdsRowPairStride +               \
        a_col_swizzle * 16 + a_lane_odd_k;                                     \
    const int a_lds_read_addr = (STAGE_BASE) + a_lds_load_offset;              \
    asm volatile("ds_read_b128 %0, %1\n\t"                                    \
                 : "=v"((A_REG).i32x4[0])                                     \
                 : "v"(a_lds_read_addr)                                      \
                 : "memory");                                                \
  } while (0)

__device__ __forceinline__ vec4_f32 fp8_mmac_16x16x32_ht(
    const vec8_i8& a, const vec8_i8& b, vec4_f32 acc) {
#if defined(__gfx938__)
    acc = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, acc, 1, 0);
#endif
    return acc;
}

__device__ __forceinline__ vec4_i32 i8_mmac_16x16x32_ht(
    const vec8_i8& a, const vec8_i8& b, vec4_i32 acc) {
#if defined(__gfx938__)
    acc = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(a, b, acc, 1, 0, 0);
#endif
    return acc;
}

// MMA traits for template-based fp8/int8 dispatch
template <bool kIsInt8>
struct GemmMmaTraits;

template <>
struct GemmMmaTraits<false> {  // FP8
    using acc_t = vec4_f32;
    static __device__ __forceinline__ acc_t mma(const vec8_i8& a, const vec8_i8& b, acc_t acc) {
        return fp8_mmac_16x16x32_ht(a, b, acc);
    }
    static __device__ __forceinline__ float to_float(float v) { return v; }
    static __device__ __forceinline__ acc_t zero() { return {0.0f, 0.0f, 0.0f, 0.0f}; }
    static __device__ __forceinline__ float rescale(float v, float rg) { return v * rg; }
};

template <>
struct GemmMmaTraits<true> {  // INT8
    using acc_t = vec4_i32;
    static __device__ __forceinline__ acc_t mma(const vec8_i8& a, const vec8_i8& b, acc_t acc) {
        return i8_mmac_16x16x32_ht(a, b, acc);
    }
    static __device__ __forceinline__ float to_float(int32_t v) { return static_cast<float>(v); }
    static __device__ __forceinline__ acc_t zero() { return {0, 0, 0, 0}; }
    static __device__ __forceinline__ int32_t rescale(int32_t v, float rg) {
        return __float2int_rn(static_cast<float>(v) * rg);
    }
};

// w6_offset: index into Marlin permuted [E, K/64, N/16, 4, 16, 16] weight
__device__ __forceinline__ int w6_offset_ht(int size_n, int n, int k) {
    const int n_band = n / 16;
    const int n_inner = n & 15;
    const int k_band = k / 64;
    const int k_mid = (k & 63) / 16;
    const int k_inner = k & 15;
    return (((k_band * (size_n / 16) + n_band) * 4 + k_mid) * 16 + n_inner) *
               16 + k_inner;
}

// ---------------------------------------------------------------------------
// High-throughput tile GEMM
//
// Template params:
//   N:          output columns (e.g. 2*intermediate_hidden for L1, hidden for L2)
//   K:          reduction dimension (e.g. hidden for L1, intermediate_hidden for L2)
//   BLOCK_M:    128
//   BLOCK_N:    256
//   BLOCK_K:    64
//   NUM_WARPS:  8 (uses all math warps)
//
// Args:
//   matrixA:       [M, K] FP8 contiguous (already offset to expert/pool block)
//   matrixB_w6:     Marlin permuted 6D weight [E, K/64, N/16, 4, 16, 16] (flat)
//   expert_b_base:  byte offset for this expert in matrixB_w6
//   matrixA_scale: [M] float per-token
//   matrixB_scale: [N] float per-channel (already offset to expert)
//   tile_m:        m tile index (0-based, each tile is BLOCK_M rows)
//   tile_n:        n tile index (0-based, each tile is BLOCK_N columns)
//   valid_m:       actual valid rows in this tile (for masking, <= BLOCK_M)
//   MatrixC:       output BF16 buffer (GMEM), layout [M, N] row-major
//   M:             max tokens for this expert (stride of A and C)
//   math_warp_offset: threadIdx.x offset of math warps (0 for epilogue blocks)
//   C_stride:      output stride (typically N)
//   a_lds:         LDS buffer for A matrix (caller-allocated, 16KB)
// ---------------------------------------------------------------------------
template <int32_t N, int K,
          int BLOCK_M = 128, int BLOCK_N = 256, int BLOCK_K = 64,
          int NUM_WARPS = 8, bool kIsInt8 = false,
          bool kAllWaveBarrier = false, int kAStagesArg = 3>
__device__ void gemm_ht_tile_bm128_bk64(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    static_assert(BLOCK_M % kMmacRows == 0 && BLOCK_N % (NUM_WARPS * kMmacCols) == 0 &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;  // 8
    constexpr int kK64Tile = 64;
    constexpr int kKLoops = BLOCK_K / 32;            // 2
    constexpr int kK64Loops = BLOCK_K / kK64Tile;    // 1
    constexpr int kNFragments = BLOCK_N / (NUM_WARPS * kMmacCols);  // 2
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;  // 128*64 = 8192
    constexpr int kAStages = kAStagesArg;
    constexpr int kALdsTotalBytes = kAStages * kAStageBytes;  // 24576
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;       // 16*64 = 1024
    constexpr int kALdsRowPairStride = 128;

    static_assert(kMSubtiles >= 8, "expects at least 8 M-subtiles (BM>=128)");
    static_assert(kNFragments >= 1 && kNFragments <= 4, "BN256/512 expects 1..4 N-fragments per warp");
    static_assert(kK64Loops == 1, "BK64 expects 1 K64 loop");

    // Stage barrier for the 3-stage pipeline.  s_barrier only waits for
    // wave arrival -- it does NOT drain the wait counters -- so the
    // explicit vmcnt(kLoadsPerTile) keeps tile k+1's A matrix_load + B
    // buffer_loads in flight across the barrier, giving the pipeline real
    // depth (same wait_vmcntN_barrier pattern as the grouped-GEMM
    // inc.cuh kernels).  stage_sync_drain() is for the last tile, whose
    // own loads must all be resident before compute.
    constexpr int kALoadsPerWarp =
        (kMSubtiles + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int kLoadsPerTile = kALoadsPerWarp + kNFragments;
    // Per-warp vmcnt budget: the allowance must match the number of
    // loads this warp itself issued last iteration.  With
    // NUM_WARPS > kMSubtiles some warps skip their A matrix_load, so
    // kLoadsPerTile would leave one extra slot of slack -- a B
    // buffer_load straggler from the previous tile could then stay in
    // flight across the barrier and feed stale registers to mma.
    // n_a_issued: A matrix_loads issued by an A-loading warp in the
    // half-iteration that just elapsed (0 or 2 per k-tile); the
    // allowance keeps exactly that half's loads (A + B fragments) in
    // flight.
    auto stage_sync = [&](int n_a_issued) {
        const int my_wid = (threadIdx.x - math_warp_offset) / kWarpSize;
        const bool full_a =
            (my_wid * kMmacRows < BLOCK_M) &&
            (kALoadsPerWarp == 1 ||
             my_wid * kMmacRows + NUM_WARPS * kMmacRows < BLOCK_M);
        if constexpr (kAllWaveBarrier) {
            if (full_a && n_a_issued)
                wait_vmcnt_all_barrier_ht<kALoadsPerWarp + kNFragments>();
            else
                wait_vmcnt_all_barrier_ht<kNFragments>();
        } else {
            if (full_a && n_a_issued)
                wait_vmcnt_subset_barrier_ht<kALoadsPerWarp + kNFragments>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
            else
                wait_vmcnt_subset_barrier_ht<kNFragments>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };
    auto stage_sync_drain = [&]() {
        if constexpr (kAllWaveBarrier) {
            wait_vmcnt_all_barrier_ht<0>();
        } else {
            wait_vmcnt_subset_barrier_ht<0>(
                p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };
    // Split M-subtiles into two halves to reduce peak VGPR pressure.
    // a_regs[8] = 64 VGPRs; by processing 4 at a time, peak drops to 32 VGPRs.
    constexpr int kMSubtilesHalf = kMSubtiles / 2;  // 4

    // Remap thread index for math warps
    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;

    const int n_block_start = tile_n * BLOCK_N;
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    // A LDS addresses
    // With NUM_WARPS < kMSubtiles (e.g. 4 warps for BM=128), each warp loads
    // 2 A tiles (a_load_m0, a_load_m1) to cover all BLOCK_M rows.
    constexpr int kNumALoadsPerWarp = (kMSubtiles + NUM_WARPS - 1) / NUM_WARPS;
    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m0 = warp_id * kMmacRows;
    const int a_load_m1 = a_load_m0 + NUM_WARPS * kMmacRows;
    const int a_lds_write_bytes0 = a_load_m0 * BLOCK_K;
    const int a_lds_write_bytes1 = a_load_m1 * BLOCK_K;
    // For backward compatibility with existing code using a_lds_write_bytes
    const int a_lds_write_bytes = a_lds_write_bytes0;

    // N base for this warp
    const int logical_n_base =
        n_block_start + warp_id * kNFragments * kMmacCols + (lane_id & 15);

    // B byte offset is affine in the K-tile index: b_k = KT*64+lane_group*16
    // stays inside one 64-row band, so b_k_band==KT, b_k_mid==lane_group and
    // b_k_inner==0.  Hoist the per-fragment constant out of the K loop --
    // each buffer_load then costs one IADD instead of ~10 ALU ops.
    const int b_stride_bytes = b_n_tiles * 4 * kMmacCols * 16;
    // Per-frag byte offset is affine: logical_n advances exactly one
    // 16-col group per fragment, so b_off(f) = b_off0 + f*(4*16*16).
    // One scalar instead of a kNFragments table (the f*step term folds
    // into the load's immediate add).
    const int b_off_base0 = expert_b_base +
        ((logical_n_base / kMmacCols) * 4 + lane_group) * (kMmacCols * 16) +
        (logical_n_base & (kMmacCols - 1)) * 16;
    constexpr int kBOffFragStep = 4 * kMmacCols * 16;

    // Accumulators
    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    // LDS read parameters
    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    // B register double-buffer (explicit names to avoid private memory)
    // Support kNFragments=2 (BN256) or kNFragments=4 (BN512)
    BOperandHT b_even0, b_even1;
    BOperandHT b_odd0, b_odd1;
    BOperandHT b_even2, b_even3;
    BOperandHT b_odd2, b_odd3;

    // Helper macros to select B register by fragment index
    #define B_EVEN_AT_HT(IDX) \
        ((IDX) == 0 ? b_even0 : (IDX) == 1 ? b_even1 : (IDX) == 2 ? b_even2 : b_even3)
    #define B_ODD_AT_HT(IDX) \
        ((IDX) == 0 ? b_odd0 : (IDX) == 1 ? b_odd1 : (IDX) == 2 ? b_odd2 : b_odd3)

    // A matrix base pointer for this warp's rows (first tile)
    const char* a_ptr_base0 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m0) * K;
    const char* a_ptr_base1 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m1) * K;

    // Helper macro: load A tile pair to a given LDS stage base
    // STAGE_BASE is the LDS base address for the pipeline stage
    // K_OFFSET is the K offset (in bytes) for the A tile
    #define LOAD_A_TILE_PAIR_HT(STAGE_BASE, K_OFFSET) \
        do { \
            if (a_load_m0 < BLOCK_M) { \
                const char* a0 = a_ptr_base0 + static_cast<int64_t>(K_OFFSET); \
                vec4_u32 srsrc0 = make_matrix_resource_ht(a0, K); \
                MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes0, srsrc0); \
            } \
            if constexpr (kNumALoadsPerWarp > 1) { \
                if (a_load_m1 < BLOCK_M) { \
                    const char* a1 = a_ptr_base1 + static_cast<int64_t>(K_OFFSET); \
                    vec4_u32 srsrc1 = make_matrix_resource_ht(a1, K); \
                    MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes1, srsrc1); \
                } \
            } \
        } while (0)

    #define LOAD_B_TILE_HT(KT, B_SEL)                                          \
        do {                                                                   \
            _Pragma("unroll")                                                  \
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {             \
                buffer_load_i32x4_ht(matrixB_w6, B_SEL(n_frag).i32x4,          \
                                     b_off_base0 +                           \
                                     (n_frag) * kBOffFragStep +             \
                                     (KT) * b_stride_bytes);                 \
            }                                                                  \
        } while (0)

    // Compute one K-tile: stage (KT % kAStages), B regs selected by B_SEL.
    // 4-slot A-frag rotation with prefetch distance 2 + lgkmcnt(1): the read
    // for subtile m+2 stays in flight while the mmas for m issue, hiding
    // ~30cyc of LDS latency that the old lgkmcnt(0) ping-pong exposed on
    // every subtile.  Slot (m+2)&3 was last read by mma(m-2), two full
    // iterations back, so the DS_READ writeback cannot collide with an
    // in-flight operand fetch (the failed 3-slot variant only had one
    // iteration of margin).
    #define COMPUTE_TILE_HT(KT, B_SEL)                                         \
        do {                                                                   \
            const int a_stage_base_c =                                         \
                a_lds_base_addr + (((KT) % kAStages) * kAStageBytes);          \
            AOperandHT a_regs[4];                                              \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 0, a_regs[0]);           \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 1, a_regs[1]);           \
            wait_lgkmcnt_one_ht();                                             \
            wait_vmcnt7_zero_ht();                                             \
            _Pragma("unroll")                                                  \
            for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {              \
                const int compute_slot = m_loop & 3;                           \
                if (m_loop + 2 < kMSubtiles) {                                 \
                    DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, m_loop + 2,      \
                                              a_regs[(m_loop + 2) & 3]);       \
                }                                                              \
                _Pragma("unroll")                                              \
                for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {         \
                    const BOperandHT& b_reg = B_SEL(n_frag);                   \
                    acc[m_loop][n_frag] = MmaT::mma(                           \
                        a_regs[compute_slot].fp8x8[0], b_reg.fp8x8[0],         \
                        acc[m_loop][n_frag]);                                  \
                    acc[m_loop][n_frag] = MmaT::mma(                           \
                        a_regs[compute_slot].fp8x8[1], b_reg.fp8x8[1],         \
                        acc[m_loop][n_frag]);                                  \
                }                                                              \
                if (m_loop + 2 < kMSubtiles) {                                 \
                    wait_lgkmcnt_one_ht();                                     \
                } else if (m_loop + 1 < kMSubtiles) {                          \
                    wait_lgkmcnt_zero_ht();                                    \
                }                                                              \
            }                                                                  \
        } while (0)

    // =========================================================================
    // Prologue: tiles 0/1 -> stages 0/1 (depth-2 prefetch), B0/B1 -> regs
    // =========================================================================
    {
        LOAD_A_TILE_PAIR_HT(a_lds_base_addr, 0);
        LOAD_B_TILE_HT(0, B_EVEN_AT_HT);
        if (k_tiles > 1) {
            LOAD_A_TILE_PAIR_HT(a_lds_base_addr + kAStageBytes, BLOCK_K);
            LOAD_B_TILE_HT(1, B_ODD_AT_HT);
        }
    }

    // =========================================================================
    // Main loop: 3-stage A LDS pipeline, one stage barrier per K-tile.
    // Each barrier publishes this tile's LDS writes and retires the stage
    // read two tiles ago, which the k+2 A-load then refills.  B(k+2) goes
    // into the just-consumed register slot after compute.  vmcnt(N) at the
    // barrier keeps the next tile's loads in flight (s_barrier alone does
    // not drain the counters).
    // =========================================================================
    int k_tile = 0;
    for (; k_tile + 2 < k_tiles; k_tile += 2) {
        // alpha half: A(k+2) always issues here.  With 3 stages A(k+3)
        // must wait for the beta barrier (stage (k+3)%3 aliases the tile
        // compute(k) is still reading); with 4 stages stage (k+3)%4's
        // last reader was compute(k-1), already ordered by the alpha
        // barrier, so it can issue in the beta half as well -- keeping
        // the matrix_load traffic symmetric across the iteration.
        stage_sync(0);
        LOAD_A_TILE_PAIR_HT(
            a_lds_base_addr + (((k_tile + 2) % kAStages) * kAStageBytes),
            (k_tile + 2) * BLOCK_K);
        COMPUTE_TILE_HT(k_tile, B_EVEN_AT_HT);
        LOAD_B_TILE_HT(k_tile + 2, B_EVEN_AT_HT);

        // Beta barrier: required for 3 stages and measurably better for
        // 4 too -- the vmcnt wait forces B(k+1) stragglers and A(k+2) to
        // land (dropping it cost ~0.3ms @16384 in testing).
        stage_sync(1);
        if (k_tile + 3 < k_tiles) {
            LOAD_A_TILE_PAIR_HT(
                a_lds_base_addr + (((k_tile + 3) % kAStages) * kAStageBytes),
                (k_tile + 3) * BLOCK_K);
        }
        COMPUTE_TILE_HT(k_tile + 1, B_ODD_AT_HT);
        if (k_tile + 3 < k_tiles) {
            LOAD_B_TILE_HT(k_tile + 3, B_ODD_AT_HT);
        }
    }

    // =========================================================================
    // Epilogue: drain the last 1-2 tiles (loads already issued)
    // =========================================================================
    {
        const int remaining_tiles = k_tiles - k_tile;
        if (remaining_tiles >= 1) {
            if (remaining_tiles == 2) {
                stage_sync(0);  // tile k_tile+1's loads may still be in flight
            } else {
                stage_sync_drain();
            }
            COMPUTE_TILE_HT(k_tile, B_EVEN_AT_HT);
            ++k_tile;
            if (k_tile < k_tiles) {
                stage_sync_drain();
                COMPUTE_TILE_HT(k_tile, B_ODD_AT_HT);
            }
        }
    }

    // =========================================================================
    // Channel-wise dequant and output: acc * input_scale * weight_scale ?BF16
    // =========================================================================
    // Load col_scale per-fragment to reduce VGPR pressure (4 vs 8 VGPRs)

    // Check if full block is valid
    const bool full_block_valid =
        (m_block_start + BLOCK_M <= M) && (valid_m >= BLOCK_M);

    if (full_block_valid) {
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            const int row = m_block_start + m_loop * kMmacRows + lane_row;
            const float row_scale = matrixA_scale[row];
            #pragma unroll
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
                const int col_base = n_block_start +
                                     warp_id * kNFragments * kMmacCols +
                                     n_frag * kMmacCols + lane_group * kMmacVgprs;
                float col_scale[kMmacVgprs];
                #pragma unroll
                for (int elem = 0; elem < kMmacVgprs; ++elem) {
                    col_scale[elem] = matrixB_scale[col_base + elem];
                }
                const float v0 = MmaT::to_float(acc[m_loop][n_frag][0]) * row_scale * col_scale[0];
                const float v1 = MmaT::to_float(acc[m_loop][n_frag][1]) * row_scale * col_scale[1];
                const float v2 = MmaT::to_float(acc[m_loop][n_frag][2]) * row_scale * col_scale[2];
                const float v3 = MmaT::to_float(acc[m_loop][n_frag][3]) * row_scale * col_scale[3];
                *reinterpret_cast<uint64_t*>(
                    MatrixC + static_cast<int64_t>(row) * C_stride + col_base) =
                    pack_bf16x4_ht(v0, v1, v2, v3);
            }
        }
    } else {
        // Boundary handling: check each row
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            const int row = m_block_start + m_loop * kMmacRows + lane_row;
            if (row >= M || (int)(m_loop * kMmacRows + lane_row) >= valid_m) {
                continue;
            }
            const float row_scale = matrixA_scale[row];
            #pragma unroll
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
                const int col_base = n_block_start +
                                     warp_id * kNFragments * kMmacCols +
                                     n_frag * kMmacCols + lane_group * kMmacVgprs;
                float col_scale[kMmacVgprs];
                #pragma unroll
                for (int elem = 0; elem < kMmacVgprs; ++elem) {
                    col_scale[elem] = matrixB_scale[col_base + elem];
                }
                const float v0 = MmaT::to_float(acc[m_loop][n_frag][0]) * row_scale * col_scale[0];
                const float v1 = MmaT::to_float(acc[m_loop][n_frag][1]) * row_scale * col_scale[1];
                const float v2 = MmaT::to_float(acc[m_loop][n_frag][2]) * row_scale * col_scale[2];
                const float v3 = MmaT::to_float(acc[m_loop][n_frag][3]) * row_scale * col_scale[3];
                *reinterpret_cast<uint64_t*>(
                    MatrixC + static_cast<int64_t>(row) * C_stride + col_base) =
                    pack_bf16x4_ht(v0, v1, v2, v3);
            }
        }
    }
    #undef B_EVEN_AT_HT
    #undef B_ODD_AT_HT
    #undef LOAD_A_TILE_PAIR_HT
    #undef LOAD_B_TILE_HT
    #undef COMPUTE_TILE_HT
}

// ===========================================================================
// Fused variant of gemm_ht_tile_bm128_bk64 for the HT L1 GEMM: SwiGLU +
// group-wise FP8 quantization folded into the GEMM epilogue (FP8 only).
//
// The Marlin weight layout interleaves gate/up at granularity 8: each
// 16-col MMA fragment holds [gate[0..7] | up[0..7]], so lane_group 0/1 hold
// gate cols and lane_group 2/3 hold the paired up cols of the same row.
// silu(g)*u is computed by exchanging fragments with __shfl_xor(...,32);
// both partner lanes compute the same product redundantly.
//
// Each call covers BLOCK_N raw cols -> BLOCK_N/2 SwiGLU cols starting at
// out_base = tile_n * (BLOCK_N/2).  One group of BLOCK_N/2 output cols
// shares one amax/scale: NUM_WARPS*kNFragments*8 == BLOCK_N/2 holds, so a
// group always spans exactly all NUM_WARPS warps of the block.
//   - BN256 (or split BN512 call): one call = one 128-col group
//   - BN512 direct: one call = one 256-col group
//
// Contains exactly ONE s_barrier (row-amax cross-warp reduction), on top of
// the K/BLOCK_K stage barriers 鈥?callers must mirror both counts in warps
// that skip this function.
//
// Args (in addition to gemm_ht_tile_bm128_bk64):
//   l2_token_out: [pool_tokens, ih] FP8 output base (row 0)
//   l2_sf:        [pool_tokens, sf_stride] group scale output base (row 0)
//   m_idx:        pool-buffer row offset of this tile
//   ih:           row stride of l2_token_out (intermediate_hidden)
//   sf_stride:    floats per l2_sf row (ih / GROUP_SIZE)
//   smem_amax:    scratch [NUM_WARPS * BLOCK_M] floats (caller-allocated)
// ===========================================================================
template <int32_t N, int K,
          int BLOCK_M = 128, int BLOCK_N = 256, int BLOCK_K = 64,
          int NUM_WARPS = 8, bool kIsInt8 = false,
          bool kAllWaveBarrier = false, int kAStagesArg = 3,
          int kGroupSizeArg = 0, bool kAbsNColBaseArg = false,
          bool kRegOptArg = false>
__device__ void gemm_ht_tile_bm128_bk64_silu_quant(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    const int32_t M,
    const uint32_t math_warp_offset,
    uint8_t* __restrict__ l2_token_out,
    float* __restrict__ l2_sf,
    const int32_t m_idx,
    const int32_t sf_stride,
    const int32_t ih,
    float* __restrict__ smem_amax,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense,
    const float activation_clamp,
    const bool fast_math,
    const int32_t n_col_base = 0)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    // Non-uniform N-fragment split (examples at kNFragments below): a tile whose
    // width does not divide by NUM_WARPS * 16 still covers every column -- the
    // lowest warp ids simply take one fragment more than the rest.
    constexpr int kTotalFrags = BLOCK_N / kMmacCols;
    constexpr int kBaseFrags = kTotalFrags / NUM_WARPS;
    constexpr int kExtraFrags = kTotalFrags % NUM_WARPS;
    constexpr bool kNonUniformFrags = (kExtraFrags != 0);
    static_assert(BLOCK_M % kMmacRows == 0 &&
                  (kNonUniformFrags || BLOCK_N % (NUM_WARPS * kMmacCols) == 0) &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;  // 8
    constexpr int kK64Tile = 64;
    constexpr int kKLoops = BLOCK_K / 32;            // 2
    constexpr int kK64Loops = BLOCK_K / kK64Tile;    // 1
    constexpr int kNFragments = BLOCK_N / (NUM_WARPS * kMmacCols);  // 2 or 4
    // Per-warp upper bound of the split.  The 256-wide L1 tail on 12 warps has
    // 256 / 16 = 16 fragments over 12 warps: warp 0..3 takes 2 (16 out cols),
    // warp 4..11 takes 1 (8 out cols) -> 4*16 + 8*8 = 128 = one quant group.
    // Uniform tiles have kExtraFrags == 0, so every use below folds back to the
    // original constant and their codegen stays bit-identical.
    constexpr int kNFragMax = kNonUniformFrags ? (kBaseFrags + 1) : kNFragments;
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;
    constexpr int kAStages = kAStagesArg;
    // A 4-N-fragment tile keeps acc[8][4] = 128 dword live; under the
    // 1024-thread launch (192 VGPR cap) that exhausts the budget.  Three
    // rotation slots still give one full iteration of prefetch margin
    // ((m+2)%3 != m%3) and free 8 dword of live VGPR.
    constexpr bool kRegOptTile = kRegOptArg && (kNFragMax >= 4);
    constexpr int kASlots = (kRegOptTile && (kRegOptMask & 1)) ? 3 : 4;
    static_assert(kASlots >= 3, "prefetch distance 2 needs >= 3 A-frag slots");
    constexpr int kALdsTotalBytes = kAStages * kAStageBytes;
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;
    constexpr int kALdsRowPairStride = 128;

    static_assert(kMSubtiles >= 4, "expects at least 4 M-subtiles (BM>=64)");
    static_assert(kNFragMax >= 1 && kNFragMax <= 4,
                  "BN256/512 expects at most 4 N-fragments per warp");
    static_assert(kK64Loops == 1, "BK64 expects 1 K64 loop");
    static_assert(kNonUniformFrags
                      ? (kTotalFrags * 8 == BLOCK_N / 2 && kBaseFrags >= 1)
                      : (NUM_WARPS * kNFragments * 8 == BLOCK_N / 2),
                  "warps must exactly cover the SwiGLU output width");
    // Group size = SwiGLU output width covered by one amax/scale.  Defaults to
    // one group per call (BLOCK_N/2).  A caller may pass a smaller value so a
    // single tile emits several groups -- required when the L1 tile is 768
    // wide but `intermediate_hidden` only admits a 128-channel group
    // (hy4: 2048 % 384 != 0, so the 12-wave tile must write 3 groups of 128).
    constexpr int kGS = (kGroupSizeArg != 0) ? kGroupSizeArg : (BLOCK_N / 2);

    // Stage barrier for the 3-stage pipeline.  s_barrier only waits for
    // wave arrival -- it does NOT drain the wait counters -- so the
    // explicit vmcnt(kLoadsPerTile) keeps tile k+1's A matrix_load + B
    // buffer_loads in flight across the barrier, giving the pipeline real
    // depth (same wait_vmcntN_barrier pattern as the grouped-GEMM
    // inc.cuh kernels).  stage_sync_drain() is for the last tile, whose
    // own loads must all be resident before compute.
    constexpr int kALoadsPerWarp =
        (kMSubtiles + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int kLoadsPerTile = kALoadsPerWarp + kNFragments;
    // Per-warp vmcnt budget: the allowance must match the number of
    // loads this warp itself issued last iteration.  With
    // NUM_WARPS > kMSubtiles some warps skip their A matrix_load, so
    // kLoadsPerTile would leave one extra slot of slack -- a B
    // buffer_load straggler from the previous tile could then stay in
    // flight across the barrier and feed stale registers to mma.
    // n_a_issued: A matrix_loads issued by an A-loading warp in the
    // half-iteration that just elapsed (0 or 2 per k-tile); the
    // allowance keeps exactly that half's loads (A + B fragments) in
    // flight.
    auto stage_sync = [&](int n_a_issued) {
        const int my_wid = (threadIdx.x - math_warp_offset) / kWarpSize;
        const bool full_a =
            (my_wid * kMmacRows < BLOCK_M) &&
            (kALoadsPerWarp == 1 ||
             my_wid * kMmacRows + NUM_WARPS * kMmacRows < BLOCK_M);
        if constexpr (kNonUniformFrags) {
            // Non-uniform split: a warp owns kBaseFrags or kBaseFrags + 1 B
            // fragments, so its vmcnt allowance must match its own count --
            // one spare slot would let a stale B load survive the barrier and
            // feed the next tile's mma.
            const int b_frags = kBaseFrags + ((my_wid < kExtraFrags) ? 1 : 0);
            if (b_frags > kBaseFrags) {
                if constexpr (kAllWaveBarrier) {
                    if (full_a && n_a_issued)
                        wait_vmcnt_all_barrier_ht<kALoadsPerWarp + kBaseFrags + 1>();
                    else
                        wait_vmcnt_all_barrier_ht<kBaseFrags + 1>();
                } else {
                    if (full_a && n_a_issued)
                        wait_vmcnt_subset_barrier_ht<kALoadsPerWarp + kBaseFrags + 1>(
                            p_bar_count, p_bar_sense, NUM_WARPS);
                    else
                        wait_vmcnt_subset_barrier_ht<kBaseFrags + 1>(
                            p_bar_count, p_bar_sense, NUM_WARPS);
                }
            } else {
                if constexpr (kAllWaveBarrier) {
                    if (full_a && n_a_issued)
                        wait_vmcnt_all_barrier_ht<kALoadsPerWarp + kBaseFrags>();
                    else
                        wait_vmcnt_all_barrier_ht<kBaseFrags>();
                } else {
                    if (full_a && n_a_issued)
                        wait_vmcnt_subset_barrier_ht<kALoadsPerWarp + kBaseFrags>(
                            p_bar_count, p_bar_sense, NUM_WARPS);
                    else
                        wait_vmcnt_subset_barrier_ht<kBaseFrags>(
                            p_bar_count, p_bar_sense, NUM_WARPS);
                }
            }
        } else if constexpr (kAllWaveBarrier) {
            if (full_a && n_a_issued)
                wait_vmcnt_all_barrier_ht<kALoadsPerWarp + kNFragments>();
            else
                wait_vmcnt_all_barrier_ht<kNFragments>();
        } else {
            if (full_a && n_a_issued)
                wait_vmcnt_subset_barrier_ht<kALoadsPerWarp + kNFragments>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
            else
                wait_vmcnt_subset_barrier_ht<kNFragments>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };
    auto stage_sync_drain = [&]() {
        if constexpr (kAllWaveBarrier) {
            wait_vmcnt_all_barrier_ht<0>();
        } else {
            wait_vmcnt_subset_barrier_ht<0>(
                p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };

    // Split M-subtiles into two halves to reduce peak VGPR pressure.
    constexpr int kMSubtilesHalf = kMSubtiles / 2;  // 4

    // Remap thread index for math warps
    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;

    // Mixed L1 tiling passes an absolute column base: its tile_n stays inside
    // the scheduler's n-block range while the B columns still land on the
    // trailing segment.  Uniform tiles fold this straight back to the original
    // expression (kAbsNColBaseArg is a compile-time constant).
    const int n_block_start = kAbsNColBaseArg ? n_col_base : (tile_n * BLOCK_N);
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    // Fragments this warp owns.  Uniform tiles fold this to kNFragments (a
    // compile-time constant), so every loop below keeps its original trip
    // count; only the non-uniform tail evaluates the split at runtime.
    const int my_frags = kNonUniformFrags
        ? (kBaseFrags + ((warp_id < kExtraFrags) ? 1 : 0))
        : kNFragments;
    // First fragment index of this warp (16 raw columns each), the SwiGLU
    // output column base it owns (8 out cols per fragment) and its width.
    const int my_frag_start = kNonUniformFrags
        ? ((warp_id < kExtraFrags)
               ? warp_id * (kBaseFrags + 1)
               : kExtraFrags * (kBaseFrags + 1) +
                     (warp_id - kExtraFrags) * kBaseFrags)
        : warp_id * kNFragments;
    const int my_out_base = my_frag_start * 8;

    // A LDS addresses
    constexpr int kNumALoadsPerWarp = (kMSubtiles + NUM_WARPS - 1) / NUM_WARPS;
    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m0 = warp_id * kMmacRows;
    const int a_load_m1 = a_load_m0 + NUM_WARPS * kMmacRows;
    const int a_lds_write_bytes0 = a_load_m0 * BLOCK_K;
    const int a_lds_write_bytes1 = a_load_m1 * BLOCK_K;
    const int a_lds_write_bytes = a_lds_write_bytes0;

    // N base for this warp
    const int logical_n_base =
        n_block_start + my_frag_start * kMmacCols + (lane_id & 15);

    // B byte offset is affine in the K-tile index: b_k = KT*64+lane_group*16
    // stays inside one 64-row band, so b_k_band==KT, b_k_mid==lane_group and
    // b_k_inner==0.  Hoist the per-fragment constant out of the K loop --
    // each buffer_load then costs one IADD instead of ~10 ALU ops.
    const int b_stride_bytes = b_n_tiles * 4 * kMmacCols * 16;
    // Per-frag byte offset is affine: logical_n advances exactly one
    // 16-col group per fragment, so b_off(f) = b_off0 + f*(4*16*16).
    // One scalar instead of a kNFragments table (the f*step term folds
    // into the load's immediate add).
    const int b_off_base0 = expert_b_base +
        ((logical_n_base / kMmacCols) * 4 + lane_group) * (kMmacCols * 16) +
        (logical_n_base & (kMmacCols - 1)) * 16;
    constexpr int kBOffFragStep = 4 * kMmacCols * 16;

    // A1: descriptor is invariant over the K loop.  expert_b_base stays
    // inside b_off_base0 -- adding it here too would double-count it.
    ht_rsrc_t b_rsrc = make_buffer_resource_ht(matrixB_w6);

    // Accumulators.  The non-uniform split sizes this by the per-warp maximum,
    // so the light warps carry one idle fragment's worth of VGPRs (8 for BM128).
    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragMax];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragMax; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    // LDS read parameters
    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    // B register quad-buffer: two sets per parity so the next phase's B
    // tiles issue at the top of the current one (lead ~1 full iteration
    // instead of ~half).  Named sets keep the selector compile-time
    // constant (a runtime index would spill to local memory).
    BOperandHT b_even0, b_even1, b_even2, b_even3;
    BOperandHT b_odd0, b_odd1, b_odd2, b_odd3;

    #define B_EVEN_AT_HT(IDX) \
        ((IDX) == 0 ? b_even0 : (IDX) == 1 ? b_even1 : (IDX) == 2 ? b_even2 : b_even3)
    #define B_ODD_AT_HT(IDX) \
        ((IDX) == 0 ? b_odd0 : (IDX) == 1 ? b_odd1 : (IDX) == 2 ? b_odd2 : b_odd3)

    const char* a_ptr_base0 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m0) * K;
    const char* a_ptr_base1 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m1) * K;

    #define LOAD_A_TILE_PAIR_HT(STAGE_BASE, K_OFFSET) \
        do { \
            if (a_load_m0 < BLOCK_M) { \
                const char* a0 = a_ptr_base0 + static_cast<int64_t>(K_OFFSET); \
                vec4_u32 srsrc0 = make_matrix_resource_ht(a0, K); \
                MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes0, srsrc0); \
            } \
            if constexpr (kNumALoadsPerWarp > 1) { \
                if (a_load_m1 < BLOCK_M) { \
                    const char* a1 = a_ptr_base1 + static_cast<int64_t>(K_OFFSET); \
                    vec4_u32 srsrc1 = make_matrix_resource_ht(a1, K); \
                    MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes1, srsrc1); \
                } \
            } \
        } while (0)

    // Fragment trip count is kNFragMax with a guard that folds away on
    // uniform tiles (kNonUniformFrags == false), so their codegen is
    // unchanged; the non-uniform tail only issues the fragments its warp owns.
    #define LOAD_B_TILE_HT(KT, B_SEL)                                          \
        do {                                                                   \
            _Pragma("unroll")                                                  \
            for (int n_frag = 0; n_frag < kNFragMax; ++n_frag) {               \
                if (!kNonUniformFrags || n_frag < my_frags) {                  \
                    buffer_load_i32x4_rsrc_ht(b_rsrc, B_SEL(n_frag).i32x4,      \
                                         b_off_base0 +                         \
                                         (n_frag) * kBOffFragStep +           \
                                         (KT) * b_stride_bytes);               \
                }                                                              \
            }                                                                  \
        } while (0)

    // Compute one K-tile: stage (KT % kAStages), B regs selected by B_SEL.
    // 4-slot A-frag rotation with prefetch distance 2 + lgkmcnt(1): the read
    // for subtile m+2 stays in flight while the mmas for m issue, hiding
    // ~30cyc of LDS latency that the old lgkmcnt(0) ping-pong exposed on
    // every subtile.  Slot (m+2)&3 was last read by mma(m-2), two full
    // iterations back, so the DS_READ writeback cannot collide with an
    // in-flight operand fetch (the failed 3-slot variant only had one
    // iteration of margin).
    #define COMPUTE_TILE_HT(KT, B_SEL)                                         \
        do {                                                                   \
            const int a_stage_base_c =                                         \
                a_lds_base_addr + (((KT) % kAStages) * kAStageBytes);          \
            AOperandHT a_regs[kASlots];                                              \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 0, a_regs[0]);           \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 1, a_regs[1]);           \
            wait_lgkmcnt_one_ht();                                             \
            wait_vmcnt7_zero_ht();                                             \
            _Pragma("unroll")                                                  \
            for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {              \
                const int compute_slot = m_loop % kASlots;                           \
                if (m_loop + 2 < kMSubtiles) {                                 \
                    DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, m_loop + 2,      \
                                              a_regs[(m_loop + 2) % kASlots]);       \
                }                                                              \
                _Pragma("unroll")                                              \
                for (int n_frag = 0; n_frag < kNFragMax; ++n_frag) {           \
                    if (!kNonUniformFrags || n_frag < my_frags) {              \
                        const BOperandHT& b_reg = B_SEL(n_frag);               \
                        acc[m_loop][n_frag] = MmaT::mma(                       \
                            a_regs[compute_slot].fp8x8[0], b_reg.fp8x8[0],     \
                            acc[m_loop][n_frag]);                              \
                        acc[m_loop][n_frag] = MmaT::mma(                       \
                            a_regs[compute_slot].fp8x8[1], b_reg.fp8x8[1],     \
                            acc[m_loop][n_frag]);                              \
                    }                                                          \
                }                                                              \
                if (m_loop + 2 < kMSubtiles) {                                 \
                    wait_lgkmcnt_one_ht();                                     \
                } else if (m_loop + 1 < kMSubtiles) {                          \
                    wait_lgkmcnt_zero_ht();                                    \
                }                                                              \
            }                                                                  \
        } while (0)

    // =========================================================================
    // Prologue: tiles 0/1 -> stages 0/1 (depth-2 prefetch), B0/B1 -> regs
    // =========================================================================
    {
        LOAD_A_TILE_PAIR_HT(a_lds_base_addr, 0);
        LOAD_B_TILE_HT(0, B_EVEN_AT_HT);
        if (k_tiles > 1) {
            LOAD_A_TILE_PAIR_HT(a_lds_base_addr + kAStageBytes, BLOCK_K);
            LOAD_B_TILE_HT(1, B_ODD_AT_HT);
        }
    }

    // =========================================================================
    // Main loop: 4-stage A LDS pipeline (kAStages), one stage barrier per
    // half-iteration.  Baseline ordering: A(k+2) issues in the alpha half,
    // A(k+3) after the beta barrier (its stage's last reader is then
    // ordered for both 3- and 4-stage layouts).  B(k+2) goes into the
    // just-consumed register slot after compute.  vmcnt(N) at the barrier
    // keeps the half's own loads in flight (s_barrier alone does not
    // drain the counters).
    // =========================================================================
    int k_tile = 0;
    for (; k_tile + 2 < k_tiles; k_tile += 2) {
        stage_sync(1);
        LOAD_A_TILE_PAIR_HT(
            a_lds_base_addr + (((k_tile + 2) % kAStages) * kAStageBytes),
            (k_tile + 2) * BLOCK_K);
        COMPUTE_TILE_HT(k_tile, B_EVEN_AT_HT);
        LOAD_B_TILE_HT(k_tile + 2, B_EVEN_AT_HT);

        // Beta barrier: required for 3 stages and measurably better for 4
        // too -- the vmcnt wait forces B(k+1) stragglers and A(k+2) to
        // land (dropping it cost ~0.3ms @16384 in testing).
        stage_sync(1);
        if (k_tile + 3 < k_tiles) {
            LOAD_A_TILE_PAIR_HT(
                a_lds_base_addr + (((k_tile + 3) % kAStages) * kAStageBytes),
                (k_tile + 3) * BLOCK_K);
        }
        COMPUTE_TILE_HT(k_tile + 1, B_ODD_AT_HT);
        if (k_tile + 3 < k_tiles) {
            LOAD_B_TILE_HT(k_tile + 3, B_ODD_AT_HT);
        }
    }

    // =========================================================================
    // Epilogue: drain the last 1-2 tiles (loads already issued)
    // =========================================================================
    {
        const int remaining_tiles = k_tiles - k_tile;
        if (remaining_tiles >= 1) {
            if (remaining_tiles == 2) {
                stage_sync(1);  // tile k_tile+1's loads may still be in flight
            } else {
                stage_sync_drain();
            }
            COMPUTE_TILE_HT(k_tile, B_EVEN_AT_HT);
            ++k_tile;
            if (k_tile < k_tiles) {
                stage_sync_drain();
                COMPUTE_TILE_HT(k_tile, B_ODD_AT_HT);
            }
        }
    }

    // =========================================================================
    // Fused SwiGLU + group-wise FP8 quant epilogue
    // -------------------------------------------------------------------------
    // Fragment (m_loop, n_frag): row = m_block_start + m_loop*16 + lane_row,
    // raw cols warp_id*kNFragments*16 + n_frag*16 + lane_group*4 + [0..3].
    // Granularity-8 interleave: lane_group<2 -> gate, >=2 -> paired up.
    // SwiGLU out col of raw col c: (c/16)*8 + c%8 -> warp w covers out cols
    // [warp_id*kNFragments*8, warp_id*kNFragments*8 + kNFragments*8).
    // =========================================================================
    constexpr int kOutPerWarp = kNFragments * 8;
    static_assert(kNonUniformFrags || (kGS % kOutPerWarp == 0),
                  "quant group must cover a whole number of warps");
    // A non-uniform tile emits exactly one group over all NUM_WARPS warps, so
    // heavy and light warps take part in the same amax reduction.
    static_assert(!kNonUniformFrags || (kGS == kTotalFrags * 8),
                  "non-uniform split requires the tile to emit exactly one group");
    constexpr int kWarpsPerGroup =
        kNonUniformFrags ? static_cast<int>(NUM_WARPS) : (kGS / kOutPerWarp);
    static_assert(kWarpsPerGroup <= static_cast<int>(NUM_WARPS),
                  "one quant group must not span more warps than the tile has");
    const int out_base = kAbsNColBaseArg
        ? (n_col_base / 2) : (tile_n * (BLOCK_N / 2));
    const int group_warp_base = (warp_id / kWarpsPerGroup) * kWarpsPerGroup;
    // One group per kWarpsPerGroup warps: when kGS == BLOCK_N/2 that is every
    // warp and group_idx == tile_n; a smaller kGS makes the tile emit several
    // groups (hy4 L1 12-wave: 768-wide tile -> 3 groups of 128).
    // The single-group form keeps the exact pre-L1W12 expression so the
    // production epilogue generates identical code (no extra
    // warp_id/kWarpsPerGroup divide, which the compiler cannot fold because it
    // does not know warp_id < NUM_WARPS).
    int group_idx;
    if constexpr (kWarpsPerGroup == static_cast<int>(NUM_WARPS)) {
        group_idx = out_base / kGS;
    } else {
        group_idx = out_base / kGS + warp_id / kWarpsPerGroup;
    }

    // Out-col base of this lane within its warp: lg0->0, lg1->4, lg2/3 mirror
    // lg0/lg1 (they recompute the same SwiGLU values for the other half).

    // Phase 1: dequant + SwiGLU, keep BF16-packed silu values in registers
    // (silu_pk, kMSubtiles*kNFragments u64 = 64 regs for BN512) instead of a
    // GMEM roundtrip 鈥?same scheme as the LL fused epilogue.  acc dies
    // progressively as each (m,n) fragment is consumed; the packed values
    // carry the same bf16 precision the GMEM staging used.  N-fragments are
    // processed in two halves so acc[m][first-half] dies early.
    constexpr int kFragHalf = (kNFragMax + 1) / 2;
    // Row-round count of the fused epilogue.  2 (historical) keeps
    // silu_pk[kMSubtiles/2][kNFragments] live (32 VGPR for BM128/BN512); 4
    // cuts that to 16 VGPR -- the headroom that removes the last VGPR spills
    // of the 12-wave instantiation -- at the cost of 2 extra row-amax
    // barriers per tile.
    // Four row rounds are only safe when the kernel has no mirror warps:
    // mega_moe_kernel_unified.cuh counts exactly two row-amax barriers per
    // L1 call in its mirror loop.  8-math-warp layouts (BN512 and the L1
    // tail) keep two rounds even when bit 1 is set.
    constexpr int kEpilogueMRounds = (kMSubtiles <= 4) ? 1
        : ((kRegOptArg && (kRegOptMask & 2)) ? 4 : 2);
    // Process the epilogue in two row-half rounds when the tile
    // has many M-subtiles: each round keeps only
    // kMHalf*kNFragments packed BF16 silu values live
    // (silu_pk[4][4] = 32 regs for BM128/BN512 instead of 64),
    // roughly halving epilogue peak VGPR pressure at the cost of
    // one extra row-amax barrier per tile.  Tiles with <=4
    // M-subtiles (BM64) keep a single round.
    constexpr int kMHalf = (kMSubtiles > 4) ? (kMSubtiles / kEpilogueMRounds) : kMSubtiles;
    // smem_amax is overlaid on the (dead) a_lds region when NUM_WARPS >
    // kMSubtiles or the caller uses a 4-stage pipeline -- wait for every
    // warp's stage reads to land before writing it.
    if constexpr (NUM_WARPS > kMSubtiles || kAStages > 3) {
        if constexpr (kAllWaveBarrier) {
            wait_lgkmcnt0_all_barrier_ht();
        } else {
            wait_lgkmcnt0_subset_barrier_ht(p_bar_count, p_bar_sense, NUM_WARPS);
        }
    }
    // Opaque fence: derived per-row address tables ((m_idx+row)*ih,
    // row*K, (m_idx+row)*sf) must rematerialize inside the epilogue
    // instead of living across the whole K loop as early-hoisted
    // temporaries.  '+s' keeps the pointers in SGPRs (they are uniform)
    // while making them opaque so SCEV cannot pre-form the row table.
    asm volatile("" : "+s"(l2_token_out), "+s"(l2_sf), "+s"(matrixA_scale),
                      "+s"(matrixB_scale));
    #pragma unroll
    for (int mh = 0; mh < (kMSubtiles + kMHalf - 1) / kMHalf; ++mh) {
    // Phase 1: dequant + SwiGLU -> packed BF16 silu values in regs.
    float part[kMHalf];
    uint64_t silu_pk[kMHalf][kNFragMax];
    #pragma unroll
    for (int m_loc = 0; m_loc < kMHalf; ++m_loc) part[m_loc] = 0.0f;
    #pragma unroll
    for (int half = 0; half * kFragHalf < kNFragMax; ++half) {
        #pragma unroll
        for (int m_loc = 0; m_loc < kMHalf; ++m_loc) {
            const int m_loop = mh * kMHalf + m_loc;
            if (m_loop >= kMSubtiles) break;
            const int row_local = m_loop * kMmacRows + lane_row;
            const int row = m_block_start + row_local;
            const bool row_ok = (row_local < valid_m) && (row < M);
            const float row_scale = row_ok ? matrixA_scale[row] : 0.0f;
            #pragma unroll
            for (int n_frag = half * kFragHalf;
                 n_frag < min((half + 1) * kFragHalf, kNFragMax); ++n_frag) {
                if (kNonUniformFrags && n_frag >= my_frags) break;
                const int col_base = n_block_start +
                    my_frag_start * kMmacCols +
                    n_frag * kMmacCols + lane_group * kMmacVgprs;
                uint64_t sv;
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; ++i) {
                    const float v = MmaT::to_float(acc[m_loop][n_frag][i])
                        * row_scale * matrixB_scale[col_base + i];
                    const float partner = __shfl_xor(v, 32);
                    // lg<2: v=gate, partner=up; lg>=2: v=up, partner=gate.
                    const float g = (lane_group < 2) ? v : partner;
                    const float u = (lane_group < 2) ? partner : v;
                    const float s = swiglu_nv(g, u, activation_clamp, fast_math);
                    part[m_loc] = fmaxf(part[m_loc], fabsf(s));
                    reinterpret_cast<hip_bfloat16*>(&sv)[i] = hip_bfloat16(s);
                }
                silu_pk[m_loc][n_frag] = sv;
            }
        }
    }
    #pragma unroll
    for (int m_loc = 0; m_loc < kMHalf; ++m_loc) {
        const int m_loop = mh * kMHalf + m_loc;
        if (m_loop >= kMSubtiles) break;
        const int row_local = m_loop * kMmacRows + lane_row;
        const int row = m_block_start + row_local;
        const bool row_ok = (row_local < valid_m) && (row < M);
        float p = row_ok ? part[m_loc] : 0.0f;
        // Both partner lanes produce identical silu values; row amax over the
        // warp's fragment cols needs only the {r, r+16, r+32, r+48} quad.
        p = fmaxf(p, __shfl_xor(p, 16));
        p = fmaxf(p, __shfl_xor(p, 32));
        if (lane_id < 16) {
            // Transposed [row][warp] layout: the cross-warp reduction
            // below then reads a contiguous float run (vectorizable)
            // instead of a strided gather over warp-major slots.
            smem_amax[row_local * NUM_WARPS + warp_id] = p;
        }
    }

    // Row-amax cross-warp reduction.  lgkmcnt(0) drains the smem_amax
    // ds_writes; the silu values themselves stay in registers.  With the
    // all-wave s_barrier non-math warps mirror this call; with the subset
    // barrier only the math warps participate.
    if constexpr (kAllWaveBarrier) {
        wait_lgkmcnt0_all_barrier_ht();
    } else {
        wait_lgkmcnt0_subset_barrier_ht(p_bar_count, p_bar_sense, NUM_WARPS);
    }

    // Phase 2: quantize the register-resident silu values, pack, repack to
    // 16B stores.  Accumulators are dead here -- only silu_pk, scales and
    // addresses stay live.
#ifndef DG_MEGA_L1_NOQUANT
    #pragma unroll
    for (int m_loc = 0; m_loc < kMHalf; ++m_loc) {
        const int m_loop = mh * kMHalf + m_loc;
        if (m_loop >= kMSubtiles) break;
        const int row_local = m_loop * kMmacRows + lane_row;
        const int row = m_block_start + row_local;
        const bool row_ok = (row_local < valid_m) && (row < M);
        float row_amax = 0.0f;
        static_assert(kWarpsPerGroup % 4 == 0,
                      "row-amax vectorized read needs groups of 4 warps");
        #pragma unroll
        for (int w = 0; w < kWarpsPerGroup; w += 4) {
            const float4 a4 = *reinterpret_cast<const float4*>(
                &smem_amax[row_local * NUM_WARPS + group_warp_base + w]);
            row_amax = fmaxf(row_amax,
                fmaxf(fmaxf(a4.x, a4.y), fmaxf(a4.z, a4.w)));
        }
        const float qmax = kIsInt8 ? 127.0f : 448.0f;
        const float clamped_amax = fmaxf(row_amax, kIsInt8 ? 1.0e-6f : 1.0e-4f);
        const float scale = clamped_amax / qmax;
        const float inv_scale = qmax / clamped_amax;

        uint32_t packed[kNFragMax];
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragMax; ++n_frag) {
            if (kNonUniformFrags && n_frag >= my_frags) break;
            const hip_bfloat16* svb =
                reinterpret_cast<const hip_bfloat16*>(&silu_pk[m_loc][n_frag]);
            float q[kMmacVgprs];
            #pragma unroll
            for (int i = 0; i < kMmacVgprs; ++i) {
                const float s = static_cast<float>(svb[i]);
                q[i] = quant_clamp<kIsInt8>(s * inv_scale, qmax);
            }
            uint32_t p = 0;
            if constexpr (kIsInt8) {
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; ++i) {
                    p |= (static_cast<uint32_t>(static_cast<uint8_t>(
                              static_cast<int8_t>(__float2int_rn(q[i]))))
                          << (i * 8));
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; i += 2) {
                    float2 f2{q[i], q[i + 1]};
                    auto fp8x2 = __hip_cvt_float2_to_fp8x2(
                        f2, __HIP_SATFINITE, __HIP_E4M3);
                    p |= (static_cast<uint32_t>(
                              *reinterpret_cast<uint16_t*>(&fp8x2)) << (i * 8));
                }
            }
            packed[n_frag] = p;
        }

        // Repack: the warp's out cols are spread over lanes r/r+16 (frag
        // pairs).  A two-fragment warp gathers one 16B chunk, the light warps
        // of a non-uniform split own 8 columns and emit one 8B store.  All
        // indices stay compile-time so packed[] never spills to local memory.
        if (row_ok) {
            if constexpr (kNonUniformFrags) {
                #pragma unroll
                for (int pair = 0; pair < kNFragMax / 2; ++pair) {
                    if (pair * 2 + 1 < my_frags) {
                        uint4 v4;
                        v4.x = __shfl(packed[pair * 2 + 0], lane_id);
                        v4.y = __shfl(packed[pair * 2 + 0], lane_id + 16);
                        v4.z = __shfl(packed[pair * 2 + 1], lane_id);
                        v4.w = __shfl(packed[pair * 2 + 1], lane_id + 16);
                        if (lane_id < 16) {
                            *reinterpret_cast<uint4*>(
                                l2_token_out +
                                static_cast<int64_t>(m_idx + row_local) * ih +
                                out_base + my_out_base + pair * 16) = v4;
                        }
                    }
                }
                #pragma unroll
                for (int f = 0; f < kNFragMax; ++f) {
                    if ((my_frags & 1) && f == my_frags - 1) {
                        uint2 v2;
                        v2.x = __shfl(packed[f], lane_id);
                        v2.y = __shfl(packed[f], lane_id + 16);
                        if (lane_id < 16) {
                            *reinterpret_cast<uint2*>(
                                l2_token_out +
                                static_cast<int64_t>(m_idx + row_local) * ih +
                                out_base + my_out_base + f * 8) = v2;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int chunk = 0; chunk < kOutPerWarp / 16; ++chunk) {
                    uint4 v4;
                    v4.x = __shfl(packed[chunk * 2 + 0], lane_id);
                    v4.y = __shfl(packed[chunk * 2 + 0], lane_id + 16);
                    v4.z = __shfl(packed[chunk * 2 + 1], lane_id);
                    v4.w = __shfl(packed[chunk * 2 + 1], lane_id + 16);
                    if (lane_id < 16) {
                        *reinterpret_cast<uint4*>(
                            l2_token_out +
                            static_cast<int64_t>(m_idx + row_local) * ih +
                            out_base + warp_id * kOutPerWarp + chunk * 16) = v4;
                    }
                }
            }
        }
        // One lane per row per group writes the scale (all lanes of the
        // group share row_amax; guard to the first warp of the group).
        if (lane_id < 16 && row_ok &&
            warp_id % kWarpsPerGroup == 0) {
            l2_sf[static_cast<int64_t>(m_idx + row_local) * sf_stride
                  + group_idx] = scale;
        }
    }
#else
    (void)silu_pk; (void)group_warp_base; (void)kWarpsPerGroup;
#endif
    }  // mh round

    #undef B_EVEN_AT_HT
    #undef B_ODD_AT_HT
    #undef LOAD_A_TILE_PAIR_HT
    #undef LOAD_B_TILE_HT
    #undef COMPUTE_TILE_HT
}

// ===========================================================================
// BM=64/BK=128 GEMM tile (migrated from m_grouped_fp8_gemm_nt_contiguous
// bm64_bn256_bk128_nloop2). Uses 4 math warps, matrix_load_128x16_b8,
// double K64 sub-loops per BK128 tile, kNFragments=4, kMSubtiles=4.
// ===========================================================================

// DS read for BM=64/BK=128: offset within a K64 segment of a BK128 LDS tile.
// LDS layout: [kMSubtiles=4][BK=128] bytes, each M-subtile is 16*128 = 2048 B.
// Within a K64 segment, the row-pair stride is BK=128 (not 64).
#define DS_READ_A_M16_BK128_FROM_LDS_HT(STAGE_BASE, M_LOOP, K64_LOOP, A_REG,    \
                                       LDS_TILE_BYTES, ROW_PAIR_STRIDE)          \
  do {                                                                         \
    const int a_lds_load_base =                                                 \
        (M_LOOP) * (LDS_TILE_BYTES) + (K64_LOOP) * 64 * 16;                    \
    const int a_lds_load_offset =                                              \
        a_lds_load_base + a_lane_row_pair * (ROW_PAIR_STRIDE) +                \
        a_col_swizzle * 16 + a_lane_odd_k;                                     \
    const int a_lds_read_addr = (STAGE_BASE) + a_lds_load_offset;              \
    asm volatile("ds_read_b128 %0, %1\n\t"                                    \
                 : "=v"((A_REG).i32x4[0])                                     \
                 : "v"(a_lds_read_addr)                                      \
                 : "memory");                                                \
  } while (0)

template <int32_t N, int K,
          int BLOCK_M = 64, int BLOCK_N = 256, int BLOCK_K = 128,
          int NUM_WARPS = 4, bool kIsInt8 = false,
          bool kAllWaveBarrier = false>
__device__ void gemm_ht_tile_bm64_bk128(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    static_assert(BLOCK_M % kMmacRows == 0 && BLOCK_N % (NUM_WARPS * kMmacCols) == 0 &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;  // 4
    constexpr int kK64Tile = 64;
    constexpr int kK64Loops = BLOCK_K / kK64Tile;    // 2
    constexpr int kNFragmentsPerWarp = 2;
    constexpr int kNLoopNum = BLOCK_N / (NUM_WARPS * kNFragmentsPerWarp * kMmacCols); // 2
    constexpr int kNFragments = kNLoopNum * kNFragmentsPerWarp;  // 4
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;  // 64*128 = 8192
    constexpr int kAStages = 2;
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;  // 16*128 = 2048
    constexpr int kALdsRowPairStride = BLOCK_K;          // 128

    static_assert(kMSubtiles == 4, "BM64 expects 4 M-subtiles");
    static_assert(kNFragments == 4, "BN256 with 4 warps expects 4 N-fragments");
    static_assert(kK64Loops == 2, "BK128 expects 2 K64 loops");

    // Stage barrier: all-wave s_barrier (callers mirror it for non-math
    // warps) or the math-warp subset barrier.
    auto stage_sync = [&]() {
        if constexpr (kAllWaveBarrier) {
            wait_vmcnt0_all_barrier_ht();
        } else {
            wait_vmcnt4_subset_barrier_ht(p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;

    const int n_block_start = tile_n * BLOCK_N;
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m = warp_id * kMmacRows;
    const int a_lds_write_bytes = a_load_m * BLOCK_K;

    const int logical_n_base =
        n_block_start + warp_id * kNFragments * kMmacCols + (lane_id & 15);

    // B byte offset is affine in the K-tile index: b_k = KT*64+lane_group*16
    // stays inside one 64-row band, so b_k_band==KT, b_k_mid==lane_group and
    // b_k_inner==0.  Hoist the per-fragment constant out of the K loop --
    // each buffer_load then costs one IADD instead of ~10 ALU ops.
    const int b_stride_bytes = b_n_tiles * 4 * kMmacCols * 16;
    // Per-frag byte offset is affine: logical_n advances exactly one
    // 16-col group per fragment, so b_off(f) = b_off0 + f*(4*16*16).
    // One scalar instead of a kNFragments table (the f*step term folds
    // into the load's immediate add).
    const int b_off_base0 = expert_b_base +
        ((logical_n_base / kMmacCols) * 4 + lane_group) * (kMmacCols * 16) +
        (logical_n_base & (kMmacCols - 1)) * 16;
    constexpr int kBOffFragStep = 4 * kMmacCols * 16;

    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    // B register double-buffer: 4 fragments, each with even/odd K64 halves
    BOperandHT b_even0[kNFragments], b_even1[kNFragments];
    BOperandHT b_odd0[kNFragments], b_odd1[kNFragments];

    const char* a_ptr_base =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m) * K;

    // Helper: load B tile (2 K64 halves) for a given k_base
    auto load_b_tile_bn128 = [&](int k_base, BOperandHT (&b0)[kNFragments],
                                                BOperandHT (&b1)[kNFragments]) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int logical_n = logical_n_base + n_frag * kMmacCols;
            int b_offset0 = expert_b_base +
                w6_offset_ht(N, logical_n, k_base + lane_group * 16);
            buffer_load_i32x4_ht(matrixB_w6, b0[n_frag].i32x4, b_offset0);
        }
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int logical_n = logical_n_base + n_frag * kMmacCols;
            int b_offset1 = expert_b_base +
                w6_offset_ht(N, logical_n, k_base + kK64Tile + lane_group * 16);
            buffer_load_i32x4_ht(matrixB_w6, b1[n_frag].i32x4, b_offset1);
        }
    };

    // Helper: compute one K64 segment from LDS
    auto compute_k64 = [&](const char* a_stage, int k64_loop,
                           const BOperandHT (&b_reg)[kNFragments]) {
        AOperandHT a_regs[kMSubtiles];
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            DS_READ_A_M16_BK128_FROM_LDS_HT(
                static_cast<int>(reinterpret_cast<size_t>(a_stage)),
                m_loop, k64_loop, a_regs[m_loop],
                kALdsTileBytes, kALdsRowPairStride);
        }
        wait_lgkmcnt_zero_ht();
        if (k64_loop == 0) {
            wait_vmcnt7_zero_ht();
        } else {
            wait_vmcnt5_zero_ht();
        }
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            #pragma unroll
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
                acc[m_loop][n_frag] = MmaT::mma(
                    a_regs[m_loop].fp8x8[0], b_reg[n_frag].fp8x8[0],
                    acc[m_loop][n_frag]);
                acc[m_loop][n_frag] = MmaT::mma(
                    a_regs[m_loop].fp8x8[1], b_reg[n_frag].fp8x8[1],
                    acc[m_loop][n_frag]);
            }
        }
    };

    // Prologue: load tile0 to even stage + prefetch B (2 K64 halves)
    {
        const char* a_ptr0 = a_ptr_base;
        vec4_u32 a_srsrc0 = make_matrix_resource_ht(a_ptr0, K);
        const int lds_addr0 = a_lds_base_addr + a_lds_write_bytes;
        MATRIX_LOAD_128X16_B8_T_LDS_HT(lds_addr0, a_srsrc0);
        load_b_tile_bn128(0, b_even0, b_even1);
        stage_sync();
    }

    // Main loop: double-buffered A (LDS) and B (registers)
    int k_tile = 0;
    for (; k_tile + 2 < k_tiles; k_tile += 2) {
        // Load odd tile to stage 1
        {
            const char* a_odd_ptr =
                a_ptr_base + static_cast<int64_t>(k_tile + 1) * BLOCK_K;
            vec4_u32 a_odd_srsrc = make_matrix_resource_ht(a_odd_ptr, K);
            const int odd_lds_addr =
                a_lds_base_addr + kAStageBytes + a_lds_write_bytes;
            MATRIX_LOAD_128X16_B8_T_LDS_HT(odd_lds_addr, a_odd_srsrc);
        }
        load_b_tile_bn128((k_tile + 1) * BLOCK_K, b_odd0, b_odd1);

        // Compute even tile (stage 0): 2 K64 loops
        {
            const char* a_stage = a_lds;
            compute_k64(a_stage, 0, b_even0);
            compute_k64(a_stage, 1, b_even1);
        }
        stage_sync();

        // Load even tile to stage 0 (for next iteration)
        {
            const char* a_even_ptr =
                a_ptr_base + static_cast<int64_t>(k_tile + 2) * BLOCK_K;
            vec4_u32 a_even_srsrc = make_matrix_resource_ht(a_even_ptr, K);
            const int even_lds_addr = a_lds_base_addr + a_lds_write_bytes;
            MATRIX_LOAD_128X16_B8_T_LDS_HT(even_lds_addr, a_even_srsrc);
        }
        load_b_tile_bn128((k_tile + 2) * BLOCK_K, b_even0, b_even1);

        // Compute odd tile (stage 1): 2 K64 loops
        {
            const char* a_stage = a_lds + kAStageBytes;
            compute_k64(a_stage, 0, b_odd0);
            compute_k64(a_stage, 1, b_odd1);
        }
        stage_sync();
    }

    // Epilogue: drain remaining tiles
    const int remaining_tiles = k_tiles - k_tile;
    if (remaining_tiles == 1) {
        // Compute even tile (no more loads)
        const char* a_stage = a_lds;
        compute_k64(a_stage, 0, b_even0);
        compute_k64(a_stage, 1, b_even1);
    } else {
        // 2 remaining tiles: load odd, compute even, compute odd
        {
            const char* a_tail_odd_ptr =
                a_ptr_base + static_cast<int64_t>(k_tile + 1) * BLOCK_K;
            vec4_u32 a_tail_odd_srsrc =
                make_matrix_resource_ht(a_tail_odd_ptr, K);
            const int tail_odd_lds_addr =
                a_lds_base_addr + kAStageBytes + a_lds_write_bytes;
            MATRIX_LOAD_128X16_B8_T_LDS_HT(tail_odd_lds_addr, a_tail_odd_srsrc);
        }
        load_b_tile_bn128((k_tile + 1) * BLOCK_K, b_odd0, b_odd1);

        // Compute even tile
        {
            const char* a_stage = a_lds;
            compute_k64(a_stage, 0, b_even0);
            compute_k64(a_stage, 1, b_even1);
        }
        stage_sync();

        // Compute odd tile
        {
            const char* a_stage = a_lds + kAStageBytes;
            compute_k64(a_stage, 0, b_odd0);
            compute_k64(a_stage, 1, b_odd1);
        }
    }

    // Channel-wise dequant and output: acc * input_scale * weight_scale -> BF16
    float col_scale[kNFragments][kMmacVgprs];
    #pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int col_base = n_block_start +
                             warp_id * kNFragments * kMmacCols +
                             n_frag * kMmacCols + lane_group * kMmacVgprs;
        #pragma unroll
        for (int elem = 0; elem < kMmacVgprs; ++elem) {
            col_scale[n_frag][elem] = matrixB_scale[col_base + elem];
        }
    }

    const bool full_block_valid =
        (m_block_start + BLOCK_M <= M) && (valid_m >= BLOCK_M);

    if (full_block_valid) {
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            const int row = m_block_start + m_loop * kMmacRows + lane_row;
            const float row_scale = matrixA_scale[row];
            #pragma unroll
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
                const int col_base = n_block_start +
                                     warp_id * kNFragments * kMmacCols +
                                     n_frag * kMmacCols + lane_group * kMmacVgprs;
                const int col = col_base;
                const float value0 = MmaT::to_float(acc[m_loop][n_frag][0]) * row_scale *
                                     col_scale[n_frag][0];
                const float value1 = MmaT::to_float(acc[m_loop][n_frag][1]) * row_scale *
                                     col_scale[n_frag][1];
                const float value2 = MmaT::to_float(acc[m_loop][n_frag][2]) * row_scale *
                                     col_scale[n_frag][2];
                const float value3 = MmaT::to_float(acc[m_loop][n_frag][3]) * row_scale *
                                     col_scale[n_frag][3];
                *reinterpret_cast<uint64_t*>(MatrixC +
                    static_cast<int64_t>(row) * C_stride + col) =
                    pack_bf16x4_ht(value0, value1, value2, value3);
            }
        }
    } else {
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            const int row = m_block_start + m_loop * kMmacRows + lane_row;
            if (row >= M) continue;
            const float row_scale = matrixA_scale[row];
            #pragma unroll
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
                const int col_base = n_block_start +
                                     warp_id * kNFragments * kMmacCols +
                                     n_frag * kMmacCols + lane_group * kMmacVgprs;
                const int col = col_base;
                const float value0 = MmaT::to_float(acc[m_loop][n_frag][0]) * row_scale *
                                     col_scale[n_frag][0];
                const float value1 = MmaT::to_float(acc[m_loop][n_frag][1]) * row_scale *
                                     col_scale[n_frag][1];
                const float value2 = MmaT::to_float(acc[m_loop][n_frag][2]) * row_scale *
                                     col_scale[n_frag][2];
                const float value3 = MmaT::to_float(acc[m_loop][n_frag][3]) * row_scale *
                                     col_scale[n_frag][3];
                *reinterpret_cast<uint64_t*>(MatrixC +
                    static_cast<int64_t>(row) * C_stride + col) =
                    pack_bf16x4_ht(value0, value1, value2, value3);
            }
        }
    }
}

// ===========================================================================
// Fused SwiGLU + group-wise quant variant of gemm_ht_tile_bm64_bk128 for the
// HT L1 GEMM.  Same mainloop (BM=64/BK=128, 4 math warps, kNFragments=4);
// the epilogue dequantizes, applies SwiGLU (granularity-8 gate/up
// interleave), computes a per-(row, 128-col group) amax, quantizes to
// FP8/INT8 and writes l2_token_out/l2_sf directly -- no GMEM roundtrip.
// Each call's BLOCK_N/2 = 128 output columns form exactly one quant group
// (kGS = 128); all 4 warps contribute to it (kWarpsPerGroup = 4).
// ===========================================================================
template <int32_t N, int K,
          int BLOCK_M = 64, int BLOCK_N = 256, int BLOCK_K = 128,
          int NUM_WARPS = 4, bool kIsInt8 = false,
          bool kAllWaveBarrier = false, int kAStagesArg = 3>
__device__ void gemm_ht_tile_bm64_bk128_silu_quant(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    const int32_t M,
    const uint32_t math_warp_offset,
    uint8_t* __restrict__ l2_token_out,
    float* __restrict__ l2_sf,
    const int32_t m_idx,
    const int32_t sf_stride,
    const int32_t ih,
    float* __restrict__ smem_amax,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense,
    const float activation_clamp,
    const bool fast_math)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    static_assert(BLOCK_M % kMmacRows == 0 && BLOCK_N % (NUM_WARPS * kMmacCols) == 0 &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;  // 4
    constexpr int kK64Tile = 64;
    constexpr int kK64Loops = BLOCK_K / kK64Tile;    // 2
    constexpr int kNFragmentsPerWarp = 2;
    constexpr int kNLoopNum = BLOCK_N / (NUM_WARPS * kNFragmentsPerWarp * kMmacCols); // 2
    constexpr int kNFragments = kNLoopNum * kNFragmentsPerWarp;  // 4
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;  // 64*128 = 8192
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;  // 16*128 = 2048
    constexpr int kALdsRowPairStride = BLOCK_K;          // 128

    static_assert(kMSubtiles == 4, "BM64 expects 4 M-subtiles");
    static_assert(kNFragments == 4, "BM64 expects 4 N-fragments per warp");
    static_assert(kK64Loops == 2, "BK128 expects 2 K64 loops");
    static_assert(NUM_WARPS * kNFragments * 8 == BLOCK_N / 2,
                  "warps must exactly cover the SwiGLU output width");
    constexpr int kGS = BLOCK_N / 2;  // one group per call's output width
    constexpr int kAStages = kAStagesArg;

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;

    const int n_block_start = tile_n * BLOCK_N;
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m = warp_id * kMmacRows;
    const int a_lds_write_bytes = a_load_m * BLOCK_K;
    // With NUM_WARPS > kMSubtiles (e.g. 8 warps for BM=64), warps beyond the
    // M-subtile count carry no A tile: they still issue their own B loads and
    // run the full MMA over all kMSubtiles subtiles read back from LDS.
    const bool loads_a = a_load_m < BLOCK_M;

    // Per-warp load issue counts per K-tile: 2*kNFragments B buffer_loads
    // (two K64 halves) plus one matrix A load on the loading warps only.
    // The vmcnt allowance keeps the loads issued in the just-elapsed half
    // in flight across the barrier; anything issued earlier must have
    // landed (same scheme as the bm128_bk64 tiles).
    constexpr int kBLoadsPerTile = 2 * kNFragments;
    auto stage_sync = [&](int n_a_issued) {
        if constexpr (kAllWaveBarrier) {
            if (loads_a && n_a_issued)
                wait_vmcnt_all_barrier_ht<1 + kBLoadsPerTile>();
            else
                wait_vmcnt_all_barrier_ht<kBLoadsPerTile>();
        } else {
            if (loads_a && n_a_issued)
                wait_vmcnt_subset_barrier_ht<1 + kBLoadsPerTile>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
            else
                wait_vmcnt_subset_barrier_ht<kBLoadsPerTile>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };
    auto stage_sync_drain = [&]() {
        if constexpr (kAllWaveBarrier) {
            wait_vmcnt0_all_barrier_ht();
        } else {
            wait_vmcnt_subset_barrier_ht<0>(p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };

    const int logical_n_base =
        n_block_start + warp_id * kNFragments * kMmacCols + (lane_id & 15);

    // B byte offset is affine in the K-tile index: b_k = KT*64+lane_group*16
    // stays inside one 64-row band, so b_k_band==KT, b_k_mid==lane_group and
    // b_k_inner==0.  Hoist the per-fragment constant out of the K loop --
    // each buffer_load then costs one IADD instead of ~10 ALU ops.
    const int b_stride_bytes = b_n_tiles * 4 * kMmacCols * 16;
    // Per-frag byte offset is affine: logical_n advances exactly one
    // 16-col group per fragment, so b_off(f) = b_off0 + f*(4*16*16).
    // One scalar instead of a kNFragments table (the f*step term folds
    // into the load's immediate add).
    const int b_off_base0 = expert_b_base +
        ((logical_n_base / kMmacCols) * 4 + lane_group) * (kMmacCols * 16) +
        (logical_n_base & (kMmacCols - 1)) * 16;
    constexpr int kBOffFragStep = 4 * kMmacCols * 16;

    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    BOperandHT b_even0[kNFragments], b_even1[kNFragments];
    BOperandHT b_odd0[kNFragments], b_odd1[kNFragments];

    const char* a_ptr_base =
        matrixA + static_cast<int64_t>(m_block_start +
                                         (loads_a ? a_load_m : 0)) * K;

    auto load_b_tile_bn128 = [&](int k_base, BOperandHT (&b0)[kNFragments],
                                                BOperandHT (&b1)[kNFragments]) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int logical_n = logical_n_base + n_frag * kMmacCols;
            int b_offset0 = expert_b_base +
                w6_offset_ht(N, logical_n, k_base + lane_group * 16);
            buffer_load_i32x4_ht(matrixB_w6, b0[n_frag].i32x4, b_offset0);
        }
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int logical_n = logical_n_base + n_frag * kMmacCols;
            int b_offset1 = expert_b_base +
                w6_offset_ht(N, logical_n, k_base + kK64Tile + lane_group * 16);
            buffer_load_i32x4_ht(matrixB_w6, b1[n_frag].i32x4, b_offset1);
        }
    };

    auto compute_k64 = [&](const char* a_stage, int k64_loop,
                           const BOperandHT (&b_reg)[kNFragments]) {
        AOperandHT a_regs[kMSubtiles];
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            DS_READ_A_M16_BK128_FROM_LDS_HT(
                static_cast<int>(reinterpret_cast<size_t>(a_stage)),
                m_loop, k64_loop, a_regs[m_loop],
                kALdsTileBytes, kALdsRowPairStride);
        }
        wait_lgkmcnt_zero_ht();
        // 9 loads stay in flight per half (1 A matrix_load + 8 B
        // buffer_loads); vmcnt(9)/(8) only drains loads issued >=2 halves
        // ago -- the current tile's B regs are guaranteed resident without
        // stalling the next tile's prefetch.
        if (k64_loop == 0) {
            asm volatile("s_waitcnt vmcnt(9)" ::: "memory");
        } else {
            asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
        }
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            #pragma unroll
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
                acc[m_loop][n_frag] = MmaT::mma(
                    a_regs[m_loop].fp8x8[0], b_reg[n_frag].fp8x8[0],
                    acc[m_loop][n_frag]);
                acc[m_loop][n_frag] = MmaT::mma(
                    a_regs[m_loop].fp8x8[1], b_reg[n_frag].fp8x8[1],
                    acc[m_loop][n_frag]);
            }
        }
    };

    // Prologue: tiles 0/1 -> ring stages 0/1 (depth-2 prefetch) plus the
    // matching B register tiles.  No barrier here: nothing reads the stages
    // until the first stage_sync in the main loop/epilogue.
    {
        if (loads_a) {
            vec4_u32 a_srsrc0 = make_matrix_resource_ht(a_ptr_base, K);
            MATRIX_LOAD_128X16_B8_T_LDS_HT(
                a_lds_base_addr + a_lds_write_bytes, a_srsrc0);
        }
        load_b_tile_bn128(0, b_even0, b_even1);
        if (k_tiles > 1) {
            if (loads_a) {
                const char* a_ptr1 = a_ptr_base + BLOCK_K;
                vec4_u32 a_srsrc1 = make_matrix_resource_ht(a_ptr1, K);
                MATRIX_LOAD_128X16_B8_T_LDS_HT(
                    a_lds_base_addr + kAStageBytes + a_lds_write_bytes,
                    a_srsrc1);
            }
            load_b_tile_bn128(BLOCK_K, b_odd0, b_odd1);
        }
    }

    // Main loop: kAStages-deep A LDS ring + register-level B double buffer.
    // Same structure as gemm_ht_tile_bm128_bk64_silu_quant (d8d56dd):
    //   alpha sync -> issue A(k+2) -> compute k -> issue B(k+2)
    //   beta  sync -> issue A(k+3) -> compute k+1 -> issue B(k+3)
    // A(k+2)->stage (k+2)%kAStages is safe right after alpha because its last
    // reader was compute(k-1) in the previous iteration.  A(k+3) reuses the
    // stage compute(k) just finished, so it must wait for the beta barrier.
    int k_tile = 0;
    for (; k_tile + 2 < k_tiles; k_tile += 2) {
        stage_sync(0);   // alpha
        if (loads_a) {
            const char* a_nx_ptr =
                a_ptr_base + static_cast<int64_t>(k_tile + 2) * BLOCK_K;
            vec4_u32 a_nx_srsrc = make_matrix_resource_ht(a_nx_ptr, K);
            MATRIX_LOAD_128X16_B8_T_LDS_HT(
                a_lds_base_addr + ((k_tile + 2) % kAStages) * kAStageBytes +
                    a_lds_write_bytes,
                a_nx_srsrc);
        }
        {
            const char* a_stage =
                a_lds + (k_tile % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_even0);
            compute_k64(a_stage, 1, b_even1);
        }
        load_b_tile_bn128((k_tile + 2) * BLOCK_K, b_even0, b_even1);
        stage_sync(1);   // beta
        if (k_tile + 3 < k_tiles && loads_a) {
            const char* a_nx_ptr =
                a_ptr_base + static_cast<int64_t>(k_tile + 3) * BLOCK_K;
            vec4_u32 a_nx_srsrc = make_matrix_resource_ht(a_nx_ptr, K);
            MATRIX_LOAD_128X16_B8_T_LDS_HT(
                a_lds_base_addr + ((k_tile + 3) % kAStages) * kAStageBytes +
                    a_lds_write_bytes,
                a_nx_srsrc);
        }
        {
            const char* a_stage =
                a_lds + ((k_tile + 1) % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_odd0);
            compute_k64(a_stage, 1, b_odd1);
        }
        if (k_tile + 3 < k_tiles)
            load_b_tile_bn128((k_tile + 3) * BLOCK_K, b_odd0, b_odd1);
    }

    // Epilogue: drain the last 1-2 tiles (their loads were already issued).
    const int remaining_tiles = k_tiles - k_tile;
    if (remaining_tiles >= 1) {
        if (remaining_tiles == 2) {
            stage_sync(0);
        } else {
            stage_sync_drain();
        }
        {
            const char* a_stage =
                a_lds + (k_tile % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_even0);
            compute_k64(a_stage, 1, b_even1);
        }
        ++k_tile;
        if (k_tile < k_tiles) {
            stage_sync_drain();
            const char* a_stage =
                a_lds + (k_tile % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_odd0);
            compute_k64(a_stage, 1, b_odd1);
        }
    }

    // =========================================================================
    // Fused SwiGLU + group-wise quant epilogue (same scheme as the BM128
    // variant; constants adapt: kMSubtiles=4, kNFragments=4, kOutPerWarp=32,
    // kWarpsPerGroup=4).
    // =========================================================================
    constexpr int kOutPerWarp = kNFragments * 8;
    static_assert(kGS % kOutPerWarp == 0, "quant group must cover a whole number of warps");
    constexpr int kWarpsPerGroup = kGS / kOutPerWarp;  // == NUM_WARPS
    const int out_base = tile_n * (BLOCK_N / 2);
    const int group_warp_base = (warp_id / kWarpsPerGroup) * kWarpsPerGroup;
    const int group_idx = out_base / kGS;  // == tile_n

    // smem_amax is overlaid on the (dead) a_lds region when the caller runs
    // >8 math warps or a >3-stage pipeline -- wait for every warp's stage
    // reads to land before writing it (mirrors kAmaxOnALds in the kernel;
    // the non-math mirror loop already counts this barrier).
    if constexpr (NUM_WARPS > 8 || kAStages > 3) {
        if constexpr (kAllWaveBarrier) {
            wait_lgkmcnt0_all_barrier_ht();
        } else {
            wait_lgkmcnt0_subset_barrier_ht(p_bar_count, p_bar_sense, NUM_WARPS);
        }
    }
    // Phase 1: dequant + SwiGLU, keep BF16-packed silu values in registers.
    constexpr int kFragHalf = (kNFragments + 1) / 2;
    float part[kMSubtiles];
    uint64_t silu_pk[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) part[m_loop] = 0.0f;
    #pragma unroll
    for (int half = 0; half * kFragHalf < kNFragments; ++half) {
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            const int row_local = m_loop * kMmacRows + lane_row;
            const int row = m_block_start + row_local;
            const bool row_ok = (row_local < valid_m) && (row < M);
            const float row_scale = row_ok ? matrixA_scale[row] : 0.0f;
            #pragma unroll
            for (int n_frag = half * kFragHalf;
                 n_frag < min((half + 1) * kFragHalf, kNFragments); ++n_frag) {
                const int col_base = n_block_start +
                    warp_id * kNFragments * kMmacCols +
                    n_frag * kMmacCols + lane_group * kMmacVgprs;
                uint64_t sv;
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; ++i) {
                    const float v = MmaT::to_float(acc[m_loop][n_frag][i])
                        * row_scale * matrixB_scale[col_base + i];
                    const float partner = __shfl_xor(v, 32);
                    // lg<2: v=gate, partner=up; lg>=2: v=up, partner=gate.
                    const float g = (lane_group < 2) ? v : partner;
                    const float u = (lane_group < 2) ? partner : v;
                    const float s = swiglu_nv(g, u, activation_clamp, fast_math);
                    part[m_loop] = fmaxf(part[m_loop], fabsf(s));
                    reinterpret_cast<hip_bfloat16*>(&sv)[i] = hip_bfloat16(s);
                }
                silu_pk[m_loop][n_frag] = sv;
            }
        }
    }
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row_local = m_loop * kMmacRows + lane_row;
        const int row = m_block_start + row_local;
        const bool row_ok = (row_local < valid_m) && (row < M);
        float p = row_ok ? part[m_loop] : 0.0f;
        // Both partner lanes produce identical silu values; row amax over the
        // warp's fragment cols needs only the {r, r+16, r+32, r+48} quad.
        p = fmaxf(p, __shfl_xor(p, 16));
        p = fmaxf(p, __shfl_xor(p, 32));
        if (lane_id < 16) {
            // Transposed [row][warp] layout: the cross-warp reduction
            // below then reads a contiguous float run (vectorizable)
            // instead of a strided gather over warp-major slots.
            smem_amax[row_local * NUM_WARPS + warp_id] = p;
        }
    }

    // Row-amax cross-warp reduction (all-block barrier; non-math warps
    // mirror this s_barrier per call).
    wait_lgkmcnt0_all_barrier_ht();

    // Phase 2: quantize the register-resident silu values, pack, repack to
    // 16B stores.
#ifndef DG_MEGA_L1_NOQUANT
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row_local = m_loop * kMmacRows + lane_row;
        const int row = m_block_start + row_local;
        const bool row_ok = (row_local < valid_m) && (row < M);
        float row_amax = 0.0f;
        static_assert(kWarpsPerGroup % 4 == 0,
                      "row-amax vectorized read needs groups of 4 warps");
        #pragma unroll
        for (int w = 0; w < kWarpsPerGroup; w += 4) {
            const float4 a4 = *reinterpret_cast<const float4*>(
                &smem_amax[row_local * NUM_WARPS + group_warp_base + w]);
            row_amax = fmaxf(row_amax,
                fmaxf(fmaxf(a4.x, a4.y), fmaxf(a4.z, a4.w)));
        }
        const float qmax = kIsInt8 ? 127.0f : 448.0f;
        const float clamped_amax = fmaxf(row_amax, kIsInt8 ? 1.0e-6f : 1.0e-4f);
        const float scale = clamped_amax / qmax;
        const float inv_scale = qmax / clamped_amax;

        uint32_t packed[kNFragments];
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const hip_bfloat16* svb =
                reinterpret_cast<const hip_bfloat16*>(&silu_pk[m_loop][n_frag]);
            float q[kMmacVgprs];
            #pragma unroll
            for (int i = 0; i < kMmacVgprs; ++i) {
                const float s = static_cast<float>(svb[i]);
                q[i] = quant_clamp<kIsInt8>(s * inv_scale, qmax);
            }
            uint32_t p = 0;
            if constexpr (kIsInt8) {
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; ++i) {
                    p |= (static_cast<uint32_t>(static_cast<uint8_t>(
                              static_cast<int8_t>(__float2int_rn(q[i]))))
                          << (i * 8));
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; i += 2) {
                    float2 f2{q[i], q[i + 1]};
                    auto fp8x2 = __hip_cvt_float2_to_fp8x2(
                        f2, __HIP_SATFINITE, __HIP_E4M3);
                    p |= (static_cast<uint32_t>(
                              *reinterpret_cast<uint16_t*>(&fp8x2)) << (i * 8));
                }
            }
            packed[n_frag] = p;
        }

        // Repack: row's kOutPerWarp out cols are spread over lanes r/r+16
        // (frag pairs). Gather into lane r for 16B chunks.
        if (row_ok) {
            #pragma unroll
            for (int chunk = 0; chunk < kOutPerWarp / 16; ++chunk) {
                uint4 v4;
                v4.x = __shfl(packed[chunk * 2 + 0], lane_id);
                v4.y = __shfl(packed[chunk * 2 + 0], lane_id + 16);
                v4.z = __shfl(packed[chunk * 2 + 1], lane_id);
                v4.w = __shfl(packed[chunk * 2 + 1], lane_id + 16);
                if (lane_id < 16) {
                    *reinterpret_cast<uint4*>(
                        l2_token_out +
                        static_cast<int64_t>(m_idx + row_local) * ih +
                        out_base + warp_id * kOutPerWarp + chunk * 16) = v4;
                }
            }
        }
        // One lane per row per group writes the scale (all lanes of the
        // group share row_amax; guard to the first warp of the group).
        if (lane_id < 16 && row_ok &&
            warp_id % kWarpsPerGroup == 0) {
            l2_sf[static_cast<int64_t>(m_idx + row_local) * sf_stride
                  + group_idx] = scale;
        }
    }
#else
    (void)silu_pk; (void)group_warp_base; (void)kWarpsPerGroup;
#endif
}
// (FP8 only).  A-scale is [M, K/GROUP_SIZE] per-token-per-group instead of
// per-token.  acc accumulates raw FP8 partials; at each GROUP_SIZE boundary
// the accumulator is rescaled in place by rg = s_{g-1}/s_g (telescoping to
// sum s_j*P_j), and the epilogue multiplies by s_last.
//
// Scale handling: the caller pre-fills a per-tile [BLOCK_M, K_OUT_ITERS]
// table in SMEM before the GEMM runs, storing boundary ratios
// r_g = s_{g-1}/s_g (slot 0 holds s_last for the epilogue).  The main loop
// holds no persistent scale registers 鈥?at each group-start iter a single
// LDS read per row supplies r_g, and the rescale multiply is interleaved
// into the MMA pipeline.  This avoids the register-ring pressure (32+ live
// VGPRs), per-boundary GMEM loads, and per-boundary frcp chains.
// ===========================================================================

template <int32_t N, int K, int32_t GROUP_SIZE,
          int BLOCK_M = 128, int BLOCK_N = 256, int BLOCK_K = 64,
          int NUM_WARPS = 8, bool kIsInt8 = false,
          bool kAllWaveBarrier = false, int kAStagesArg = 3,
          bool kRegOptArg = false>
__device__ void gemm_ht_tile_bm128_bk64_groupwise(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride,
    const float* __restrict__ smem_scale,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense,
    const int32_t n_block_base = -1)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    static_assert(BLOCK_M % kMmacRows == 0 && BLOCK_N % (NUM_WARPS * kMmacCols) == 0 &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;  // 8
    constexpr int kK64Tile = 64;
    constexpr int kKLoops = BLOCK_K / 32;            // 2
    constexpr int kK64Loops = BLOCK_K / kK64Tile;    // 1
    constexpr int kNFragments = BLOCK_N / (NUM_WARPS * kMmacCols);  // 2 or 4
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;
    constexpr int kAStages = kAStagesArg;
    // A 4-N-fragment tile keeps acc[8][4] = 128 dword live; under the
    // 1024-thread launch (192 VGPR cap) that exhausts the budget.  Three
    // rotation slots still give one full iteration of prefetch margin
    // ((m+2)%3 != m%3) and free 8 dword of live VGPR.
    constexpr bool kRegOptTile = kRegOptArg && (kNFragments >= 4);
    constexpr int kASlots = (kRegOptTile && (kRegOptMask & 1)) ? 3 : 4;
    static_assert(kASlots >= 3, "prefetch distance 2 needs >= 3 A-frag slots");
    constexpr int kALdsTotalBytes = kAStages * kAStageBytes;
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;
    constexpr int kALdsRowPairStride = 128;
    constexpr int K_OUT_ITERS = K / GROUP_SIZE;        // 24 for K=3072,GS=128
    static_assert(GROUP_SIZE % BLOCK_K == 0 && K % GROUP_SIZE == 0,
                  "K/GROUP_SIZE/BLOCK_K must tile evenly");
    constexpr int K_IN_ITERS = GROUP_SIZE / BLOCK_K;   // 2 for GS=128,BK=64

    static_assert(kMSubtiles >= 4, "expects at least 4 M-subtiles (BM>=64)");
    static_assert(kNFragments >= 1 && kNFragments <= 4, "BN256/512 expects 1..4 N-fragments per warp");
    static_assert(kK64Loops == 1, "BK64 expects 1 K64 loop");
    static_assert(K_IN_ITERS % 2 == 0,
                  "group must span whole main-loop iters (2 K-tiles each)");

    // vmcnt(kLoadsPerTile) leaves the next tile's A matrix_load + B
    // buffer_loads in flight across the barrier; s_barrier alone only
    // waits for wave arrival and does not drain the counters.
    constexpr int kALoadsPerWarp =
        (kMSubtiles + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int kLoadsPerTile = kALoadsPerWarp + kNFragments;
    // Per-warp vmcnt budget: the allowance must match the number of
    // loads this warp itself issued last iteration.  With
    // NUM_WARPS > kMSubtiles some warps skip their A matrix_load, so
    // kLoadsPerTile would leave one extra slot of slack -- a B
    // buffer_load straggler from the previous tile could then stay in
    // flight across the barrier and feed stale registers to mma.
    // n_a_issued: A matrix_loads issued by an A-loading warp in the
    // half-iteration that just elapsed (0 or 2 per k-tile); the
    // allowance keeps exactly that half's loads (A + B fragments) in
    // flight.
    auto stage_sync = [&](int n_a_issued) {
        const int my_wid = (threadIdx.x - math_warp_offset) / kWarpSize;
        const bool full_a =
            (my_wid * kMmacRows < BLOCK_M) &&
            (kALoadsPerWarp == 1 ||
             my_wid * kMmacRows + NUM_WARPS * kMmacRows < BLOCK_M);
        if constexpr (kAllWaveBarrier) {
            if (full_a && n_a_issued)
                wait_vmcnt_all_barrier_ht<kALoadsPerWarp + kNFragments>();
            else
                wait_vmcnt_all_barrier_ht<kNFragments>();
        } else {
            if (full_a && n_a_issued)
                wait_vmcnt_subset_barrier_ht<kALoadsPerWarp + kNFragments>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
            else
                wait_vmcnt_subset_barrier_ht<kNFragments>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };
    auto stage_sync_drain = [&]() {
        if constexpr (kAllWaveBarrier) {
            wait_vmcnt_all_barrier_ht<0>();
        } else {
            wait_vmcnt_subset_barrier_ht<0>(
                p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;

    // Tile column base: n_block_base >= 0 carries an absolute column (used
    // when L2_SHAPE_N is not an exact multiple of BLOCK_N and the last tile
    // is shifted back so it overlaps its predecessor instead of running past
    // the end of the row).
    const int n_block_start = (n_block_base >= 0) ? n_block_base : tile_n * BLOCK_N;
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    constexpr int kNumALoadsPerWarp = (kMSubtiles + NUM_WARPS - 1) / NUM_WARPS;
    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m0 = warp_id * kMmacRows;
    const int a_load_m1 = a_load_m0 + NUM_WARPS * kMmacRows;
    const int a_lds_write_bytes0 = a_load_m0 * BLOCK_K;
    const int a_lds_write_bytes1 = a_load_m1 * BLOCK_K;
    const int a_lds_write_bytes = a_lds_write_bytes0;

    const int logical_n_base =
        n_block_start + warp_id * kNFragments * kMmacCols + (lane_id & 15);

    // B byte offset is affine in the K-tile index: b_k = KT*64+lane_group*16
    // stays inside one 64-row band, so b_k_band==KT, b_k_mid==lane_group and
    // b_k_inner==0.  Hoist the per-fragment constant out of the K loop --
    // each buffer_load then costs one IADD instead of ~10 ALU ops.
    const int b_stride_bytes = b_n_tiles * 4 * kMmacCols * 16;
    // Per-frag byte offset is affine: logical_n advances exactly one
    // 16-col group per fragment, so b_off(f) = b_off0 + f*(4*16*16).
    // One scalar instead of a kNFragments table (the f*step term folds
    // into the load's immediate add).
    const int b_off_base0 = expert_b_base +
        ((logical_n_base / kMmacCols) * 4 + lane_group) * (kMmacCols * 16) +
        (logical_n_base & (kMmacCols - 1)) * 16;
    constexpr int kBOffFragStep = 4 * kMmacCols * 16;

    // A1: descriptor is invariant over the K loop.  expert_b_base stays
    // inside b_off_base0 -- adding it here too would double-count it.
    ht_rsrc_t b_rsrc = make_buffer_resource_ht(matrixB_w6);

    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    // B register quad-buffer: two sets per parity so the next phase's B
    // tiles issue at the top of the current one (lead ~1 full iteration
    // instead of ~half).  Named sets keep the selector compile-time
    // constant (a runtime index would spill to local memory).
    BOperandHT b_even0, b_even1, b_even2, b_even3;
    BOperandHT b_odd0, b_odd1, b_odd2, b_odd3;

    #define B_EVEN_AT_HT(IDX) \
        ((IDX) == 0 ? b_even0 : (IDX) == 1 ? b_even1 : (IDX) == 2 ? b_even2 : b_even3)
    #define B_ODD_AT_HT(IDX) \
        ((IDX) == 0 ? b_odd0 : (IDX) == 1 ? b_odd1 : (IDX) == 2 ? b_odd2 : b_odd3)

    const char* a_ptr_base0 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m0) * K;
    const char* a_ptr_base1 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m1) * K;

    #define LOAD_A_TILE_PAIR_HT(STAGE_BASE, K_OFFSET) \
        do { \
            if (a_load_m0 < BLOCK_M) { \
                const char* a0 = a_ptr_base0 + static_cast<int64_t>(K_OFFSET); \
                vec4_u32 srsrc0 = make_matrix_resource_ht(a0, K); \
                MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes0, srsrc0); \
            } \
            if constexpr (kNumALoadsPerWarp > 1) { \
                if (a_load_m1 < BLOCK_M) { \
                    const char* a1 = a_ptr_base1 + static_cast<int64_t>(K_OFFSET); \
                    vec4_u32 srsrc1 = make_matrix_resource_ht(a1, K); \
                    MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes1, srsrc1); \
                } \
            } \
        } while (0)

    // ---------------------------------------------------------------------
    // Groupwise scale state: the caller pre-fills SMEM with this tile's
    // [BLOCK_M, K_OUT_ITERS] table of *boundary ratios* r_g = s_{g-1}/s_g
    // (slot 0 holds s_last for the epilogue).  The main loop keeps ZERO
    // persistent scale registers 鈥?each boundary reads r_g with a single
    // LDS load per row, no frcp in the warp loop.  Invalid rows are padded
    // with 1.0f by the fill, giving rg == 1 (outputs are masked downstream).
    //   smem_scale[(m*kMmacRows + lane_row) * K_OUT_ITERS + g] = r_row(g)
    // ---------------------------------------------------------------------
    #define SMEM_SCALE_ROW(m) \
        (smem_scale + (m * kMmacRows + lane_row) * K_OUT_ITERS)
#ifdef DG_MEGA_GW_NORESCALE
    // Diagnostic build: skip the boundary rescale multiply entirely.
    // Results are wrong (each group's partial stays in its own scale
    // units); used only to measure the rescale's share of L2 runtime.
    #define RESCALE_GW_M(m, rg) do { (void)(rg); } while (0)
#else
    #define RESCALE_GW_M(m, rg)                                              \
        do {                                                                 \
            _Pragma("unroll")                                                \
            for (int f = 0; f < kNFragments; ++f) {                          \
                _Pragma("unroll")                                            \
                for (int i = 0; i < kMmacVgprs; ++i) {                       \
                    acc[m][f][i] = MmaT::rescale(acc[m][f][i], rg);          \
                }                                                            \
            }                                                                \
        } while (0)
#endif
    #define RESCALE_GW(g)                                                    \
        do {                                                                 \
            _Pragma("unroll")                                                \
            for (int m = 0; m < kMSubtiles; ++m) {                           \
                const float rgm = SMEM_SCALE_ROW(m)[g];                      \
                RESCALE_GW_M(m, rgm);                                        \
            }                                                                \
        } while (0)

    #define LOAD_B_TILE_HT(KT, B_SEL)                                          \
        do {                                                                   \
            _Pragma("unroll")                                                  \
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {             \
                buffer_load_i32x4_rsrc_ht(b_rsrc, B_SEL(n_frag).i32x4,          \
                                     b_off_base0 +                           \
                                     (n_frag) * kBOffFragStep +             \
                                     (KT) * b_stride_bytes);                 \
            }                                                                  \
        } while (0)

    // Compute one K-tile: stage (KT % kAStages), B regs via B_SEL.  When
    // GW_BOUNDARY is true the per-row boundary ratios in gw_rg[] rescale
    // acc[m] just before the row's MMAs.
    // A-operand pipeline: same 4-slot rotation / prefetch distance 2 /
    // lgkmcnt(1) as the L1 pass -- the read for subtile m+2 stays in flight
    // while the MMAs for m issue, instead of one full LDS round-trip per
    // subtile.  Slot (m+2)&3 was last read by mma(m-2), two iterations back,
    // so the DS_READ writeback cannot collide with an in-flight fetch.
    #define COMPUTE_TILE_GW_HT(KT, B_SEL, GW_BOUNDARY)                         \
        do {                                                                   \
            const int a_stage_base_c =                                         \
                a_lds_base_addr + (((KT) % kAStages) * kAStageBytes);          \
            AOperandHT a_regs[kASlots];                                              \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 0, a_regs[0]);           \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 1, a_regs[1]);           \
            wait_lgkmcnt_one_ht();                                             \
            wait_vmcnt7_zero_ht();                                             \
            _Pragma("unroll")                                                  \
            for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {              \
                const int compute_slot = m_loop % kASlots;                           \
                if (m_loop + 2 < kMSubtiles) {                                 \
                    DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, m_loop + 2,      \
                                              a_regs[(m_loop + 2) % kASlots]);       \
                }                                                              \
                if (GW_BOUNDARY) {                                             \
                    RESCALE_GW_M(m_loop, gw_rg[m_loop]);                       \
                }                                                              \
                _Pragma("unroll")                                              \
                for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {         \
                    const BOperandHT& b_reg = B_SEL(n_frag);                   \
                    acc[m_loop][n_frag] = MmaT::mma(                           \
                        a_regs[compute_slot].fp8x8[0], b_reg.fp8x8[0],         \
                        acc[m_loop][n_frag]);                                  \
                    acc[m_loop][n_frag] = MmaT::mma(                           \
                        a_regs[compute_slot].fp8x8[1], b_reg.fp8x8[1],         \
                        acc[m_loop][n_frag]);                                  \
                }                                                              \
                if (m_loop + 2 < kMSubtiles) {                                 \
                    wait_lgkmcnt_one_ht();                                     \
                } else if (m_loop + 1 < kMSubtiles) {                          \
                    wait_lgkmcnt_zero_ht();                                    \
                }                                                              \
            }                                                                  \
        } while (0)

    // =========================================================================
    // Prologue: tiles 0/1 -> stages 0/1 (depth-2 prefetch), B0/B1 -> set A
    // =========================================================================
    {
        LOAD_A_TILE_PAIR_HT(a_lds_base_addr, 0);
        LOAD_B_TILE_HT(0, B_EVEN_AT_HT);
        if (k_tiles > 1) {
            LOAD_A_TILE_PAIR_HT(a_lds_base_addr + kAStageBytes, BLOCK_K);
            LOAD_B_TILE_HT(1, B_ODD_AT_HT);
        }
    }

    // =========================================================================
    // Main loop: 4-stage A LDS pipeline (kAStages), one stage barrier per
    // half-iteration.  Baseline ordering: A(k+2) in the alpha half,
    // A(k+3) after the beta barrier.  A GROUP boundary can only land on an
    // even tile (K_IN_ITERS is even), so only the B_EVEN compute carries
    // the rescale.
    // =========================================================================
    int k_tile = 0;
    for (; k_tile + 2 < k_tiles; k_tile += 2) {
        stage_sync(1);
        LOAD_A_TILE_PAIR_HT(
            a_lds_base_addr + (((k_tile + 2) % kAStages) * kAStageBytes),
            (k_tile + 2) * BLOCK_K);

        const bool gw_boundary =
            (k_tile > 0 && (k_tile % K_IN_ITERS) == 0);
        float gw_rg[kMSubtiles];
        if (gw_boundary) {
            const int g = k_tile / K_IN_ITERS;
            #pragma unroll
            for (int m = 0; m < kMSubtiles; ++m) {
                gw_rg[m] = SMEM_SCALE_ROW(m)[g];
            }
        }
        COMPUTE_TILE_GW_HT(k_tile, B_EVEN_AT_HT, gw_boundary);
        LOAD_B_TILE_HT(k_tile + 2, B_EVEN_AT_HT);

        stage_sync(1);
        if (k_tile + 3 < k_tiles) {
            LOAD_A_TILE_PAIR_HT(
                a_lds_base_addr + (((k_tile + 3) % kAStages) * kAStageBytes),
                (k_tile + 3) * BLOCK_K);
        }
        COMPUTE_TILE_GW_HT(k_tile + 1, B_ODD_AT_HT, false);
        if (k_tile + 3 < k_tiles) {
            LOAD_B_TILE_HT(k_tile + 3, B_ODD_AT_HT);
        }
    }

    // =========================================================================
    // Epilogue: drain the last 1-2 tiles (loads already issued)
    // =========================================================================
    {
        const int remaining_tiles = k_tiles - k_tile;
        if (remaining_tiles >= 1) {
            if (remaining_tiles == 2) {
                stage_sync(1);
            } else {
                stage_sync_drain();
            }
            const bool gw_boundary =
                (k_tile > 0 && (k_tile % K_IN_ITERS) == 0);
            float gw_rg[kMSubtiles];
            if (gw_boundary) {
                const int g = k_tile / K_IN_ITERS;
                #pragma unroll
                for (int m = 0; m < kMSubtiles; ++m) {
                    gw_rg[m] = SMEM_SCALE_ROW(m)[g];
                }
            }
            COMPUTE_TILE_GW_HT(k_tile, B_EVEN_AT_HT, gw_boundary);
            ++k_tile;
            if (k_tile < k_tiles) {
                stage_sync_drain();
                COMPUTE_TILE_GW_HT(k_tile, B_ODD_AT_HT, false);
            }
        }
    }

    // =========================================================================
    // Dequant and output: acc * s_last(row) * col_scale -> BF16
    // =========================================================================
    const bool full_block_valid =
        (m_block_start + BLOCK_M <= M) && (valid_m >= BLOCK_M);

    // Column scales are m-loop invariant: hoist them out (one load per
    // fragment instead of one per row).
    float col_scale[kNFragments][kMmacVgprs];
    #pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int col_base = n_block_start +
                             warp_id * kNFragments * kMmacCols +
                             n_frag * kMmacCols + lane_group * kMmacVgprs;
        #pragma unroll
        for (int elem = 0; elem < kMmacVgprs; ++elem) {
            col_scale[n_frag][elem] = matrixB_scale[col_base + elem];
        }
    }

    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row = m_block_start + m_loop * kMmacRows + lane_row;
        if (!full_block_valid &&
            (row >= M || m_loop * kMmacRows + lane_row >= valid_m)) {
            continue;
        }
        const float row_scale = SMEM_SCALE_ROW(m_loop)[0];  // slot 0 = s_last
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int col_base = n_block_start +
                                 warp_id * kNFragments * kMmacCols +
                                 n_frag * kMmacCols + lane_group * kMmacVgprs;
            const float v0 = MmaT::to_float(acc[m_loop][n_frag][0]) * row_scale * col_scale[n_frag][0];
            const float v1 = MmaT::to_float(acc[m_loop][n_frag][1]) * row_scale * col_scale[n_frag][1];
            const float v2 = MmaT::to_float(acc[m_loop][n_frag][2]) * row_scale * col_scale[n_frag][2];
            const float v3 = MmaT::to_float(acc[m_loop][n_frag][3]) * row_scale * col_scale[n_frag][3];
            *reinterpret_cast<uint64_t*>(
                MatrixC + static_cast<int64_t>(row) * C_stride + col_base) =
                pack_bf16x4_ht(v0, v1, v2, v3);
        }
    }

    #undef B_EVEN_AT_HT
    #undef B_ODD_AT_HT
    #undef LOAD_A_TILE_PAIR_HT
    #undef LOAD_B_TILE_HT
    #undef COMPUTE_TILE_GW_HT
    #undef SMEM_SCALE_ROW
    #undef RESCALE_GW_M
    #undef RESCALE_GW
}

// ===========================================================================
// Dedicated BLOCK_M=256 GEMM for split-HT (gfx938), L1 SwiGLU+quant variant.
//
// Warp layout: 8 math warps (like the proven BM128/8-warp mapping); each
// warp covers ALL 256 rows (16 M-subtiles) x 32 N columns, holding
// acc[16][2] = 128 acc VGPRs.  The earlier 16-warp 2x8-panel variant kept
// acc at 64 VGPRs but fetched every B tile twice (both panels share the
// same 32-col set) and measured ~2.3x slower than BM128 at 16384 tokens.
// Fewer math warps also raise the per-thread VGPR budget (768-thread
// launch -> up to 256 VGPR/thread), which acc[16][2] needs.
//
// BM256-specific structure:
//   - A-LDS is 2-stage (32KB; 3 stages would not fit LDS with the other
//     kernel buffers).  Each warp issues TWO matrix_load_64x16 ops per
//     stage: subtiles w*16 and w*16+128.
//   - Pipeline: A prefetch depth 1 / B depth 2.  Stage s((k+1)&1) is only
//     refilled after the top-of-tile barrier guarantees every warp's
//     compute(k-1) reads are done; vmcnt(2) at the barrier drains the two
//     A(k+1) matrix_loads while allowing the newest B(k+2) pair to stay
//     in flight.
//   - B tiles are fetched once per column set (8 warps x 32 cols = 256).
//
// Epilogue: identical SwiGLU + group quant to the BM128 variant; every
// warp owns all 16 row subtiles.  smem_amax holds NUM_WARPS*BLOCK_M =
// 8*256 = 2048 floats (8KB).
// ===========================================================================
template <int32_t N, int K,
          int BLOCK_N = 256, int BLOCK_K = 64,
          bool kIsInt8 = false>
__device__ void gemm_ht_tile_bm256_bk64_silu_quant(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    const int32_t M,
    const uint32_t math_warp_offset,
    uint8_t* __restrict__ l2_token_out,
    float* __restrict__ l2_sf,
    const int32_t m_idx,
    const int32_t sf_stride,
    const int32_t ih,
    float* __restrict__ smem_amax,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense,
    const float activation_clamp,
    const bool fast_math)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    constexpr int BLOCK_M = 256;
    // 16 math warps, N-split: each warp owns a distinct 16-col set
    // (1 fragment) and computes ALL 256 rows (16 subtiles), so
    // acc[16][1] = 64 VGPRs -- same per-lane accumulator profile as the
    // proven BM128 kernel.  B is loaded once per column set: the old
    // 2x8 M-panel layout duplicated every B tile (2x GMEM/L2 traffic),
    // and the 8-warp all-rows variant needed acc[16][2] = 128 regs and
    // spilled ~932B/thread (456ms @16384).
    constexpr int NUM_WARPS = 16;
    static_assert(BLOCK_M % kMmacRows == 0 && BLOCK_N % (NUM_WARPS * kMmacCols) == 0 &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;      // 16
    constexpr int kK64Tile = 64;
    constexpr int kNFragments = BLOCK_N / (NUM_WARPS * kMmacCols);  // 1
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;    // 16KB
    constexpr int kAStages = 2;
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;        // 1KB
    constexpr int kALdsRowPairStride = 128;

    static_assert(BLOCK_K == kK64Tile, "BM256 path expects BK=64");
    static_assert(kNFragments == 1, "BM256 path expects BN=256 (1 frag/warp)");
    static_assert(NUM_WARPS * kNFragments * 8 == BLOCK_N / 2,
                  "warps must exactly cover the SwiGLU output width");
    constexpr int kGS = BLOCK_N / 2;  // one group per call's output width

    // stage_sync leaves the newest 2 loads (the B(k+1) pair) in flight across
    // the barrier; s_barrier only waits for wave arrival, not the counters.
    // stage_sync_drain is for the last tile, whose own loads must all be
    // resident before compute.
    auto stage_sync = [&]() { wait_vmcnt_all_barrier_ht<2>(); };
    auto stage_sync_drain = [&]() { wait_vmcnt_all_barrier_ht<0>(); };

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);   // 0..15
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;
    const int n_warp = warp_id;                 // 0..15: N-column owner

    const int n_block_start = tile_n * BLOCK_N;
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    // Each warp matrix_loads exactly one 16-row subtile (w*16); 16 warps
    // cover all 256 rows.
    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m0 = warp_id * kMmacRows;
    const int a_lds_write_bytes0 = a_load_m0 * BLOCK_K;

    // N base for this warp (each warp owns a distinct 16-col set)
    const int logical_n_base =
        n_block_start + n_warp * kNFragments * kMmacCols + (lane_id & 15);

    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    BOperandHT b_even0, b_even1;
    BOperandHT b_odd0, b_odd1;

    #define B_EVEN_AT_BM256(IDX) ((IDX) == 0 ? b_even0 : b_even1)
    #define B_ODD_AT_BM256(IDX) ((IDX) == 0 ? b_odd0 : b_odd1)

    const char* a_ptr_base0 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m0) * K;

    // One matrix_load per warp per stage: subtile w*16.
    #define LOAD_A_TILE_BM256(STAGE_BASE, K_OFFSET) \
        do { \
            const char* a0 = a_ptr_base0 + static_cast<int64_t>(K_OFFSET); \
            vec4_u32 srsrc0 = make_matrix_resource_ht(a0, K); \
            MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes0, srsrc0); \
        } while (0)

    #define LOAD_B_TILE_BM256(KT, B_SEL) \
        do { \
            _Pragma("unroll") \
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) { \
                const int logical_n = logical_n_base + n_frag * kMmacCols; \
                const int n_band = logical_n / kMmacCols; \
                const int n_inner = logical_n & (kMmacCols - 1); \
                const int b_k = (KT) * BLOCK_K + lane_group * 16; \
                const int b_k_band = b_k / kK64Tile; \
                const int b_k_mid = (b_k & (kK64Tile - 1)) / 16; \
                const int b_k_inner = b_k & 15; \
                const int b_tile = \
                    (b_k_band * b_n_tiles + n_band) * 4 + b_k_mid; \
                const int b_offset = expert_b_base + \
                    (b_tile * kMmacCols + n_inner) * 16 + b_k_inner; \
                /* Plain compiler-visible vector load.  The inline-asm \
                   buffer_load_dwordx4 used by the BM128 path miscompiled \
                   under this 16-warp/1024-thread instantiation (B operand \
                   regs came back overwritten with stale data incl. 0x80 \
                   FNUZ-NaN bytes), so keep the plain load here. */ \
                B_SEL(n_frag).i32x4 = *reinterpret_cast<const vec4_i32*>( \
                    matrixB_w6 + b_offset); \
            } \
        } while (0)

    // Compute one K-tile from stage (KT & 1): all 16 subtiles live in the
    // shared 16KB stage.  Same proven 2-slot A-frag ping-pong as BM128.
    #define COMPUTE_TILE_BM256(KT, B_SEL) \
        do { \
            const int a_stage_base_c = a_lds_base_addr + \
                ((KT) & 1) * kAStageBytes; \
            AOperandHT a_regs[2]; \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 0, a_regs[0]); \
            wait_lgkmcnt_zero_ht(); \
            wait_vmcnt5_zero_ht(); \
            _Pragma("unroll") \
            for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) { \
                const int compute_slot = m_loop & 1; \
                const int load_slot = compute_slot ^ 1; \
                if (m_loop + 1 < kMSubtiles) { \
                    DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, m_loop + 1, \
                                              a_regs[load_slot]); \
                } \
                _Pragma("unroll") \
                for (int n_frag = 0; n_frag < kNFragments; ++n_frag) { \
                    const BOperandHT& b_reg = B_SEL(n_frag); \
                    acc[m_loop][n_frag] = MmaT::mma( \
                        a_regs[compute_slot].fp8x8[0], b_reg.fp8x8[0], \
                        acc[m_loop][n_frag]); \
                    acc[m_loop][n_frag] = MmaT::mma( \
                        a_regs[compute_slot].fp8x8[1], b_reg.fp8x8[1], \
                        acc[m_loop][n_frag]); \
                } \
                if (m_loop + 1 < kMSubtiles) { \
                    wait_lgkmcnt_zero_ht(); \
                } \
            } \
        } while (0)

    // =========================================================================
    // Prologue: A(0)->s0, B(0)->even, B(1)->odd (B kept two tiles deep).
    // =========================================================================
    LOAD_A_TILE_BM256(a_lds_base_addr, 0);
    LOAD_B_TILE_BM256(0, B_EVEN_AT_BM256);
    if (k_tiles > 1) {
        LOAD_B_TILE_BM256(1, B_ODD_AT_BM256);
    }

    // =========================================================================
    // Main loop: 2-stage A pipeline, one barrier per K-tile.  The barrier at
    // the top of iteration k guarantees all warps finished compute(k-1), so
    // stage s((k+1)&1) is safe to refill with A(k+1).
    // =========================================================================
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        // vmcnt(2) leaves the B(k+1) pair (the newest ops) in flight and
        // drains A(k); on the last tile no B(k+1) was issued, so A(k) would
        // be among the two newest and could stay unlanded -- drain fully.
        if (k_tile + 1 < k_tiles) {
            stage_sync();
        } else {
            stage_sync_drain();
        }
        if (k_tile + 1 < k_tiles) {
            LOAD_A_TILE_BM256(
                a_lds_base_addr + (((k_tile + 1) & 1) * kAStageBytes),
                (k_tile + 1) * BLOCK_K);
        }
        if (k_tile & 1) {
            COMPUTE_TILE_BM256(k_tile, B_ODD_AT_BM256);
        } else {
            COMPUTE_TILE_BM256(k_tile, B_EVEN_AT_BM256);
        }
        if (k_tile + 2 < k_tiles) {
            if (k_tile & 1) {
                LOAD_B_TILE_BM256(k_tile + 2, B_ODD_AT_BM256);
            } else {
                LOAD_B_TILE_BM256(k_tile + 2, B_EVEN_AT_BM256);
            }
        }
    }

    // =========================================================================
    // Fused SwiGLU + group-wise FP8 quant epilogue (full-tile mapping)
    // Fragment (m_loop, n_frag): row = m_block_start + m_loop*16 +
    // lane_row; raw cols n_warp*32 + n_frag*16 + lane_group*4.
    // =========================================================================
    constexpr int kOutPerWarp = kNFragments * 8;         // 16 out cols
    static_assert(kGS % kOutPerWarp == 0, "quant group must cover a whole number of warps");
    constexpr int kWarpsPerGroup = kGS / kOutPerWarp;    // 8 = NUM_WARPS
    const int out_base = tile_n * (BLOCK_N / 2);
    const int group_warp_base = 0;                       // all warps, one group
    const int group_idx = out_base / kGS;                // == tile_n

    constexpr int kFragHalf = (kNFragments + 1) / 2;
    float part[kMSubtiles];
    uint64_t silu_pk[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) part[m_loop] = 0.0f;
    #pragma unroll
    for (int half = 0; half * kFragHalf < kNFragments; ++half) {
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            const int row_local = m_loop * kMmacRows + lane_row;
            const int row_blk = row_local;
            const int row = m_block_start + row_blk;
            const bool row_ok = (row_blk < valid_m) && (row < M);
            const float row_scale = row_ok ? matrixA_scale[row] : 0.0f;
            #pragma unroll
            for (int n_frag = half * kFragHalf;
                 n_frag < min((half + 1) * kFragHalf, kNFragments); ++n_frag) {
                const int col_base = n_block_start +
                    n_warp * kNFragments * kMmacCols +
                    n_frag * kMmacCols + lane_group * kMmacVgprs;
                uint64_t sv;
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; ++i) {
                    const float v = MmaT::to_float(acc[m_loop][n_frag][i])
                        * row_scale * matrixB_scale[col_base + i];
                    const float partner = __shfl_xor(v, 32);
                    const float g = (lane_group < 2) ? v : partner;
                    const float u = (lane_group < 2) ? partner : v;
                    const float s = swiglu_nv(g, u, activation_clamp, fast_math);
                    part[m_loop] = fmaxf(part[m_loop], fabsf(s));
                    reinterpret_cast<hip_bfloat16*>(&sv)[i] = hip_bfloat16(s);
                }
                silu_pk[m_loop][n_frag] = sv;
            }
        }
    }
    // Row-amax reduction via a 256-entry atomic-max table (1KB, overlaid
    // on the count/amax scratch): 16 warps x 256 rows as a [warp][row]
    // table would need 16KB which does not fit the SMEM budget.  p >= 0,
    // so unsigned compare == float compare.
    uint32_t* const amax_u = reinterpret_cast<uint32_t*>(smem_amax);
    for (int i = math_tid; i < BLOCK_M; i += NUM_WARPS * kWarpSize) {
        amax_u[i] = 0;
    }
    wait_lgkmcnt0_all_barrier_ht();
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row_local = m_loop * kMmacRows + lane_row;
        const int row_blk = row_local;
        const int row = m_block_start + row_blk;
        const bool row_ok = (row_blk < valid_m) && (row < M);
        float p = row_ok ? part[m_loop] : 0.0f;
        p = fmaxf(p, __shfl_xor(p, 16));
        p = fmaxf(p, __shfl_xor(p, 32));
        if (lane_id < 16) {
            atomicMax(amax_u + row_local, __float_as_uint(p));
        }
    }

    // Publish the amax table to all warps.
    wait_lgkmcnt0_all_barrier_ht();

#ifndef DG_MEGA_L1_NOQUANT
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row_local = m_loop * kMmacRows + lane_row;
        const int row_blk = row_local;
        const int row = m_block_start + row_blk;
        const bool row_ok = (row_blk < valid_m) && (row < M);
        const float row_amax = __uint_as_float(amax_u[row_local]);
        const float qmax = kIsInt8 ? 127.0f : 448.0f;
        const float clamped_amax = fmaxf(row_amax, kIsInt8 ? 1.0e-6f : 1.0e-4f);
        const float scale = clamped_amax / qmax;
        const float inv_scale = qmax / clamped_amax;

        uint32_t packed[kNFragments];
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const hip_bfloat16* svb =
                reinterpret_cast<const hip_bfloat16*>(&silu_pk[m_loop][n_frag]);
            float q[kMmacVgprs];
            #pragma unroll
            for (int i = 0; i < kMmacVgprs; ++i) {
                const float s = static_cast<float>(svb[i]);
                q[i] = quant_clamp<kIsInt8>(s * inv_scale, qmax);
            }
            uint32_t p = 0;
            if constexpr (kIsInt8) {
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; ++i) {
                    p |= (static_cast<uint32_t>(static_cast<uint8_t>(
                              static_cast<int8_t>(__float2int_rn(q[i]))))
                          << (i * 8));
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kMmacVgprs; i += 2) {
                    float2 f2{q[i], q[i + 1]};
                    auto fp8x2 = __hip_cvt_float2_to_fp8x2(
                        f2, __HIP_SATFINITE, __HIP_E4M3);
                    p |= (static_cast<uint32_t>(
                              *reinterpret_cast<uint16_t*>(&fp8x2)) << (i * 8));
                }
            }
            packed[n_frag] = p;
        }

        if (row_ok) {
            if constexpr (kNFragments == 1) {
                // 8 output cols per warp = one 8B store per row.
                uint2 v2;
                v2.x = __shfl(packed[0], lane_id);
                v2.y = __shfl(packed[0], lane_id + 16);
                if (lane_id < 16) {
                    *reinterpret_cast<uint2*>(
                        l2_token_out +
                        static_cast<int64_t>(m_idx + row_blk) * ih +
                        out_base + n_warp * kOutPerWarp) = v2;
                }
            } else {
            #pragma unroll
            for (int chunk = 0; chunk < kOutPerWarp / 16; ++chunk) {
                uint4 v4;
                v4.x = __shfl(packed[chunk * 2 + 0], lane_id);
                v4.y = __shfl(packed[chunk * 2 + 0], lane_id + 16);
                v4.z = __shfl(packed[chunk * 2 + 1], lane_id);
                v4.w = __shfl(packed[chunk * 2 + 1], lane_id + 16);
                if (lane_id < 16) {
                    *reinterpret_cast<uint4*>(
                        l2_token_out +
                        static_cast<int64_t>(m_idx + row_blk) * ih +
                        out_base + n_warp * kOutPerWarp + chunk * 16) = v4;
                }
            }
            }
        }
        if (lane_id < 16 && row_ok && warp_id == 0) {
            l2_sf[static_cast<int64_t>(m_idx + row_blk) * sf_stride
                  + group_idx] = scale;
        }
    }
#else
    (void)silu_pk; (void)group_warp_base; (void)kWarpsPerGroup;
#endif

    #undef B_EVEN_AT_BM256
    #undef B_ODD_AT_BM256
    #undef LOAD_A_TILE_BM256
    #undef LOAD_B_TILE_BM256
    #undef COMPUTE_TILE_BM256
    (void)p_bar_count; (void)p_bar_sense;
}

// ===========================================================================
// Dedicated BLOCK_M=256 group-wise dequant GEMM for the split-HT L2 GEMM.
// Same 8-math-warp organization as gemm_ht_tile_bm256_bk64_silu_quant
// (each warp covers all 256 rows x 32 cols, acc[16][2] = 128 VGPRs,
// 2-stage A-LDS, A-depth-1/B-depth-2 pipeline, single B fetch).
//
// Unlike the BM128 variant the per-row boundary-ratio table is NOT
// pre-staged in SMEM (a [256x24] float table would not fit the LDS budget);
// the ratios r_g = s_{g-1}/s_g are read straight from the global l2_sf
// buffer via matrixA_scale (layout [row][K_OUT_ITERS]).  Slot 0 semantics
// are preserved by reading s_last = s[K_OUT_ITERS-1] in the epilogue.
//
// Output is a COMPACT [BLOCK_M x BLOCK_N] BF16 tile: the caller passes
// C_stride == BLOCK_N and MatrixC already offset for this n-tile.
// ===========================================================================
template <int32_t N, int K, int32_t GROUP_SIZE,
          int BLOCK_N = 256, int BLOCK_K = 64,
          bool kIsInt8 = false>
__device__ void gemm_ht_tile_bm256_bk64_groupwise(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,   // [rows, K_OUT_ITERS] group sf
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    constexpr int BLOCK_M = 256;
    // 16 math warps, N-split: each warp owns a distinct 16-col set
    // (1 fragment) and computes all 256 rows -> acc[16][1] = 64 VGPRs.
    // B is loaded once per column set (no panel duplication).
    constexpr int NUM_WARPS = 16;
    static_assert(BLOCK_M % kMmacRows == 0 && BLOCK_N % (NUM_WARPS * kMmacCols) == 0 &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;      // 16
    constexpr int kK64Tile = 64;
    constexpr int kNFragments = BLOCK_N / (NUM_WARPS * kMmacCols);  // 1
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;
    constexpr int kALdsRowPairStride = 128;
    constexpr int K_OUT_ITERS = K / GROUP_SIZE;        // 24
    static_assert(GROUP_SIZE % BLOCK_K == 0 && K % GROUP_SIZE == 0,
                  "K/GROUP_SIZE/BLOCK_K must tile evenly");
    constexpr int K_IN_ITERS = GROUP_SIZE / BLOCK_K;   // 2

    static_assert(BLOCK_K == kK64Tile, "BM256 path expects BK=64");
    static_assert(BLOCK_N == 256, "BM256 path expects BN=256");
    static_assert(K_IN_ITERS == 2, "expects GROUP_SIZE=2*BLOCK_K");

    // Same 2-deep B-in-flight semantics as the channelwise variant above,
    // on the all-wave s_barrier mirrored by the non-math warps.
    auto stage_sync = [&]() {
        wait_vmcnt_all_barrier_ht<2>();
    };
    auto stage_sync_drain = [&]() {
        wait_vmcnt_all_barrier_ht<0>();
    };

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;
    const int n_warp = warp_id;

    const int n_block_start = tile_n * BLOCK_N;
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    // Each warp matrix_loads exactly one 16-row subtile (w*16).
    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m0 = warp_id * kMmacRows;
    const int a_lds_write_bytes0 = a_load_m0 * BLOCK_K;

    const int logical_n_base =
        n_block_start + n_warp * kNFragments * kMmacCols + (lane_id & 15);

    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    BOperandHT b_even0, b_even1;
    BOperandHT b_odd0, b_odd1;

    #define B_EVEN_AT_BM256(IDX) ((IDX) == 0 ? b_even0 : b_even1)
    #define B_ODD_AT_BM256(IDX) ((IDX) == 0 ? b_odd0 : b_odd1)

    const char* a_ptr_base0 =
        matrixA + static_cast<int64_t>(m_block_start + a_load_m0) * K;

    #define LOAD_A_TILE_BM256(STAGE_BASE, K_OFFSET) \
        do { \
            const char* a0 = a_ptr_base0 + static_cast<int64_t>(K_OFFSET); \
            vec4_u32 srsrc0 = make_matrix_resource_ht(a0, K); \
            MATRIX_LOAD_64X16_B8_T_LDS_HT((STAGE_BASE) + a_lds_write_bytes0, srsrc0); \
        } while (0)

    // Group-wise scale reads go straight to GMEM (the l2_sf table), no SMEM
    // staging: r_g = s_{g-1}/s_g per row; s_last = s[K_OUT_ITERS-1].
    #define GW_ROW_SF(m) \
        (matrixA_scale + \
         ((m) * kMmacRows + lane_row) * K_OUT_ITERS)
    #define GW_RG(m, g) (GW_ROW_SF(m)[(g) - 1] / GW_ROW_SF(m)[g])

    #define RESCALE_GW_M_BM256(m, rg) \
        do { \
            _Pragma("unroll") \
            for (int f = 0; f < kNFragments; ++f) { \
                _Pragma("unroll") \
                for (int i = 0; i < kMmacVgprs; ++i) { \
                    acc[m][f][i] = MmaT::rescale(acc[m][f][i], rg); \
                } \
            } \
        } while (0)

    #define LOAD_B_TILE_BM256(KT, B_SEL) \
        do { \
            _Pragma("unroll") \
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) { \
                const int logical_n = logical_n_base + n_frag * kMmacCols; \
                const int n_band = logical_n / kMmacCols; \
                const int n_inner = logical_n & (kMmacCols - 1); \
                const int b_k = (KT) * BLOCK_K + lane_group * 16; \
                const int b_k_band = b_k / kK64Tile; \
                const int b_k_mid = (b_k & (kK64Tile - 1)) / 16; \
                const int b_k_inner = b_k & 15; \
                const int b_tile = \
                    (b_k_band * b_n_tiles + n_band) * 4 + b_k_mid; \
                const int b_offset = expert_b_base + \
                    (b_tile * kMmacCols + n_inner) * 16 + b_k_inner; \
                /* Plain compiler-visible vector load -- the BM128 inline-asm \
                   buffer_load_dwordx4 miscompiles under the 16-warp/1024- \
                   thread instantiation (B regs return stale/NaN data). */ \
                B_SEL(n_frag).i32x4 = *reinterpret_cast<const vec4_i32*>( \
                    matrixB_w6 + b_offset); \
            } \
        } while (0)

    #define COMPUTE_TILE_GW_BM256(KT, B_SEL, GW_BOUNDARY) \
        do { \
            const int a_stage_base_c = a_lds_base_addr + \
                ((KT) & 1) * kAStageBytes; \
            AOperandHT a_regs[2]; \
            DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, 0, a_regs[0]); \
            wait_lgkmcnt_zero_ht(); \
            wait_vmcnt5_zero_ht(); \
            _Pragma("unroll") \
            for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) { \
                const int compute_slot = m_loop & 1; \
                const int load_slot = compute_slot ^ 1; \
                if (m_loop + 1 < kMSubtiles) { \
                    DS_READ_A_M16_FROM_LDS_HT(a_stage_base_c, m_loop + 1, \
                                              a_regs[load_slot]); \
                } \
                if (GW_BOUNDARY) { \
                    RESCALE_GW_M_BM256(m_loop, gw_rg[m_loop]); \
                } \
                _Pragma("unroll") \
                for (int n_frag = 0; n_frag < kNFragments; ++n_frag) { \
                    const BOperandHT& b_reg = B_SEL(n_frag); \
                    acc[m_loop][n_frag] = MmaT::mma( \
                        a_regs[compute_slot].fp8x8[0], b_reg.fp8x8[0], \
                        acc[m_loop][n_frag]); \
                    acc[m_loop][n_frag] = MmaT::mma( \
                        a_regs[compute_slot].fp8x8[1], b_reg.fp8x8[1], \
                        acc[m_loop][n_frag]); \
                } \
                if (m_loop + 1 < kMSubtiles) { \
                    wait_lgkmcnt_zero_ht(); \
                } \
            } \
        } while (0)

    LOAD_A_TILE_BM256(a_lds_base_addr, 0);
    LOAD_B_TILE_BM256(0, B_EVEN_AT_BM256);
    if (k_tiles > 1) {
        LOAD_B_TILE_BM256(1, B_ODD_AT_BM256);
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        if (k_tile + 1 < k_tiles) {
            stage_sync();
        } else {
            stage_sync_drain();
        }
        if (k_tile + 1 < k_tiles) {
            LOAD_A_TILE_BM256(
                a_lds_base_addr + (((k_tile + 1) & 1) * kAStageBytes),
                (k_tile + 1) * BLOCK_K);
        }
        // A group boundary can only land on an even tile (K_IN_ITERS == 2).
        const bool gw_boundary =
            (k_tile > 0 && (k_tile % K_IN_ITERS) == 0);
        float gw_rg[kMSubtiles];
        if (gw_boundary) {
            const int g = k_tile / K_IN_ITERS;
            #pragma unroll
            for (int m = 0; m < kMSubtiles; ++m) {
                const int row_blk_g =
                    m * kMmacRows + lane_row;
                const int row_g = m_block_start + row_blk_g;
                gw_rg[m] =
                    (row_blk_g < valid_m && row_g < M) ? GW_RG(m, g) : 1.0f;
            }
        }
        if (k_tile & 1) {
            COMPUTE_TILE_GW_BM256(k_tile, B_ODD_AT_BM256, gw_boundary);
        } else {
            COMPUTE_TILE_GW_BM256(k_tile, B_EVEN_AT_BM256, gw_boundary);
        }
        if (k_tile + 2 < k_tiles) {
            if (k_tile & 1) {
                LOAD_B_TILE_BM256(k_tile + 2, B_ODD_AT_BM256);
            } else {
                LOAD_B_TILE_BM256(k_tile + 2, B_EVEN_AT_BM256);
            }
        }
    }

    // =========================================================================
    // Dequant and output: acc * s_last(row) * col_scale -> BF16, compact tile
    // =========================================================================
    const bool full_block_valid =
        (m_block_start + BLOCK_M <= M) && (valid_m >= BLOCK_M);

    float col_scale[kNFragments][kMmacVgprs];
    #pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int col_base = n_warp * kNFragments * kMmacCols +
                             n_frag * kMmacCols + lane_group * kMmacVgprs;
        #pragma unroll
        for (int elem = 0; elem < kMmacVgprs; ++elem) {
            col_scale[n_frag][elem] = matrixB_scale[n_block_start + col_base + elem];
        }
    }

    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row_blk = m_loop * kMmacRows + lane_row;
        const int row = m_block_start + row_blk;
        if (!full_block_valid &&
            (row >= M || row_blk >= valid_m)) {
            continue;
        }
        const float row_scale = GW_ROW_SF(m_loop)[K_OUT_ITERS - 1];  // s_last
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int col_base = n_warp * kNFragments * kMmacCols +
                                 n_frag * kMmacCols + lane_group * kMmacVgprs;
            const float v0 = MmaT::to_float(acc[m_loop][n_frag][0]) * row_scale * col_scale[n_frag][0];
            const float v1 = MmaT::to_float(acc[m_loop][n_frag][1]) * row_scale * col_scale[n_frag][1];
            const float v2 = MmaT::to_float(acc[m_loop][n_frag][2]) * row_scale * col_scale[n_frag][2];
            const float v3 = MmaT::to_float(acc[m_loop][n_frag][3]) * row_scale * col_scale[n_frag][3];
            *reinterpret_cast<uint64_t*>(
                MatrixC + static_cast<int64_t>(row_blk) * C_stride + col_base) =
                pack_bf16x4_ht(v0, v1, v2, v3);
        }
    }

    #undef B_EVEN_AT_BM256
    #undef B_ODD_AT_BM256
    #undef LOAD_A_TILE_BM256
    #undef LOAD_B_TILE_BM256
    #undef COMPUTE_TILE_GW_BM256
    #undef GW_ROW_SF
    #undef GW_RG
    #undef RESCALE_GW_M_BM256
    (void)math_warp_offset;
}

// ===========================================================================
// Group-wise dequant variant of gemm_ht_tile_bm64_bk128 for the HT L2 GEMM.
// Same rescale scheme as gemm_ht_tile_bm128_bk64_groupwise: acc holds raw
// partials; at each GROUP_SIZE boundary acc is multiplied by the boundary
// ratio r_g = s_{g-1}/s_g (SMEM table; slot 0 = s_last for the epilogue).
// For BK=128/GS=128 every K-tile is a group boundary (K_IN_ITERS=1).
// ===========================================================================
template <int32_t N, int K, int32_t GROUP_SIZE,
          int BLOCK_M = 64, int BLOCK_N = 256, int BLOCK_K = 128,
          int NUM_WARPS = 4, bool kIsInt8 = false,
          bool kAllWaveBarrier = false, int kAStagesArg = 3>
__device__ void gemm_ht_tile_bm64_bk128_groupwise(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB_w6,
    int expert_b_base,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride,
    const float* __restrict__ smem_scale,
    char* a_lds,
    volatile uint32_t* p_bar_count,
    volatile uint32_t* p_bar_sense)
{
    constexpr int kMmacRows = 16;
    constexpr int kMmacCols = 16;
    constexpr int kMmacVgprs = 4;
    static_assert(BLOCK_M % kMmacRows == 0 && BLOCK_N % (NUM_WARPS * kMmacCols) == 0 &&
                  BLOCK_K % 64 == 0 && N % kMmacCols == 0,
                  "tile dims must divide evenly by the MMA fragment sizes");
    constexpr int kMSubtiles = BLOCK_M / kMmacRows;  // 4
    constexpr int kK64Tile = 64;
    constexpr int kK64Loops = BLOCK_K / kK64Tile;    // 2
    constexpr int kNFragmentsPerWarp = 2;
    constexpr int kNLoopNum = BLOCK_N / (NUM_WARPS * kNFragmentsPerWarp * kMmacCols);
    constexpr int kNFragments = kNLoopNum * kNFragmentsPerWarp;  // 4
    constexpr int kAStageBytes = BLOCK_M * BLOCK_K;  // 8192
    constexpr int kALdsTileBytes = kMmacRows * BLOCK_K;  // 2048
    constexpr int kALdsRowPairStride = BLOCK_K;          // 128
    constexpr int K_OUT_ITERS = K / GROUP_SIZE;        // 24 for K=3072,GS=128
    static_assert(GROUP_SIZE % BLOCK_K == 0 && K % GROUP_SIZE == 0,
                  "K/GROUP_SIZE/BLOCK_K must tile evenly");
    constexpr int K_IN_ITERS = GROUP_SIZE / BLOCK_K;   // 1 for GS=128,BK=128

    static_assert(kMSubtiles == 4, "BM64 expects 4 M-subtiles");
    static_assert(kNFragments == 4, "BM64 expects 4 N-fragments per warp");
    static_assert(kK64Loops == 2, "BK128 expects 2 K64 loops");
    constexpr int kAStages = kAStagesArg;

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_id = broadcast_lane0(math_tid / kWarpSize);
    const int lane_id = math_tid % kWarpSize;
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;

    const int n_block_start = tile_n * BLOCK_N;
    const int m_block_start = tile_m * BLOCK_M;
    const int k_tiles = K / BLOCK_K;
    const int b_n_tiles = N / kMmacCols;

    const int a_lds_base_addr = static_cast<int>(reinterpret_cast<size_t>(a_lds));
    const int a_load_m = warp_id * kMmacRows;
    const int a_lds_write_bytes = a_load_m * BLOCK_K;
    // With NUM_WARPS > kMSubtiles (e.g. 8 warps for BM=64), warps beyond the
    // M-subtile count carry no A tile: they still issue their own B loads and
    // run the full MMA over all kMSubtiles subtiles read back from LDS.
    const bool loads_a = a_load_m < BLOCK_M;

    // Per-warp load issue counts per K-tile: 2*kNFragments B buffer_loads
    // (two K64 halves) plus one matrix A load on the loading warps only.
    // The vmcnt allowance keeps the loads issued in the just-elapsed half
    // in flight across the barrier; anything issued earlier must have
    // landed (same scheme as the bm128_bk64 tiles).
    constexpr int kBLoadsPerTile = 2 * kNFragments;
    auto stage_sync = [&](int n_a_issued) {
        if constexpr (kAllWaveBarrier) {
            if (loads_a && n_a_issued)
                wait_vmcnt_all_barrier_ht<1 + kBLoadsPerTile>();
            else
                wait_vmcnt_all_barrier_ht<kBLoadsPerTile>();
        } else {
            if (loads_a && n_a_issued)
                wait_vmcnt_subset_barrier_ht<1 + kBLoadsPerTile>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
            else
                wait_vmcnt_subset_barrier_ht<kBLoadsPerTile>(
                    p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };
    auto stage_sync_drain = [&]() {
        if constexpr (kAllWaveBarrier) {
            wait_vmcnt0_all_barrier_ht();
        } else {
            wait_vmcnt_subset_barrier_ht<0>(p_bar_count, p_bar_sense, NUM_WARPS);
        }
    };

    const int logical_n_base =
        n_block_start + warp_id * kNFragments * kMmacCols + (lane_id & 15);

    // B byte offset is affine in the K-tile index: b_k = KT*64+lane_group*16
    // stays inside one 64-row band, so b_k_band==KT, b_k_mid==lane_group and
    // b_k_inner==0.  Hoist the per-fragment constant out of the K loop --
    // each buffer_load then costs one IADD instead of ~10 ALU ops.
    const int b_stride_bytes = b_n_tiles * 4 * kMmacCols * 16;
    // Per-frag byte offset is affine: logical_n advances exactly one
    // 16-col group per fragment, so b_off(f) = b_off0 + f*(4*16*16).
    // One scalar instead of a kNFragments table (the f*step term folds
    // into the load's immediate add).
    const int b_off_base0 = expert_b_base +
        ((logical_n_base / kMmacCols) * 4 + lane_group) * (kMmacCols * 16) +
        (logical_n_base & (kMmacCols - 1)) * 16;
    constexpr int kBOffFragStep = 4 * kMmacCols * 16;

    using MmaT = GemmMmaTraits<kIsInt8>;
    typename MmaT::acc_t acc[kMSubtiles][kNFragments];
    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            acc[m_loop][n_frag] = MmaT::zero();
        }
    }

    const int a_lane_row_pair = (lane_id & 15) >> 1;
    const int a_lane_col_group = lane_id >> 4;
    const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3;
    const int a_lane_odd_k = (lane_id & 1) * kK64Tile;

    BOperandHT b_even0[kNFragments], b_even1[kNFragments];
    BOperandHT b_odd0[kNFragments], b_odd1[kNFragments];

    const char* a_ptr_base =
        matrixA + static_cast<int64_t>(m_block_start +
                                         (loads_a ? a_load_m : 0)) * K;

    auto load_b_tile_bn128 = [&](int k_base, BOperandHT (&b0)[kNFragments],
                                                BOperandHT (&b1)[kNFragments]) {
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int logical_n = logical_n_base + n_frag * kMmacCols;
            int b_offset0 = expert_b_base +
                w6_offset_ht(N, logical_n, k_base + lane_group * 16);
            buffer_load_i32x4_ht(matrixB_w6, b0[n_frag].i32x4, b_offset0);
        }
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int logical_n = logical_n_base + n_frag * kMmacCols;
            int b_offset1 = expert_b_base +
                w6_offset_ht(N, logical_n, k_base + kK64Tile + lane_group * 16);
            buffer_load_i32x4_ht(matrixB_w6, b1[n_frag].i32x4, b_offset1);
        }
    };

    // Groupwise scale table: smem_scale[(m*kMmacRows + lane_row) *
    // K_OUT_ITERS + g] = r_row(g); slot 0 holds s_last for the epilogue.
    #define SMEM_SCALE_ROW_BM64(m) \
        (smem_scale + (m * kMmacRows + lane_row) * K_OUT_ITERS)
    #define RESCALE_GW_BM64(g)                                            \
        do {                                                              \
            _Pragma("unroll")                                             \
            for (int m = 0; m < kMSubtiles; ++m) {                        \
                const float rgm = SMEM_SCALE_ROW_BM64(m)[g];              \
                _Pragma("unroll")                                         \
                for (int f = 0; f < kNFragments; ++f) {                   \
                    _Pragma("unroll")                                     \
                    for (int i = 0; i < kMmacVgprs; ++i) {                \
                        acc[m][f][i] = MmaT::rescale(acc[m][f][i], rgm);  \
                    }                                                     \
                }                                                         \
            }                                                             \
        } while (0)

    auto compute_k64 = [&](const char* a_stage, int k64_loop,
                           const BOperandHT (&b_reg)[kNFragments]) {
        AOperandHT a_regs[kMSubtiles];
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            DS_READ_A_M16_BK128_FROM_LDS_HT(
                static_cast<int>(reinterpret_cast<size_t>(a_stage)),
                m_loop, k64_loop, a_regs[m_loop],
                kALdsTileBytes, kALdsRowPairStride);
        }
        wait_lgkmcnt_zero_ht();
        // 9 loads stay in flight per half (1 A matrix_load + 8 B
        // buffer_loads); vmcnt(9)/(8) only drains loads issued >=2 halves
        // ago -- the current tile's B regs are guaranteed resident without
        // stalling the next tile's prefetch.
        if (k64_loop == 0) {
            asm volatile("s_waitcnt vmcnt(9)" ::: "memory");
        } else {
            asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
        }
        #pragma unroll
        for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
            #pragma unroll
            for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
                acc[m_loop][n_frag] = MmaT::mma(
                    a_regs[m_loop].fp8x8[0], b_reg[n_frag].fp8x8[0],
                    acc[m_loop][n_frag]);
                acc[m_loop][n_frag] = MmaT::mma(
                    a_regs[m_loop].fp8x8[1], b_reg[n_frag].fp8x8[1],
                    acc[m_loop][n_frag]);
            }
        }
    };

    // Prologue: tiles 0/1 -> ring stages 0/1 (depth-2 prefetch) plus the
    // matching B register tiles.  No barrier here: nothing reads the stages
    // until the first stage_sync in the main loop/epilogue.
    {
        if (loads_a) {
            vec4_u32 a_srsrc0 = make_matrix_resource_ht(a_ptr_base, K);
            MATRIX_LOAD_128X16_B8_T_LDS_HT(
                a_lds_base_addr + a_lds_write_bytes, a_srsrc0);
        }
        load_b_tile_bn128(0, b_even0, b_even1);
        if (k_tiles > 1) {
            if (loads_a) {
                const char* a_ptr1 = a_ptr_base + BLOCK_K;
                vec4_u32 a_srsrc1 = make_matrix_resource_ht(a_ptr1, K);
                MATRIX_LOAD_128X16_B8_T_LDS_HT(
                    a_lds_base_addr + kAStageBytes + a_lds_write_bytes,
                    a_srsrc1);
            }
            load_b_tile_bn128(BLOCK_K, b_odd0, b_odd1);
        }
    }

    // Main loop: kAStages-deep A LDS ring + register-level B double buffer.
    // Same structure as gemm_ht_tile_bm128_bk64_groupwise (d8d56dd):
    //   alpha sync -> issue A(k+2) -> rescale/compute k -> issue B(k+2)
    //   beta  sync -> issue A(k+3) -> rescale/compute k+1 -> issue B(k+3)
    // A(k+2)->stage (k+2)%kAStages is safe right after alpha because its last
    // reader was compute(k-1) in the previous iteration.  A(k+3) reuses the
    // stage compute(k) just finished, so it must wait for the beta barrier.
    int k_tile = 0;
    for (; k_tile + 2 < k_tiles; k_tile += 2) {
        stage_sync(0);   // alpha
        if (loads_a) {
            const char* a_nx_ptr =
                a_ptr_base + static_cast<int64_t>(k_tile + 2) * BLOCK_K;
            vec4_u32 a_nx_srsrc = make_matrix_resource_ht(a_nx_ptr, K);
            MATRIX_LOAD_128X16_B8_T_LDS_HT(
                a_lds_base_addr + ((k_tile + 2) % kAStages) * kAStageBytes +
                    a_lds_write_bytes,
                a_nx_srsrc);
        }
        // Group-boundary rescale for the even tile (group k_tile).
        if (k_tile % K_IN_ITERS == 0 && k_tile > 0) {
            RESCALE_GW_BM64(k_tile / K_IN_ITERS);
        }
        {
            const char* a_stage =
                a_lds + (k_tile % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_even0);
            compute_k64(a_stage, 1, b_even1);
        }
        load_b_tile_bn128((k_tile + 2) * BLOCK_K, b_even0, b_even1);
        stage_sync(1);   // beta
        if (k_tile + 3 < k_tiles && loads_a) {
            const char* a_nx_ptr =
                a_ptr_base + static_cast<int64_t>(k_tile + 3) * BLOCK_K;
            vec4_u32 a_nx_srsrc = make_matrix_resource_ht(a_nx_ptr, K);
            MATRIX_LOAD_128X16_B8_T_LDS_HT(
                a_lds_base_addr + ((k_tile + 3) % kAStages) * kAStageBytes +
                    a_lds_write_bytes,
                a_nx_srsrc);
        }
        // Group-boundary rescale for the odd tile (group k_tile+1).
        if ((k_tile + 1) % K_IN_ITERS == 0) {
            RESCALE_GW_BM64((k_tile + 1) / K_IN_ITERS);
        }
        {
            const char* a_stage =
                a_lds + ((k_tile + 1) % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_odd0);
            compute_k64(a_stage, 1, b_odd1);
        }
        if (k_tile + 3 < k_tiles)
            load_b_tile_bn128((k_tile + 3) * BLOCK_K, b_odd0, b_odd1);
    }

    // Epilogue: drain the last 1-2 tiles (their loads were already issued).
    const int remaining_tiles = k_tiles - k_tile;
    if (remaining_tiles >= 1) {
        if (remaining_tiles == 2) {
            stage_sync(0);
        } else {
            stage_sync_drain();
        }
        if (k_tile % K_IN_ITERS == 0 && k_tile > 0) {
            RESCALE_GW_BM64(k_tile / K_IN_ITERS);
        }
        {
            const char* a_stage =
                a_lds + (k_tile % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_even0);
            compute_k64(a_stage, 1, b_even1);
        }
        ++k_tile;
        if (k_tile < k_tiles) {
            stage_sync_drain();
            if (k_tile % K_IN_ITERS == 0) {
                RESCALE_GW_BM64(k_tile / K_IN_ITERS);
            }
            const char* a_stage =
                a_lds + (k_tile % kAStages) * kAStageBytes;
            compute_k64(a_stage, 0, b_odd0);
            compute_k64(a_stage, 1, b_odd1);
        }
    }

    // =========================================================================
    // Dequant and output: acc * s_last(row) * col_scale -> BF16
    // =========================================================================
    const bool full_block_valid =
        (m_block_start + BLOCK_M <= M) && (valid_m >= BLOCK_M);

    float col_scale[kNFragments][kMmacVgprs];
    #pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int col_base = n_block_start +
                             warp_id * kNFragments * kMmacCols +
                             n_frag * kMmacCols + lane_group * kMmacVgprs;
        #pragma unroll
        for (int elem = 0; elem < kMmacVgprs; ++elem) {
            col_scale[n_frag][elem] = matrixB_scale[col_base + elem];
        }
    }

    #pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row = m_block_start + m_loop * kMmacRows + lane_row;
        if (!full_block_valid &&
            (row >= M || m_loop * kMmacRows + lane_row >= valid_m)) {
            continue;
        }
        const float row_scale = SMEM_SCALE_ROW_BM64(m_loop)[0];  // slot 0 = s_last
        #pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
            const int col_base = n_block_start +
                                 warp_id * kNFragments * kMmacCols +
                                 n_frag * kMmacCols + lane_group * kMmacVgprs;
            const float v0 = MmaT::to_float(acc[m_loop][n_frag][0]) * row_scale * col_scale[n_frag][0];
            const float v1 = MmaT::to_float(acc[m_loop][n_frag][1]) * row_scale * col_scale[n_frag][1];
            const float v2 = MmaT::to_float(acc[m_loop][n_frag][2]) * row_scale * col_scale[n_frag][2];
            const float v3 = MmaT::to_float(acc[m_loop][n_frag][3]) * row_scale * col_scale[n_frag][3];
            *reinterpret_cast<uint64_t*>(
                MatrixC + static_cast<int64_t>(row) * C_stride + col_base) =
                pack_bf16x4_ht(v0, v1, v2, v3);
        }
    }

    #undef SMEM_SCALE_ROW_BM64
    #undef RESCALE_GW_BM64
}

} // namespace deep_gemm::mega::hcu
