// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// GEMM device function for HCU gfx938 MFMA FP8.
// Ported from V1 csrc/kernels/unified_moe/gemm_device_funcs.cuh
//
// Key characteristics (from V1):
//   - MFMA instruction: __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts
//   - BLOCK_M=16, BLOCK_N=256, BLOCK_K=64, NUM_WARPS=4, WARP_N=16, N_REPEAT=4
//   - Direct global_load_dwordx4 to registers (no LDS staging for A/B)
//   - A-operand lane shuffle: shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16)
//   - Software pipeline PIPE_STAGE=2
//   - Channel-wise dequant: rC[n][i] * input_scale * weight_scale[n][i]
//     input_scale: per-token (1 float/token)
//     weight_scale: per-channel (1 float per N column)

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "params.h"
#include "warp_reduce.cuh"
#include "activation_device.cuh"  // swiglu_nv

namespace deep_gemm::mega::hcu {

// Vector types (matching V1 fp8_masked_utils.h)
template <typename scalar_t, size_t len>
using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

using vec4_bf16 = vec<unsigned short, 4>;

union v_type {
    vec<int8_t, 8> int8_arr[2];
    vec<int32_t, 4> int_arr[1];
};

// MFMA FP8 16x16x32: accumulate C += A * B (float accumulator)
static __device__ vec<float, 4> mmac_(const vec<int8_t, 8>& v1,
                                       const vec<int8_t, 8>& v2,
                                       vec<float, 4>& v3) {
#if defined(__gfx938__)
    v3 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(v1, v2, v3, 1, 0);
#endif
    return v3;
}

// MFMA INT8 16x16x32: accumulate C += A * B (int32 accumulator)
static __device__ vec<int32_t, 4> mmac_i8_(const vec<int8_t, 8>& v1,
                                            const vec<int8_t, 8>& v2,
                                            vec<int32_t, 4>& v3) {
#if defined(__gfx938__)
    v3 = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(v1, v2, v3, 1, 0, 0);
#endif
    return v3;
}

// Accumulator type by quantization mode (FP8 -> float acc, INT8 -> int32 acc).
template <bool I8> struct GemmCwTraits;
template <> struct GemmCwTraits<false> { using acc_t = vec<float, 4>;   using elem_t = float; };
template <> struct GemmCwTraits<true>  { using acc_t = vec<int32_t, 4>; using elem_t = int32_t; };

// MMA dispatch by accumulator type.
static __device__ __forceinline__ vec<float, 4> mma_cw(
    const vec<int8_t, 8>& a, const vec<int8_t, 8>& b, vec<float, 4>& c) {
    return mmac_(a, b, c);
}
static __device__ __forceinline__ vec<int32_t, 4> mma_cw(
    const vec<int8_t, 8>& a, const vec<int8_t, 8>& b, vec<int32_t, 4>& c) {
    return mmac_i8_(a, b, c);
}

// Hardware all-wave barrier that first drains LDS ops.  Use when the
// barrier publishes SMEM data to other warps (e.g. the row-amax reduction
// table): writers ds_write before it, readers consume afterwards.
__device__ __forceinline__ void wait_lgkmcnt0_all_barrier() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n\t" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// =========================================================================
// Tile-level GEMM: computes a single (expert, m_tile, n_tile) tile.
// Adapted from V1 gemm_channelwise_stage for the fused MegaMoE kernel.
//
// Called by 4 math warps (256 threads). The thread index is remapped:
//   math_warp_idx = (thread_idx - math_warp_offset) / 64  (0-3)
//   math_lane     = (thread_idx - math_warp_offset) % 64  (0-63)
//
// Template params:
//   N:          output columns (e.g. 2*intermediate_hidden for L1, hidden for L2)
//   K:          reduction dimension (e.g. hidden for L1, intermediate_hidden for L2)
//   BLOCK_M:    tile M (16)
//   BLOCK_N:    tile N (256)
//   BLOCK_K:    tile K (64)
//   WARP_N:     16
//   NUM_WARPS:  4
//
// Args:
//   matrixA:       [M, K] FP8 for this expert (already offset to expert)
//   matrixB:       [N, K] FP8 for this expert (already offset to expert)
//   matrixA_scale: [M] float per-token (already offset to expert)
//   matrixB_scale: [N] float per-channel (already offset to expert)
//   tile_m:        m tile index (0-based for this expert's tiles)
//   tile_n:        n tile index
//   valid_m:       actual valid rows in this tile (for masking, <= BLOCK_M)
//   MatrixC:       output BF16 buffer (GMEM or SMEM), layout [M, N] row-major
//   M:             max tokens for this expert (stride of A and C)
//   math_warp_offset: threadIdx.x offset of math warps (e.g. 640 for warp 10)
// =========================================================================
template <int32_t N, int K,
          int BLOCK_M = 16, int BLOCK_N = 256, int BLOCK_K = 64,
          int WARP_N = 16, int NUM_WARPS = 4,
          int32_t PIPE_STAGE_T = 2>
__device__ void gemm_channelwise_tile(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride = N)
{
    constexpr int N_TILES = N / BLOCK_N;
    constexpr int K_ITERS = K / BLOCK_K;
    constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
    constexpr int WARP_SIZE = kWarpSize;
    constexpr int MMA_N = 16;
    constexpr int MMA_C_REGS = 4;
    // M_TILES: number of 16-row MFMA tiles in M dimension (BLOCK_M / 16)
    constexpr int M_TILES = BLOCK_M / 16;
    static_assert(N % BLOCK_N == 0 && K % BLOCK_K == 0,
                  "GEMM shape must tile evenly by BLOCK_N/BLOCK_K");
    static_assert(BLOCK_N % (NUM_WARPS * WARP_N) == 0,
                  "BLOCK_N must cover all warps evenly");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of the 16-row MMA tile");

    // Load info (matching V1)
    constexpr int LD_BITS_PER_THREAD = 128;
    constexpr int ELEMENT_BITS = sizeof(char) * 8;
    constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
    constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
    constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
    constexpr int CACHELINE_BITS = 512;
    constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
    constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
    constexpr int LD_M_ROW_STRIDE = K;
    constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;

    constexpr int32_t PIPE_STAGE = PIPE_STAGE_T;

    // Remap thread index for math warps
    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_idx = math_tid / WARP_SIZE;
    const int lane_id = math_tid % WARP_SIZE;

    int ld_row = lane_id % LD_ROWS;
    int ld_col = lane_id / LD_ROWS;
    int ld_row_N = ld_row;
    int ld_row_ = lane_id / LD_COLS;
    int ld_col_ = lane_id % LD_COLS;

    const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

    // Single tile computation (no expert iteration — expert already offset)
    // M_TILES > 1: replicate A registers and accumulators for each 16-row M-tile
    v_type rA[PIPE_STAGE][M_TILES];
    v_type rB[PIPE_STAGE][N_REPEAT];
    vec<float, MMA_C_REGS> rC[M_TILES][N_REPEAT] = {};

    const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
    const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE;
    const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;

    const char* cur_M_base = matrixA + tile_m * BLOCK_M * K;
    const char* cur_N_base = matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
    const char* cur_M_warp_base = cur_M_base;
    const char* cur_N_warp_base = cur_N_base + warp_offset_N;
    hip_bfloat16* cur_matrixC_warp_base = MatrixC
        + tile_m * BLOCK_M * C_stride + tile_n * BLOCK_N + warp_idx * MMA_N;

    // Channel-wise: input scale per token (one per M-tile)
    float input_scale[M_TILES];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        input_scale[m] = 0.0f;
        if (ld_row + m * 16 < valid_m) {
            int32_t sa_offset = tile_m * BLOCK_M + ld_row + m * 16;
            input_scale[m] = matrixA_scale[sa_offset];
        }
    }

    // Channel-wise: weight scales as vec<float, 4>
    vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
    for (int i = 0; i < N_REPEAT; ++i) {
        const vec<float, MMA_C_REGS>* p_ws =
            reinterpret_cast<const vec<float, MMA_C_REGS>*>(
                matrixB_scale + tile_n * BLOCK_N
                + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
        weight_scale[i] = p_ws[0];
    }

    // Software pipeline: preload PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = 0; k < (PIPE_STAGE - 1); k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }
    }

    // Main loop: load next + compute current
    #pragma unroll
    for (int k = (PIPE_STAGE - 1); k < K_ITERS; k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }

        __builtin_amdgcn_sched_barrier(0);

        // Shuffle A values for previous iteration + MMA for each M-tile
        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mmac_(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                 rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mmac_(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                 rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // Epilogue: drain remaining PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = K_ITERS; k < K_ITERS + (PIPE_STAGE - 1); k++) {
        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mmac_(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                 rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mmac_(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                 rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // Channel-wise dequant: rC[m][n][i] * input_scale[m] * weight_scale[n][i]
    // Only store if the row is valid
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        if (ld_row + m * 16 < valid_m) {
            for (int n = 0; n < N_REPEAT; n++) {
                vec4_bf16 st;
                for (int32_t i = 0; i < MMA_C_REGS; i++) {
                    float r_st = float(rC[m][n][i]) * input_scale[m] * weight_scale[n][i];
                    hip_bfloat16 tmp = hip_bfloat16(r_st);
                    st[i] = reinterpret_cast<unsigned short&>(tmp);
                }
                int32_t st_lane_offset = (ld_row + m * 16) * C_stride + ld_col * MMA_C_REGS;
                vec4_bf16* p_st = reinterpret_cast<vec4_bf16*>(
                    cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
                p_st[0] = st;
            }
        }
    }
}

// =========================================================================
// Fused variant of gemm_channelwise_tile for the L1 GEMM: SwiGLU + group-wise
// quantization folded into the GEMM epilogue (FP8 or INT8 via kIsInt8).
//
// Requires N_REPEAT == 2 and BLOCK_N == 2 * (NUM_WARPS * WARP_N).  Weights
// use granularity-8 gate/up interleave, so within a fragment's 16 raw cols
// the gate col c (< 8) pairs with up col c + 8; the partner value lives in
// lane ^ 32 (one __shfl_xor per element, same scheme as the HT epilogue).
//
// Fragment mapping (same as gemm_channelwise_tile): lane l holds row
// (l % 16 + m * 16), columns ((l / 16) * 4 + i) inside the warp's 16-column
// MMA slice, so the 4 lanes {r, r+16, r+32, r+48} cover a row's 16 warp-cols.
// Row-wise amax needs a cross-warp reduction over NUM_WARPS warps via
// smem_amax + subset_warp_barrier (only math warps run this function).
//
// Args (in addition to gemm_channelwise_tile):
//   l2_token_out: [pool_tokens, IH] FP8 output, row base = m_idx, col base =
//                 n_blk * (BLOCK_N/2)
//   l2_sf:        [pool_tokens, IH/GROUP_SIZE] group scale output
//   m_idx:        row offset of this tile in the pool buffers
//   n_blk:        N-block index = group index in l2_sf
//   sf_stride:    floats per l2_sf row (IH / GROUP_SIZE)
//   ih:           row stride of l2_token_out
//   smem_amax:    scratch [NUM_WARPS * BLOCK_M] floats
//
// Contains exactly ONE s_barrier (row-amax cross-warp reduction). Callers
// must mirror it with one s_barrier in warps that skip this function
// (same pattern as the HT kernel's subset barriers).
// =========================================================================
template <int32_t N, int K,
          int BLOCK_M = 16, int BLOCK_N = 256, int BLOCK_K = 64,
          int WARP_N = 16, int NUM_WARPS = 4,
          int32_t PIPE_STAGE_T = 2,
          bool kIsInt8 = false>
__device__ void gemm_channelwise_tile_silu_quant(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t valid_m,
    const int32_t M,
    const uint32_t math_warp_offset,
    uint8_t* __restrict__ l2_token_out,
    float* __restrict__ l2_sf,
    const int32_t m_idx,
    const int32_t n_blk,
    const int32_t sf_stride,
    const int32_t ih,
    float* __restrict__ smem_amax,
    const float activation_clamp,
    const bool fast_math,
    hip_bfloat16* __restrict__ dbg_v = nullptr)
{
    constexpr int N_TILES = N / BLOCK_N;
    constexpr int K_ITERS = K / BLOCK_K;
    constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
    constexpr int WARP_SIZE = kWarpSize;
    constexpr int MMA_N = 16;
    constexpr int MMA_C_REGS = 4;
    constexpr int M_TILES = BLOCK_M / 16;
    static_assert(N % BLOCK_N == 0 && K % BLOCK_K == 0,
                  "GEMM shape must tile evenly by BLOCK_N/BLOCK_K");
    static_assert(BLOCK_N % (NUM_WARPS * WARP_N) == 0,
                  "BLOCK_N must cover all warps evenly");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of the 16-row MMA tile");
    static_assert(N_REPEAT == 2, "fused silu requires gate/up pairing in-thread");

    constexpr int LD_BITS_PER_THREAD = 128;
    constexpr int ELEMENT_BITS = sizeof(char) * 8;
    constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
    constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
    constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
    constexpr int CACHELINE_BITS = 512;
    constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
    constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
    constexpr int LD_M_ROW_STRIDE = K;
    constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;

    constexpr int32_t PIPE_STAGE = PIPE_STAGE_T;

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_idx = math_tid / WARP_SIZE;
    const int lane_id = math_tid % WARP_SIZE;

    int ld_row = lane_id % LD_ROWS;
    int ld_col = lane_id / LD_ROWS;
    int ld_row_N = ld_row;
    int ld_row_ = lane_id / LD_COLS;
    int ld_col_ = lane_id % LD_COLS;

    const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

    v_type rA[PIPE_STAGE][M_TILES];
    v_type rB[PIPE_STAGE][N_REPEAT];
    typename GemmCwTraits<kIsInt8>::acc_t rC[M_TILES][N_REPEAT] = {};

    const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
    const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE;
    const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;

    const char* cur_M_base = matrixA + tile_m * BLOCK_M * K;
    const char* cur_M_warp_base = cur_M_base;
    const char* cur_N_warp_base = matrixB + warp_offset_N;

    float input_scale[M_TILES];  // per-token L1 input dequant scale
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        input_scale[m] = 0.0f;
        if (ld_row + m * 16 < valid_m) {
            int32_t sa_offset = tile_m * BLOCK_M + ld_row + m * 16;
            input_scale[m] = matrixA_scale[sa_offset];
        }
    }

    vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
    for (int i = 0; i < N_REPEAT; ++i) {
        const vec<float, MMA_C_REGS>* p_ws =
            reinterpret_cast<const vec<float, MMA_C_REGS>*>(
                matrixB_scale
                + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
        weight_scale[i] = p_ws[0];
    }

    // Software pipeline: preload PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = 0; k < (PIPE_STAGE - 1); k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }
    }

    // Main loop: load next + compute current
    #pragma unroll
    for (int k = (PIPE_STAGE - 1); k < K_ITERS; k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }

        __builtin_amdgcn_sched_barrier(0);

        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // Epilogue: drain remaining PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = K_ITERS; k < K_ITERS + (PIPE_STAGE - 1); k++) {
        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // ============ Fused SwiGLU + group-wise quant epilogue (FP8/INT8) ======
    // Granularity-8 gate/up interleave: within a fragment's 16 raw cols,
    // cols c<8 are gate and c+8 the paired up.  Lane group lg=lane_id/16
    // holds cols lg*4..+3, so the gate/up partner lives in lane ^32;
    // both halves recompute the same silu(g)*u product.  Out col of frag n
    // (warp w), col c: n_blk*HALF_BN + n*(HALF_BN/N_REPEAT) + w*8 + c%8.
    float silu_v[M_TILES][N_REPEAT][MMA_C_REGS];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        float lane_amax = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_REPEAT; ++n) {
            #pragma unroll
            for (int i = 0; i < MMA_C_REGS; ++i) {
                const float v = static_cast<float>(rC[m][n][i])
                    * input_scale[m] * weight_scale[n][i];
#ifdef DG_MEGA_DUMP_SILU
                // Precision bisect: dump the dequantized accumulator `v` at its
                // raw (interleaved) L1 column, [pool_row<128][N] bf16.
                if (dbg_v != nullptr) {
                    const int32_t dbg_row = m_idx + ld_row + m * 16;
                    if (dbg_row < gridDim.x * kHTL2ScratchBlockM) {
                        const int32_t raw_col = n_blk * BLOCK_N
                            + (n * NUM_WARPS + warp_idx) * 16 + ld_col * 4 + i;
                        dbg_v[dbg_row * N + raw_col] = hip_bfloat16(v);
                    }
                }
#endif
                const float partner = __shfl_xor(v, 32);
                const float g = (ld_col < 2) ? v : partner;
                const float u = (ld_col < 2) ? partner : v;
                const float s = swiglu_nv(g, u, activation_clamp, fast_math);
                silu_v[m][n][i] = s;
                lane_amax = fmaxf(lane_amax, fabsf(s));
            }
        }
        if (ld_row + m * 16 >= valid_m) lane_amax = 0.0f;
        // Row r is held by lanes {r, r+16, r+32, r+48}: reduce via xor shuffles.
        lane_amax = fmaxf(lane_amax, __shfl_xor(lane_amax, 16));
        lane_amax = fmaxf(lane_amax, __shfl_xor(lane_amax, 32));
        if (lane_id < 16) {
            smem_amax[warp_idx * BLOCK_M + m * 16 + lane_id] = lane_amax;
        }
    }

    // Row-amax cross-warp reduction barrier (math warps only; callers mirror
    // one s_barrier in non-math warps, same as the HT subset-barrier
    // pattern).  lgkmcnt(0) drains the smem_amax ds_writes so readers on
    // other warps observe them after the barrier.
    wait_lgkmcnt0_all_barrier();

    // Per-thread row amax across all math warps, then quantize + store.
    constexpr int HALF_BN = BLOCK_N / 2;
    constexpr int FRAG_OUT = HALF_BN / N_REPEAT;  // out cols per fragment
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        const int32_t row = m * 16 + ld_row;
        float row_amax = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            row_amax = fmaxf(row_amax, smem_amax[w * BLOCK_M + row]);
        }
        const float qmax = kIsInt8 ? 127.0f : 448.0f;
        const float clamped_amax = fmaxf(row_amax, kIsInt8 ? 1.0e-6f : 1.0e-4f);
        const float scale = clamped_amax / qmax;
        const float inv_scale = qmax / clamped_amax;

        if (row < valid_m) {
            #pragma unroll
            for (int n = 0; n < N_REPEAT; ++n) {
                float q[MMA_C_REGS];
                #pragma unroll
                for (int i = 0; i < MMA_C_REGS; ++i) {
                    q[i] = quant_clamp<kIsInt8>(silu_v[m][n][i] * inv_scale, qmax);
                }
                uint32_t packed = 0;
                if constexpr (kIsInt8) {
                    #pragma unroll
                    for (int i = 0; i < MMA_C_REGS; ++i) {
                        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(
                                       static_cast<int8_t>(__float2int_rn(q[i]))))
                                   << (i * 8));
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < MMA_C_REGS; i += 2) {
                        float2 f2{q[i], q[i + 1]};
                        auto fp8x2 = __hip_cvt_float2_to_fp8x2(f2, __HIP_SATFINITE, __HIP_E4M3);
                        packed |= (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&fp8x2))
                                   << (i * 8));
                    }
                }
                // Out bytes 0-3 live in lg0 lanes (row r -> lane r), 4-7 in
                // lg1 (lane r+16); lg2/lg3 are duplicates.  Gather into
                // lane r for one 8B store per fragment.
                uint2 packed2;
                packed2.x = __shfl(packed, lane_id & 15);
                packed2.y = __shfl(packed, (lane_id & 15) + 16);
                if (lane_id < 16) {
                    *reinterpret_cast<uint2*>(
                        l2_token_out + (m_idx + row) * ih
                        + n_blk * HALF_BN + n * FRAG_OUT + warp_idx * 8) = packed2;
                }
            }
        }
        if (lane_id < 16 && row < valid_m) {
            l2_sf[(m_idx + row) * sf_stride + n_blk] = scale;
        }
    }
}

// =========================================================================
// Warp-K split variant of gemm_channelwise_tile_silu_quant (masked-GEMM
// inspired): the NUM_WARPS math warps are decomposed as
//   warp_n_id = warp_idx / WARP_K  (BLOCK_N/(NUM_WARPS/WARP_K) cols each)
//   warp_k_id = warp_idx % WARP_K  (contiguous half of the K iters each)
// so each warp runs K_ITERS/WARP_K iterations, halving the serial
// load-latency chain and doubling per-iteration loads in flight.
//
// After the K loop the partial accumulators of the warp_k_id==1 warps are
// pushed through `smem_red` (>= M_TILES*8KB; the L1-phase l2_out staging
// buffer is used by the caller) in N_REPEAT/2 frag-pair rounds and added
// into the warp_k_id==0 partner's rC.  Contains exactly
// (WARP_K == 2 ? 3 : 0) + 1 s_barriers: three for the two reduction rounds
// (write0|read+write1|read) plus the usual row-amax barrier.
// =========================================================================
template <int32_t N, int K,
          int BLOCK_M = 16, int BLOCK_N = 256, int BLOCK_K = 64,
          int WARP_N = 16, int NUM_WARPS = 4,
          int32_t PIPE_STAGE_T = 2,
          int WARP_K = 2,
          bool kIsInt8 = false>
__device__ void gemm_channelwise_tile_silu_quant_wk(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t valid_m,
    const int32_t M,
    const uint32_t math_warp_offset,
    uint8_t* __restrict__ l2_token_out,
    float* __restrict__ l2_sf,
    const int32_t m_idx,
    const int32_t n_blk,
    const int32_t sf_stride,
    const int32_t ih,
    float* __restrict__ smem_amax,
    float* __restrict__ smem_red,
    const float activation_clamp,
    const bool fast_math)
{
    static_assert(NUM_WARPS % WARP_K == 0, "NUM_WARPS must divide WARP_K");
    constexpr int WARP_N_NUM = NUM_WARPS / WARP_K;
    constexpr int N_TILES = N / BLOCK_N;
    constexpr int K_ITERS = K / BLOCK_K;
    constexpr int K_LOCAL = K_ITERS / WARP_K;
    constexpr int32_t N_REPEAT = (BLOCK_N / (WARP_N_NUM * WARP_N));
    constexpr int WARP_SIZE = kWarpSize;
    constexpr int MMA_N = 16;
    constexpr int MMA_C_REGS = 4;
    constexpr int M_TILES = BLOCK_M / 16;
    static_assert(N % BLOCK_N == 0 && K % BLOCK_K == 0,
                  "GEMM shape must tile evenly by BLOCK_N/BLOCK_K");
    static_assert(BLOCK_N % (WARP_N_NUM * WARP_N) == 0,
                  "BLOCK_N must cover all warps evenly");
    static_assert(K_ITERS % WARP_K == 0,
                  "K iterations must split evenly across WARP_K");
    static_assert(N_REPEAT % 2 == 0 && N_REPEAT >= 2,
                  "fused silu requires in-thread gate/up frag pairs");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of the 16-row MMA tile");
    // Gate/up pairing is intra-fragment: col c pairs with c^8 (lane ^32).

    constexpr int LD_BITS_PER_THREAD = 128;
    constexpr int ELEMENT_BITS = sizeof(char) * 8;
    constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
    constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
    constexpr int CACHELINE_BITS = 512;
    constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
    constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
    constexpr int LD_M_ROW_STRIDE = K;
    constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;

    constexpr int32_t PIPE_STAGE = PIPE_STAGE_T;

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_idx = math_tid / WARP_SIZE;
    const int lane_id = math_tid % WARP_SIZE;
    const int warp_k_id = warp_idx % WARP_K;
    const int warp_n_id = warp_idx / WARP_K;

    int ld_row = lane_id % LD_ROWS;
    int ld_col = lane_id / LD_ROWS;
    int ld_row_N = ld_row;
    int ld_row_ = lane_id / LD_COLS;
    int ld_col_ = lane_id % LD_COLS;

    const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

    v_type rA[PIPE_STAGE][M_TILES];
    v_type rB[PIPE_STAGE][N_REPEAT];
    typename GemmCwTraits<kIsInt8>::acc_t rC[M_TILES][N_REPEAT] = {};

    const int32_t warp_offset_N = warp_n_id * LD_ELEMENTS_PER_WARP;
    const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE;
    const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;

    const char* cur_M_base = matrixA + tile_m * BLOCK_M * K;
    const char* cur_M_warp_base =
        cur_M_base + warp_k_id * K_LOCAL * LD_M_ITERK_STRIDE;
    const char* cur_N_warp_base = matrixB + warp_offset_N
        + warp_k_id * K_LOCAL * LD_N_ITERK_STRIDE;

    float input_scale[M_TILES];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        input_scale[m] = 0.0f;
        if (ld_row + m * 16 < valid_m) {
            int32_t sa_offset = tile_m * BLOCK_M + ld_row + m * 16;
            input_scale[m] = matrixA_scale[sa_offset];
        }
    }

    vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
    for (int i = 0; i < N_REPEAT; ++i) {
        const vec<float, MMA_C_REGS>* p_ws =
            reinterpret_cast<const vec<float, MMA_C_REGS>*>(
                matrixB_scale
                + (i * WARP_N_NUM + warp_n_id) * MMA_N + ld_col * MMA_C_REGS);
        weight_scale[i] = p_ws[0];
    }

    // Software pipeline: preload PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = 0; k < (PIPE_STAGE - 1); k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * WARP_N_NUM * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }
    }

    // Main loop: load next + compute current (K_LOCAL iterations)
    #pragma unroll
    for (int k = (PIPE_STAGE - 1); k < K_LOCAL; k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * WARP_N_NUM * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }

        __builtin_amdgcn_sched_barrier(0);

        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // Drain remaining PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = K_LOCAL; k < K_LOCAL + (PIPE_STAGE - 1); k++) {
        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // ============ K-partial reduction across WARP_K warps via smem_red ====
    // Per round: warp_k_id==1 warps write M_TILES*2 frags (M_TILES*8 floats
    // per lane); warp_k_id==0 warps add them after a barrier.  smem_red must
    // provide WARP_N_NUM*64*M_TILES*8 floats (= M_TILES*8KB for WARP_K=2,
    // NUM_WARPS=8) — exactly one half of the acc set per round.
    {
        constexpr int LANE_FLOATS = M_TILES * 2 * MMA_C_REGS;
        using ElemT = typename GemmCwTraits<kIsInt8>::elem_t;
        ElemT* red = reinterpret_cast<ElemT*>(smem_red);
        const int slot = (warp_n_id * WARP_SIZE + lane_id) * LANE_FLOATS;
        #pragma unroll
        for (int rd = 0; rd < N_REPEAT / 2; ++rd) {
            if (warp_k_id == 1) {
                #pragma unroll
                for (int m = 0; m < M_TILES; ++m) {
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        #pragma unroll
                        for (int i = 0; i < MMA_C_REGS; ++i) {
                            red[slot + (m * 2 + f) * MMA_C_REGS + i] =
                                rC[m][rd * 2 + f][i];
                        }
                    }
                }
            }
            wait_lgkmcnt0_all_barrier();
            if (warp_k_id == 0) {
                #pragma unroll
                for (int m = 0; m < M_TILES; ++m) {
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        #pragma unroll
                        for (int i = 0; i < MMA_C_REGS; ++i) {
                            rC[m][rd * 2 + f][i] +=
                                red[slot + (m * 2 + f) * MMA_C_REGS + i];
                        }
                    }
                }
            }
            // Separate this round's reads from next round's writes.
            if (rd + 1 < N_REPEAT / 2) {
                asm volatile("s_barrier\n\t" ::: "memory");
            }
        }
    }

    // From here on only warp_k_id==0 warps hold complete accumulators.
    // warp_k_id==1 warps still execute the row-amax barrier (caller mirrors
    // the same barrier count in non-math warps).

    // ============ Fused SwiGLU + group-wise quant epilogue (FP8/INT8) ======
    // Granularity-8 gate/up interleave: within a fragment's 16 raw cols,
    // cols c<8 are gate and c+8 the paired up.  Lane group lg=lane_id/16
    // holds cols lg*4..+3, so the gate/up partner lives in lane ^32; both
    // halves recompute the same silu(g)*u product.  Out col of frag n
    // (warp_n wn): n_blk*HALF_BN + n*(HALF_BN/N_REPEAT) + wn*8 + c%8.
    float silu_v[M_TILES][N_REPEAT][MMA_C_REGS];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        float lane_amax = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_REPEAT; ++n) {
            #pragma unroll
            for (int i = 0; i < MMA_C_REGS; ++i) {
                const float v = static_cast<float>(rC[m][n][i])
                    * input_scale[m] * weight_scale[n][i];
                const float partner = __shfl_xor(v, 32);
                const float gv = (ld_col < 2) ? v : partner;
                const float uv = (ld_col < 2) ? partner : v;
                const float s = swiglu_nv(gv, uv, activation_clamp, fast_math);
                silu_v[m][n][i] = s;
                lane_amax = fmaxf(lane_amax, fabsf(s));
            }
        }
        if (warp_k_id != 0 || ld_row + m * 16 >= valid_m) lane_amax = 0.0f;
        lane_amax = fmaxf(lane_amax, __shfl_xor(lane_amax, 16));
        lane_amax = fmaxf(lane_amax, __shfl_xor(lane_amax, 32));
        if (warp_k_id == 0 && lane_id < 16) {
            smem_amax[warp_n_id * BLOCK_M + m * 16 + lane_id] = lane_amax;
        }
    }

    // Row-amax cross-warp reduction barrier (all math warps; k1 warps only
    // participate in the barrier).
    wait_lgkmcnt0_all_barrier();

    constexpr int HALF_BN = BLOCK_N / 2;
    constexpr int FRAG_OUT = HALF_BN / N_REPEAT;  // out cols per fragment
    if (warp_k_id == 0) {
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        const int32_t row = m * 16 + ld_row;
        float row_amax = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARP_N_NUM; ++w) {
            row_amax = fmaxf(row_amax, smem_amax[w * BLOCK_M + row]);
        }
        const float qmax = kIsInt8 ? 127.0f : 448.0f;
        const float clamped_amax = fmaxf(row_amax, kIsInt8 ? 1.0e-6f : 1.0e-4f);
        const float scale = clamped_amax / qmax;
        const float inv_scale = qmax / clamped_amax;

        if (row < valid_m) {
            #pragma unroll
            for (int n = 0; n < N_REPEAT; ++n) {
                float q[MMA_C_REGS];
                #pragma unroll
                for (int i = 0; i < MMA_C_REGS; ++i) {
                    q[i] = quant_clamp<kIsInt8>(silu_v[m][n][i] * inv_scale, qmax);
                }
                uint32_t packed = 0;
                if constexpr (kIsInt8) {
                    #pragma unroll
                    for (int i = 0; i < MMA_C_REGS; ++i) {
                        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(
                                       static_cast<int8_t>(__float2int_rn(q[i]))))
                                   << (i * 8));
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < MMA_C_REGS; i += 2) {
                        float2 f2{q[i], q[i + 1]};
                        auto fp8x2 = __hip_cvt_float2_to_fp8x2(f2, __HIP_SATFINITE, __HIP_E4M3);
                        packed |= (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&fp8x2))
                                   << (i * 8));
                    }
                }
                // Out bytes 0-3 live in lg0 lanes (row r -> lane r), 4-7 in
                // lg1 (lane r+16); lg2/lg3 are duplicates.  Gather into
                // lane r for one 8B store per fragment.
                uint2 packed2;
                packed2.x = __shfl(packed, lane_id & 15);
                packed2.y = __shfl(packed, (lane_id & 15) + 16);
                if (lane_id < 16) {
                    *reinterpret_cast<uint2*>(
                        l2_token_out + (m_idx + row) * ih
                        + n_blk * HALF_BN + n * FRAG_OUT
                        + warp_n_id * 8) = packed2;
                }
            }
        }
        if (lane_id < 16 && row < valid_m) {
            l2_sf[(m_idx + row) * sf_stride + n_blk] = scale;
        }
    }
    }
}

// =========================================================================
// INT8 variant of gemm_channelwise_tile.
// Uses __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts (int32 accumulator)
// instead of the fp8 MMA (float accumulator). The epilogue converts int32
// accumulator to float before applying per-token and per-channel scales.
// =========================================================================
template <int32_t N, int K,
          int BLOCK_M = 16, int BLOCK_N = 256, int BLOCK_K = 64,
          int WARP_N = 16, int NUM_WARPS = 4,
          int32_t PIPE_STAGE_T = 2>
__device__ void gemm_channelwise_tile_i8(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB,
    const float* __restrict__ matrixA_scale,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride = N)
{
    constexpr int N_TILES = N / BLOCK_N;
    constexpr int K_ITERS = K / BLOCK_K;
    constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
    constexpr int WARP_SIZE = kWarpSize;
    constexpr int MMA_N = 16;
    constexpr int MMA_C_REGS = 4;
    constexpr int M_TILES = BLOCK_M / 16;
    static_assert(N % BLOCK_N == 0 && K % BLOCK_K == 0,
                  "GEMM shape must tile evenly by BLOCK_N/BLOCK_K");
    static_assert(BLOCK_N % (NUM_WARPS * WARP_N) == 0,
                  "BLOCK_N must cover all warps evenly");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of the 16-row MMA tile");

    constexpr int LD_BITS_PER_THREAD = 128;
    constexpr int ELEMENT_BITS = sizeof(char) * 8;
    constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
    constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
    constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
    constexpr int CACHELINE_BITS = 512;
    constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
    constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
    constexpr int LD_M_ROW_STRIDE = K;
    constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;

    constexpr int32_t PIPE_STAGE = PIPE_STAGE_T;

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_idx = math_tid / WARP_SIZE;
    const int lane_id = math_tid % WARP_SIZE;

    int ld_row = lane_id % LD_ROWS;
    int ld_col = lane_id / LD_ROWS;
    int ld_row_N = ld_row;
    int ld_row_ = lane_id / LD_COLS;
    int ld_col_ = lane_id % LD_COLS;

    const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

    v_type rA[PIPE_STAGE][M_TILES];
    v_type rB[PIPE_STAGE][N_REPEAT];
    vec<int32_t, MMA_C_REGS> rC[M_TILES][N_REPEAT] = {};

    const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
    const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE;
    const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;

    const char* cur_M_base = matrixA + tile_m * BLOCK_M * K;
    const char* cur_N_base = matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
    const char* cur_M_warp_base = cur_M_base;
    const char* cur_N_warp_base = cur_N_base + warp_offset_N;
    hip_bfloat16* cur_matrixC_warp_base = MatrixC
        + tile_m * BLOCK_M * C_stride + tile_n * BLOCK_N + warp_idx * MMA_N;

    float input_scale[M_TILES];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        input_scale[m] = 0.0f;
        if (ld_row + m * 16 < valid_m) {
            int32_t sa_offset = tile_m * BLOCK_M + ld_row + m * 16;
            input_scale[m] = matrixA_scale[sa_offset];
        }
    }

    vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
    for (int i = 0; i < N_REPEAT; ++i) {
        const vec<float, MMA_C_REGS>* p_ws =
            reinterpret_cast<const vec<float, MMA_C_REGS>*>(
                matrixB_scale + tile_n * BLOCK_N
                + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
        weight_scale[i] = p_ws[0];
    }

    #pragma unroll
    for (int k = 0; k < (PIPE_STAGE - 1); k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }
    }

    #pragma unroll
    for (int k = (PIPE_STAGE - 1); k < K_ITERS; k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }

        __builtin_amdgcn_sched_barrier(0);

        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mmac_i8_(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                    rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mmac_i8_(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                    rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    #pragma unroll
    for (int k = K_ITERS; k < K_ITERS + (PIPE_STAGE - 1); k++) {
        int prev_k = k - (PIPE_STAGE - 1);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mmac_i8_(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                    rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mmac_i8_(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                    rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // INT8 dequant: float(int32_acc) * input_scale * weight_scale
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        if (ld_row + m * 16 < valid_m) {
            for (int n = 0; n < N_REPEAT; n++) {
                vec4_bf16 st;
                for (int32_t i = 0; i < MMA_C_REGS; i++) {
                    float r_st = static_cast<float>(rC[m][n][i]) * input_scale[m] * weight_scale[n][i];
                    hip_bfloat16 tmp = hip_bfloat16(r_st);
                    st[i] = reinterpret_cast<unsigned short&>(tmp);
                }
                int32_t st_lane_offset = (ld_row + m * 16) * C_stride + ld_col * MMA_C_REGS;
                vec4_bf16* p_st = reinterpret_cast<vec4_bf16*>(
                    cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
                p_st[0] = st;
            }
        }
    }
}

// =========================================================================
// Group-wise quantized GEMM tile: computes a single (expert, m_tile, n_tile) tile
// with group-wise A scales (group_size=128 by default).
//
// Uses a double-layer K loop structure (adapted from kGroupGemmBlockWise_):
//   outer loop iterates over groups (K_OUT_ITERS = K / GROUP_SIZE)
//   inner loop iterates over K-tiles within a group (K_IN_ITERS = GROUP_SIZE / BLOCK_K)
//
//   C[m,n] = sum_g( a_scale[m,g] * sum_{k in group g}(A[m,k] * B[n,k]) ) * b_scale[n]
//
// Single-accumulator formulation (ratio rescale): keep acc'_g = sum_{j<=g}
// s_j*P_j / s_g in rC. At each group boundary, rC *= (s_{g-1}/s_g) before the
// group's MMAs, so acc' telescopes and the final result is s_{G-1}*acc'.
// This avoids a second accumulator array (rC_final), halving the VGPR cost
// vs the reset-and-accumulate formulation — critical since the fused kernel
// runs at the 256-VGPR cap.
//
// Scales are pre-staged by the caller into an SMEM table of *boundary
// ratios*: smem_scale[row*K_OUT_ITERS_p1 + 0] = s_{G-1} (epilogue scale)
// and smem_scale[row*K_OUT_ITERS_p1 + g] = s_{g-1}/s_g for g >= 1, with
// invalid rows padded to 1.0f.  The math loop then needs only one LDS
// read per row per boundary — no GMEM scale loads, no frcp, and no
// persistent scale registers.
//
// Args:
//   matrixA:       [M, K] FP8 for this expert
//   matrixB:        [N, K] FP8 for this expert
//   matrixB_scale:  [N] float per-channel
//   smem_scale:     [BLOCK_M, K/GROUP_SIZE (+1 pad)] boundary-ratio table
//   tile_m, tile_n, valid_m, MatrixC, M, math_warp_offset, C_stride: same as channelwise
// =========================================================================
template <int32_t N, int K, int32_t GROUP_SIZE = 128,
          int BLOCK_M = 16, int BLOCK_N = 256, int BLOCK_K = 64,
          int WARP_N = 16, int NUM_WARPS = 4,
          int32_t PIPE_STAGE_T = 2,
          bool kIsInt8 = false>
__device__ void gemm_groupwise_tile(
    const char* __restrict__ matrixA,
    const char* __restrict__ matrixB,
    const float* __restrict__ matrixB_scale,
    const int32_t tile_m,
    const int32_t tile_n,
    const int32_t valid_m,
    hip_bfloat16* __restrict__ MatrixC,
    const int32_t M,
    const uint32_t math_warp_offset,
    const int32_t C_stride,
    const float* __restrict__ smem_scale)
{
    constexpr int N_TILES = N / BLOCK_N;
    constexpr int K_ITERS = K / BLOCK_K;
    constexpr int K_OUT_ITERS = K / GROUP_SIZE;       // group count (e.g., 3072/128=24)
    constexpr int K_IN_ITERS = GROUP_SIZE / BLOCK_K;  // K-tiles per group (e.g., 128/64=2)
    constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
    constexpr int WARP_SIZE = kWarpSize;
    constexpr int MMA_N = 16;
    constexpr int MMA_C_REGS = 4;
    constexpr int M_TILES = BLOCK_M / 16;
    static_assert(N % BLOCK_N == 0 && K % BLOCK_K == 0,
                  "GEMM shape must tile evenly by BLOCK_N/BLOCK_K");
    static_assert(BLOCK_N % (NUM_WARPS * WARP_N) == 0,
                  "BLOCK_N must cover all warps evenly");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of the 16-row MMA tile");
    static_assert(GROUP_SIZE % BLOCK_K == 0, "GROUP_SIZE must be divisible by BLOCK_K");
    static_assert(K % GROUP_SIZE == 0, "K must be divisible by GROUP_SIZE");

    // Load info
    constexpr int LD_BITS_PER_THREAD = 128;
    constexpr int ELEMENT_BITS = sizeof(char) * 8;
    constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
    constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
    constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
    constexpr int CACHELINE_BITS = 512;
    constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
    constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
    constexpr int LD_M_ROW_STRIDE = K;
    constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
    constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;

    constexpr int32_t PIPE_STAGE = PIPE_STAGE_T;

    const uint32_t math_tid = threadIdx.x - math_warp_offset;
    const int warp_idx = math_tid / WARP_SIZE;
    const int lane_id = math_tid % WARP_SIZE;

    int ld_row = lane_id % LD_ROWS;
    int ld_col = lane_id / LD_ROWS;
    int ld_row_N = ld_row;
    int ld_row_ = lane_id / LD_COLS;
    int ld_col_ = lane_id % LD_COLS;

    const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

    const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
    const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE;
    const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;

    const char* cur_M_base = matrixA + tile_m * BLOCK_M * K;
    const char* cur_N_base = matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
    const char* cur_M_warp_base = cur_M_base;
    const char* cur_N_warp_base = cur_N_base + warp_offset_N;
    hip_bfloat16* cur_matrixC_warp_base = MatrixC
        + tile_m * BLOCK_M * C_stride + tile_n * BLOCK_N + warp_idx * MMA_N;

    // Weight scales (per-channel, loaded once — same as channelwise)
    vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
    for (int i = 0; i < N_REPEAT; ++i) {
        const vec<float, MMA_C_REGS>* p_ws =
            reinterpret_cast<const vec<float, MMA_C_REGS>*>(
                matrixB_scale + tile_n * BLOCK_N
                + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
        weight_scale[i] = p_ws[0];
    }

    // Pipeline buffers (reused across groups)
    v_type rA[PIPE_STAGE][M_TILES];
    v_type rB[PIPE_STAGE][N_REPEAT];

    // =====================================================================
    // Flat K loop with in-place ratio rescale at group boundaries.
    // rC holds acc'_g = sum_{j<=g} s_j*P_j / s_g; before the first MMA of
    // group g, rC *= s_{g-1}/s_g, telescoping to sum s_j*P_j at the end.
    // This keeps the load pipeline running across group boundaries (unlike
    // a per-group preload/drain structure, which stalls every K_IN_ITERS).
    // For INT8 the accumulator is int32: the ratio rescale rounds back to
    // int (same scheme as the HT groupwise i8 path).
    // =====================================================================
    typename GemmCwTraits<kIsInt8>::acc_t rC[M_TILES][N_REPEAT] = {};

    // Boundary-ratio table in SMEM (filled by the caller — see header):
    // slot 0 = s_{G-1} for the epilogue, slot g >= 1 = s_{g-1}/s_g.
    // Stride is padded by 1 so the 16 per-lane row reads at a boundary
    // land on distinct LDS banks.  Invalid rows were padded with 1.0f,
    // making the rescale a no-op (outputs are masked downstream).
    constexpr int kScaleStride = K_OUT_ITERS + 1;

    // Rescale at group boundary before this tile's MMAs: one LDS read
    // per m-subtile row — no GMEM scale loads, no frcp in the math loop.
    // For INT8 the accumulator is int32: the ratio rescale rounds back
    // to int (same scheme as the HT groupwise i8 path).
    #define GROUP_BOUNDARY_HT(PREV_K)                                               \
        do {                                                                        \
            if (((PREV_K) % K_IN_ITERS) == 0 && (PREV_K) > 0) {                     \
                const int g_ = (PREV_K) / K_IN_ITERS;                               \
                _Pragma("unroll")                                                   \
                for (int m = 0; m < M_TILES; ++m) {                                 \
                    const float rg_ =                                               \
                        smem_scale[(ld_row + m * 16) * kScaleStride + g_];          \
                    _Pragma("unroll")                                               \
                    for (int n = 0; n < N_REPEAT; ++n) {                            \
                        _Pragma("unroll")                                           \
                        for (int i = 0; i < MMA_C_REGS; ++i) {                      \
                            if constexpr (kIsInt8) {                                \
                                rC[m][n][i] = __float2int_rn(                       \
                                    static_cast<float>(rC[m][n][i]) * rg_);         \
                            } else {                                                \
                                rC[m][n][i] *= rg_;                                 \
                            }                                                       \
                        }                                                           \
                    }                                                               \
                }                                                                   \
            }                                                                       \
        } while (0)

    // Software pipeline: preload PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = 0; k < (PIPE_STAGE - 1); k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        #pragma unroll
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }
    }

    // Main loop: load next + compute current (identical pipeline to channelwise)
    #pragma unroll
    for (int k = (PIPE_STAGE - 1); k < K_ITERS; k++) {
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const vec<int32_t, 4>* addr_a = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_M_warp_base + m * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % PIPE_STAGE][m].int_arr[0] = addr_a[0];
        }
        #pragma unroll
        for (int n = 0; n < N_REPEAT; ++n) {
            const vec<int32_t, 4>* addr_b = reinterpret_cast<const vec<int32_t, 4>*>(
                cur_N_warp_base + k * LD_N_ITERK_STRIDE
                + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
            rB[k % PIPE_STAGE][n].int_arr[0] = addr_b[0];
        }

        __builtin_amdgcn_sched_barrier(0);

        // Rescale at group boundary before this tile's MMAs
        const int prev_k = k - (PIPE_STAGE - 1);
        GROUP_BOUNDARY_HT(prev_k);

        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // Drain remaining PIPE_STAGE-1 iterations
    #pragma unroll
    for (int k = K_ITERS; k < K_ITERS + (PIPE_STAGE - 1); k++) {
        const int prev_k = k - (PIPE_STAGE - 1);
        GROUP_BOUNDARY_HT(prev_k);
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            for (int32_t r = 0; r < 4; ++r) {
                rA[prev_k % PIPE_STAGE][m].int_arr[0][r] =
                    __shfl(rA[prev_k % PIPE_STAGE][m].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n) {
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[0],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[0], rC[m][n]);
                rC[m][n] = mma_cw(rA[prev_k % PIPE_STAGE][m].int8_arr[1],
                                  rB[prev_k % PIPE_STAGE][n].int8_arr[1], rC[m][n]);
            }
        }
    }

    // Output: rC * last_group_scale (slot 0) * weight_scale -> BF16
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        if (ld_row + m * 16 < valid_m) {
            const float s_last = smem_scale[(ld_row + m * 16) * kScaleStride];
            for (int n = 0; n < N_REPEAT; n++) {
                vec4_bf16 st;
                for (int32_t i = 0; i < MMA_C_REGS; i++) {
                    float r_st = static_cast<float>(rC[m][n][i])
                        * s_last
                        * weight_scale[n][i];
                    hip_bfloat16 tmp = hip_bfloat16(r_st);
                    st[i] = reinterpret_cast<unsigned short&>(tmp);
                }
                int32_t st_lane_offset = (ld_row + m * 16) * C_stride + ld_col * MMA_C_REGS;
                vec4_bf16* p_st = reinterpret_cast<vec4_bf16*>(
                    cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
                p_st[0] = st;
            }
        }
    }

    #undef GROUP_BOUNDARY_HT
}

} // namespace deep_gemm::mega::hcu
