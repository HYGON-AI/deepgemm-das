// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// Fused pre-processing kernel for HCU gfx938 MegaMoE.
// Combines bf16 -> FP8 channelwise quantization + topk copy into a single kernel,
// writing directly into the symmetric buffer.
//

#include <cstdint>
#include <hip/hip_runtime.h>
#include "params.h"

namespace deep_gemm::mega::hcu {

// ============================================================================
// Device helpers (bf16 conversion, FP8 packing, wave reduce)
// ============================================================================

__device__ static inline float bf16_bits_to_float(const uint16_t x) {
#if defined(__gfx938__)
    return __builtin_hcu_cvt_f32_bf16(x, false, 0, false);
#else
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
#endif
}

__device__ static inline float clip_fp8_e4m3fn(const float x) {
    return fminf(448.0f, fmaxf(-448.0f, x));
}

__device__ static inline uint32_t pack4_fp8_e4m3fn(
    const float x0, const float x1,
    const float x2, const float x3) {
#if defined(__gfx938__) || defined(__gfx94__) || defined(__gfx12__)
    uint32_t packed = 0;
    packed = __builtin_hcu_cvt_pk_fp8_f32(
        clip_fp8_e4m3fn(x0), clip_fp8_e4m3fn(x1), packed, false);
    packed = __builtin_hcu_cvt_pk_fp8_f32(
        clip_fp8_e4m3fn(x2), clip_fp8_e4m3fn(x3), packed, true);
    return packed;
#else
    // Fallback: manual packing (not used on gfx938)
    auto cvt = [](float v) -> uint8_t {
        // Simple truncation-based e4m3 (fallback only)
        v = clip_fp8_e4m3fn(v);
        uint32_t bits = __float_as_uint(v);
        return static_cast<uint8_t>(bits >> 24);
    };
    return cvt(x0) | (cvt(x1) << 8) | (cvt(x2) << 16) | (cvt(x3) << 24);
#endif
}

__device__ static inline float wave_shuffle_down_float(const float value, const int lane_delta) {
    const int32_t remote = __builtin_amdgcn_ds_bpermute(
        (__lane_id() + lane_delta) << 2,
        __builtin_bit_cast(int32_t, value));
    return __builtin_bit_cast(float, remote);
}

__device__ static inline float wave_broadcast_lane0_float(const float value) {
    const int32_t lane0 = __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int32_t, value));
    return __builtin_bit_cast(float, lane0);
}

__device__ static inline float wave_reduce_max_64(float value) {
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        value = fmaxf(value, wave_shuffle_down_float(value, offset));
    return wave_broadcast_lane0_float(value);
}

// ============================================================================
// stage_topk_route: copy topk_idx and topk_weights to output buffers
// ============================================================================

template <bool TopkIdxI64, bool TopkWeightsBf16>
__device__ static inline void stage_topk_route(
    const void* __restrict__ topk_idx,
    const void* __restrict__ topk_weights,
    int64_t* __restrict__ out_topk_idx,
    float* __restrict__ out_topk_weights,
    const int64_t route_offset) {
    if constexpr (TopkIdxI64) {
        out_topk_idx[route_offset] = static_cast<const int64_t*>(topk_idx)[route_offset];
    } else {
        out_topk_idx[route_offset] =
            static_cast<int64_t>(static_cast<const int32_t*>(topk_idx)[route_offset]);
    }
    if constexpr (TopkWeightsBf16) {
        out_topk_weights[route_offset] =
            bf16_bits_to_float(static_cast<const uint16_t*>(topk_weights)[route_offset]);
    } else {
        out_topk_weights[route_offset] = static_cast<const float*>(topk_weights)[route_offset];
    }
}

// ============================================================================
// mega_moe_preprocess_kernel
//   Fuses: bf16 -> FP8 channelwise quant + topk copy into sym buffer.
//   One block per row (token).
// ============================================================================

template <int Threads, bool TopkIdxI64, bool TopkWeightsBf16, bool kIsInt8 = false>
__global__ __launch_bounds__(Threads)
void mega_moe_preprocess_kernel(
    const uint16_t* __restrict__ x_bf16,
    const void* __restrict__ topk_idx,
    const void* __restrict__ topk_weights,
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    int64_t* __restrict__ out_topk_idx,
    float* __restrict__ out_topk_weights,
    const int rows,
    const int hidden,
    const int topk) {
#if !defined(__gfx938__)
    // MegaMoE device code is gfx938-only; empty stub for other archs.
    return;
#else
    static_assert(Threads % kWarpSize == 0,
                  "Threads must tile whole waves for wave_maxes");
    __shared__ float wave_maxes[Threads / kWarpSize];
    __shared__ float row_scale;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int64_t x_row_offset = static_cast<int64_t>(row) * hidden;
    const int64_t topk_row_offset = static_cast<int64_t>(row) * topk;

    // Stage topk idx/weights (first topk threads)
    if (tid < topk) {
        const int64_t route_offset = topk_row_offset + tid;
        stage_topk_route<TopkIdxI64, TopkWeightsBf16>(
            topk_idx, topk_weights, out_topk_idx, out_topk_weights, route_offset);
    }

    // Pass 1: compute row amax (8 elements per thread per iteration)
    float local_max = 0.0f;
    for (int col = tid * 8; col < hidden; col += Threads * 8) {
        const uint64_t packed =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col);
        const uint64_t packed_hi =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 4);
        const float x0 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed)));
        const float x1 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed >> 16)));
        const float x2 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed >> 32)));
        const float x3 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed >> 48)));
        const float x4 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi)));
        const float x5 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 16)));
        const float x6 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 32)));
        const float x7 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 48)));
        local_max = fmaxf(local_max,
                          fmaxf(fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)),
                                fmaxf(fmaxf(x4, x5), fmaxf(x6, x7))));
    }

    // Block-level reduce max
    const float wave_max = wave_reduce_max_64(local_max);
    if ((tid & (kWarpSize - 1)) == 0)
        wave_maxes[tid / kWarpSize] = wave_max;
    __syncthreads();

    float block_max = tid < (Threads / kWarpSize) ? wave_maxes[tid] : 0.0f;
    if (tid < kWarpSize)
        block_max = wave_reduce_max_64(block_max);
    if (tid == 0) {
        // INT8: scale = amax / 127.0;  FP8: scale = amax / 448.0
        if constexpr (kIsInt8) {
            row_scale = fmaxf(block_max, 1.0e-6f) / 127.0f;
        } else {
            row_scale = fmaxf(block_max, 1.0e-4f) / 448.0f;
        }
        out_scale[row] = row_scale;
    }
    __syncthreads();

    // Pass 2: quantize and write
    for (int col = tid * 8; col < hidden; col += Threads * 8) {
        const uint64_t packed =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col);
        const uint64_t packed_hi =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 4);
        const float x0 = bf16_bits_to_float(static_cast<uint16_t>(packed)) / row_scale;
        const float x1 = bf16_bits_to_float(static_cast<uint16_t>(packed >> 16)) / row_scale;
        const float x2 = bf16_bits_to_float(static_cast<uint16_t>(packed >> 32)) / row_scale;
        const float x3 = bf16_bits_to_float(static_cast<uint16_t>(packed >> 48)) / row_scale;
        const float x4 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi)) / row_scale;
        const float x5 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 16)) / row_scale;
        const float x6 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 32)) / row_scale;
        const float x7 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 48)) / row_scale;
        if constexpr (kIsInt8) {
            // INT8: round + clamp to [-127, 127], pack as signed bytes
            auto q8 = [](float v) -> uint8_t {
                float q = rintf(v);
                q = fminf(fmaxf(q, -127.0f), 127.0f);
                return static_cast<uint8_t>(static_cast<int8_t>(static_cast<int32_t>(q)));
            };
            uint64_t out_packed = 0;
            out_packed |= static_cast<uint64_t>(q8(x0));
            out_packed |= static_cast<uint64_t>(q8(x1)) << 8;
            out_packed |= static_cast<uint64_t>(q8(x2)) << 16;
            out_packed |= static_cast<uint64_t>(q8(x3)) << 24;
            out_packed |= static_cast<uint64_t>(q8(x4)) << 32;
            out_packed |= static_cast<uint64_t>(q8(x5)) << 40;
            out_packed |= static_cast<uint64_t>(q8(x6)) << 48;
            out_packed |= static_cast<uint64_t>(q8(x7)) << 56;
            *reinterpret_cast<uint64_t*>(out_fp8 + x_row_offset + col) = out_packed;
        } else {
            uint32_t* out_vec = reinterpret_cast<uint32_t*>(out_fp8 + x_row_offset + col);
            out_vec[0] = pack4_fp8_e4m3fn(x0, x1, x2, x3);
            out_vec[1] = pack4_fp8_e4m3fn(x4, x5, x6, x7);
        }
    }
#endif // !defined(__gfx938__)
}

// ============================================================================
// Host launcher
// ============================================================================

static inline void launch_mega_moe_preprocess(
    const void* x_bf16,
    const void* topk_idx,
    const void* topk_weights,
    void* out_fp8,
    float* out_scale,
    int64_t* out_topk_idx,
    float* out_topk_weights,
    const int rows,
    const int hidden,
    const int topk,
    const bool topk_idx_i64,
    const bool topk_weights_bf16,
    const bool is_int8 = false) {
    constexpr int kThreads = 256;
    TORCH_CHECK(hidden % 8 == 0,
                "preprocess reads 8 bf16 per thread-step; hidden=", hidden);
    TORCH_CHECK(topk <= kThreads,
                "preprocess stages topk on the first topk threads; topk=", topk);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(std::max(1, rows));
    const dim3 block(kThreads);

#define LAUNCH_PREPROCESS(TOPK_I64, TOPK_W_BF16, IS_INT8) \
    do { \
        hipLaunchKernelGGL( \
            (mega_moe_preprocess_kernel<kThreads, TOPK_I64, TOPK_W_BF16, IS_INT8>), \
            grid, block, 0, stream, \
            static_cast<const uint16_t*>(x_bf16), \
            topk_idx, topk_weights, \
            static_cast<uint8_t*>(out_fp8), \
            out_scale, out_topk_idx, out_topk_weights, \
            rows, hidden, topk); \
    } while (0)

#define LAUNCH_PREPROCESS_TOPK(TOPK_I64, TOPK_W_BF16) \
    do { \
        if (is_int8) { \
            LAUNCH_PREPROCESS(TOPK_I64, TOPK_W_BF16, true); \
        } else { \
            LAUNCH_PREPROCESS(TOPK_I64, TOPK_W_BF16, false); \
        } \
    } while (0)

    if (topk_idx_i64) {
        if (topk_weights_bf16) {
            LAUNCH_PREPROCESS_TOPK(true, true);
        } else {
            LAUNCH_PREPROCESS_TOPK(true, false);
        }
    } else {
        if (topk_weights_bf16) {
            LAUNCH_PREPROCESS_TOPK(false, true);
        } else {
            LAUNCH_PREPROCESS_TOPK(false, false);
        }
    }
#undef LAUNCH_PREPROCESS
#undef LAUNCH_PREPROCESS_TOPK
}

} // namespace deep_gemm::mega::hcu
