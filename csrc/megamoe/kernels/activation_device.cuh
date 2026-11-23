// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// SwiGLU + per-token Channel-Wise FP8 quantization for HCU gfx938.
// Ported from V1 csrc/kernels/unified_moe/activation_device_funcs.cuh:98-167
// (silu_and_mul_quant_channelwise).
//
// Behavior (matching V1 exactly):
//   - Input: gate and up halves (each intermediate_size elements)
//   - Compute: result = SiLU(gate) * up = gate / (1 + exp(-gate)) * up
//   - Per-token absmax: reduce over ALL intermediate_size elements
//   - Single scale per token: scale = max(amax, 1e-4) / 448.0
//   - Quantize: fp32_to_fp8_e4m3(clamp(result * (448.0 / amax), -448, 448))
//   - Output: FP8 E4M3 [num_max_pool_tokens, intermediate_hidden]
//   - Output scale: float [num_max_pool_tokens] (one per token)

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include "params.h"
#include "warp_reduce.cuh"

namespace deep_gemm::mega::hcu {

// FP32 to FP8 E4M3 conversion (from V1)
static __device__ __forceinline__ uint8_t fp32_to_fp8_e4m3(float value) {
    float2 packed_input{value, 0.0f};
    auto packed = __hip_cvt_float2_to_fp8x2(packed_input, __HIP_SATFINITE, __HIP_E4M3);
    return reinterpret_cast<uint8_t*>(&packed)[0];
}

// FP32 to INT8 conversion (symmetric, clamp to [-127, 127])
// Matches DeepEP _quantize_low_latency_layered quant_type=1:
//   scale = 127.0 / amax; xq = round(x * scale).clamp(-127, 127).to(int8)
static __device__ __forceinline__ int8_t fp32_to_int8(float value) {
    float q = rintf(value);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return static_cast<int8_t>(static_cast<int32_t>(q));
}


// ============================================================================
// Quantization clamp (DG_MEGA_QUANT_NOCLAMP, default 1 = clamp removed).
//   INT8: the explicit clamp is REQUIRED -- a float->int cast wraps on overflow.
//   FP8 : redundant -- __hip_cvt_float2_to_fp8x2(..., __HIP_SATFINITE, ...)
//         already saturates to +-448, so the two min/max were pure overhead
//         (2 ALU ops per output element in every L1 epilogue).
//   Only difference: a NaN input now converts to canonical NaN instead of +-448;
//   the fused path never produces NaNs (tests assert non-finite == 0).
// Set DG_MEGA_QUANT_NOCLAMP=0 (build flag) to restore the old always-clamp form.
// ============================================================================
#ifndef DG_MEGA_QUANT_NOCLAMP
#define DG_MEGA_QUANT_NOCLAMP 1
#endif

template <bool kIsInt8>
static __device__ __forceinline__ float quant_clamp(float x, float qmax) {
#if DG_MEGA_QUANT_NOCLAMP
    if constexpr (kIsInt8) return fminf(fmaxf(x, -qmax), qmax);
    else return x;
#else
    return fminf(fmaxf(x, -qmax), qmax);
#endif
}

// ============================================================================
// NV SM100-aligned SwiGLU for the fused GEMM epilogues
// (sm100_fp8_fp4_mega_moe.cuh):
//   1. (2026-11-17) the bf16 round-trip of gate/up was DROPPED: the fused path
//      keeps f32 accumulators, so the two cvt instructions per column were pure
//      overhead.  It used to mimic NV storing accumulators as bf16 in TMEM
//      (and the ref path's bf16 l1_out); the small numeric difference is inside
//      the test tolerance.
//   2. gate upper-clamped at +activation_clamp; up clamped to
//      [-activation_clamp, +activation_clamp].  activation_clamp=+inf
//      (the API default) degenerates to a no-op; fminf/fmaxf NaN
//      semantics make a NaN clamp a no-op too.
//   3. silu(gate) * up:
//        fast_math  -> __expf + approximate reciprocal
//        !fast_math -> expf + IEEE division
// ============================================================================
static __device__ __forceinline__ float swiglu_nv(
    float g, float u, float activation_clamp, bool fast_math)
{
    g = fminf(g, activation_clamp);
    u = fminf(fmaxf(u, -activation_clamp), activation_clamp);
    const float denom = 1.0f + (fast_math ? __expf(-g) : expf(-g));
    return (fast_math ? g * __frcp_rn(denom) : g / denom) * u;
}

} // namespace deep_gemm::mega::hcu
