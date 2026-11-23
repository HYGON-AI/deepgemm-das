// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "mqa_logits/common/types.hpp"
#include "mqa_logits/common/window.hpp"
#include "mqa_logits/common/epilogue.hpp"

namespace deepgemm::mqa::int8 {
using common::classify_window;
using common::F4;
using common::fill_empty_tile;
using common::I2;
using common::I4;
using common::load_window;
using common::output_vector_aligned;
using common::small_window;
using common::store1;
using common::store_logits4;
using common::tail_rows;
using common::TileWindow;
using common::Window;

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Public row views guarantee only four-byte alignment, as in FP8.
// memcpy avoids asserting the stronger C++ vector-pointer alignment.
template <class Vector, class Scalar>
__device__ __forceinline__ Vector load_aligned4(const Scalar *src) {
    Vector value;
    const auto *aligned = static_cast<const Scalar *>(__builtin_assume_aligned(src, 4));
    __builtin_memcpy(&value, aligned, sizeof(value));
    return value;
}

// Rebase at the tile so every lane offset fits in 32 bits even for a large KV.
__device__ __forceinline__ I4 make_kv_resource(const void *tile) {
    const uint64_t address = reinterpret_cast<uint64_t>(tile);
    return {static_cast<int>(address), static_cast<int>(address >> 32), -1, 0x00020000};
}

// KV fragments are reused from registers across heads. SLC can reduce cache
// pollution from streaming KV reads and preserve cache capacity for Q/weights,
// which is the expected reason for the measured speedup.
// Only KV uses SLC. The builtin preserves compiler-managed load dependencies;
// GLC stays off and Q/scales retain their ordinary four-byte-aligned loads.
__device__ __forceinline__ I2 load_kv_slc(I4 resource, int offset) {
    return __builtin_amdgcn_buffer_load_dwordx2(resource, 0, offset, false, true);
}

// Signed INT8 MMAC shares the non-LIT layout on gfx936 and gfx938.
__device__ __forceinline__ void mmac_i8(const I2 &a, const I2 &b, I4 &c) {
#if defined(__gfx936__) || defined(__gfx938__)
    c = __builtin_hcu_mmac_i32_16x16x32_i8(a, b, c);
#endif
}

// For signed INT8 and D=128, every partial dot is in [-2^21, 2^21].
// Adding its integer value to these bits keeps the FP32 exponent unchanged,
// so bit_cast<float>(biased_dot) - 1.5*2^23 is exact. Clamp before subtracting.
constexpr int kDotBias = 0x4b400000;
constexpr float kDotBiasValue = 12582912.0f;
static_assert(128 * 128 * 128 < (1 << 22));

__device__ __forceinline__ I4 dot_init() {
    return I4{kDotBias, kDotBias, kDotBias, kDotBias};
}

// Two independent FP32 pairs; no change to the reduction order.
__device__ __forceinline__ F4 pk_mul4(F4 a, F4 b) {
    F4 result;
    result.xy = __builtin_hcu_pk_mul_f32(a.xy, b.xy);
    result.zw = __builtin_hcu_pk_mul_f32(a.zw, b.zw);
    return result;
}

__device__ __forceinline__ void relu_weighted_sum(const I4 &dot, float w, F4 &accum) {
    using pair_t = float __attribute__((ext_vector_type(2)));
    const pair_t weight = {w, w};
    const pair_t negative_bias = {-kDotBiasValue, -kDotBiasValue};
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
        pair_t value = {__builtin_bit_cast(float, max(dot[i], kDotBias)),
                        __builtin_bit_cast(float, max(dot[i + 1], kDotBias))};
        value = __builtin_hcu_pk_add_f32(value, negative_bias);
        pair_t sum = {accum[i], accum[i + 1]};
        sum = __builtin_hcu_pk_fma_f32(value, weight, sum);
        accum[i] = sum[0];
        accum[i + 1] = sum[1];
    }
}

// Shared by dense and compact kernels.
__device__ __forceinline__ void weighted(const I4 &d, const F4 &w, F4 &a) {
    // Preserve the original per-head rounding and head reduction order.
    using pair_t = float __attribute__((ext_vector_type(2)));
    const pair_t negative_bias = {-kDotBiasValue, -kDotBiasValue};
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
        pair_t value = {__builtin_bit_cast(float, max(d[i], kDotBias)),
                        __builtin_bit_cast(float, max(d[i + 1], kDotBias))};
        value = __builtin_hcu_pk_add_f32(value, negative_bias);
        const pair_t weight = {w[i], w[i + 1]};
        pair_t sum = {a[i], a[i + 1]};
        sum = __builtin_hcu_pk_fma_f32(value, weight, sum);
        a[i] = sum[0];
        a[i + 1] = sum[1];
    }
}

#endif
} // namespace deepgemm::mqa::int8
