// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// BF16 device primitives shared by every BF16 kernel impl: aligned vector reads,
// K buffer descriptors, MMAC wrappers, ReLU/PK-FMA helpers and Q staging.
// Device code only; no Tensor adapter, auto table or host launch.

#pragma once
#include "mqa_logits/common/types.hpp"
#include "mqa_logits/common/window.hpp"
#include "mqa_logits/common/epilogue.hpp"

namespace deepgemm::mqa::bf16 {
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

using B4 = short __attribute__((ext_vector_type(4)));

// All BF16 schedules use SLC for KV. Rebase the descriptor at each
// tile so the 32-bit byte offsets stay bounded even for large tensors.
__device__ __forceinline__ I4 make_kv_resource(const uint16_t *tile) {
    const uint64_t address = reinterpret_cast<uint64_t>(tile);
    return {static_cast<int>(address), static_cast<int>(address >> 32),
            -1, 0x00020000};
}
__device__ __forceinline__ I4 load_kv4(I4 resource, int byte_offset) {
    return __builtin_amdgcn_buffer_load_dwordx4(resource, 0, byte_offset, false, true);
}
__device__ __forceinline__ B4 load_kv2(I4 resource, int byte_offset) {
    const I2 raw = __builtin_amdgcn_buffer_load_dwordx2(resource, 0, byte_offset, false, true);
    B4 value;
    __builtin_memcpy(&value, &raw, sizeof(raw));
    return value;
}

__device__ __forceinline__ void mmac_bf16(const B4 &a, const B4 &b, F4 &c) {
#if defined(__gfx936__) || defined(__gfx938__)
    c = __builtin_hcu_mmac_f32_16x16x16_bf16(a, b, c);
#endif
    return;
}
// Keep the independent zero-C input; let aicc allocate and schedule the builtin result.
__device__ __forceinline__ F4 mmac_first(const B4 &a, const B4 &b, const F4 &zero) {
    F4 result = zero;
#if defined(__gfx936__) || defined(__gfx938__)
    result = __builtin_hcu_mmac_f32_16x16x16_bf16(a, b, zero);
#endif
    return result;
}
// An ordered comparison clamps negative values while preserving NaNs.
// Keep this in compiler-visible code so MMAC consumer waits are scheduled.
__device__ __forceinline__ float relu(float x) {
    return x < 0.f ? 0.f : x;
}
__device__ __forceinline__ void relu_weighted_sum(const F4 &dot, float w, F4 &accum) {
    using F2 = float __attribute__((ext_vector_type(2)));
    F4 value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        value[i] = relu(dot[i]);
    accum.xy = __builtin_hcu_pk_fma_f32(F2{w, w}, value.xy, accum.xy);
    accum.zw = __builtin_hcu_pk_fma_f32(F2{w, w}, value.zw, accum.zw);
    return;
}

// Requires NaN-free values; the integer clamp normalizes -0 to +0.
// Pass a scalar to bit_cast: DTK miscompiles bit_cast(vector[index]).
__device__ __forceinline__ float relu_non_nan(float x) {
    return __builtin_bit_cast(
        float, __builtin_elementwise_max(__builtin_bit_cast(int, x), 0));
}

__device__ __forceinline__ void weighted_sum_only(const F4 &value, float w, F4 &accum) {
    using F2 = float __attribute__((ext_vector_type(2)));
    accum.xy = __builtin_hcu_pk_fma_f32(value.xy, F2{w, w}, accum.xy);
    accum.zw = __builtin_hcu_pk_fma_f32(value.zw, F2{w, w}, accum.zw);
    return;
}

__device__ __forceinline__ void prefetch4(I4 *dst, const uint16_t *ptr, int offset, bool valid) {
    offset *= 2; // BF16 element offset to bytes
    using resource_t = uint32_t __attribute__((ext_vector_type(4)));
    const uint64_t address = reinterpret_cast<uint64_t>(ptr);
    resource_t resource = {
        static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address >> 32)) + 0x41000000u,
        0x80000000u, 0x00020000u};
    if (valid) {
        // Q uses no explicit SLC modifier, following the measured gfx936 policy.
        // KV enables SLC separately in load_kv4/load_kv2.
        // Expose the loaded value to the compiler for dependency scheduling.
        // The caller waits for VMEM before publishing these vectors through LDS.
        *dst = __builtin_bit_cast(
            I4, __builtin_amdgcn_buffer_load_dwordx4(
                    __builtin_bit_cast(I4, resource), 0, offset, false, false));
    }
    return;
}

// Shared by dense and compact kernels.
template <int H, bool Pair>
__device__ __forceinline__ void weighted(const F4 &dot, const F4 &w, F4 &accum, int lane) {
    // Pair adjacent independent components, preserving each head accumulator's order.
    // Padded heads must not update accum: zero Q times infinite K can be NaN.
    using F2 = float __attribute__((ext_vector_type(2)));
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
        const F2 value = {relu(dot[i]), relu(dot[i + 1])};
        const F2 weight = {w[i], w[i + 1]};
        const F2 initial = {accum[i], accum[i + 1]};
        const F2 updated = __builtin_hcu_pk_fma_f32(weight, value, initial);
        if (Pair || H >= 16 || lane / 16 + i * 4 < H)
            accum[i] = updated[0];
        if (Pair || H >= 16 || lane / 16 + (i + 1) * 4 < H)
            accum[i + 1] = updated[1];
    }
    return;
}

#endif
} // namespace deepgemm::mqa::bf16
