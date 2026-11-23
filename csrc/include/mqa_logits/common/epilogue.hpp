// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "mqa_logits/common/types.hpp"
#include "mqa_logits/common/params.hpp"
#include "mqa_logits/common/window.hpp"

namespace deepgemm::mqa::common {
__device__ __forceinline__ void store4(float *destination, F4 value) {
    // DCC must see the store to protect its source registers from early reuse.
    // An opaque global_store ASM miscompiled the last M16's aligned writeback.
    *reinterpret_cast<F4 *>(destination) = value;
}

__device__ __forceinline__ void store1(float *destination, float value) { *destination = value; }

// The input/scales retain their global KV column. Only the output is rebased.
// This helper is used by no-clean/compact epilogues; dense clean fast paths
// retain their existing stores and mask handling.
template <class Params>
__device__ __forceinline__ void store_window_value(const Params &p, int64_t row,
                                                   int64_t col, float value) {
    if (row < 0 || row >= p.m || col < 0 || col >= p.n)
        return;
    const Window w = load_window(p.ks, p.ke, p.n, row);
    if (col < w.start || col >= w.end)
        return;
    const int64_t out_col = p.output_compressed ? col - w.start : col;
    if (p.output_compressed && uint64_t(out_col) >= uint64_t(p.output_width))
        return;
    store1(p.out + row * p.out_stride_m + out_col, value);
}

template <class Params>
__device__ __forceinline__ void store_window4(const Params &p, int64_t row,
                                              int64_t col, F4 value) {
    if (row < 0 || row >= p.m)
        return;
    const Window w = load_window(p.ks, p.ke, p.n, row);
    if (col >= 0 && col + 3 < p.n && w.start <= col && col + 3 < w.end) {
        const int64_t out_col = p.output_compressed ? col - w.start : col;
        if (!p.output_compressed || uint64_t(out_col + 3) < uint64_t(p.output_width)) {
            float *dst = p.out + row * p.out_stride_m + out_col;
            if ((reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
                store4(dst, value);
                return;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        store_window_value(p, row, col + j, value[j]);
}

// AlignN is the logical N divisibility. An arbitrary caller-owned output may
// have weaker base/row alignment; retain scalar fallback without touching padding.
template <int AlignN>
__device__ __forceinline__ bool output_vector_aligned(const float *out, int64_t stride) {
    static_assert(AlignN == 4 || AlignN == 2 || AlignN == 1);
    return ((uintptr_t(out) | (uint64_t(stride) * sizeof(float))) & (AlignN * sizeof(float) - 1)) ==
           0;
}

__device__ __forceinline__ void store_logits4(float *out, F4 value, bool vector_out) {
    // The caller guards the packet's first column; N%4 guarantees all four.
    if (vector_out)
        store4(out, value);
    else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            store1(out + j, value[j]);
    }
}

template <int AlignN, int Waves = 4>
__device__ __forceinline__ void fill_empty_tile(float *out, int64_t stride, int64_t m0, int64_t n0,
                                                int valid_m, int valid_n, int wave, int lane) {
    const float ni = -__builtin_inff();
    const bool vector_out = output_vector_aligned<AlignN>(out, stride);
    for (int row = wave; row < valid_m; row += Waves) {
        for (int col = AlignN * lane; col < valid_n; col += AlignN * 64) {
            float *dst = out + (m0 + row) * stride + n0 + col;
            if constexpr (AlignN == 4)
                store_logits4(dst, F4{ni, ni, ni, ni}, vector_out);
            else if constexpr (AlignN == 2) {
                const F2 value = {ni, ni};
                __builtin_memcpy(dst, &value, sizeof(value));
            } else
                store1(dst, ni);
        }
    }
}

} // namespace deepgemm::mqa::common
