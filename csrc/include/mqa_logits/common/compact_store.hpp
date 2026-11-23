// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mqa_logits/common/window.hpp"

namespace deepgemm::mqa::common {
struct RowWindow { int32_t start, end; };
struct EmptyWindow {};

template <bool Packet = true, class Params>
__device__ __forceinline__ TileWindow compact_window(const Params &p, int64_t m0,
    int64_t n0, int vm, int vn, int lane, RowWindow &row) {
    bool any = false, covered = true;
    row = {0, 0};
    if (lane < vm) {
        const int32_t n = int32_t(p.n);
        int32_t s = p.ks ? p.ks[m0 + lane] : 0;
        int32_t e = p.ke ? p.ke[m0 + lane] : n;
        s = s < 0 ? 0 : (s > n ? n : s);
        e = e < 0 ? 0 : (e > n ? n : e);
        const int64_t limit = int64_t(s) + int64_t(p.output_width);
        row = {s, int32_t(e < limit ? e : limit)};
        any = row.start < row.end && row.start < int32_t(n0 + vn) && row.end > int32_t(n0);
        // The full fast path writes complete four-float packets, including N tails.
        covered = row.start <= int32_t(n0) &&
            int64_t(row.end) >= n0 + (Packet ? ((vn + 3) & ~3) : vn);
    }
    // Conditional loads must reconverge before full-wave collectives.
    return {__ballot(any) != 0, __ballot(!covered) == 0};
}

__device__ __forceinline__ RowWindow compact_shuffle(RowWindow w, int row, bool full = false) {
    const int32_t start = __shfl(w.start, row, 64);
    const int32_t end = full ? 0 : __shfl(w.end, row, 64);
    return {start, end};
}

template <int Row>
__device__ __forceinline__ RowWindow compact_readlane(RowWindow w) {
    return {int32_t(__builtin_amdgcn_readlane(w.start, Row)),
            int32_t(__builtin_amdgcn_readlane(w.end, Row))};
}

template <class Params, class Value>
__device__ __forceinline__ void compact_store4(const Params &p, int64_t row,
    int64_t col, Value value, RowWindow w, bool tile_full) {
    static_assert(sizeof(Value) == 16);
    const int32_t c = int32_t(col);
    const int64_t packet_index = row * p.out_stride_m + int64_t(c - w.start);
    if (tile_full || (w.start <= c && c + 3 < w.end)) {
        float *dst = p.out + packet_index;
        // Float-aligned memcpy permits an unaligned compact start without
        // imposing the vector type's stronger alignment requirement.
        __builtin_memcpy(dst, &value, 16);
    } else if (w.start <= c + 3 && c < w.end) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (w.start <= c + j && c + j < w.end)
                p.out[packet_index + j] = value[j];
    }
}

template <class Params>
__device__ __forceinline__ void compact_store1(const Params &p, int64_t row,
    int64_t col, float value, RowWindow w, bool tile_full) {
    if (tile_full || (w.start <= col && col < w.end))
        p.out[row * p.out_stride_m + col - w.start] = value;
}
} // namespace deepgemm::mqa::common
