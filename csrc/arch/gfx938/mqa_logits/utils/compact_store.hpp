// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "window.hpp"
#include <type_traits>
namespace deepgemm::mqa::fp8 {
struct RowWindow { int32_t start, end; };
struct EmptyWindow {};
template <bool Packet = true, class Params>
__device__ __forceinline__ TileWindow compact_window(const Params &p, int64_t m0,
    int64_t n0, int vm, int vn, int lane, RowWindow &row) {
    bool any = false, covered = true;
    row = {0, 0};
    if (lane < vm) {
        // Host N is a nonnegative int32. Bounds and tile columns fit int32;
        // widen only the capacity sum, which can exceed INT32_MAX.
        const int32_t n = int32_t(p.n);
        int32_t s = p.ks ? p.ks[m0 + lane] : 0;
        int32_t e = p.ke ? p.ke[m0 + lane] : n;
        s = s < 0 ? 0 : (s > n ? n : s);
        e = e < 0 ? 0 : (e > n ? n : e);
        const int64_t limit = int64_t(s) + int64_t(p.output_width);
        row = {s, int32_t(e < limit ? e : limit)};
        any = row.start < row.end && row.start < int32_t(n0 + vn) && row.end > int32_t(n0);
        // Packets must fit the last partial N tile; scalar QH needs no rounding.
        covered = row.start <= int32_t(n0) && int64_t(row.end) >= n0 + (Packet ? ((vn + 3) & ~3) : vn);
    }
    // Reconverge after conditional loads before the full-wave collectives.
    return {__ballot(any) != 0, __ballot(!covered) == 0};
}
__device__ __forceinline__ RowWindow compact_shuffle(RowWindow w, int row, bool full = false) {
    const int32_t start = __shfl(w.start, row, 64);
    // full is uniform across the wave; the collective remains converged.
    const int32_t end = full ? 0 : __shfl(w.end, row, 64);
    return {start, end};
}
template <int Row>
__device__ __forceinline__ RowWindow compact_readlane(RowWindow w) {
    return {int32_t(__builtin_amdgcn_readlane(w.start, Row)),
            int32_t(__builtin_amdgcn_readlane(w.end, Row))};
}
__device__ __forceinline__ void compact_store4(const MqaFp8Params &p, int64_t row,
    int64_t col, F4 value, RowWindow w, bool tile_full) {
    // Every consumer's first column is a multiple of four and below N;
    // its last column is <= INT32_MAX, including the physical tail packet.
    const int32_t c = int32_t(col);
    if (tile_full || (w.start <= c && c + 3 < w.end)) {
        float *dst = p.out + row * p.out_stride_m + int64_t(c - w.start);
        // AICC emits global_store_dwordx4 for this 16-byte memcpy, including
        // float-aligned destinations. Avoid splitting identical store paths.
        __builtin_memcpy(dst, &value, 16);
    } else if (w.start <= c + 3 && c < w.end) {
        // Entirely outside packets need no scalar boundary predicates.
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (w.start <= c + j && c + j < w.end)
                store1(p.out + row * p.out_stride_m + int64_t(c + j - w.start), value[j]);
    }
}
} // namespace deepgemm::mqa::fp8
