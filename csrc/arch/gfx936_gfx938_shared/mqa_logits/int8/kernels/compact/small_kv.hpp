// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../utils/ops.hpp"
#include "mqa_logits/common/params.hpp"
#include "mqa_logits/common/compact_store.hpp"

namespace deepgemm::mqa::int8 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Heads form MMAC columns. H8 can place two adjacent queries in 16 columns.
template <int H, int TN, int WAVES, bool PAIR, bool EvenM, bool EvenN>
__device__ __forceinline__ void run_compact(const int8_t *__restrict__ q, const int8_t *__restrict__ k,
                                    const float *__restrict__ weights,
                                    const int32_t *__restrict__ ks, const int32_t *__restrict__ ke,
                                    const float *__restrict__ scales, float *__restrict__ out,
                                    int m, int n, int64_t weights_stride_m, int64_t out_stride_m,
                                    int64_t output_width) {
    static_assert(!PAIR || H == 8);
    constexpr int Q = PAIR ? 2 : 1, NT = TN / 16;
    const int lane = threadIdx.x % 64, row = (blockIdx.x * WAVES + threadIdx.x / 64) * Q;
    const int begin = blockIdx.y * TN;
    if (!EvenM && row >= m)
        return;
    int lo[Q], hi[Q];

    const int vn = EvenN ? TN : tail_rows<TN>(int64_t(n) - begin);

    TileWindow tile{false, true};
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        lo[j] = hi[j] = 0;
        // A wave handles one query (or a pair); the row predicate is uniform.
        if (EvenM || row + j < m) {
            int32_t s = ks ? ks[row + j] : 0;
            int32_t e = ke ? ke[row + j] : n;
            s = s < 0 ? 0 : (s > n ? n : s);
            e = e < 0 ? 0 : (e > n ? n : e);
            const int64_t limit = int64_t(s) + output_width;
            e = int32_t(e < limit ? e : limit);
            lo[j] = __builtin_amdgcn_readfirstlane(s);
            hi[j] = __builtin_amdgcn_readfirstlane(e);
            tile.any |= lo[j] < hi[j] && lo[j] < begin + vn && hi[j] > begin;
            tile.full &= lo[j] <= begin && hi[j] >= begin + vn;
        }
    }
    const bool valid = tile.any, tile_full = tile.full;
    if (!valid)
        return;
    const I4 kv_resource = make_kv_resource(k + int64_t(begin) * 128);
    I2 kr[NT][4];
    F4 acc[NT];
    constexpr bool PrefetchScale = (H == 8 && WAVES == 2 && !PAIR && !EvenN) || (H == 16 && TN == 32);
    float prefetched_scale[NT]{};
#pragma unroll
    for (int t = 0; t < NT; t++) {
        acc[t] = {0, 0, 0, 0};
        const int col = begin + t * 16 + lane % 16;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            kr[t][d] = {0, 0};
            if (EvenN || col < n)
                kr[t][d] = load_kv_slc(
                    kv_resource, (col - begin) * 128 + lane / 16 * 8 + d * 32);
        }
    }
    if constexpr (H <= 16) {
#pragma unroll
        for (int h = 0; h < H; h += 16) {
            I2 qr[4];
            const int qrrow = row + (PAIR ? (lane % 16) / 8 : 0),
                      head = h + (PAIR ? lane % 8 : lane % 16);
#pragma unroll
            for (int d = 0; d < 4; d++) {
                qr[d] = {0, 0};
                if ((EvenM || qrrow < m) && head < H)
                    qr[d] = load_aligned4<I2>(q + (int64_t(qrrow) * H + head) * 128 + lane / 16 * 8 +
                                              d * 32);
            }
            F4 w;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int colh = h + lane / 16 + e * 4, wr = row + (PAIR ? colh / 8 : 0),
                          wh = PAIR ? colh % 8 : colh;
                w[e] = ((EvenM || wr < m) && wh < H) ? weights[int64_t(wr) * weights_stride_m + wh] : 0;
            }
#pragma unroll
            for (int t = 0; t < NT; t++) {
                I4 dot = dot_init();
#pragma unroll
                for (int d = 0; d < 4; d++)
                    mmac_i8(kr[t][d], qr[d], dot);
                if constexpr (PrefetchScale) {
                    // The one-head-block cases have no next-head Q to overlap.
                    // Let their PK chain and unchanged head reduction cover the
                    // final scale load instead; only in-range KV columns are read.
                    __builtin_amdgcn_sched_barrier(0);
                    const int col = begin + t * 16 + lane % 16;
                    if (EvenN || col < n)
                        prefetched_scale[t] = scales[col];
                    __builtin_hcu_sched_barrier(0);
                }
                weighted(dot, w, acc[t]);
            }
        }
    } else if constexpr (H == 128 && TN == 32 && !EvenN) {
#pragma unroll
    for (int h = 0; h < H; h += 16) {
        I2 qr[4];
        const int qrrow = row + (PAIR ? (lane % 16) / 8 : 0),
                  head = h + (PAIR ? lane % 8 : lane % 16);
#pragma unroll
        for (int d = 0; d < 4; d++) {
            qr[d] = {0, 0};
            if ((EvenM || qrrow < m) && head < H)
                qr[d] = load_aligned4<I2>(q + (int64_t(qrrow) * H + head) * 128 + lane / 16 * 8 +
                                          d * 32);
        }
        F4 w;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int colh = h + lane / 16 + e * 4, wr = row + (PAIR ? colh / 8 : 0),
                      wh = PAIR ? colh % 8 : colh;
            w[e] = ((EvenM || wr < m) && wh < H) ? weights[int64_t(wr) * weights_stride_m + wh] : 0;
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            I4 dot = dot_init();
#pragma unroll
            for (int d = 0; d < 4; d++)
                mmac_i8(kr[t][d], qr[d], dot);
            weighted(dot, w, acc[t]);
        }
    }
    } else {
        constexpr int HeadBatches = (H + 15) / 16;
        I2 queries[HeadBatches][4];
        F4 head_weights[HeadBatches];
        auto load_head = [&](int h, I2 (&qr)[4], F4 &w) {
        const int qrrow = row + (PAIR ? (lane % 16) / 8 : 0),
                  head = h + (PAIR ? lane % 8 : lane % 16);
#pragma unroll
        for (int d = 0; d < 4; d++) {
            qr[d] = {0, 0};
            if ((EvenM || qrrow < m) && head < H)
                qr[d] = load_aligned4<I2>(q + (int64_t(qrrow) * H + head) * 128 + lane / 16 * 8 +
                                          d * 32);
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int colh = h + lane / 16 + e * 4, wr = row + (PAIR ? colh / 8 : 0),
                      wh = PAIR ? colh % 8 : colh;
            w[e] = ((EvenM || wr < m) && wh < H) ? weights[int64_t(wr) * weights_stride_m + wh] : 0;
        }
        };
        // Expose all independent Q/weight loads before their consumers.
        // This boundary constrains scheduling, but emits no hardware barrier.
#pragma unroll
        for (int b = 0; b < HeadBatches; b++)
            load_head(b * 16, queries[b], head_weights[b]);
        __builtin_hcu_sched_barrier(0);
#pragma unroll
        for (int b = 0; b < HeadBatches; b++) {
#pragma unroll
            for (int t = 0; t < NT; t++) {
                I4 dot = dot_init();
#pragma unroll
                for (int d = 0; d < 4; d++)
                    mmac_i8(kr[t][d], queries[b][d], dot);
                weighted(dot, head_weights[b], acc[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const int col = begin + t * 16 + lane % 16;
#pragma unroll
        for (int j = 0; j < Q; j++) {
            float v = PAIR ? (acc[t][j * 2] + acc[t][j * 2 + 1])
                           : ((acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3]));
            v += __shfl_xor(v, 16, 64);
            v += __shfl_xor(v, 32, 64);
            if (lane < 16 && (EvenN || col < n) && (EvenM || row + j < m)) {
                float value;
                value = v * (PrefetchScale ? prefetched_scale[t] : scales[col]);
                if (tile_full || (lo[j] <= col && col < hi[j]))
                    store1(out + int64_t(row + j) * out_stride_m + (col - lo[j]), value);
            }
        }
    }
}

template <int H, int TN, int WAVES, bool PAIR, bool EvenM, bool EvenN>
__global__ __launch_bounds__(WAVES * 64) void small_kv(MqaInt8Params p) {
    run_compact<H, TN, WAVES, PAIR, EvenM, EvenN>(
        reinterpret_cast<const int8_t *>(p.q), reinterpret_cast<const int8_t *>(p.kv),
        p.weights, p.ks, p.ke, p.kv_scale, p.out, int(p.m), int(p.n), p.weights_stride_m,
        p.out_stride_m, p.output_width);
}
#endif
} // namespace deepgemm::mqa::int8
