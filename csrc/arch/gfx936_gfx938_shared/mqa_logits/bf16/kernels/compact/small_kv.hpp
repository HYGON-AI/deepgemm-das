// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// BF16 compact small-KV kernel small_kv.
// Device code; compact launcher only.

#pragma once
#include "../../utils/ops.hpp"
#include "mqa_logits/common/params.hpp"
#include "mqa_logits/common/compact_store.hpp"

namespace deepgemm::mqa::bf16 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Heads form MMAC columns. H8 can place two adjacent queries in 16 columns.
template <int H, int TN, int WAVES, bool PAIR, bool EvenM, bool EvenN>
__device__ __forceinline__ void run_compact(const uint16_t *__restrict__ q, const uint16_t *__restrict__ k,
                                    const float *__restrict__ weights,
                                    const int32_t *__restrict__ ks, const int32_t *__restrict__ ke,
                                    float *__restrict__ out,
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
        // Each row predicate is wave-uniform; readfirstlane stays converged.
        if (EvenM || row + j < m) {
            const Window w = load_window(ks, ke, n, row + j);
            const int64_t limit = w.start + output_width;
            const int end = int(w.end < limit ? w.end : limit);
            lo[j] = __builtin_amdgcn_readfirstlane(int(w.start));
            hi[j] = __builtin_amdgcn_readfirstlane(end);
            tile.any |= lo[j] < hi[j] && lo[j] < begin + vn && hi[j] > begin;
            tile.full &= lo[j] <= begin && hi[j] >= begin + vn;
        }
    }
    const bool valid = tile.any, tile_full = tile.full;
    if (!valid)
        return;
    const I4 kv_resource = make_kv_resource(k + int64_t(begin) * 128);
    B4 kr[NT][8];
    F4 acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        acc[t] = {0, 0, 0, 0};
        const int col = begin + t * 16 + lane % 16;
#pragma unroll
        for (int d = 0; d < 8; d++) {
            kr[t][d] = {0, 0, 0, 0};
            if (EvenN || col < n)
                kr[t][d] = load_kv2(kv_resource, (col - begin) * 256 + lane / 16 * 8 + d * 32);
        }
    }
    if constexpr (TN == 64 && H == 64 && !EvenN) {
        // Stage at most two adjacent head blocks, then consume them in order.
#pragma unroll
        for (int h = 0; h < H; h += 32) {
            F4 dots[2][4]{};
            F4 head_weights[2];
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const int block_head = h + g * 16;
                B4 qr[8];
                const int qrrow = row + (PAIR ? (lane % 16) / 8 : 0),
                          head = block_head + (PAIR ? lane % 8 : lane % 16);
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    qr[d] = {0, 0, 0, 0};
                    if ((EvenM || qrrow < m) && head < H)
                        qr[d] = load_aligned4<B4>(
                            q + (int64_t(qrrow) * H + head) * 128 + lane / 16 * 4 + d * 16);
                }
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int colh = block_head + lane / 16 + e * 4,
                              wr = row + (PAIR ? colh / 8 : 0),
                              wh = PAIR ? colh % 8 : colh;
                    head_weights[g][e] =
                        ((EvenM || wr < m) && wh < H)
                            ? weights[int64_t(wr) * weights_stride_m + wh]
                            : 0;
                }
#pragma unroll
                for (int d = 0; d < 8; ++d) {
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        mmac_bf16(kr[t][d], qr[d], dots[g][t]);
                }
            }
            // Preserve each output component's h, h+16 FMA sequence.
#pragma unroll
            for (int g = 0; g < 2; ++g) {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    weighted<H, PAIR>(dots[g][t], head_weights[g], acc[t], lane);
            }
        }
    } else {
#pragma unroll
        for (int h = 0; h < H; h += 16) {
            B4 qr[8];
            const int qrrow = row + (PAIR ? (lane % 16) / 8 : 0),
                      head = h + (PAIR ? lane % 8 : lane % 16);
#pragma unroll
            for (int d = 0; d < 8; d++) {
                qr[d] = {0, 0, 0, 0};
                if ((EvenM || qrrow < m) && head < H)
                    qr[d] = load_aligned4<B4>(q + (int64_t(qrrow) * H + head) * 128 + lane / 16 * 4 +
                                              d * 16);
            }
            F4 w;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int colh = h + lane / 16 + e * 4, wr = row + (PAIR ? colh / 8 : 0),
                          wh = PAIR ? colh % 8 : colh;
                w[e] = ((EvenM || wr < m) && wh < H) ? weights[int64_t(wr) * weights_stride_m + wh] : 0;
            }
#if defined(__gfx938__)
            // Keep this head block's query/weight loads ahead of its dot products.
            // This compiler-only boundary avoids the 768-VGPR scheduling regression.
            if constexpr ((H == 64 || H == 128) && TN == 64 && EvenN)
                __builtin_amdgcn_sched_barrier(0);
#endif
            if constexpr (!PAIR && H <= 4 && NT >= 2 && NT % 2 == 0) {
                // For H <= 4 only component zero has real heads. Pack adjacent
                // N subtiles, not a real head with a padded head, into the PK FMA.
                // Keep each output's eight MMACs and its weighted FMA in order.
                using F2 = float __attribute__((ext_vector_type(2)));
#pragma unroll
                for (int t = 0; t < NT; t += 2) {
                    F4 dot0 = {0, 0, 0, 0};
                    F4 dot1 = {0, 0, 0, 0};
#pragma unroll
                    for (int d = 0; d < 8; d++) {
                        mmac_bf16(kr[t][d], qr[d], dot0);
                        mmac_bf16(kr[t + 1][d], qr[d], dot1);
                    }
                    const F2 value = {relu(dot0[0]), relu(dot1[0])};
                    const F2 weight = {w[0], w[0]};
                    const F2 initial = {acc[t][0], acc[t + 1][0]};
                    const F2 updated = __builtin_hcu_pk_fma_f32(weight, value, initial);
                    // Do not commit padded-head NaNs; all other acc components
                    // remain the original positive zero for the same reduction.
                    if (lane / 16 < H) {
                        acc[t][0] = updated[0];
                        acc[t + 1][0] = updated[1];
                    }
                }
            } else if constexpr (TN == 64 && H >= 32) {
                // Pack the same head component from two independent N subtiles.
                // Each output keeps its original d and ascending-head FMA order.
                using F2 = float __attribute__((ext_vector_type(2)));
#pragma unroll
                for (int t = 0; t < NT; t += 2) {
                    F4 dot0 = {0, 0, 0, 0};
                    F4 dot1 = {0, 0, 0, 0};
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        mmac_bf16(kr[t][d], qr[d], dot0);
                        mmac_bf16(kr[t + 1][d], qr[d], dot1);
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const F2 value = {relu(dot0[i]), relu(dot1[i])};
                        const F2 weight = {w[i], w[i]};
                        const F2 initial = {acc[t][i], acc[t + 1][i]};
                        const F2 updated = __builtin_hcu_pk_fma_f32(weight, value, initial);
                        // All head components are real for H >= 32; an invalid
                        // N tail component remains isolated in its packed lane.
                        acc[t][i] = updated[0];
                        acc[t + 1][i] = updated[1];
                    }
                }
            } else {
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    F4 dot = {0, 0, 0, 0};
#pragma unroll
                    for (int d = 0; d < 8; d++)
                        mmac_bf16(kr[t][d], qr[d], dot);
                    weighted<H, PAIR>(dot, w, acc[t], lane);
                }
            }
#if defined(__gfx936__)
            if constexpr (H == 128 && TN == 64 && EvenN) {
                // Consume the first head block before scheduling the remaining dots.
                // One compiler scheduling boundary limits live fragments in 768-VGPR mode.
                if (h == 0)
                    __builtin_amdgcn_sched_barrier(0);
            }
#endif
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
            if (lane < 16 && (EvenN || col < n) && (EvenM || row + j < m) &&
                (tile_full || (lo[j] <= col && col < hi[j])))
                store1(out + int64_t(row + j) * out_stride_m + (col - lo[j]), v);
        }
    }
    return;
}

template <int H, int TN, int WAVES, bool PAIR, bool EvenM, bool EvenN>
__global__ __launch_bounds__(WAVES * 64) void small_kv(MqaBf16Params p) {
    run_compact<H, TN, WAVES, PAIR, EvenM, EvenN>(
        reinterpret_cast<const uint16_t *>(p.q),
        reinterpret_cast<const uint16_t *>(p.kv), p.weights, p.ks, p.ke, p.out,
        int(p.m), int(p.n), p.weights_stride_m,
        p.out_stride_m, p.output_width);
    return;
}
#endif
} // namespace deepgemm::mqa::bf16
