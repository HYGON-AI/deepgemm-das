// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include "../../utils/small_kv.hpp"
#include <type_traits>
namespace deepgemm::mqa {
namespace fp8 {

// Independent M16/BN128 grouped-weight kernel, promoted from the measured demo.
// Keep the original M16 kernel for unsupported shapes and unmeasured modes.
// Each wave stages an N16 x D128 K fragment in 2048 B of LDS.

template <int BN, bool Clean, bool UnitScale = false, bool FullTile = false,
          bool PreloadH8 = false, bool UseK64 = false, int WGroup = 0>
__global__ __launch_bounds__(256) void mqa_m16_grouped(MqaFp8Params p) {
    static_assert(BN == 128 && Clean && !UnitScale && FullTile && !PreloadH8);
    static_assert(WGroup == 8 || WGroup == 16);
    static_assert(BN == 64 || BN == 128 || BN == 256);
    static_assert(!PreloadH8 || (BN == 64 && FullTile));
    static_assert(!UseK64 || (BN == 128 && FullTile));
    static_assert(WGroup == 0 || WGroup == 8 || WGroup == 16);
    static_assert(WGroup == 0 || (FullTile && !PreloadH8));
    constexpr int NF = BN / 64, WN = BN / 4;
    constexpr bool SplitK = BN == 128;
    // KV staging and the output swizzle share the same LDS offset.
    // Grouped weights use the separate shared array below.
    constexpr uint32_t KvOffset = PreloadH8 ? 16384u : 0u;
    constexpr uint32_t OutOffset = KvOffset;
    constexpr int LdsBytes = PreloadH8 ? 32768 : (SplitK ? 8192 : 16384);
    constexpr int RowSlots = SplitK ? 8 : 16;
    __shared__ __align__(256) uint8_t shared[LdsBytes];
    // Give grouped weights a separate LDS span with explicit ownership.
    __shared__ __align__(256) float grouped_shared[WGroup ? 2 * WGroup * 16 : 1];
    const uint32_t lds = uint32_t(reinterpret_cast<uintptr_t>(shared));
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64), lane = threadIdx.x % 64;
    const int64_t m0 = int64_t(blockIdx.y) * 16, n0 = int64_t(blockIdx.x) * BN;
    const int vm = FullTile ? 16 : tail_rows<16>(p.m - m0);
    const int vn = FullTile ? BN : tail_rows<BN>(p.n - n0);
    const TileWindow tile = small_window<Clean>(p, m0, n0, vm, vn, lane);
    if (!tile.any) {
        if constexpr (Clean)
            if constexpr (FullTile) {
                // Host guarantees complete rows and 16-byte output alignment.
                for (int i = threadIdx.x * 4; i < 16 * BN; i += 256 * 4)
                    store4(p.out + (m0 + i / BN) * p.out_stride_m + n0 + i % BN,
                           F4{-__builtin_inff(), -__builtin_inff(), -__builtin_inff(), -__builtin_inff()});
            } else {
                empty_output(p, m0, n0, vm, vn);
            }
        return;
    }
    // Load grouped weights after K has moved to VGPRs. The two head-group
    // planes stay in separate LDS storage while Q and K retain their scratch layout.
    const I4 wd = WeightBufferLoad::make_descriptor(p.weights + m0 * p.weights_stride_m);
    uint32_t grouped_woff = 0;
    float grouped_weight = 0;
    if constexpr (WGroup != 0) {
        const int wr = threadIdx.x / WGroup, wh = threadIdx.x % WGroup;
        grouped_woff = threadIdx.x < 16 * WGroup
                            ? uint32_t(wr * p.weights_stride_m + wh) * 4 : UINT32_MAX;
    }
    I4 kv[NF][2];
    if constexpr (PreloadH8) {
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int head = wave + group * 4;
            const uint8_t *q = p.q + (m0 * 8 + head) * 128;
            MLS128x16B8::matrix_load(MLS128x16B8::make_descriptor(q, 8 * 128),
                                     lds + uint32_t(head) * 2048u);
        }
        // small_kv waits for this wave's Q and KV MLS requests, drains KV to
        // VGPRs, then synchronizes waves before any Q tile is read through DS.
        small_kv<64, true>(p, lds + KvOffset, n0, vn, wave, kv);
    } else if constexpr (SplitK)
        load_kv_128_grouped<FullTile, UseK64>(p, lds, n0, vn, wave, kv);
    else
        small_kv<BN, FullTile>(p, lds, n0, vn, wave, kv);
    if constexpr (WGroup != 0)
        grouped_weight = WeightBufferLoad::load(wd, grouped_woff, 0);
    // The H8 grouped selector is exact. Its static trip count lets the compiler
    // remove the otherwise redundant next-group tests in that path.
    const int count = WGroup == 8 ? 8 : PreloadH8 ? 8 : int(p.h);
    const auto prefetch_q = [&](int h, int stage) {
        const uint8_t *q = p.q + (m0 * p.h + h) * 128;
        if (wave == 0)
            MLS128x16B8::matrix_load(
                MLS128x16B8::make_descriptor<!FullTile>(q, int(p.h) * 128, vm),
                lds + stage * 2048);
    };
    const uint32_t wo = FullTile ? uint32_t((lane % 16) * p.weights_stride_m) * 4
                                 : lane < vm ? uint32_t(lane * p.weights_stride_m) * 4 : UINT32_MAX;
    if constexpr (!PreloadH8)
        prefetch_q(0, 0);
    using LdsFloat = float __attribute__((address_space(3)));
    auto *weight_lds = reinterpret_cast<LdsFloat *>(uintptr_t(grouped_shared));
    float weight = 0;
    if constexpr (WGroup != 0) {
        finish_weight(grouped_weight);
        if (threadIdx.x < 16 * WGroup)
            weight_lds[(threadIdx.x % WGroup) * 16 + threadIdx.x / WGroup] = grouped_weight;
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
    } else {
        weight = WeightBufferLoad::load(wd, wo, 0);
        finish_weight(weight);
    }
    if constexpr (!PreloadH8)
        barrier();
    F4 sums[NF] = {};
    F4 zero = {};
    if constexpr (NF > 1) {
        // The public entry validates D=128. Preserve exact +0 bits while
        // materializing a runtime zero quartet outside the head loop.
        const float exact_zero = __builtin_bit_cast(float, uint32_t(p.d) - 128u);
        zero = F4{exact_zero, exact_zero, exact_zero, exact_zero};
    }
    // Keep the zero-C scheduling boundary; MMAC builtin dependencies manage readiness.
    __builtin_hcu_sched_barrier(0);
    float next_group_weight = 0;
#pragma clang loop unroll_count(WGroup == 8 ? 8 : 1)
    for (int local = 0; local < count; ++local) {
        const int stage = PreloadH8 ? local : (local & 1), head = local;
        I4 q0 = MLS128x16B8::ds_read_format<>(lds + stage * 2048);
        I4 q1 = MLS128x16B8::ds_read_format<1024>(lds + stage * 2048);
        float w;
        if constexpr (WGroup == 0)
            w = shuffle_weight(weight, lane % 16);
        else
            w = weight_lds[((local / WGroup) & 1) * WGroup * 16 +
                           (local % WGroup) * 16 + lane % 16];
        __builtin_hcu_sched_barrier(0);
        float pending = 0;
        if (local + 1 < count) {
            if constexpr (!PreloadH8)
                prefetch_q(head + 1, stage ^ 1);
            if constexpr (WGroup == 0)
                pending = WeightBufferLoad::load(wd, wo, uint32_t(head + 1) * 4);
        }
#pragma unroll
        for (int nf = 0; nf < NF; nf += 2) {
            F4 d0 = dot128(q0, q1, kv[nf][0], kv[nf][1], zero);
            if constexpr (NF > 1) {
                F4 d1 = dot128(q0, q1, kv[nf + 1][0], kv[nf + 1][1], zero);
                sums[nf] = accumulate_relu_ready(sums[nf], d0, F2{w, w});
                sums[nf + 1] = accumulate_relu_ready(sums[nf + 1], d1, F2{w, w});
            } else {
                // The accumulation helper supplies the single-dot scheduling boundary.
                sums[nf] = accumulate_relu_ready(sums[nf], d0, F2{w, w});
            }
        }
        if constexpr (WGroup != 0) {
            if ((local + 1) % WGroup == 0 && local + 1 < count) {
                next_group_weight =
                    WeightBufferLoad::load(wd, grouped_woff, uint32_t(local + 1) * 4);
                finish_weight(next_group_weight);
                if (threadIdx.x < 16 * WGroup)
                    weight_lds[(((local + 1) / WGroup) & 1) * WGroup * 16 +
                               (threadIdx.x % WGroup) * 16 + threadIdx.x / WGroup] =
                        next_group_weight;
            }
        } else if (local + 1 < count) {
            finish_weight(pending);
        }
        if constexpr (!PreloadH8) {
            // The original per-head weight wait also drained the next Q MLS.
            // Grouped weights remove that wait, so Q needs an explicit VM drain
            // before another wave reads the new LDS stage.
            if constexpr (WGroup != 0)
                wait_vm();
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            barrier();
        }
        if constexpr (WGroup == 0)
            weight = pending;
    }
    using LdsF4 = F4 __attribute__((address_space(3)));
    auto *out_lds = reinterpret_cast<LdsF4 *>(uintptr_t(lds + OutOffset));
    const int row = lane % 16;
    Window bounds{};
    if constexpr (Clean)
        if (row < vm && !tile.full)
            bounds = load_window(p.ks, p.ke, p.n, m0 + row);
    if constexpr (WGroup == 8 && Clean) {
        // Compile separate full-window and masked epilogues so tile.full is
        // tested once for all N fragments, rather than once per fragment.
        const auto store_fragments = [&](auto apply_mask) {
            constexpr bool ApplyMask = decltype(apply_mask)::value;
        #pragma unroll
            for (int nf = 0; nf < NF; ++nf) {
                const int c = nf * 16 + 4 * (lane / 16);
                F4 result = sums[nf];
                if constexpr (!UnitScale) {
                    F4 scales;
                    if constexpr (FullTile) {
                        scales = load_scale_buffer<4, true>(ScalarBufferLoad::make_descriptor(p.kv_scale + n0),
                                                           wave * WN + c, BN);
                    } else {
        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int nc = wave * WN + c + j;
                            scales[j] = ScalarBufferLoad::load(
                                ScalarBufferLoad::make_descriptor(p.kv_scale + n0),
                                nc < vn ? uint32_t(nc) * 4 : ScalarBufferLoad::kOOBOffset);
                        }
                    }
                    result = pk_mul(result, scales);
                }
                if constexpr (ApplyMask) {
                    const int nc = wave * WN + c;
                    const bool keep0 = (bounds.start <= n0 + nc) & (n0 + nc < bounds.end);
                    const bool keep1 = (bounds.start <= n0 + nc + 1) & (n0 + nc + 1 < bounds.end);
                    const bool keep2 = (bounds.start <= n0 + nc + 2) & (n0 + nc + 2 < bounds.end);
                    const bool keep3 = (bounds.start <= n0 + nc + 3) & (n0 + nc + 3 < bounds.end);
                    result = F4{keep0 ? result[0] : -__builtin_inff(),
                                keep1 ? result[1] : -__builtin_inff(),
                                keep2 ? result[2] : -__builtin_inff(),
                                keep3 ? result[3] : -__builtin_inff()};
                }
                // BN=128's eight column groups do not collide after the XOR is masked to 8 slots.
                const int slot = ((c ^ (row << 2)) / 4) & (RowSlots - 1);
                out_lds[wave * (16 * RowSlots) + row * RowSlots + slot] = result;
            }
        };
        if (tile.full)
            store_fragments(std::false_type{});
        else
            store_fragments(std::true_type{});
    } else {
    #pragma unroll
        for (int nf = 0; nf < NF; ++nf) {
            const int c = nf * 16 + 4 * (lane / 16);
            F4 result = sums[nf];
            if constexpr (!UnitScale) {
                F4 scales;
                if constexpr (FullTile) {
                    scales = load_scale_buffer<4, true>(ScalarBufferLoad::make_descriptor(p.kv_scale + n0),
                                                       wave * WN + c, BN);
                } else {
    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int nc = wave * WN + c + j;
                        scales[j] = ScalarBufferLoad::load(
                            ScalarBufferLoad::make_descriptor(p.kv_scale + n0),
                            nc < vn ? uint32_t(nc) * 4 : ScalarBufferLoad::kOOBOffset);
                    }
                }
                result = pk_mul(result, scales);
            }
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nc = wave * WN + c + j;
                if constexpr (Clean)
                    if (!tile.full && !(bounds.start <= n0 + nc && n0 + nc < bounds.end))
                        result[j] = -__builtin_inff();
            }
            // BN=128's eight column groups do not collide after the XOR is masked to 8 slots.
            const int slot = ((c ^ (row << 2)) / 4) & (RowSlots - 1);
            out_lds[wave * (16 * RowSlots) + row * RowSlots + slot] = result;
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
    __builtin_hcu_sched_barrier(0);
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        const int index = lane * 4 + i * 256;
        const int r = index / WN, c = index % WN;
        const int slot = ((c ^ (r << 2)) / 4) & (RowSlots - 1);
        F4 result = out_lds[wave * (16 * RowSlots) + r * RowSlots + slot];
        const int nc = wave * WN + c;
        if (FullTile || (r < vm && nc < vn)) {
            float *dst = p.out + (m0 + r) * p.out_stride_m + n0 + nc;
            if (FullTile || (nc + 3 < vn && !(uintptr_t(dst) & 15)))
                store4(dst, result);
            else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (nc + j < vn)
                        store1(dst + j, result[j]);
            }
        }
    }
}

} // namespace fp8
} // namespace deepgemm::mqa
#endif
