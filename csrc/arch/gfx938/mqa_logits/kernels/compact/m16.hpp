// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include "../../utils/small_kv.hpp"
#include "../../utils/compact_store.hpp"
namespace deepgemm::mqa::fp8::compact {

// BN=128 keeps one N16 x D128 fragment per wave in LDS (2048 B). The selected
// H8 full-tile path uses two D64 MLS requests; other paths use one D128 load.
// The next N16 reuses this span after DS retires; the CTA fits in 8192 B.

template <int BN, bool UnitScale, bool FullTile, bool PreloadH8, bool UseK64>
__global__ __launch_bounds__(256) void mqa_m16(MqaFp8Params p) {
    static_assert(BN == 64 || BN == 128 || BN == 256);
    static_assert(!PreloadH8 || (BN == 64 && FullTile));
    static_assert(!UseK64 || (BN == 128 && FullTile));
    constexpr int NF = BN / 64, WN = BN / 4;
    constexpr bool SplitK = BN == 128;
    // H8/BN64 keeps all eight Q tiles separate from four KV slots. The
    // producer waves issue Q before KV. KV moves to VGPRs before the output
    // swizzle reuses its 16 KiB LDS span; Q remains resident throughout.
    constexpr uint32_t KvOffset = PreloadH8 ? 16384u : 0u;
    constexpr uint32_t OutOffset = KvOffset;
    constexpr int LdsBytes = PreloadH8 ? 32768 : (SplitK ? 8192 : 16384);
    constexpr int RowSlots = SplitK ? 8 : 16;
    __shared__ __align__(256) uint8_t shared[LdsBytes];
    const uint32_t lds = uint32_t(reinterpret_cast<uintptr_t>(shared));
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64), lane = threadIdx.x % 64;
    const int64_t m0 = int64_t(blockIdx.y) * 16, n0 = int64_t(blockIdx.x) * BN;
    const int vm = FullTile ? 16 : tail_rows<16>(p.m - m0);
    const int vn = FullTile ? BN : tail_rows<BN>(p.n - n0);
    RowWindow cached_window;
    const TileWindow tile = [&] {
        return compact_window(p, m0, n0, vm, vn, lane, cached_window);
    }();
    if (!tile.any)
        return;
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
        load_kv_128<FullTile, UseK64>(p, lds, n0, vn, wave, kv);
    else
        small_kv<BN, FullTile>(p, lds, n0, vn, wave, kv);
    const int count = PreloadH8 ? 8 : int(p.h);
    const auto prefetch_q = [&](int h, int stage) {
        const uint8_t *q = p.q + (m0 * p.h + h) * 128;
        if (wave == 0)
            MLS128x16B8::matrix_load(
                MLS128x16B8::make_descriptor<!FullTile>(q, int(p.h) * 128, vm),
                lds + stage * 2048);
    };
    const I4 wd = WeightBufferLoad::make_descriptor(p.weights + m0 * p.weights_stride_m);
    const uint32_t wo = FullTile ? uint32_t((lane % 16) * p.weights_stride_m) * 4
                                 : lane < vm ? uint32_t(lane * p.weights_stride_m) * 4 : UINT32_MAX;
    if constexpr (!PreloadH8)
        prefetch_q(0, 0);
    float weight = WeightBufferLoad::load(wd, wo, 0);
    if constexpr (PreloadH8)
        finish_weight(weight);
    else
        wait_vm_barrier();
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
    for (int local = 0; local < count; ++local) {
        const int stage = PreloadH8 ? local : (local & 1), head = local;
        I4 q0 = MLS128x16B8::ds_read_format<>(lds + stage * 2048);
        I4 q1 = MLS128x16B8::ds_read_format<1024>(lds + stage * 2048);
        const float w = shuffle_weight(weight, lane % 16);
        __builtin_hcu_sched_barrier(0);
        float pending = 0;
        if (local + 1 < count) {
            if constexpr (!PreloadH8)
                prefetch_q(head + 1, stage ^ 1);
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
        if (local + 1 < count)
            finish_weight(pending);
        if constexpr (!PreloadH8) {
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            barrier();
        }
        weight = pending;
    }
    using LdsF4 = F4 __attribute__((address_space(3)));
    auto *out_lds = reinterpret_cast<LdsF4 *>(uintptr_t(lds + OutOffset));
    const int row = lane % 16;
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
            // Preserve the existing M16 scale-pair code generation order.
            const F2 scaled_high = __builtin_hcu_pk_mul_f32(result.zw, scales.zw);
            result.xy = __builtin_hcu_pk_mul_f32(result.xy, scales.xy);
            result.zw = scaled_high;
        }
        // BN=128's eight column groups do not collide after the XOR is masked to 8 slots.
        const int slot = ((c ^ (row << 2)) / 4) & (RowSlots - 1);
        out_lds[wave * (16 * RowSlots) + row * RowSlots + slot] = result;
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
        if (FullTile || i * 256 / WN < vm) {
            const RowWindow w = compact_shuffle(cached_window, r, tile.full);
            if (FullTile || (r < vm && nc < vn))
                compact_store4(p, m0 + r, n0 + nc, result, w, tile.full);
        }
    }
}

} // namespace deepgemm::mqa::fp8::compact
#endif
