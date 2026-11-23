// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include "../../utils/compute.hpp"
#include "../../utils/load.hpp"
#include "../../utils/m64_epilogue.hpp"
#include "../../utils/window.hpp"
#include "mqa_logits/common/params.hpp"
#include <type_traits>
#include "../../utils/compact_store.hpp"
namespace deepgemm::mqa::fp8::compact {

// One implementation; the host selects compile-time M/N alignment.
template <bool is_even_m, bool is_even_n, int AlignN, bool UnitScale>
__device__ __forceinline__ void mqa_fp8_body(MqaFp8Params p) {
    constexpr int64_t HeadDim = 128;
    __shared__ __align__(256) uint8_t shared[16384];
    const uint32_t shared_base = uint32_t(reinterpret_cast<uintptr_t>(shared));
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int lane = threadIdx.x & 63;
    const int64_t m0 = int64_t(blockIdx.y) * 64;
    const int64_t n0 = int64_t(blockIdx.x) * 256;

    const int heads = int(p.h);                   // Host validates H=1..128.
    const int64_t q_row_stride = heads * HeadDim; // Host validates contiguous Q/KV.
    int valid_m = 64, valid_n = 256;
    if constexpr (!is_even_m)
        valid_m = tail_rows<64>(p.m - m0);
    if constexpr (!is_even_n)
        valid_n = tail_rows<256>(p.n - n0);
    bool tile_full = false;
    RowWindow cached_window;
    {
        const TileWindow tile = compact_window(p, m0, n0, valid_m, valid_n, lane, cached_window);
        if (!tile.any) return;
        tile_full = tile.full;
    }
    I4 kv[4][2];
    F4 scales[4];
    const float *scale_base = nullptr;
    if constexpr (!UnitScale)
        scale_base = p.kv_scale + n0;
    const I4 scale_descriptor = ScalarBufferLoad::make_descriptor(scale_base);
    // Each wave owns 4 KiB. Two N32 x D128 phases load all of its N64 keys.
    // Each MLS reads a contiguous N16 x D128 region with the compact mfmt=0 layout.
#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            int64_t kv_row = n0 + wave * 64 + phase * 32 + n * 16;
            int valid_rows = 16;
            if constexpr (!is_even_n) {
                valid_rows = tail_rows<16>(valid_n - (wave * 64 + phase * 32 + n * 16));
                if (!valid_rows)
                    kv_row = n0; // Keep pointer formation within the input.
            }
            if (valid_rows > 0)
                MLS128x16B8::matrix_load(
                    MLS128x16B8::make_descriptor<!is_even_n>(p.kv + kv_row * 128, 128,
                                                               valid_rows),
                    shared_base + wave * 4096 + n * 2048);
        }
        if (phase == 0) {
            // Each lane holds four F4 scales, reused by its four M16 fragments.
            // The 16 lanes with equal lane/16 share the same N coordinates.
#pragma unroll
            for (int n16 = 0; n16 < 4; ++n16) {
                const int local_col = wave * 64 + n16 * 16 + 4 * (lane / 16);
                if constexpr (UnitScale)
                    scales[n16] = F4{1, 1, 1, 1};
                else
                    scales[n16] =
                        load_scale_buffer<AlignN, is_even_n>(scale_descriptor, local_col, valid_n);
            }
        }
        wait_vm();
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int d = 0; d < 2; ++d)
                kv[phase * 2 + n][d] = MLS128x16B8::ds_read_format<>(
                    shared_base + wave * 4096 + n * 2048 + d * 1024);
        }
        finish_kv_half(kv[phase * 2][0], kv[phase * 2][1], kv[phase * 2 + 1][0],
                       kv[phase * 2 + 1][1]);
    }
    // Q subsequently uses data produced by other waves; release all KV spans.
    barrier();

    // Four producers load Q[64,D128] for one head. Q rows have stride H*128.
    const int q32 = wave / 2;
    const int qk64 = wave % 2;
    int64_t q_row = m0 + q32 * 32;
    int q_valid_rows = 32;
    if constexpr (!is_even_m) {
        q_valid_rows = tail_rows<32>(valid_m - q32 * 32);
        if (!q_valid_rows) {
            q_row = m0;
            q_valid_rows = 1;
        } // Safe dummy row; its outputs are masked.
    }
    const uint8_t *q_head0 = p.q + q_row * q_row_stride + qk64 * 64;
    const int q_lds = q32 * 4096 + qk64 * 2048;
    // Rebase once per CTA in 64 bits; the host checks the 64-row byte span.
    // WeightBufferLoad preserves an OOB row offset when adding each head offset.
    const I4 weight_descriptor =
        WeightBufferLoad::make_descriptor(p.weights + m0 * p.weights_stride_m);
    uint32_t weight_offset = uint32_t(lane) * uint32_t(p.weights_stride_m) * 4;
    if constexpr (!is_even_m)
        weight_offset = lane < valid_m ? weight_offset : UINT32_MAX;
    float current_weight, pending_weight = 0;
    MLS64x32B8::matrix_load(
        MLS64x32B8::make_descriptor<!is_even_m>(q_head0, int(q_row_stride), q_valid_rows),
        shared_base + q_lds);
    current_weight = WeightBufferLoad::load(weight_descriptor, weight_offset, 0);
    wait_vm_barrier();

    F4 sums[4][4] = {};
    auto compute_head = [&](auto prefetch_tag, int head, int stage) {
        constexpr bool prefetch = decltype(prefetch_tag)::value;
        // The preceding WG barrier published this head's Q. Read its first
        // M16 and shuffle the ready W before next-head GMEM address work.
        // Matrix DS, shuffle and builtin MMAC retain compiler-managed dependencies.
        // Keep Q double buffered and preserve the paired-dot scheduling boundaries.
        I4 queries[4][2];
        float weights[2];
        const uint32_t q_stage = shared_base + stage * 8192;
        queries[0][0] = MLS64x32B8::ds_read_format<0>(q_stage);
        queries[0][1] = MLS64x32B8::ds_read_format<2048>(q_stage);
        weights[0] = shuffle_weight(current_weight, lane % 16);
        __builtin_hcu_sched_barrier(0);

        // Start the next head's weight load now. Its Q MLS is issued below,
        // after the remaining current-head Q fragments have been read.
        // The head-end wait completes both loads before weight rotation.
        if constexpr (prefetch) {
            pending_weight =
                WeightBufferLoad::load(weight_descriptor, weight_offset, uint32_t(head + 1) * 4);
        }

        // Immediate DS offsets let all four M16 reads share the stage's M0.
        auto compute_m16 = [&](auto m_tag) {
            constexpr int m16 = decltype(m_tag)::value;
            constexpr int current = m16;
            // Readiness waits are inserted by the compiler for the actual uses.
            const F2 weight_pair = {weights[current & 1], weights[current & 1]};
            auto compute_dot = [&](int n16) {
                F4 dot = {};
                dot = mmac<0>(queries[current][0], kv[n16][0], dot);
                dot = mmac<1>(queries[current][0], kv[n16][0], dot);
                dot = mmac<0>(queries[current][1], kv[n16][1], dot);
                dot = mmac<1>(queries[current][1], kv[n16][1], dot);
                // Finish this dot before scheduling its independent paired dot.
                // The second chain then covers the first dot's MMAC-to-ReLU gap.
                __builtin_amdgcn_sched_barrier(0);
                return dot;
            };
#pragma unroll
            for (int n16 = 0; n16 < 4; n16 += 2) {
                F4 dot0 = compute_dot(n16);
                if (m16 < 3 && n16 == 0) {
                    constexpr int next_m = m16 + 1;
                    if constexpr (m16 == 0) {
                        queries[1][0] = MLS64x32B8::ds_read_format<1024>(q_stage);
                        queries[1][1] = MLS64x32B8::ds_read_format<3072>(q_stage);
                        queries[2][0] = MLS64x32B8::ds_read_format<4096>(q_stage);
                        queries[2][1] = MLS64x32B8::ds_read_format<6144>(q_stage);
                        queries[3][0] = MLS64x32B8::ds_read_format<5120>(q_stage);
                        queries[3][1] = MLS64x32B8::ds_read_format<7168>(q_stage);
                    }
                    weights[(current ^ 1) & 1] = shuffle_weight(current_weight, next_m * 16 + lane % 16);
                    __builtin_hcu_sched_barrier(0);
                    if constexpr (prefetch && m16 == 0) {
                        MLS64x32B8::matrix_load(
                            MLS64x32B8::make_descriptor<!is_even_m>(q_head0 + (head + 1) * HeadDim,
                                                                     int(q_row_stride), q_valid_rows),
                            shared_base + (stage ^ 1) * 8192 + q_lds);
                    }
                }
                F4 dot1 = compute_dot(n16 + 1);
                sums[m16][n16] = accumulate_relu_ready(sums[m16][n16], dot0, weight_pair);
                sums[m16][n16 + 1] = accumulate_relu_ready(sums[m16][n16 + 1], dot1, weight_pair);
            }
            __builtin_hcu_sched_barrier(0);
        };
        compute_m16(std::integral_constant<int, 0>{});
        compute_m16(std::integral_constant<int, 1>{});
        compute_m16(std::integral_constant<int, 2>{});
        compute_m16(std::integral_constant<int, 3>{});
        // Complete both next-head loads before publishing Q and rotating W.
        if constexpr (prefetch)
            finish_weight(pending_weight);
        // The final head can defer W consumers past the barrier. Explicitly
        // finish compiler-owned LDS requests before epilogue scratch reuse.
        if constexpr (!prefetch)
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        // Tail still releases Q read by all waves before output reuses LDS.
        barrier();
        if constexpr (prefetch)
            current_weight = pending_weight;
    };
    int head = 0;
    for (; head + 2 < heads; head += 2) {
        compute_head(std::true_type{}, head, 0);
        compute_head(std::true_type{}, head + 1, 1);
    }
    if (head + 1 < heads)
        compute_head(std::true_type{}, head, 0);
    compute_head(std::false_type{}, heads - 1, (heads - 1) & 1);

    finish_scales(scales);
    // Reuse 16 KiB only after the final Q/W handoff. Each wave owns 4 KiB.
    // F4-aligned XOR swizzle redistributes the MMAC ownership into consecutive
    // lane stores. Four M16 stages cover the full 64 KiB CTA output.
    using LdsF4 = F4 __attribute__((address_space(3)));
    auto *output_lds =
        reinterpret_cast<LdsF4 *>(uintptr_t(uint32_t(reinterpret_cast<uintptr_t>(shared))));
#pragma unroll
    for (int m16 = 0; m16 < 4; ++m16) {
        const int producer_row = lane % 16;
#pragma unroll
        for (int n16 = 0; n16 < 4; ++n16) {
            const int producer_col = n16 * 16 + 4 * (lane / 16);
            const int swizzled_col = producer_col ^ (producer_row << 2);
            F4 result = pk_mul(sums[m16][n16], scales[n16]);
            output_lds[wave * 256 + producer_row * 16 + swizzled_col / 4] = result;
        }
        // This wave owns its 4 KiB region. Drain writes before cross-lane reads.
        __builtin_hcu_sched_barrier(0);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        __builtin_hcu_sched_barrier(0);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int consumer_row = 4 * r + lane / 16;
            const int consumer_col = 4 * (lane % 16);
            const int swizzled_col = consumer_col ^ (consumer_row << 2);
            const F4 result = output_lds[wave * 256 + consumer_row * 16 + swizzled_col / 4];
            const int64_t row = m0 + m16 * 16 + consumer_row;
            const int64_t col = n0 + wave * 64 + consumer_col;
            bool valid = true;
            if constexpr (!is_even_m)
                valid = m16 * 16 + consumer_row < valid_m;
            if constexpr (!is_even_n)
                valid = valid && wave * 64 + consumer_col < valid_n;
            // Uniform row-group guard: do not broadcast for four rows
            // that are all outside M; participating waves stay converged.
            if (is_even_m || m16 * 16 + 4 * r < valid_m) {
                const RowWindow w = compact_shuffle(cached_window, m16 * 16 + consumer_row, tile_full);
                if (valid)
                    compact_store4(p, row, col, result, w, tile_full);
            }
        }
        // Complete all reads before the next M16 overwrites this scratch.
        __builtin_hcu_sched_barrier(0);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        __builtin_hcu_sched_barrier(0);
    }
}

template <bool EvenM, bool EvenN, int AlignN, bool UnitScale>
__global__ __launch_bounds__(256) void mqa_fp8_64x256_kernel(MqaFp8Params p) {
    mqa_fp8_body<EvenM, EvenN, AlignN, UnitScale>(p);
}

} // namespace deepgemm::mqa::fp8::compact
#endif
