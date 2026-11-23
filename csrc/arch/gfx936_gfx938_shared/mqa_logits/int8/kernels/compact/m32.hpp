// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../utils/ops.hpp"
#include "mqa_logits/common/epilogue.hpp"
#include "mqa_logits/common/params.hpp"
#include "mqa_logits/common/compact_store.hpp"

namespace deepgemm::mqa::int8 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
// Four waves share Q fragments while retaining K and head accumulators in
// registers. Wave w owns columns [w*BN/4, (w+1)*BN/4) of all 32 rows.
//
// D permutation (as in FP8): in d-step d, MMAC lane l = 16g + j holds bytes
// [32g + 8d, 32g + 8d + 8) of its row/column j, so a lane needs 32 contiguous bytes per
// row: two 16-byte K loads and two 16-byte Q reads per M16 tile. Q is stored unpadded in
// fragment order [head][M16 tile][half][slot]; lane l reads slot l ^ (2g + half). Eight
// consecutive readers stay in one aligned group of eight slots, and the eight 16-byte
// chunks of one row land in eight distinct 16-byte bank groups.
template <int Heads, int BN, int GroupHeads, bool EvenM, bool EvenN>
__global__ __launch_bounds__(256) void m32(MqaInt8Params p) {
    constexpr int BM = 32;
    static_assert(BN == 64 || BN == 128 || BN == 256);
    static_assert(Heads % GroupHeads == 0 && BM * GroupHeads <= 64);
    // F4 packets with kv_scale read once. BN=256 keeps the scalar scatter: its
    // per-packet kv_scale wait paces the stores, and unpaced stores measured up to 7%
    // slower at H8, N=131072.
    constexpr bool F4Epilogue = BN <= 128;
    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int bx = blockIdx.y;
    const int by = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int row_begin = bx * BM;
    const int col_begin = by * BN;

    constexpr int WN = BN / 4, NT = BN / 64, MT = BM / 16;
    // One head group: GroupHeads x BM unpadded 128-byte Q rows, then their weights.
    constexpr int Rows = BM * GroupHeads;
    constexpr int StageBytes = Rows * 128;
    constexpr int InputBytes = StageBytes + Rows * 4;
    constexpr int OutputBytes = 4 * 16 * WN * 4;
    // Final Q/weight reads complete before the same storage becomes output scratch.
    __shared__ __align__(16) int8_t shared[InputBytes > OutputBytes ? InputBytes : OutputBytes];
    float *weights_lds = reinterpret_cast<float *>(shared + StageBytes);
    const int valid_m = EvenM ? BM : tail_rows<BM>(int64_t(p.m) - row_begin);
    const int valid_n = EvenN ? BN : tail_rows<BN>(int64_t(p.n) - col_begin);
    common::RowWindow cached_window;
    TileWindow tile{true, true};
    tile = common::compact_window(p, row_begin, col_begin, valid_m, valid_n, lane, cached_window);
    const bool tile_full = tile.full;
    if (!tile.any)
        return;
    const I4 kv_resource = make_kv_resource(p.kv + int64_t(col_begin) * 128);
    // F4 epilogue: B column j of N16 tile t holds key t*16 + 4*(j%4) + j/4, so a lane's
    // accumulator holds four consecutive keys.
    const int key = F4Epilogue ? lane % 4 * 4 + lane % 16 / 4 : lane % 16;
    I4 kr[NT][2];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int col = col_begin + wave * WN + t * 16 + key;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            kr[t][half] = I4{0, 0, 0, 0};
            if (EvenN || col < p.n)
                kr[t][half] = __builtin_bit_cast(
                    I4, __builtin_amdgcn_buffer_load_dwordx4(
                            kv_resource, 0, (col - col_begin) * 128 + lane / 16 * 32 + half * 16,
                            false, true));
        }
    }
    // F4 epilogue: every packet a lane stores has the same columns, so kv_scale is read once.
    constexpr int Slots = WN / 4; // F4 packets per scratch row
    const int out_col = col_begin + wave * WN + 4 * (lane % Slots);
    F4 scale = {0.f, 0.f, 0.f, 0.f};
    if constexpr (F4Epilogue) {
        if (EvenN || out_col + 3 < p.n)
            scale = load_aligned4<F4>(p.kv_scale + out_col);
        else {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                if (out_col + e < p.n)
                    scale[e] = p.kv_scale[out_col + e];
        }
    }
    F4 accum[MT][NT] = {};
    // Thread i = tid + 256c copies 16-byte chunk i%8 of row i/8 = (head i/8/BM, query
    // i/8%BM): eight lanes read one complete 128-byte row. Rows past M repeat the last
    // valid query; their outputs are never stored.
    constexpr int Copies = (Rows * 8 + 255) / 256;
    const int8_t *q_block =
        reinterpret_cast<const int8_t *>(p.q) + int64_t(row_begin) * Heads * 128;
    auto q_slot = [](int l, int half) { return l ^ (2 * (l / 16) + half); };
    int q_src[Copies], q_dst[Copies];
#pragma unroll
    for (int c = 0; c < Copies; ++c) {
        const int i = tid + c * 256, chunk = i % 8, h = i / 8 / BM, r = i / 8 % BM;
        const int src_row = EvenM ? r : min(r, valid_m - 1);
        q_src[c] = (src_row * Heads + h) * 128 + chunk * 16;
        q_dst[c] = h * (BM * 128) +
                   ((r / 16 * 2 + chunk % 2) * 64 + q_slot(chunk / 2 * 16 + r % 16, chunk % 2)) *
                       16;
    }
    // Each copy covers 256 chunks; a 32-row head contributes exactly 256.
    auto load_q = [&](int group, I4 (&q)[Copies]) {
#pragma unroll
        for (int c = 0; c < Copies; ++c)
            q[c] = __builtin_bit_cast(I4, __builtin_amdgcn_buffer_load_dwordx4(
                                              make_kv_resource(q_block + group * 128), 0,
                                              q_src[c], false, false));
    };
    auto store_q = [&](const I4 (&q)[Copies]) {
#pragma unroll
        for (int c = 0; c < Copies; ++c)
            *reinterpret_cast<I4 *>(shared + q_dst[c]) = q[c];
    };
    // Reuse the bias C value for the first MMAC of every head.
    // The compiler materializes the VGPR C operand required by gfx936 MMAC.
    I4 dot_bias = dot_init();
    __builtin_amdgcn_sched_barrier(0);
    // One d-step of the (M16, N16) tile t; d-step 0 starts from the bias register.
    auto mma = [&](int d, const I4 &qd, int t, I4 &dot) {
        const I2 qr = d % 2 ? I2{qd[2], qd[3]} : I2{qd[0], qd[1]};
        const I4 &kd = kr[t][d / 2];
        const I2 k = d % 2 ? I2{kd[2], kd[3]} : I2{kd[0], kd[1]};
        if (d == 0) {
            // Preserve bias C; allow the compiler to allocate D.
            dot = __builtin_hcu_mmac_i32_16x16x32_i8(qr, k, dot_bias);
            // Keep machine instructions from moving across the first MMAC.
            __builtin_amdgcn_sched_barrier(0);
        } else {
            mmac_i8(qr, k, dot);
        }
    };
    // Weight of (head tid / BM, query tid % BM) of a group.
    const bool w_active = tid < Rows && (EvenM || row_begin + tid % BM < p.m);
    const float *w_src =
        p.weights + int64_t(row_begin + tid % BM) * p.weights_stride_m + tid / BM;
    auto load_weight = [&](int group) { return w_active ? w_src[group] : 0.f; };
    I4 next_q[Copies] = {};
    load_q(0, next_q);
    float next_weight = load_weight(0);
#pragma unroll 1
    for (int group = 0; group < Heads; group += GroupHeads) {
        store_q(next_q);
        if (tid < Rows)
            weights_lds[tid] = next_weight;
        __syncthreads();
        if (group + GroupHeads < Heads) {
            load_q(group + GroupHeads, next_q);
            next_weight = load_weight(group + GroupHeads);
        }
#pragma unroll
        for (int h = 0; h < GroupHeads; ++h) {
            const int8_t *head_q = shared + h * (BM * 128);
            I4 q[MT][2];
#pragma unroll
            for (int r = 0; r < MT; ++r) {
#pragma unroll
                for (int half = 0; half < 2; ++half)
                    q[r][half] = *reinterpret_cast<const I4 *>(
                        head_q + (r * 2 + half) * 1024 + q_slot(lane, half) * 16);
            }
            I4 dot[MT][NT];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
#pragma unroll
                for (int r = 0; r < MT; ++r) {
#pragma unroll
                    for (int t = 0; t < NT; ++t)
                        mma(d, q[r][d / 2], t, dot[r][t]);
                }
            }
#pragma unroll
            for (int r = 0; r < MT; ++r) {
                const float weight = weights_lds[h * BM + r * 16 + lane % 16];
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    relu_weighted_sum(dot[r][t], weight, accum[r][t]);
            }
        }
        // Publish all final Q reads before the next producer reuses LDS.
        __syncthreads();
    }

    // Masks and writes one scaled F4 packet: row out_row, keys [packet_col, packet_col + 4).
    using StoreWindow = common::RowWindow;
    auto emit = [&](int out_row, int packet_col, F4 result, const StoreWindow &bounds) {
        common::compact_store4(p, out_row, packet_col, result, bounds, tile_full);
    };
    if constexpr (F4Epilogue) {
        // An F4-aligned XOR swizzle turns the MMAC row ownership into consecutive-lane row
        // segments. For eight consecutive lanes both phases touch eight distinct 16-byte
        // bank groups.
        auto swizzle = [](int row) { return Slots == 4 ? (row >> 1) & 3 : row & (Slots - 1); };
        using LdsF4 = F4 __attribute__((address_space(3)));
        const uint32_t lds_base = uint32_t(reinterpret_cast<uintptr_t>(shared));
        auto *output = reinterpret_cast<LdsF4 *>(uintptr_t(lds_base)) + wave * 16 * Slots;
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const int producer_row = lane % 16;
#pragma unroll
            for (int t = 0; t < NT; ++t)
                output[producer_row * Slots + ((t * 4 + lane / 16) ^ swizzle(producer_row))] =
                    accum[r][t];
            __builtin_hcu_sched_barrier(0);
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            __builtin_hcu_sched_barrier(0);
#pragma unroll
            for (int packet = 0; packet < NT; ++packet) {
                const int consumer_row = packet * (64 / Slots) + lane / Slots;
                const F4 value =
                    output[consumer_row * Slots + ((lane % Slots) ^ swizzle(consumer_row))];
                const int out_row = row_begin + r * 16 + consumer_row;
                StoreWindow bounds{};
                if (EvenM || row_begin + r * 16 + packet * (64 / Slots) < p.m)
                    bounds = common::compact_shuffle(cached_window, r * 16 + consumer_row, tile_full);
                if ((EvenM || out_row < p.m) && (EvenN || out_col < p.n)) {
                    const F4 result = pk_mul4(value, scale);

                    emit(out_row, out_col, result, bounds);
                }
            }
            // Drain reads before the next M16 overwrites this wave's scratch.
            __builtin_hcu_sched_barrier(0);
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            __builtin_hcu_sched_barrier(0);
        }
    } else {
        auto &output = *reinterpret_cast<float(*)[4][16][WN]>(shared);
        // Non-LIT accumulators own columns a+4*e, a=lane/16.
        // Scatter them into an XOR-swizzled row, then gather consecutive F4 packets.
        // Each wave owns a disjoint scratch region; no inter-wave exchange is needed.
        constexpr int RowShift = BN == 64 ? 2 : BN == 128 ? 1 : 0;
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const int producer_row = lane % 16;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int producer_col = t * 16 + lane / 16 + e * 4;
                    const int swizzled_col = producer_col ^ ((producer_row >> RowShift) << 2);
                    output[wave][producer_row][swizzled_col] = accum[r][t][e];
                }
            }
            __builtin_hcu_sched_barrier(0);
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            __builtin_hcu_sched_barrier(0);
#pragma unroll
            for (int packet = 0; packet < NT; ++packet) {
                const int consumer_row = packet * (64 / Slots) + lane / Slots;
                const int consumer_col = 4 * (lane % Slots);
                const int swizzled_col = consumer_col ^ ((consumer_row >> RowShift) << 2);
                F4 result =
                    *reinterpret_cast<const F4 *>(&output[wave][consumer_row][swizzled_col]);
                const int out_row = row_begin + r * 16 + consumer_row;
                StoreWindow bounds{};
                if (EvenM || row_begin + r * 16 + packet * (64 / Slots) < p.m)
                    bounds = common::compact_shuffle(cached_window, r * 16 + consumer_row, tile_full);
                if ((EvenM || out_row < p.m) && (EvenN || out_col < p.n)) {

                    if (EvenN || out_col + 3 < p.n)
                        result = pk_mul4(result, load_aligned4<F4>(p.kv_scale + out_col));
                    else {
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            if (out_col + e < p.n)
                                result[e] *= p.kv_scale[out_col + e];
                    }
                    emit(out_row, out_col, result, bounds);
                }
            }
            // Drain reads before the next M16 overwrites this wave's scratch.
            __builtin_hcu_sched_barrier(0);
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            __builtin_hcu_sched_barrier(0);
        }
    }
}
#endif
} // namespace deepgemm::mqa::int8
