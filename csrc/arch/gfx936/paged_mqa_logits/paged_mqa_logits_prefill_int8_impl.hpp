// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "../../gfx936_gfx938_shared/paged_mqa_logits/paged_mqa_logits_int8_impl.hpp"

namespace deepgemm {

// 32 query tokens x 256 KV positions, native INT8 MMAC. Each wave
// retains 64 KV rows in registers and reuses them across all query heads.
// Two Q buffers overlap the next Q vector load with current-head MMAC
// while retaining one CTA barrier per head.
template <int NumHeads, int BlockKV>
__global__ __launch_bounds__(256) void paged_mqa_logits_prefill_int8_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ packed_cache,
    const float* __restrict__ weights, const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ block_table, float* __restrict__ logits,
    int tokens, int max_context_len, int num_physical_pages) {
    constexpr int Rows = 32, Cols = 256, Dim = 128, Threads = 256;
    constexpr int Pages = Cols / BlockKV;
    static_assert(BlockKV == 4 || BlockKV == 8 || BlockKV == 16 || BlockKV == 32 || BlockKV == 64);
    const int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    const int row_start = blockIdx.y * Rows, col_start = blockIdx.x * Cols;
    __shared__ int lengths[Rows], page_ids[Pages], tile_max_len;
    __shared__ float scales[Cols];
    __shared__ __align__(16) union {
        struct {
            int8_t q[2][Rows][136];
            // Two groups of eight heads keep each head's row weights adjacent.
            float weights[2][8][Rows];
        } input;
        // Wave-private output tiles reuse the input allocation.
        float output[4][16][36];
    } shared;

    int length = 0;
    if (wave == 0 && lane < Rows && row_start + lane < tokens) {
        length = context_lens[row_start + lane];
    }
    // Complete predicated context loads before publication and reduction.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    if (wave == 0) {
        if (lane < Rows) lengths[lane] = length;
        #pragma unroll
        for (int delta = 32; delta > 0; delta /= 2) {
            length = ::max(length, __shfl_down(length, delta));
        }
        if (lane == 0) tile_max_len = length;
    }
    __syncthreads();

    // This branch is CTA-uniform and derived from all 32 endpoints, so it
    // remains valid for graph replay with nonmonotonic or changed lengths.
    if (col_start >= tile_max_len) {
        for (int index = tid; index < Rows * Cols; index += Threads) {
            const int row = row_start + index / Cols, col = col_start + index % Cols;
            if (row < tokens && col < max_context_len) {
                logits[static_cast<int64_t>(row) * max_context_len + col] = -INFINITY;
            }
        }
        return;
    }

    int page_id = -1;
    if (wave == 0 && lane < Pages && col_start + lane * BlockKV < max_context_len &&
        col_start + lane * BlockKV < tile_max_len) {
        // One wave reads consecutive logical pages of row zero. Other waves
        // obtain these IDs exclusively through CTA shared memory.
        page_id = block_table[col_start / BlockKV + lane];
    }
    // Complete predicated page-table loads before publishing page IDs.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    if (wave == 0 && lane < Pages) page_ids[lane] = page_id;
    __syncthreads();

    int8x8_t key[4][4];
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int local_col = wave * 64 + n * 16 + lane % 16;
        const int physical = page_ids[local_col / BlockKV];
        const bool valid = col_start + local_col < max_context_len &&
                           col_start + local_col < tile_max_len &&
                           physical >= 0 && physical < num_physical_pages;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            key[n][k] = {};
            if (valid) {
                const int64_t offset = static_cast<int64_t>(physical) * BlockKV * (Dim + 4) +
                                       (local_col % BlockKV) * Dim + (lane / 16) * 8 + k * 32;
                key[n][k] = *reinterpret_cast<const int8x8_t*>(packed_cache + offset);
            }
        }
    }
    // Complete all predicated K loads before reusing their registers across heads.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");

    const auto load_weight_group = [&](int first_head, int buffer) {
        const int row = tid / 8, head = tid % 8;
        float value = 0.0f;
        if (row_start + row < tokens) {
            value = weights[static_cast<int64_t>(row_start + row) * NumHeads + first_head + head];
        }
        // Retire predicated weight loads before writing the shared group.
        asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
        shared.input.weights[buffer][head][row] = value;
    };
    load_weight_group(0, 0);
    if (wave == 0) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int col = lane + n * 64, physical = page_ids[col / BlockKV];
            float scale = 0.0f;
            if (col_start + col < max_context_len && col_start + col < tile_max_len &&
                physical >= 0 && physical < num_physical_pages) {
                const auto* page_scales = reinterpret_cast<const float*>(
                    packed_cache + static_cast<int64_t>(physical) * BlockKV * (Dim + 4) + BlockKV * Dim);
                scale = page_scales[col % BlockKV];
            }
            scales[col] = scale;
        }
    }
    // Publish initial weights and four scale elements per wave-0 lane.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    __syncthreads();

    fp32x4 result[2][4] = {};
    const int load_row = tid / 8, load_col = (tid % 8) * 16;
    const auto load_query = [&](int head) {
        // Preserve the 16-byte transfer as four words to avoid byte repacking.
        v4i value = {};
        if (row_start + load_row < tokens) {
            value = *reinterpret_cast<const v4i*>(
                q + (static_cast<int64_t>(row_start + load_row) * NumHeads + head) * Dim + load_col);
        }
        return value;
    };
    v4i initial_query = load_query(0);
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    // The padded row guarantees 8-byte alignment; memcpy permits split stores.
    __builtin_memcpy(&shared.input.q[0][load_row][load_col], &initial_query, sizeof(initial_query));
    __syncthreads();
    #pragma unroll 1
    for (int head = 0; head < NumHeads; ++head) {
        if (head % 8 == 0 && head + 8 < NumHeads) {
            // The current Q barrier retires all readers of the group that
            // occupied this slot two groups ago. The next Q barrier will
            // publish this write, seven heads before its first consumer.
            load_weight_group(head + 8, 1 - (head / 8) % 2);
        }

        v4i next_query = {};
        if (head + 1 < NumHeads) next_query = load_query(head + 1);
        __builtin_amdgcn_sched_barrier(0);

        int32x4_t dot[2][4] = {};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                const auto operand = *reinterpret_cast<const int8x8_t*>(
                    &shared.input.q[head % 2][lane % 16 + m * 16][lane / 16 * 8 + k * 32]);
                builtin_i8_mmac(operand, key[0][k], dot[m][0]);
                builtin_i8_mmac(operand, key[1][k], dot[m][1]);
                builtin_i8_mmac(operand, key[2][k], dot[m][2]);
                builtin_i8_mmac(operand, key[3][k], dot[m][3]);
            }
        }
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float weight = shared.input.weights[(head / 8) % 2][head % 8][lane % 16 + m * 16];
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                // Integer dots are exact in FP32 for D128. Clamp before conversion,
                // then expose adjacent FP32 pairs to packed FMA.
                fp32x4 activated;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    activated[j] = static_cast<float>(::max(dot[m][n][j], 0));
                }
                const fp32x2 pair = {weight, weight};
                result[m][n].xy = activated.xy * pair + result[m][n].xy;
                result[m][n].zw = activated.zw * pair + result[m][n].zw;
            }
        }
        if (head + 1 < NumHeads) {
            // Retire the next Q load before publication. The alternate slot
            // was last consumed before the preceding CTA barrier.
            asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
            __builtin_memcpy(&shared.input.q[(head + 1) % 2][load_row][load_col],
                             &next_query, sizeof(next_query));
            __syncthreads();
        }
    }

    // All waves finish reading Q/weights before the union becomes output.
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int producer_row = lane % 16;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = wave * 64 + half * 32 + n * 16 + lane / 16 + j * 4;
                    shared.output[wave][producer_row][n * 16 + lane / 16 + j * 4] =
                        col_start + col < lengths[m * 16 + producer_row]
                            ? result[m][half * 2 + n][j] * scales[col] : -INFINITY;
                }
            }
            // Each wave owns a private transpose tile. Retire LDS producers and
            // consumers before that wave reuses it for the next output fragment.
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = m * 16 + r * 8 + lane / 8;
                const int local_col = (lane % 8) * 4;
                const int col = wave * 64 + half * 32 + local_col;
                if (row_start + row < tokens && col_start + col < max_context_len) {
                    const fp32x4 value = *reinterpret_cast<const fp32x4*>(
                        &shared.output[wave][r * 8 + lane / 8][local_col]);
                    auto* output = logits + static_cast<int64_t>(row_start + row) * max_context_len + col_start + col;
                    if (max_context_len % 4 == 0 && col_start + col + 3 < max_context_len) {
                        *reinterpret_cast<fp32x4*>(output) = value;
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            if (col_start + col + j < max_context_len) output[j] = value[j];
                        }
                    }
                }
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        }
    }
}

}  // namespace deepgemm
