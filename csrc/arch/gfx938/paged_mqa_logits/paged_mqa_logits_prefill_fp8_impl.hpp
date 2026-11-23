// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "paged_mqa_logits_fp8_impl.hpp"
#include <type_traits>

namespace deepgemm {

// gfx938 FP8 prefill for 4/8-token pages: 32 queries x 128 KV positions
// using native E4M3FN MMAC. Each wave retains 32 KV rows in registers
// and reuses them across all query heads.
// Two Q buffers prevent overwriting data still consumed by another wave.
template <int NumHeads, int BlockKV>
__device__ __forceinline__ void paged_mqa_logits_prefill_fp8_vector_device(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ packed_cache,
    const float* __restrict__ weights, const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ block_table, float* __restrict__ logits,
    int tokens, int max_context_len, int num_physical_pages) {
    constexpr int Rows = 32, Cols = 128, Dim = 128, Threads = 256;
    constexpr int Pages = Cols / BlockKV;
    static_assert(BlockKV == 4 || BlockKV == 8 || BlockKV == 16 || BlockKV == 32 || BlockKV == 64);
    const int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    const int row_start = blockIdx.y * Rows, col_start = blockIdx.x * Cols;
    __shared__ int lengths[Rows], page_ids[Pages], tile_max_len;
    __shared__ float scales[Cols];
    __shared__ __align__(16) union {
        struct {
            uint8_t q[2][Rows][136];
            // Keep two groups of eight heads, with adjacent row weights per head.
            float weights[2][8][Rows];
        } input;
        // Reuse the transpose buffer for two 64-column output halves.
        float output[Rows][68];
    } shared;

    int length = 0;
    if (wave == 0 && lane < Rows && row_start + lane < tokens) {
        length = context_lens[row_start + lane];
    }
    // Drain predicated context loads before publishing lengths and reducing them.
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
    // Publish page IDs only after predicated page-table loads complete.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    if (wave == 0 && lane < Pages) page_ids[lane] = page_id;
    __syncthreads();

    fp8x8 key[2][4];
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        const int local_col = wave * 32 + n * 16 + lane % 16;
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
                key[n][k] = *reinterpret_cast<const fp8x8*>(packed_cache + offset);
            }
        }
    }
    // Retain completed K loads in registers for all query heads.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");

    const auto load_weight_group = [&](int first_head, int buffer) {
        const int row = tid / 8, head = tid % 8;
        float value = 0.0f;
        if (row_start + row < tokens) {
            value = weights[static_cast<int64_t>(row_start + row) * NumHeads + first_head + head];
        }
        // Drain predicated weight loads before the LDS write.
        asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
        shared.input.weights[buffer][head][row] = value;
    };
    load_weight_group(0, 0);
    if (wave == 0) {
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
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
    // Complete scale loads before the CTA publishes weights and scales.
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    __syncthreads();

    fp32x4 result[2][2] = {};
    const int load_row = tid / 8, load_col = (tid % 8) * 16;
    #pragma unroll 1
    for (int head = 0; head < NumHeads; ++head) {
        // Keep the 16-byte transfer in 32-bit lanes: AICC otherwise expands
        // the predicated byte vector into byte extraction/repacking per head.
        v4i query = {};
        if (row_start + load_row < tokens) {
            query = *reinterpret_cast<const v4i*>(
                q + (static_cast<int64_t>(row_start + load_row) * NumHeads + head) * Dim + load_col);
        }
        // Complete Q loads before writing the shared buffer.
        asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
        // The 136-byte LDS row stride guarantees only 8-byte alignment.
        __builtin_memcpy(&shared.input.q[head % 2][load_row][load_col], &query, sizeof(query));
        __syncthreads();

        if (head % 8 == 0 && head + 8 < NumHeads) {
            // The current Q barrier retires all readers of the group that
            // occupied this slot two groups ago. The next Q barrier will
            // publish this write, seven heads before its first consumer.
            load_weight_group(head + 8, 1 - (head / 8) % 2);
        }

        fp32x4 dot[2][2] = {};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                const auto operand = *reinterpret_cast<const fp8x8*>(
                    &shared.input.q[head % 2][lane % 16 + m * 16][lane / 16 * 8 + k * 32]);
                builtin_fp8_mmac(operand, key[0][k], dot[m][0]);
                builtin_fp8_mmac(operand, key[1][k], dot[m][1]);
            }
        }
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float weight = shared.input.weights[(head / 8) % 2][head % 8][lane % 16 + m * 16];
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    result[m][n][j] += fmaxf(static_cast<float>(dot[m][n][j]), 0.0f) * weight;
                }
            }
        }
    }

    // All waves finish reading Q/weights before the union becomes output.
    __syncthreads();
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        if (wave / 2 == half) {
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        shared.output[lane % 16 + m * 16][(wave % 2) * 32 + n * 16 + lane / 16 + j * 4] =
                            result[m][n][j];
                    }
                }
            }
        }
        __syncthreads();
        for (int index = tid * 4; index < Rows * 64; index += Threads * 4) {
            const int row = index / 64, local_col = index % 64;
            const int col = half * 64 + local_col;
            if (row_start + row < tokens && col_start + col < max_context_len) {
                fp32x4 value = *reinterpret_cast<const fp32x4*>(&shared.output[row][local_col]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    value[j] = col_start + col + j < lengths[row] ? value[j] * scales[col + j] : -INFINITY;
                }
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
        // Every wave finishes reading the first half before another wave
        // overwrites the shared transpose buffer with the second half.
        if (half == 0) __syncthreads();
    }
}

__device__ __forceinline__ void paged_mqa_prefill_fp8_wait_vm() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void paged_mqa_prefill_fp8_wait_lds() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// gfx938 med3(infinity, dot, zero) matches fmaxf(dot, zero), including
// signaling NaNs and signed zero. Operand order is significant: placing
// infinity first avoids the extra canonicalization of native max.
__device__ __forceinline__ float paged_mqa_prefill_fp8_relu(float dot) {
    float value;
    const float upper = INFINITY;
    asm volatile("v_med3_f32 %0, %2, %1, 0"
                 : "=&v"(value) : "v"(dot), "s"(upper));
    return value;
}

// Keep both FP32 pairs live so the compiler can emit packed head FMA.
__device__ __forceinline__ fp32x4 paged_mqa_prefill_fp8_accumulate(
    fp32x4 sum, fp32x4 dot, float weight) {
    fp32x4 activated;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        activated[j] = paged_mqa_prefill_fp8_relu(dot[j]);
    }
    const fp32x2 pair = {weight, weight};
    sum.xy = activated.xy * pair + sum.xy;
    sum.zw = activated.zw * pair + sum.zw;
    return sum;
}

template <int Offset>
__device__ __forceinline__ fp8x16 paged_mqa_prefill_fp8_matrix_read(const uint8_t* source) {
    using LdsInt = int __attribute__((address_space(3)));
    auto* address = reinterpret_cast<LdsInt*>(reinterpret_cast<uintptr_t>(source));
    return __builtin_bit_cast(fp8x16, __builtin_hcu_ds_read_matrix_trans_format_i8(address, Offset, 3, 1, 0));
}

__device__ __forceinline__ void paged_mqa_prefill_fp8_load_q32(uint8_t* destination, const v4u& descriptor) {
    const uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(destination)) | 0x80000000u;
    asm volatile("s_nop 0\n\tmatrix_load_64x32_b8 %0, %1, moffset:0 t lds"
                 : : "s"(descriptor), "s"(offset) : "memory");
}

// A fixed lane offset and scalar head offset let the next weight request
// overlap current-head MMAC without rebuilding a per-lane address.
__device__ __forceinline__ float paged_mqa_prefill_fp8_weight(
    const v4u& descriptor, uint32_t offset, uint32_t head_offset) {
    float value;
    asm volatile("buffer_load_dword %0, %1, %2, %3 offen"
                 : "=&v"(value) : "v"(offset), "s"(descriptor), "s"(head_offset) : "memory");
    return value;
}

__device__ __forceinline__ void paged_mqa_prefill_fp8_finish_weight(float& value) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)" : "+v"(value) : : "memory");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ fp32x4 paged_mqa_prefill_fp8_mmac(
    fp8x8 query, fp8x8 key, fp32x4 accumulator) {
    const intx2 a = __builtin_bit_cast(intx2, query), b = __builtin_bit_cast(intx2, key);
    fp32x4 result;
    asm volatile("v_mmac_f32_16x16x32_fp8_fp8 %0, %1, %2, %3 lit"
                 : "=&v"(result) : "v"(a), "v"(b), "v"(accumulator));
    return result;
}

// A 64-query x 256-KV tile uses D128 as the MMAC reduction dimension.
// Each lane accumulates weighted ReLU results across heads for its outputs.
// Paged K stays in registers; each N16 MLS stays within one physical page.
template <int NumHeads, int BlockKV, bool EvenRows, bool EvenCols>
__device__ __forceinline__ void paged_mqa_logits_prefill_fp8_mls_device(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ packed_cache,
    const float* __restrict__ weights, const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ block_table, float* __restrict__ logits,
    int tokens, int max_context_len, int num_physical_pages) {
    constexpr int Rows = 64, Cols = 256, Dim = 128;
    static_assert(BlockKV == 16 || BlockKV == 32 || BlockKV == 64);
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64), lane = threadIdx.x % 64;
    const int row_start = blockIdx.y * Rows, col_start = blockIdx.x * Cols;
    __shared__ __align__(1024) uint8_t shared[16384];
    const uint32_t shared_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(shared));
    const int valid_rows = EvenRows ? Rows : ::min(Rows, tokens - row_start);
    const int valid_cols = EvenCols ? Cols : ::min(Cols, max_context_len - col_start);
    int length = 0;
    if (lane < valid_rows) {
        length = context_lens[row_start + lane];
    }
    // Every wave examines the same rows. This exit remains CTA-uniform for
    // arbitrary context updates during graph replay, including empty tiles.
    const bool active_tile = __ballot(length > col_start) != 0;
    const bool full_tile = __ballot(lane < valid_rows && length < col_start + valid_cols) == 0;
    if (!active_tile) {
        for (int row = wave; row < valid_rows; row += 4) {
            for (int col = lane * 4; col < valid_cols; col += 256) {
                auto* output = logits + static_cast<int64_t>(row_start + row) * max_context_len + col_start + col;
                if (EvenCols || (max_context_len % 4 == 0 && col + 3 < valid_cols)) {
                    *reinterpret_cast<fp32x4*>(output) = fp32x4{-INFINITY, -INFINITY, -INFINITY, -INFINITY};
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (col + j < valid_cols) output[j] = -INFINITY;
                    }
                }
            }
        }
        return;
    }
    f8x16_t key[4][2] = {};
    fp32x4 scales[4] = {};
    // Page IDs remain in wave-uniform registers, keeping LDS within 16 KiB.
    // Two N16 transfers use disjoint wave-private LDS slots. Retire both
    // matrix reads before the next pair reuses those slots.
    #pragma unroll
    for (int n = 0; n < 4; n += 2) {
        int physical[2];
        bool valid[2];
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int first_col = wave * 64 + (n + part) * 16;
            int page = -1;
            if (first_col < valid_cols) {
                page = block_table[col_start / BlockKV + first_col / BlockKV];
            }
            physical[part] = __builtin_amdgcn_readfirstlane(page);
            valid[part] = first_col < valid_cols && physical[part] >= 0 && physical[part] < num_physical_pages;
            if (valid[part]) {
                const auto* address = packed_cache + static_cast<int64_t>(physical[part]) * BlockKV * (Dim + 4) +
                                      first_col % BlockKV * Dim;
                matrix_load_128x16_b8_t(shared + wave * 4096 + part * 2048,
                                      make_mls_128x16_b8_srsrc(address, Dim, 0));
            }
        }
        paged_mqa_prefill_fp8_wait_vm();
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            if (valid[part]) {
                auto* destination = shared + wave * 4096 + part * 2048;
                key[n + part][0].data = paged_mqa_prefill_fp8_matrix_read<0>(destination);
                key[n + part][1].data = paged_mqa_prefill_fp8_matrix_read<1024>(destination);
            }
        }
        paged_mqa_prefill_fp8_wait_lds();
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int local_col = wave * 64 + (n + part) * 16 + (lane / 16) * 4;
            if (valid[part] && local_col < valid_cols) {
                const auto* source = reinterpret_cast<const float*>(packed_cache +
                    static_cast<int64_t>(physical[part]) * BlockKV * (Dim + 4) + BlockKV * Dim) + local_col % BlockKV;
                if constexpr (EvenCols) {
                    scales[n + part] = *reinterpret_cast<const fp32x4*>(source);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (local_col + j < valid_cols) {
                            scales[n + part][j] = source[j];
                        }
                    }
                }
            }
        }
    }
    paged_mqa_prefill_fp8_wait_vm();
    __syncthreads();

    const int q32 = wave / 2, qk64 = wave % 2;
    const int q_rows = ::min(32, ::max(0, valid_rows - q32 * 32));
    const auto issue_query = [&](int head, int stage) {
        if (q_rows > 0) {
            const auto* source = q + (static_cast<int64_t>(row_start + q32 * 32) * NumHeads + head) * Dim +
                                 qk64 * 64;
            auto descriptor = make_mls_128x16_b8_srsrc(source, NumHeads * Dim, 32 - q_rows);
            descriptor[3] |= 1u << 17;
            paged_mqa_prefill_fp8_load_q32(shared + stage * 8192 + q32 * 4096 + qk64 * 2048, descriptor);
        }
    };
    const auto* weight_base = weights + static_cast<int64_t>(row_start) * NumHeads;
    v4u weight_descriptor = {};
    *reinterpret_cast<uint64_t*>(&weight_descriptor) = reinterpret_cast<uint64_t>(weight_base);
    weight_descriptor[2] = 0xffffffffu;
    weight_descriptor[3] = 0x00020000u;
    const uint32_t weight_offset = lane < valid_rows ? lane * NumHeads * sizeof(float) : 0xffffffffu;
    issue_query(0, 0);
    float current_weight = paged_mqa_prefill_fp8_weight(weight_descriptor, weight_offset, 0);
    paged_mqa_prefill_fp8_finish_weight(current_weight);
    __syncthreads();
    fp32x4 sums[4][4] = {};
    const auto compute_head = [&](auto prefetch_tag, int head, int stage_index) {
        constexpr bool Prefetch = decltype(prefetch_tag)::value;
        const auto* stage = shared + stage_index * 8192;
        f8x16_t query[2][2];
        float row_weights[2];
        query[0][0].data = paged_mqa_prefill_fp8_matrix_read<0>(stage);
        query[0][1].data = paged_mqa_prefill_fp8_matrix_read<2048>(stage);
        row_weights[0] = __shfl(current_weight, lane % 16);
        __builtin_amdgcn_sched_barrier(0);
        float pending_weight = 0;
        if constexpr (Prefetch) {
            issue_query(head + 1, stage_index ^ 1);
            pending_weight = paged_mqa_prefill_fp8_weight(
                weight_descriptor, weight_offset, (head + 1) * sizeof(float));
        }
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int current = m % 2;
            const auto compute_dot = [&](int n) {
                fp32x4 dot = {};
                if constexpr (!Prefetch) {
                    // Inline MMAC needs an explicit zero-C RAW delay at the
                    // first dot of the peeled final head.
                    if (m == 0 && n == 0) {
                        asm volatile("s_nop 2" : "+v"(dot));
                    }
                }
                dot = paged_mqa_prefill_fp8_mmac(query[current][0].front, key[n][0].front, dot);
                dot = paged_mqa_prefill_fp8_mmac(query[current][0].rear, key[n][0].rear, dot);
                dot = paged_mqa_prefill_fp8_mmac(query[current][1].front, key[n][1].front, dot);
                return paged_mqa_prefill_fp8_mmac(query[current][1].rear, key[n][1].rear, dot);
            };
            #pragma unroll
            for (int n = 0; n < 4; n += 2) {
                const fp32x4 dot0 = compute_dot(n);
                if (m < 3 && n == 0) {
                    const auto* next = stage + ((m + 1) / 2) * 4096 + ((m + 1) % 2) * 1024;
                    query[current ^ 1][0].data = paged_mqa_prefill_fp8_matrix_read<0>(next);
                    query[current ^ 1][1].data = paged_mqa_prefill_fp8_matrix_read<2048>(next);
                    row_weights[current ^ 1] = __shfl(current_weight, (m + 1) * 16 + lane % 16);
                    __builtin_amdgcn_sched_barrier(0);
                }
                const fp32x4 dot1 = compute_dot(n + 1);
                __builtin_amdgcn_sched_barrier(0);
                sums[m][n] = paged_mqa_prefill_fp8_accumulate(sums[m][n], dot0, row_weights[current]);
                sums[m][n + 1] = paged_mqa_prefill_fp8_accumulate(sums[m][n + 1], dot1, row_weights[current]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        if constexpr (Prefetch) {
            paged_mqa_prefill_fp8_finish_weight(pending_weight);
        } else {
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
            // Finish accumulation before the epilogue creates output packets.
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    asm volatile("" : "+v"(sums[m][n]));
                }
            }
        }
        __syncthreads();
        if constexpr (Prefetch) {
            current_weight = pending_weight;
        }
    };
    // Two heads keep both LDS stage addresses constant in the rolled loop.
    #pragma unroll 1
    for (int head = 0; head < NumHeads - 2; head += 2) {
        compute_head(std::true_type{}, head, 0);
        compute_head(std::true_type{}, head + 1, 1);
    }
    compute_head(std::true_type{}, NumHeads - 2, 0);
    compute_head(std::false_type{}, NumHeads - 1, 1);

    // Wave-private XOR-swizzled FP32 packets transpose MMAC ownership into
    // consecutive global stores without a CTA barrier for each output stage.
    using LdsF4 = fp32x4 __attribute__((address_space(3)));
    auto* output_lds = reinterpret_cast<LdsF4*>(static_cast<uintptr_t>(shared_base));
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int producer_row = lane % 16;
        int row_length = max_context_len;
        if (!full_tile) {
            row_length = __shfl(length, m * 16 + producer_row);
        }
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int producer_col = n * 16 + lane / 16 * 4;
            fp32x4 value = sums[m][n] * scales[n];
            if (!full_tile) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (col_start + wave * 64 + producer_col + j >= row_length) {
                        value[j] = -INFINITY;
                    }
                }
            }
            output_lds[wave * 256 + producer_row * 16 + (producer_col ^ (producer_row * 4)) / 4] = value;
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = r * 4 + lane / 16, col = (lane % 16) * 4;
            if (m * 16 + row < valid_rows && wave * 64 + col < valid_cols) {
                const fp32x4 value = output_lds[wave * 256 + row * 16 + (col ^ (row * 4)) / 4];
                auto* output = logits + static_cast<int64_t>(row_start + m * 16 + row) * max_context_len +
                               col_start + wave * 64 + col;
                if (EvenCols || (max_context_len % 4 == 0 && wave * 64 + col + 3 < valid_cols)) {
                    *reinterpret_cast<fp32x4*>(output) = value;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (wave * 64 + col + j < valid_cols) output[j] = value[j];
                    }
                }
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
    }
}

template <int NumHeads, int BlockKV, bool EvenRows = false, bool EvenCols = false>
__global__ __launch_bounds__(256) void paged_mqa_logits_prefill_fp8_kernel(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ packed_cache,
    const float* __restrict__ weights, const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ block_table, float* __restrict__ logits,
    int tokens, int max_context_len, int num_physical_pages) {
    if constexpr (BlockKV >= 16) {
        paged_mqa_logits_prefill_fp8_mls_device<NumHeads, BlockKV, EvenRows, EvenCols>(
            q, packed_cache, weights, context_lens, block_table, logits, tokens, max_context_len, num_physical_pages);
    } else {
        // Small pages cross physical boundaries inside a 16-row MLS tile.
        // Gather bytes using the native vector/MMAC implementation instead.
        paged_mqa_logits_prefill_fp8_vector_device<NumHeads, BlockKV>(
            q, packed_cache, weights, context_lens, block_table, logits, tokens, max_context_len, num_physical_pages);
    }
}

}  // namespace deepgemm
