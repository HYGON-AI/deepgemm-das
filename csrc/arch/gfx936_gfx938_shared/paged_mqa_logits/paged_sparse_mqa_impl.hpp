// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "paged_mqa_logits/paged_sparse_mqa.h"

namespace deepgemm {
using SparseMqaBf16x4 = short __attribute__((ext_vector_type(4)));
using SparseMqaFloat4 = float __attribute__((ext_vector_type(4)));
__device__ inline int64_t sparse_mqa_physical_block(const PagedSparseMqaMapping& a, int row, int slot) {
    const int length = a.lens[int64_t(row) * a.lens_stride];
    const int64_t count = min(int64_t(a.capacity), (int64_t(length) + a.block - 1) / a.block);
    if (slot >= count || length <= 0)
        return -1;
    const int logical = a.candidates[int64_t(row) * a.candidate_stride + slot];
    const int64_t token = int64_t(logical) * a.block;
    // Length determines the active slot count. As in upstream metadata,
    // graph replay may shrink it while retaining the allocated candidates.
    // The compute kernel applies token-level masking when requested.
    if (logical < 0 || token / a.page >= a.table_cols)
        __builtin_trap();
    const int page = a.table[int64_t(row) * a.table_stride + token / a.page];
    if (page < 0)
        __builtin_trap();
    return int64_t(page) * a.page + token % a.page;
}

__device__ inline SparseMqaFloat4 sparse_mqa_bf16_mmac(SparseMqaBf16x4 a, SparseMqaBf16x4 b, SparseMqaFloat4 c) {
#if defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false);
#elif defined(__gfx936__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#else
    return c; // Runtime entry rejects architectures other than gfx936/gfx938.
#endif
}

template <bool FP4> __device__ __forceinline__ SparseMqaBf16x4 sparse_mqa_decode_bf16(const uint8_t* p, int d) {
    unsigned word;
    if constexpr (FP4) {
        uint16_t v;
        __builtin_memcpy(&v, p + d / 2, sizeof(v));
        word = v;
    } else {
        __builtin_memcpy(&word, p + d, sizeof(word));
    }
    SparseMqaBf16x4 result;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (FP4) {
            const unsigned code = (word >> (j * 4)) & 15, mag = code & 7;
            const unsigned value = mag ? 0x3f00 + ((mag >= 2 ? mag : 0) << 6) : 0;
            result[j] = ((code & 8) << 12) | value;
        } else {
            const unsigned code = (word >> (j * 8)) & 255, mag = code & 7, exp = (code >> 3) & 15;
            const unsigned sub = __float_as_uint(float(mag) * (1.f / 512.f)) >> 16;
            unsigned value = exp ? ((exp + 120) << 7) | (mag << 4) : sub;
            value = ((code & 127) == 127) ? 0x7fc0 : value;
            result[j] = ((code & 128) << 8) | value;
        }
    }
    return result;
}

__device__ __forceinline__ bool sparse_mqa_token_address(const PagedSparseMqaAddressArgs& opt, int row, int t,
                                                         int64_t& page, int& offset) {
    const PagedSparseMqaArgs& a = opt.base;
    const PagedSparseMqaMapping& m = a.mapping;
    const int slot = t >> opt.block_shift, in_block = t & (m.block - 1);
    if (slot >= m.capacity)
        return false;
    const int64_t physical = a.metadata[int64_t(row) * m.capacity + slot];
    if (physical < 0)
        return false;
    const uint64_t token = physical + in_block;
    page = opt.page_shift >= 0 ? token >> opt.page_shift : token / unsigned(m.page);
    offset = token - uint64_t(page) * m.page;
    if (a.mask_tail) {
        const int logical = m.candidates[int64_t(row) * m.candidate_stride + slot];
        if ((int64_t(logical) << opt.block_shift) + in_block >= m.lens[int64_t(row) * m.lens_stride])
            return false;
    }
    if (page >= a.pages)
        __builtin_trap();
    return true;
}

using SparseMqaUint4 = unsigned __attribute__((ext_vector_type(4)));

__device__ __forceinline__ SparseMqaUint4 sparse_mqa_load16(const uint8_t* p) {
    // The public API also accepts unaligned bases/strides. Preserve that path
    // while allowing a single vector load for naturally aligned operands.
    if ((reinterpret_cast<uintptr_t>(p) & 15) == 0)
        return *reinterpret_cast<const SparseMqaUint4*>(p);
    SparseMqaUint4 value;
    __builtin_memcpy(&value, p, 16);
    return value;
}

template <bool FP4> __device__ __forceinline__ SparseMqaBf16x4 sparse_mqa_scaled_bf16(unsigned word, unsigned e) {
    SparseMqaBf16x4 result;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (FP4) {
            const unsigned code = (word >> (4 * j)) & 15, mag = code & 7;
            const int bits = 0x3f00 + ((mag >= 2 ? mag : 0) << 6) + (int(e) - 127) * 128;
            result[j] = ((code & 8) << 12) | (mag ? bits : 0);
        } else {
            const unsigned code = (word >> (8 * j)) & 255, mag = code & 7, exp = (code >> 3) & 15;
            unsigned bits = exp ? ((exp + 120) << 7) | (mag << 4) : (__float_as_uint(float(mag) * (1.f / 512.f)) >> 16);
            bits = (code & 127) == 127 ? 0x7fc0 : bits;
            result[j] = ((code & 128) << 8) | ((bits && bits < 0x7f80) ? int(bits) + (int(e) - 127) * 128 : bits);
        }
    }
    return result;
}

__device__ __forceinline__ bool sparse_mqa_group_address(const PagedSparseMqaAddressArgs& opt, int row, int t,
                                                         int64_t& page, int& offset) {
    const PagedSparseMqaArgs& a = opt.base;
    const PagedSparseMqaMapping& m = a.mapping;
    const int lane = threadIdx.x & 63, pos = lane & 15;
    const int slot = t >> opt.block_shift, inblock = t & (m.block - 1);
    int64_t physical = -1;
    if (lane < 16 && inblock == 0 && slot < m.capacity)
        physical = a.metadata[int64_t(row) * m.capacity + slot];
    const int source = pos & ~(m.block - 1);
    const unsigned lo = __shfl(unsigned(physical), source, 64);
    const unsigned hi = __shfl(unsigned(uint64_t(physical) >> 32), source, 64);
    physical = int64_t(uint64_t(hi) << 32 | lo);
    if (slot >= m.capacity || physical < 0)
        return false;
    const uint64_t token = physical + inblock;
    page = opt.page_shift >= 0 ? token >> opt.page_shift : token / unsigned(m.page);
    offset = token - uint64_t(page) * m.page;
    if (a.mask_tail) {
        const int logical = m.candidates[int64_t(row) * m.candidate_stride + slot];
        if ((int64_t(logical) << opt.block_shift) + inblock >= m.lens[int64_t(row) * m.lens_stride])
            return false;
    }
    if (page >= a.pages)
        __builtin_trap();
    return true;
}

// Private compact plan, in int64 words:
// [R*M physical offsets][4-word header: count, reuse, reserved, reserved]
// [R live token counts][R*ceil(tiles_per_row/Reuse) task descriptors].
// Descriptor: row[31:0], first tile[47:32], end tile[63:48].
// A fixed-size allocation and fixed launches allow counts to change in graphs.
__device__ inline int sparse_mqa_live_tokens(const PagedSparseMqaMapping& m, int row) {
    const int64_t length = max(m.lens[int64_t(row) * m.lens_stride], 0);
    return int(min(int64_t(m.capacity), (length + m.block - 1) / m.block)) * m.block;
}
template <int Reuse> __device__ inline int sparse_mqa_task_count(const PagedSparseMqaMapping& m, int row) {
    return (sparse_mqa_live_tokens(m, row) + kPagedSparseMqaTileTokens * Reuse - 1) /
           (kPagedSparseMqaTileTokens * Reuse);
}
template <int Reuse> __global__ void paged_sparse_mqa_metadata_kernel(PagedSparseMqaMapping m, int64_t* out) {
    const int row = blockIdx.x, tid = threadIdx.x;
    auto* header = out + int64_t(m.rows) * m.capacity;
    auto* lengths = header + 4;
    auto* tasks = lengths + m.rows;
    __shared__ uint64_t partial[4], base;
    if (tid == 0) {
        lengths[row] = sparse_mqa_live_tokens(m, row);
    }
    uint64_t sum = 0;
    for (int r = tid; r < row; r += 256)
        sum += sparse_mqa_task_count<Reuse>(m, r);
    for (int delta = 32; delta; delta /= 2)
        sum += __shfl_down(sum, delta, 64);
    if ((tid & 63) == 0)
        partial[tid / 64] = sum;
    for (int slot = tid; slot < m.capacity; slot += 256)
        out[int64_t(row) * m.capacity + slot] = sparse_mqa_physical_block(m, row, slot);
    __syncthreads();
    if (tid == 0) {
        base = partial[0] + partial[1] + partial[2] + partial[3];
        if (row == m.rows - 1) {
            header[0] = base + sparse_mqa_task_count<Reuse>(m, row);
            header[1] = Reuse;
            header[2] = 0;
            header[3] = 0;
        }
    }
    __syncthreads();
    const int tiles = (sparse_mqa_live_tokens(m, row) + kPagedSparseMqaTileTokens - 1) / kPagedSparseMqaTileTokens;
    for (int chunk = tid; chunk < (tiles + Reuse - 1) / Reuse; chunk += 256) {
        uint64_t position = base + chunk;
        const int first = chunk * Reuse, last = min(first + Reuse, tiles);
        tasks[position] = uint64_t(unsigned(row)) | (uint64_t(first) << 32) | (uint64_t(last) << 48);
    }
}

template <bool FP4, int HT> __global__ void paged_sparse_mqa_bf16_kernel(PagedSparseMqaKernelArgs args) {
    const PagedSparseMqaAddressArgs& opt = args.opt;
    const PagedSparseMqaArgs& a = opt.base;
    const PagedSparseMqaMapping& m = a.mapping;
    constexpr int Payload = FP4 ? 64 : 128;
    const int lane = threadIdx.x & 63, pos = lane & 15, gl = lane >> 4, wave = threadIdx.x / 64;
    const auto* header = a.metadata + int64_t(m.rows) * m.capacity;
    const auto* tasks = header + 4 + m.rows;
    // The task list excludes dead suffixes. Initialize those with all 256
    // threads, disjoint from the rounded tiles visited by the main loop.
    for (int row = blockIdx.x; row < m.rows; row += gridDim.x) {
        const int first = ((int(header[4 + row]) + 63) / 64) * 64;
        for (int t = first + threadIdx.x; t < m.capacity * m.block; t += 256)
            a.out[int64_t(row) * a.out_row + t] = -INFINITY;
    }
    for (uint64_t task = blockIdx.x; task < uint64_t(header[0]); task += gridDim.x) {
        const uint64_t entry = tasks[task];
        const unsigned row = unsigned(entry);
        const int first = (entry >> 32) & 65535, last = entry >> 48;
        SparseMqaBf16x4 q[HT][8];
        float weights[HT][4];
        bool qnormal = true;
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const int head = h * 16 + pos;
            const unsigned packed = head < a.heads ? a.scales[int64_t(row) * a.sf_row + head * a.sf_head] : 0x7f7f7f7fu;
            const unsigned qe = (packed >> (gl * 8)) & 255;
            qnormal &= qe >= 96 && qe <= 160;
            SparseMqaUint4 lo{0, 0, 0, 0}, hi{0, 0, 0, 0};
            if (head < a.heads) {
                const uint8_t* qp = a.q + int64_t(row) * a.q_row + head * a.q_head + gl * (FP4 ? 16 : 32);
                lo = sparse_mqa_load16(qp);
                if constexpr (!FP4)
                    hi = sparse_mqa_load16(qp + 16);
            }
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const unsigned word = FP4 ? lo[s / 2] >> ((s & 1) * 16) : s < 4 ? lo[s] : hi[s - 4];
                q[h][s] = sparse_mqa_scaled_bf16<FP4>(word, qe);
                // FP8's Q decode also hoists exponent/sign intermediates.
                // Keep each completed fragment ahead of the next expansion.
                if constexpr (!FP4)
                    asm volatile("" : "+v"(lo), "+v"(hi) : "v"(q[h][s]));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int wh = h * 16 + j * 4 + gl;
                weights[h][j] = wh < a.heads ? a.weights[int64_t(row) * a.weight_row + wh * a.weight_head] : 0.f;
            }
        }
        qnormal = __all(qnormal);
#pragma unroll 1
        for (int tile = first; tile < last; ++tile) {
            const int t = tile * kPagedSparseMqaTileTokens + wave * 16 + pos;
            int64_t page = 0;
            int offset = 0;
            const bool live = sparse_mqa_group_address(opt, row, t, page, offset);
            const uint64_t flag = (uint64_t(row) * a.tiles_per_row + tile) * 4 + wave;
            {
                if (!__any(live)) {
                    if (lane == 0)
                        args.exceptional[flag] = 0;
                    if (gl == 0 && t < m.capacity * m.block)
                        a.out[int64_t(row) * a.out_row + t] = -INFINITY;
                    continue;
                }
            }
            if (!live) {
                page = 0;
                offset = 0;
            }
            uint32_t ks = 0x7f7f7f7fu;
            if (live)
                __builtin_memcpy(&ks, a.kv + page * a.kv_stride + int64_t(m.page) * Payload + int64_t(offset) * 4, 4);
            const unsigned ke = (ks >> (gl * 8)) & 255;
            const bool normal = qnormal && __all(ke >= 96 && ke <= 160);
            if (lane == 0)
                args.exceptional[flag] = !normal;
            if (!normal)
                continue;
            SparseMqaUint4 lo{0, 0, 0, 0}, hi{0, 0, 0, 0};
            if (live) {
                const uint8_t* kp = a.kv + page * a.kv_stride + int64_t(offset) * Payload + gl * (FP4 ? 16 : 32);
                lo = sparse_mqa_load16(kp);
                if constexpr (!FP4)
                    hi = sparse_mqa_load16(kp + 16);
            }
            SparseMqaFloat4 dots[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h)
                dots[h] = SparseMqaFloat4{0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                // Tie the next packed operand to the previous MMAC result.
                // This emits no instruction, but prevents hoisting all eight
                // nibble expansions ahead of their consumers (observed in ISA).
                if constexpr (HT == 1) {
                    if constexpr (FP4)
                        asm volatile("" : "+v"(lo) : "v"(dots[0][0]));
                    else
                        asm volatile("" : "+v"(lo), "+v"(hi) : "v"(dots[0][0]));
                } else {
                    if constexpr (FP4)
                        asm volatile("" : "+v"(lo) : "v"(dots[0][0]), "v"(dots[1][0]));
                    else
                        asm volatile("" : "+v"(lo), "+v"(hi) : "v"(dots[0][0]), "v"(dots[1][0]));
                }
                const unsigned word = FP4 ? lo[s / 2] >> ((s & 1) * 16) : s < 4 ? lo[s] : hi[s - 4];
                const SparseMqaBf16x4 k = sparse_mqa_scaled_bf16<FP4>(word, ke);
#pragma unroll
                for (int h = 0; h < HT; ++h)
                    dots[h] = sparse_mqa_bf16_mmac(k, q[h][s], dots[h]);
            }
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < HT; ++h) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (h * 16 + j * 4 + gl < a.heads) {
                        const float v = dots[h][j];
                        sum += (isnan(v) ? v : fmaxf(v, 0.f)) * weights[h][j];
                    }
                }
            }
            sum += __shfl_xor(sum, 16, 64);
            sum += __shfl_xor(sum, 32, 64);
            if (gl == 0 && t < m.capacity * m.block)
                a.out[int64_t(row) * a.out_row + t] = live ? sum : -INFINITY;
        }
    }
}

template <bool FP4, int HT> __global__ void paged_sparse_mqa_bf16_exception_kernel(PagedSparseMqaKernelArgs args) {
    const PagedSparseMqaAddressArgs& opt = args.opt;
    const PagedSparseMqaArgs& a = opt.base;
    const PagedSparseMqaMapping& m = a.mapping;
    const int lane = threadIdx.x & 63, pos = lane & 15, gl = lane >> 4, wave = threadIdx.x / 64;
    const auto* header = a.metadata + int64_t(m.rows) * m.capacity;
    const auto* tasks = header + 4 + m.rows;
    for (uint64_t work = blockIdx.x; work < uint64_t(header[0]); work += gridDim.x) {
        const uint64_t entry = tasks[work];
        const unsigned row = unsigned(entry);
        const int first = (entry >> 32) & 65535, last = entry >> 48;
        // One coalesced flag read replaces 16 per-tile address / predicate
        // iterations on the common all-normal path. No host-side readback.
        const bool exceptional =
            lane < (last - first) * 4 && args.exceptional[(uint64_t(row) * a.tiles_per_row + first) * 4 + lane];
        if (!__any(exceptional))
            continue;
        for (int tile = first; tile < last; ++tile) {
            const uint64_t task = uint64_t(row) * a.tiles_per_row + tile;
            if (!args.exceptional[task * 4 + wave])
                continue;
            const int t = tile * kPagedSparseMqaTileTokens + wave * 16 + pos;
            int64_t page = 0;
            int offset = 0;
            const bool live = sparse_mqa_token_address(opt, row, t, page, offset);
            if (!live) {
                page = 0;
                offset = 0;
            }
            constexpr int Payload = FP4 ? 64 : 128;
            const uint8_t* kp = a.kv + page * a.kv_stride + int64_t(offset) * Payload;
            uint32_t ks = 0x7f7f7f7fu, qs[HT];
            if (live)
                __builtin_memcpy(&ks, a.kv + page * a.kv_stride + int64_t(m.page) * Payload + int64_t(offset) * 4, 4);
            SparseMqaFloat4 dots[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h) {
                const int head = h * 16 + pos;
                qs[h] = head < a.heads ? a.scales[int64_t(row) * a.sf_row + head * a.sf_head] : 0x7f7f7f7fu;
                dots[h] = SparseMqaFloat4{0, 0, 0, 0};
            }
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                if constexpr (HT == 1)
                    asm volatile("" : "+v"(kp), "+v"(ks), "+v"(qs[0]) : "v"(dots[0]));
                else
                    asm volatile("" : "+v"(kp), "+v"(ks), "+v"(qs[0]), "+v"(qs[1]) : "v"(dots[0]), "v"(dots[1]));
                const unsigned ke = (ks >> (g * 8)) & 255;
                const SparseMqaBf16x4 ka =
                    live ? sparse_mqa_decode_bf16<FP4>(kp, g * 32 + gl * 4) : SparseMqaBf16x4{0, 0, 0, 0};
                const SparseMqaBf16x4 kb =
                    live ? sparse_mqa_decode_bf16<FP4>(kp, g * 32 + 16 + gl * 4) : SparseMqaBf16x4{0, 0, 0, 0};
#pragma unroll
                for (int h = 0; h < HT; ++h) {
                    const int head = h * 16 + pos;
                    const uint8_t* qp = a.q + int64_t(row) * a.q_row + min(head, a.heads - 1) * a.q_head;
                    asm volatile("" : "+v"(qp) : "v"(dots[h]));
                    const SparseMqaBf16x4 qa =
                        head < a.heads ? sparse_mqa_decode_bf16<FP4>(qp, g * 32 + gl * 4) : SparseMqaBf16x4{0, 0, 0, 0};
                    const SparseMqaBf16x4 qb = head < a.heads ? sparse_mqa_decode_bf16<FP4>(qp, g * 32 + 16 + gl * 4)
                                                              : SparseMqaBf16x4{0, 0, 0, 0};
                    const SparseMqaFloat4 raw =
                        sparse_mqa_bf16_mmac(kb, qb, sparse_mqa_bf16_mmac(ka, qa, SparseMqaFloat4{0, 0, 0, 0}));
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int qe = (__shfl(qs[h], j * 4 + gl, 64) >> (g * 8)) & 255;
                        dots[h][j] += (qe == 255 || ke == 255) ? NAN : ldexpf(raw[j], qe + int(ke) - 254);
                    }
                }
            }
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < HT; ++h) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int wh = h * 16 + j * 4 + gl;
                    if (wh < a.heads) {
                        const float v = dots[h][j];
                        sum += (isnan(v) ? v : fmaxf(v, 0.f)) *
                               a.weights[int64_t(row) * a.weight_row + wh * a.weight_head];
                    }
                }
            }
            sum += __shfl_xor(sum, 16, 64);
            sum += __shfl_xor(sum, 32, 64);
            if (gl == 0 && t < m.capacity * m.block)
                a.out[int64_t(row) * a.out_row + t] = live ? sum : -INFINITY;
        }
    }
}
} // namespace deepgemm
