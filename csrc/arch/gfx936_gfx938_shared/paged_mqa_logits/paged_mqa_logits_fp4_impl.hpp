// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "paged_mqa_logits/paged_mqa_logits.h"
#include "paged_mqa_logits/cta_schedule.h"

namespace deepgemm {
using PagedFp4Float4 = float __attribute__((ext_vector_type(4)));
using PagedFp4Bf16x4 = short __attribute__((ext_vector_type(4)));
using PagedFp4Uint4 = unsigned __attribute__((ext_vector_type(4)));

__device__ inline int paged_fp4_first_row(int route, int n, bool paired) {
    return paired ? route / ((n + 1) / 2) * n + route % ((n + 1) / 2) * 2 : route;
}
__device__ inline int paged_fp4_route_length(const int* lens, int route, int n, bool paired) {
    const int first = paged_fp4_first_row(route, n, paired);
    int length = ::max(lens[first], 0);
    if (paired && first % n + 1 < n)
        length = ::max(length, lens[first + 1]);
    return length;
}

struct PagedFp4Bf16 {
    using Fragment = PagedFp4Bf16x4;
    static constexpr int Steps = 8;
    __device__ static int pivot(unsigned) {
        return 127;
    }
    __device__ static bool normal_weight(float) {
        return true;
    }
    __device__ static float prepare_weight(float value, int) {
        return value;
    }
    __device__ static float finish(float value, int) {
        return value;
    }
    __device__ static bool normal(unsigned x) {
        const unsigned low = x & 0x7f7f7f7fu;
        const unsigned below = ~((low + 0x20202020u) | x);
        const unsigned above = (low + 0x5f5f5f5fu) & x;
        return ((below | above) & 0x80808080u) == 0;
    }
    __device__ static Fragment convert(PagedFp4Uint4 bits, int step, int e, int) {
        const unsigned word = bits[step / 2] >> ((step & 1) * 16);
        // Magnitudes 0,.5,1,1.5,2,3,4,6 form two byte tables. The exponent's
        // parity determines the low bytes; adjacent exponent pairs share the
        // high byte. Four nibbles are expanded with byte-permute instructions.
        const unsigned low0 = unsigned((e - 1) & 1) * 0x00008000u + unsigned(e & 1) * 0x80800000u + 0x40000000u;
        const unsigned low1 = unsigned((e + 1) & 1) * 0x00008080u + unsigned((e + 2) & 1) * 0x80800000u + 0x40004000u;
        const unsigned high0 = unsigned((e - 1) >> 1) * 0x00000100u + unsigned(e >> 1) * 0x01010000u;
        const unsigned high1 = unsigned((e + 1) >> 1) * 0x00000101u + unsigned((e + 2) >> 1) * 0x01010000u;
        const unsigned code = __builtin_amdgcn_perm(word >> 4, word, 0x05010400u);
        const unsigned index = code & 0x07070707u;
        const unsigned low = __builtin_amdgcn_perm(low1, low0, index);
        const unsigned high = __builtin_amdgcn_perm(high1, high0, index) | ((code & 0x08080808u) << 4);
        const unsigned packed[2] = {
            __builtin_amdgcn_perm(high, low, 0x05010400u), __builtin_amdgcn_perm(high, low, 0x07030602u)};
        return __builtin_bit_cast(Fragment, packed);
    }
    __device__ static PagedFp4Float4 mma(Fragment k, Fragment q, PagedFp4Float4 c) {
#if defined(__gfx936__)
        return __builtin_amdgcn_mmac_f32_16x16x16bf16(k, q, c);
#elif defined(__gfx938__)
        return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(k, q, c, false, false);
#else
        return c;
#endif
    }
};

// gfx936 uses exact FP16 expansion in its normal range. Wider MX scales
// retain the BF16/precise fallback below.
struct PagedFp4Fp16 : PagedFp4Bf16 {
    static constexpr int ConvertSteps = 2;
    using Fragment = __fp16 __attribute__((ext_vector_type(4)));
    __device__ static bool normal(unsigned x) {
        const unsigned low = x & 0x7f7f7f7fu;
        const unsigned below = ~((low + 0x0e0e0e0eu) | x);
        const unsigned above = (low + 0x73737373u) & x;
        return ((below | above) & 0x80808080u) == 0;
    }
    __device__ static Fragment convert(PagedFp4Uint4 bits, int step, int e, int) {
        const unsigned word = bits[step / 2] >> ((step & 1) * 16);
        const unsigned low =
            unsigned((e - 113) * 4) * 0x00000100u + unsigned((e - 112) * 4) * 0x01010000u + 0x02000000u;
        const unsigned high =
            unsigned((e - 111) * 4) * 0x00000101u + unsigned((e - 110) * 4) * 0x01010000u + 0x02000200u;
        const unsigned code = __builtin_amdgcn_perm(word >> 4, word, 0x05010400u);
        const unsigned bytes = __builtin_amdgcn_perm(high, low, code & 0x07070707u) | ((code & 0x08080808u) << 4);
        const unsigned packed[2] = {
            __builtin_amdgcn_perm(bytes, 0, 0x05000400u), __builtin_amdgcn_perm(bytes, 0, 0x07000600u)};
        return __builtin_bit_cast(Fragment, packed);
    }
    __device__ static void convert_block(PagedFp4Uint4 bits, int step, int e, int, Fragment* out) {
        const unsigned word = bits[step / 2];
        const unsigned low =
            unsigned((e - 113) * 4) * 0x00000100u + unsigned((e - 112) * 4) * 0x01010000u + 0x02000000u;
        const unsigned high =
            unsigned((e - 111) * 4) * 0x00000101u + unsigned((e - 110) * 4) * 0x01010000u + 0x02000200u;
        const unsigned even = __builtin_amdgcn_perm(high, low, word & 0x07070707u) | ((word & 0x08080808u) << 4);
        const unsigned odd = __builtin_amdgcn_perm(high, low, (word >> 4) & 0x07070707u) | (word & 0x80808080u);
        const unsigned first[2] = {
            __builtin_amdgcn_perm(odd, even, 0x040c000cu), __builtin_amdgcn_perm(odd, even, 0x050c010cu)};
        const unsigned second[2] = {
            __builtin_amdgcn_perm(odd, even, 0x060c020cu), __builtin_amdgcn_perm(odd, even, 0x070c030cu)};
        out[0] = __builtin_bit_cast(Fragment, first);
        out[1] = __builtin_bit_cast(Fragment, second);
    }
    __device__ static PagedFp4Float4 mma(Fragment k, Fragment q, PagedFp4Float4 c) {
#if defined(__gfx936__)
        return __builtin_amdgcn_mmac_f32_16x16x16f16(k, q, c);
#elif defined(__gfx938__)
        return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(k, q, c, false, false);
#else
        return c;
#endif
    }
};

template <int TableBlock = 0>
__device__ inline const uint8_t* paged_fp4_token(const PagedMqaFp4Args& a, int row, int t, int& offset) {
    const int table_row = a.paired ? row - (row % a.rows_per_request) % 2 : row;
    constexpr int Shift = TableBlock == 64 ? 6 : 3;
    const int shift = TableBlock ? Shift : a.table_shift;
    const int block = a.table[int64_t(table_row) * a.table_stride + (t >> shift)];
    if constexpr (TableBlock == 64) {
        offset = t & 63;
        return a.kv + int64_t(block) * a.kv_stride;
    } else {
        const int64_t slot = (int64_t(block) << shift) + (t & ((1 << shift) - 1));
        offset = slot & 63;
        return a.kv + (slot >> 6) * a.kv_stride;
    }
}

// Match LightOp's finite BF16 decode domain, including subnormals. Invalid
// MX scale 255 and FP4 values overflowing BF16 decode to zero in this API.
__device__ inline double paged_fp4_exact(unsigned code, unsigned scale) {
    const unsigned m = code & 7;
    if (m == 0 || scale == 255)
        return 0;
    const unsigned base = 0x3f00 + ((m >= 2 ? m : 0) << 6);
    if (int(base >> 7) + int(scale) - 127 >= 255)
        return 0;
    const double raw = m == 1 ? 0.5 : m < 4 ? m * 0.5 : m < 6 ? m - 2.0 : 2.0 * m - 8.0;
    return ldexp((code & 8) ? -raw : raw, int(scale) - 127);
}
template <int HT>
struct PagedFp4Fallback {
    PagedFp4Uint4 qbits[HT];
    int qexp[HT], loaded_row = -1;
    float weights[HT][4];
    bool normal_q = false;
    __device__ __forceinline__ void score(PagedMqaFp4Args a, int row, int start) {
        const int lane = threadIdx.x & 63, pos = lane & 15, group = lane >> 4;
        const int length = ::min(a.width, a.lens[row]), t = start + pos;
        if (loaded_row != row) {
            loaded_row = row;
            normal_q = true;
#pragma unroll
            for (int h = 0; h < HT; ++h) {
                const int head = h * 16 + pos;
                const unsigned sf = a.sf[int64_t(row) * a.heads + head];
                normal_q &= PagedFp4Bf16::normal(sf);
                qexp[h] = (sf >> (group * 8)) & 255;
                qbits[h] =
                    *reinterpret_cast<const PagedFp4Uint4*>(a.q + (int64_t(row) * a.heads + head) * 64 + group * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    weights[h][j] = a.weights[int64_t(row) * a.weight_stride + h * 16 + j * 4 + group];
            }
            normal_q = __all(normal_q);
        }
        int offset = 0;
        const uint8_t* page = t < length ? paged_fp4_token(a, row, t, offset) : a.kv;
        const unsigned ks = t < length ? *reinterpret_cast<const unsigned*>(page + 4096 + offset * 4) : 0x7f7f7f7f;
        if (normal_q && __all(PagedFp4Bf16::normal(ks))) {
            auto bits = t < length ? *reinterpret_cast<const PagedFp4Uint4*>(page + offset * 64 + group * 16)
                                   : PagedFp4Uint4{0, 0, 0, 0};
            PagedFp4Float4 dot[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h)
                dot[h] = {0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                asm volatile("" : "+v"(bits) : "v"(dot[0]));
                const auto k = PagedFp4Bf16::convert(bits, s, (ks >> (group * 8)) & 255, 127);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
                    asm volatile("" : "+v"(qbits[h]) : "v"(dot[h]));
                    const auto q = PagedFp4Bf16::convert(qbits[h], s, qexp[h], 127);
                    dot[h] = PagedFp4Bf16::mma(k, q, dot[h]);
                }
            }
            float value = 0;
#pragma unroll
            for (int h = 0; h < HT; ++h)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    value = fmaf(fmaxf(dot[h][j], 0.f), weights[h][j], value);
            value += __shfl_xor(value, 16, 64);
            value += __shfl_xor(value, 32, 64);
            if (group == 0 && t < length)
                a.output[int64_t(row) * a.output_stride + t] = value;
            return;
        }
        if (t >= length)
            return;
        const uint8_t* kp = page + offset * 64;
        double sum = 0;
        for (int h = group; h < a.heads; h += 4) {
            const uint8_t* qp = a.q + (int64_t(row) * a.heads + h) * 64;
            const unsigned qs = a.sf[int64_t(row) * a.heads + h];
            double dot = 0;
            for (int d = 0; d < 128; ++d) {
                const unsigned qv = (qp[d / 2] >> ((d & 1) * 4)) & 15;
                const unsigned kv = (kp[d / 2] >> ((d & 1) * 4)) & 15;
                dot += paged_fp4_exact(qv, (qs >> ((d / 32) * 8)) & 255) *
                       paged_fp4_exact(kv, (ks >> ((d / 32) * 8)) & 255);
            }
            sum += fmax(dot, 0.) * double(a.weights[int64_t(row) * a.weight_stride + h]);
        }
        sum += __shfl_xor(sum, 16, 64);
        sum += __shfl_xor(sum, 32, 64);
        if (group == 0)
            a.output[int64_t(row) * a.output_stride + t] = float(sum);
    }
};

// Keep only one converted Q representation live. Each task reuses Q across
// adjacent KV tiles. Exceptional values use the exact fallback below.
template <class Math, int HT, int Rows = 1, int TableBlock = 64>
__device__ __forceinline__ void paged_mqa_fp4_compute(PagedMqaFp4Args a) {
    const int lane = threadIdx.x & 63, pos = lane & 15, group = lane >> 4, wave = threadIdx.x >> 6;
    // Use shorter grid tasks when four splits would leave resident CTAs idle.
    const int splits = !a.schedule && gridDim.x > uint64_t(a.routes) * ((int64_t(a.width) + 1023) / 1024) ? 1 : 4;
    const int chunks = (int64_t(a.width) + splits * 256 - 1) / (splits * 256);
    const int64_t tasks = a.schedule ? a.schedule_ctas : int64_t(a.routes) * chunks;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int begin_route = a.schedule ? a.schedule[2 * task] : task / chunks;
        if (begin_route >= a.routes)
            break;
        const int end_route = a.schedule ? a.schedule[2 * task + 2] : begin_route;
        const int begin_split = a.schedule ? a.schedule[2 * task + 1] : (task % chunks) * splits;
        const int end_split = a.schedule ? a.schedule[2 * task + 3] : ::min(begin_split + splits, (a.width + 255) / 256);
        for (int route = begin_route; route <= end_route && route < a.routes; ++route) {
            const int first = paged_fp4_first_row(route, a.rows_per_request, Rows == 2);
            const int count = Rows == 2 && first % a.rows_per_request + 1 < a.rows_per_request ? 2 : 1;
            const int split0 = route == begin_route ? begin_split : 0;
            const int split1 =
                route == end_route
                    ? end_split
                    : (::min(paged_fp4_route_length(a.lens, route, a.rows_per_request, Rows == 2), a.width) + 255) / 256;
            int length[Rows], reach = 0;
#pragma unroll
            for (int i = 0; i < Rows; ++i) {
                length[i] = i < count ? ::min(a.width, a.lens[first + i]) : 0;
                reach = ::max(reach, length[i]);
            }
            if (split0 * 256 >= reach)
                continue;
            typename Math::Fragment q[Rows][HT][Math::Steps];
            float weight[Rows][HT][4];
            bool normal[Rows];
#pragma unroll
            for (int i = 0; i < Rows; ++i) {
                normal[i] = false;
                if (split0 * 256 >= length[i])
                    continue;
                normal[i] = true;
#pragma unroll
                for (int h = 0; h < HT; ++h) {
                    const int head = h * 16 + pos;
                    const unsigned sf = a.sf[int64_t(first + i) * a.heads + head];
                    normal[i] &= Math::normal(sf);
                    const int pivot = Math::pivot(sf), exp = (sf >> (group * 8)) & 255;
                    const auto bits = *reinterpret_cast<const PagedFp4Uint4*>(
                        a.q + (int64_t(first + i) * a.heads + head) * 64 + group * 16);
#pragma unroll
                    for (int step = 0; step < Math::Steps; ++step)
                        q[i][h][step] = Math::convert(bits, step, exp, pivot);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float w = a.weights[int64_t(first + i) * a.weight_stride + h * 16 + j * 4 + group];
                        normal[i] &= Math::normal_weight(w);
                        weight[i][h][j] = Math::prepare_weight(w, __shfl(pivot, j * 4 + group, 64));
                    }
                }
                normal[i] = __all(normal[i]);
            }
            for (int split = split0; split < ::min(split1, (reach + 255) / 256); ++split) {
                PagedFp4Uint4 payload[4];
                unsigned scales[4];
#pragma unroll
                for (int tile = 0; tile < 4; ++tile) {
                    const int t = split * 256 + wave * 64 + tile * 16 + pos;
                    payload[tile] = {0, 0, 0, 0};
                    scales[tile] = 0x7f7f7f7f;
                    if (t < reach) {
                        int offset;
                        const uint8_t* page = paged_fp4_token<TableBlock>(a, first, t, offset);
                        scales[tile] = *reinterpret_cast<const unsigned*>(page + 4096 + offset * 4);
                        payload[tile] = *reinterpret_cast<const PagedFp4Uint4*>(page + offset * 64 + group * 16);
                    }
                }
#pragma unroll
                for (int tile = 0; tile < 4; ++tile) {
                    const int start = split * 256 + wave * 64 + tile * 16, t = start + pos;
                    if (start >= reach)
                        break;
                    const unsigned sf = scales[tile];
                    const int pivot = Math::pivot(sf), exp = (sf >> (group * 8)) & 255;
                    bool fast[Rows], any_fast = false;
#pragma unroll
                    for (int i = 0; i < Rows; ++i) {
                        fast[i] = start < length[i] && normal[i] && __all(t >= length[i] || Math::normal(sf));
                        any_fast |= fast[i];
                        if (lane == 0 && start < length[i])
                            a.exceptional[int64_t(first + i) * a.tiles + start / 16] = !fast[i];
                    }
                    if (!any_fast)
                        continue;
                    auto bits = payload[tile];
                    PagedFp4Float4 dots[Rows][HT];
#pragma unroll
                    for (int i = 0; i < Rows; ++i)
#pragma unroll
                        for (int h = 0; h < HT; ++h)
                            dots[i][h] = {0, 0, 0, 0};
#pragma unroll
                    for (int step = 0; step < Math::Steps; step += Math::ConvertSteps) {
                        asm volatile("" : "+v"(bits) : "v"(dots[0][0]));
                        typename Math::Fragment k[Math::ConvertSteps];
                        Math::convert_block(bits, step, exp, pivot, k);
#pragma unroll
                        for (int i = 0; i < Rows; ++i) {
                            if (!fast[i])
                                continue;
#pragma unroll
                            for (int j = 0; j < Math::ConvertSteps; ++j)
#pragma unroll
                                for (int h = 0; h < HT; ++h)
                                    dots[i][h] = Math::mma(k[j], q[i][h][step + j], dots[i][h]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < Rows; ++i) {
                        if (!fast[i])
                            continue;
                        float sum = 0;
#pragma unroll
                        for (int h = 0; h < HT; ++h)
#pragma unroll
                            for (int j = 0; j < 4; ++j)
                                sum = fmaf(fmaxf(dots[i][h][j], 0.f), weight[i][h][j], sum);
                        sum += __shfl_xor(sum, 16, 64);
                        sum += __shfl_xor(sum, 32, 64);
                        if (group == 0 && t < length[i])
                            a.output[int64_t(first + i) * a.output_stride + t] = Math::finish(sum, pivot);
                    }
                }
            }
        }
    }
}

// Small grids retain packed Q rather than expanded fragments. This bounds
// live registers while handling both normal and exceptional values in one launch.
template <class Math, int HT, int TableBlock>
struct PagedFp4Small {
    PagedFp4Uint4 qbits[HT];
    int qexp[HT], qpivot[HT], loaded_row = -1;
    float weights[HT][4];
    bool normal_q = false;
    __device__ __forceinline__ void score(PagedMqaFp4Args a, int row, int start) {
        const int lane = threadIdx.x & 63, pos = lane & 15, group = lane >> 4;
        const int length = ::min(a.width, a.lens[row]), t = start + pos;
        if (loaded_row != row) {
            loaded_row = row;
            normal_q = true;
#pragma unroll
            for (int h = 0; h < HT; ++h) {
                const int head = h * 16 + pos;
                const unsigned sf = a.sf[int64_t(row) * a.heads + head];
                normal_q &= Math::normal(sf);
                qexp[h] = (sf >> (group * 8)) & 255;
                qpivot[h] = Math::pivot(sf);
                qbits[h] =
                    *reinterpret_cast<const PagedFp4Uint4*>(a.q + (int64_t(row) * a.heads + head) * 64 + group * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float w = a.weights[int64_t(row) * a.weight_stride + h * 16 + j * 4 + group];
                    normal_q &= Math::normal_weight(w);
                    weights[h][j] = Math::prepare_weight(w, __shfl(qpivot[h], j * 4 + group, 64));
                }
            }
            normal_q = __all(normal_q);
        }
        int offset = 0;
        const uint8_t* page = t < length ? paged_fp4_token<TableBlock>(a, row, t, offset) : a.kv;
        const unsigned sf = t < length ? *reinterpret_cast<const unsigned*>(page + 4096 + offset * 4) : 0x7f7f7f7f;
        if (normal_q && __all(Math::normal(sf))) {
            const int pivot = Math::pivot(sf), exp = (sf >> (group * 8)) & 255;
            auto bits = t < length ? *reinterpret_cast<const PagedFp4Uint4*>(page + offset * 64 + group * 16)
                                   : PagedFp4Uint4{0, 0, 0, 0};
            PagedFp4Float4 dot[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h)
                dot[h] = {0, 0, 0, 0};
#pragma unroll
            for (int step = 0; step < Math::Steps; step += Math::ConvertSteps) {
                asm volatile("" : "+v"(bits) : "v"(dot[0]));
                typename Math::Fragment k[Math::ConvertSteps];
                Math::convert_block(bits, step, exp, pivot, k);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
                    asm volatile("" : "+v"(qbits[h]) : "v"(dot[h]));
                    typename Math::Fragment q[Math::ConvertSteps];
                    Math::convert_block(qbits[h], step, qexp[h], qpivot[h], q);
#pragma unroll
                    for (int j = 0; j < Math::ConvertSteps; ++j)
                        dot[h] = Math::mma(k[j], q[j], dot[h]);
                }
            }
            float value = 0;
#pragma unroll
            for (int h = 0; h < HT; ++h)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    value = fmaf(fmaxf(dot[h][j], 0.f), weights[h][j], value);
            value += __shfl_xor(value, 16, 64);
            value += __shfl_xor(value, 32, 64);
            if (group == 0 && t < length)
                a.output[int64_t(row) * a.output_stride + t] = Math::finish(value, pivot);
        } else {
            PagedFp4Fallback<HT> fallback;
            fallback.score(a, row, start);
        }
    }
};

// A small grid covers every visible tile directly. Larger grids consume the
// persistent schedule; both retain the same lengths and row grouping semantics.
template <class Math, int HT, int Rows = 1, int TableBlock = 64, bool Small = false>
__global__ void paged_mqa_fp4_kernel(PagedMqaFp4Args a) {
    if constexpr (!Small) {
        paged_mqa_fp4_compute<Math, HT, Rows, TableBlock>(a);
    } else {
        PagedFp4Small<Math, HT, TableBlock> scorer;
        const int first = paged_fp4_first_row(blockIdx.y, a.rows_per_request, Rows == 2);
        const int count = Rows == 2 && first % a.rows_per_request + 1 < a.rows_per_request ? 2 : 1;
        const int base = int(blockIdx.x) * 256 + int(threadIdx.x >> 6) * 64;
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
            const int row = first + i;
#pragma unroll
            for (int tile = 0; tile < 4; ++tile) {
                const int start = base + tile * 16;
                if (i < count && start < ::min(a.width, a.lens[row]))
                    scorer.score(a, row, start);
            }
        }
    }
}

template <int HT>
__global__ void paged_mqa_fp4_exception_kernel(PagedMqaFp4Args a) {
    const int lane = threadIdx.x & 63;
    PagedFp4Fallback<HT> fallback;
    // One flag per lane, so normal traffic scans contiguous bytes rather than
    // serially issuing a wave-wide load for every 16-token tile.
    for (int64_t base = int64_t(blockIdx.x) * 256 + (threadIdx.x & ~63); base < int64_t(a.rows) * a.tiles;
         base += int64_t(gridDim.x) * 256) {
        const int64_t probe = base + lane;
        bool active = false;
        if (probe < int64_t(a.rows) * a.tiles) {
            const int row = probe / a.tiles, start = (probe % a.tiles) * 16;
            if (start < ::min(a.width, a.lens[row]))
                active = a.exceptional[probe];
        }
        unsigned long long pending = __ballot(active);
        while (pending) {
            const int64_t task = base + __ffsll(pending) - 1;
            pending &= pending - 1;
            const int row = task / a.tiles, start = (task % a.tiles) * 16;
            fallback.score(a, row, start);
        }
    }
}

// Both architectures share the same shape-specialized launch policy.
template <class Math, int HT, int Rows = 1, int TableBlock = 64, bool Small = false>
static void paged_fp4_launch_typed(PagedMqaFp4Args a, int device, hipStream_t stream) {
    const uint64_t tasks = uint64_t(a.routes) * ((int64_t(a.width) + 255) / 256);
    dim3 grid;
    if constexpr (Small) {
        grid = dim3((a.width + 255) / 256, a.routes);
    } else {
        grid = a.schedule ? dim3(std::min(uint64_t(a.schedule_ctas), tasks))
                          : paged_mqa_persistent_grid<1>(
                                dim3(std::min(tasks, uint64_t(UINT32_MAX))),
                                paged_mqa_fp4_kernel<Math, HT, Rows, TableBlock, Small>,
                                256,
                                0,
                                device,
                                false);
    }
    paged_mqa_fp4_kernel<Math, HT, Rows, TableBlock, Small><<<grid, 256, 0, stream>>>(a);
    if constexpr (!Small)
        paged_mqa_fp4_exception_kernel<HT><<<grid, 256, 0, stream>>>(a);
}

template <class Math, int HT, int Rows = 1, int TableBlock = 64>
static void paged_fp4_launch(PagedMqaFp4Args a, int device, hipStream_t stream) {
    if (a.small_kernel)
        paged_fp4_launch_typed<Math, HT, Rows, TableBlock, true>(a, device, stream);
    else
        paged_fp4_launch_typed<Math, HT, Rows, TableBlock, false>(a, device, stream);
}

template <class Math>
static void launch_paged_mqa_fp4(PagedMqaFp4Args a, int device, hipStream_t stream) {
    if (a.table_shift == 3) {
        if (a.heads == 16)
            paged_fp4_launch<Math, 1, 1, 8>(a, device, stream);
        else if (a.heads == 32)
            paged_fp4_launch<Math, 2, 1, 8>(a, device, stream);
        else
            paged_fp4_launch<Math, 4, 1, 8>(a, device, stream);
    } else if (a.paired && a.heads == 16)
        paged_fp4_launch<Math, 1, 2>(a, device, stream);
    else if (a.paired && a.heads == 32)
        paged_fp4_launch<Math, 2, 2>(a, device, stream);
    else if (a.heads == 16)
        paged_fp4_launch<Math, 1>(a, device, stream);
    else if (a.heads == 32)
        paged_fp4_launch<Math, 2>(a, device, stream);
    else
        paged_fp4_launch<Math, 4>(a, device, stream);
}
} // namespace deepgemm
