// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "../../gfx936_gfx938_shared/paged_mqa_logits/paged_sparse_mqa_impl.hpp"

namespace deepgemm {
using SparseMqaInt2 = int __attribute__((ext_vector_type(2)));

// gfx938 uses the same E4M3FN encoding as the public MXFP8 payload. E2M1
// values also embed exactly in E4M3FN, without applying their UE8M0 scales.
template <bool FP4> __device__ __forceinline__ SparseMqaInt2 sparse_mqa_fp8_fragment(const uint8_t* p, int d) {
    SparseMqaInt2 result;
    if constexpr (!FP4)
        __builtin_memcpy(&result, p + d, 8);
    else {
        unsigned bits;
        __builtin_memcpy(&bits, p + d / 2, 4);
        const unsigned even =
            __builtin_amdgcn_perm(0x4c484440u, 0x3c383000u, bits & 0x07070707u) | ((bits & 0x08080808u) << 4);
        const unsigned odd =
            __builtin_amdgcn_perm(0x4c484440u, 0x3c383000u, (bits >> 4) & 0x07070707u) | (bits & 0x80808080u);
        result = {int(__builtin_amdgcn_perm(odd, even, 0x05010400u)),
                  int(__builtin_amdgcn_perm(odd, even, 0x07030602u))};
    }
    return result;
}
__device__ __forceinline__ SparseMqaFloat4 sparse_mqa_fp8_mmac(SparseMqaInt2 k, SparseMqaInt2 q) {
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(k, q, SparseMqaFloat4{0, 0, 0, 0}, false, false);
}

// E2M1 magnitudes [0,.5,1,1.5,2,3,4,6] remain exactly representable in
// E4M3FN for delta in [-6,5]. The pivot guard is based on tensor contents.
__device__ __forceinline__ SparseMqaInt2 sparse_mqa_rescale_fp4_to_fp8(unsigned bits, int delta) {
    const unsigned hi = 0x4c484440u - unsigned(delta) * 0x08080808u;
    const unsigned lo = 0x3c383000u - unsigned(delta) * 0x08080800u;
    const unsigned even = __builtin_amdgcn_perm(hi, lo, bits & 0x07070707u) | ((bits & 0x08080808u) << 4);
    const unsigned odd = __builtin_amdgcn_perm(hi, lo, (bits >> 4) & 0x07070707u) | (bits & 0x80808080u);
    return {int(__builtin_amdgcn_perm(odd, even, 0x05010400u)), int(__builtin_amdgcn_perm(odd, even, 0x07030602u))};
}
__device__ __forceinline__ int sparse_mqa_scale_min4(int e) {
    e = min(e, __shfl_xor(e, 16, 64));
    return min(e, __shfl_xor(e, 32, 64));
}
__device__ __forceinline__ int sparse_mqa_scale_max4(int e) {
    e = max(e, __shfl_xor(e, 16, 64));
    return max(e, __shfl_xor(e, 32, 64));
}

// Native group arithmetic remains valid for every UE8M0 exponent, including
// reciprocal extremes. Keep four waves and cache raw Q / weights across tiles.
template <bool FP4, int HT, bool Exceptional = false>
__global__ void paged_sparse_mqa_fp8_group_kernel(PagedSparseMqaAddressArgs opt, const uint8_t* flags = nullptr) {
    const PagedSparseMqaArgs& a = opt.base;
    const PagedSparseMqaMapping& m = a.mapping;
    constexpr int Payload = FP4 ? 64 : 128;
    const int lane = threadIdx.x & 63, pos = lane & 15, gl = lane >> 4, wave = threadIdx.x / 64;
    const int64_t* header = a.metadata + int64_t(m.rows) * m.capacity;
    const uint64_t count = uint64_t(header[0]);
    if constexpr (!Exceptional) {
        // Coalesced suffix initialization is disjoint from live rounded tiles.
        // Empty queries have no task descriptor but still get a complete output.
        for (int row = blockIdx.x; row < m.rows; row += gridDim.x) {
            const int first = ((int(header[4 + row]) + 63) / 64) * 64;
            for (int t = first + threadIdx.x; t < m.capacity * m.block; t += 256)
                a.out[int64_t(row) * a.out_row + t] = -INFINITY;
        }
    }
    for (uint64_t task = blockIdx.x; task < count; task += gridDim.x) {
        unsigned row;
        int first, last;
        {
            const uint64_t entry = header[4 + m.rows + task];
            row = unsigned(entry);
            first = (entry >> 32) & 65535;
            last = entry >> 48;
        }
        if constexpr (Exceptional) {
            const bool flagged =
                lane < (last - first) * 4 && flags[(uint64_t(row) * a.tiles_per_row + first) * 4 + lane];
            if (!__any(flagged))
                continue;
        }
        SparseMqaInt2 q[HT][4];
        unsigned qs[HT][4];
        float weights[HT][4];
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const int head = h * 16 + pos;
            const unsigned packed = head < a.heads ? a.scales[int64_t(row) * a.sf_row + head * a.sf_head] : 0x7f7f7f7fu;
#pragma unroll
            for (int g = 0; g < 4; ++g)
                q[h][g] = head < a.heads ? sparse_mqa_fp8_fragment<FP4>(a.q + int64_t(row) * a.q_row + head * a.q_head,
                                                                        g * 32 + gl * 8)
                                         : SparseMqaInt2{0, 0};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int wh = h * 16 + j * 4 + gl;
                qs[h][j] = __shfl(packed, j * 4 + gl, 64);
                weights[h][j] = wh < a.heads ? a.weights[int64_t(row) * a.weight_row + wh * a.weight_head] : 0.f;
            }
        }
#pragma unroll 1
        for (int tile = first; tile < last; ++tile) {
            if constexpr (Exceptional)
                if (!flags[(uint64_t(row) * a.tiles_per_row + tile) * 4 + wave])
                    continue;
            const int t = tile * kPagedSparseMqaTileTokens + wave * 16 + pos;
            int64_t page = 0;
            int offset = 0;
            const bool live = sparse_mqa_group_address(opt, row, t, page, offset);
            if (!__any(live)) {
                if (gl == 0 && t < m.capacity * m.block)
                    a.out[int64_t(row) * a.out_row + t] = -INFINITY;
                continue;
            }
            if (!live) {
                page = 0;
                offset = 0;
            }
            const uint8_t* kp = a.kv + page * a.kv_stride + int64_t(offset) * Payload;
            unsigned ks = 0x7f7f7f7fu;
            if (live)
                __builtin_memcpy(&ks, a.kv + page * a.kv_stride + int64_t(m.page) * Payload + int64_t(offset) * 4, 4);
            SparseMqaFloat4 dots[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h)
                dots[h] = SparseMqaFloat4{0, 0, 0, 0};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                SparseMqaInt2 k = live ? sparse_mqa_fp8_fragment<FP4>(kp, g * 32 + gl * 8) : SparseMqaInt2{0, 0};
                // Allow loads to overlap, but consume a group accumulator
                // before issuing the next one. Do not serialize the VMEM pipe.
                if constexpr (HT == 1)
                    asm volatile("" : "+v"(k) : "v"(dots[0]));
                else
                    asm volatile("" : "+v"(k) : "v"(dots[0]), "v"(dots[1]));
                const unsigned ke = (ks >> (g * 8)) & 255;
#pragma unroll
                for (int h = 0; h < HT; ++h) {
                    const SparseMqaFloat4 raw = sparse_mqa_fp8_mmac(k, q[h][g]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        unsigned packed = qs[h][j];
                        asm volatile("" : "+v"(packed) : "v"(raw));
                        const int qe = (packed >> (g * 8)) & 255;
                        dots[h][j] += (qe == 255 || ke == 255) ? NAN : ldexpf(raw[j], qe + int(ke) - 254);
                    }
                }
            }
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < HT; ++h) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (h * 16 + j * 4 + gl < a.heads) {
                        const float v = dots[h][j];
                        sum += (isnan(v) ? v : fmaxf(v, 0.f)) * weights[h][j];
                    }
            }
            sum += __shfl_xor(sum, 16, 64);
            sum += __shfl_xor(sum, 32, 64);
            if (gl == 0 && t < m.capacity * m.block)
                a.out[int64_t(row) * a.out_row + t] = live ? sum : -INFINITY;
        }
    }
}

template <int HT> __global__ void paged_sparse_mqa_fp4_rescaled_kernel(PagedSparseMqaKernelArgs args) {
    const PagedSparseMqaAddressArgs& opt = args.opt;
    const PagedSparseMqaArgs& a = opt.base;
    const PagedSparseMqaMapping& m = a.mapping;
    const int lane = threadIdx.x & 63, pos = lane & 15, gl = lane >> 4, wave = threadIdx.x / 64;
    const auto* header = a.metadata + int64_t(m.rows) * m.capacity;
    const auto* tasks = header + 4 + m.rows;
    for (int row = blockIdx.x; row < m.rows; row += gridDim.x) {
        const int first = ((int(header[4 + row]) + 63) / 64) * 64;
        for (int t = first + threadIdx.x; t < m.capacity * m.block; t += 256)
            a.out[int64_t(row) * a.out_row + t] = -INFINITY;
    }
    for (uint64_t task = blockIdx.x; task < uint64_t(header[0]); task += gridDim.x) {
        const uint64_t entry = tasks[task];
        const unsigned row = unsigned(entry);
        const int first = (entry >> 32) & 65535, last = entry >> 48;
        SparseMqaInt2 q[HT][4];
        int qexps[HT][4];
        float weights[HT][4];
        bool qnormal = true;
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const int head = h * 16 + pos;
            const unsigned packed = head < a.heads ? a.scales[int64_t(row) * a.sf_row + head * a.sf_head] : 0x7f7f7f7fu;
            const int qe = (packed >> (gl * 8)) & 255;
            const int lo = sparse_mqa_scale_min4(qe), hi = sparse_mqa_scale_max4(qe), pivot = min(hi, lo + 5);
            qnormal &= lo >= 96 && hi <= 160 && hi - lo <= 11;
            SparseMqaUint4 bits{0, 0, 0, 0};
            if (head < a.heads)
                bits = sparse_mqa_load16(a.q + int64_t(row) * a.q_row + head * a.q_head + gl * 16);
#pragma unroll
            for (int s = 0; s < 4; ++s)
                q[h][s] = sparse_mqa_rescale_fp4_to_fp8(bits[s], pivot - qe);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int wh = h * 16 + j * 4 + gl;
                qexps[h][j] = __shfl(pivot, j * 4 + gl, 64);
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
            if (!__any(live)) {
                if (lane == 0)
                    args.exceptional[flag] = 0;
                if (gl == 0 && t < m.capacity * m.block)
                    a.out[int64_t(row) * a.out_row + t] = -INFINITY;
                continue;
            }
            if (!live) {
                page = 0;
                offset = 0;
            }
            unsigned ks = 0x7f7f7f7fu;
            if (live)
                __builtin_memcpy(&ks, a.kv + page * a.kv_stride + int64_t(m.page) * 64 + int64_t(offset) * 4, 4);
            const int ke = (ks >> (gl * 8)) & 255;
            const int lo = sparse_mqa_scale_min4(ke), hi = sparse_mqa_scale_max4(ke), pivot = min(hi, lo + 5);
            const bool normal = qnormal && __all(lo >= 96 && hi <= 160 && hi - lo <= 11);
            if (lane == 0)
                args.exceptional[flag] = !normal;
            if (!normal)
                continue;
            SparseMqaUint4 bits{0, 0, 0, 0};
            if (live)
                bits = sparse_mqa_load16(a.kv + page * a.kv_stride + int64_t(offset) * 64 + gl * 16);
            SparseMqaFloat4 dots[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h)
                dots[h] = SparseMqaFloat4{0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                if constexpr (HT == 1)
                    asm volatile("" : "+v"(bits) : "v"(dots[0][0]));
                else
                    asm volatile("" : "+v"(bits) : "v"(dots[0][0]), "v"(dots[1][0]));
                const SparseMqaInt2 k = sparse_mqa_rescale_fp4_to_fp8(bits[s], pivot - ke);
#pragma unroll
                for (int h = 0; h < HT; ++h)
                    dots[h] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(k, q[h][s], dots[h], false, false);
            }
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < HT; ++h) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (h * 16 + j * 4 + gl < a.heads) {
                        const float v = ldexpf(dots[h][j], qexps[h][j] + pivot - 254);
                        sum += (isnan(v) ? v : fmaxf(v, 0.f)) * weights[h][j];
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
