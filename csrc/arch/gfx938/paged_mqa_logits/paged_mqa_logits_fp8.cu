// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "paged_mqa_logits/cta_schedule.h"
#include "paged_mqa_logits/paged_mqa_logits.h"
#include "../../gfx936_gfx938_shared/paged_mqa_logits/paged_mqa_logits_dispatch.hpp"
#include "paged_mqa_logits_fp8_impl.hpp"
#include "../../gfx936_gfx938_shared/paged_mqa_logits/paged_mqa_logits_fp4_impl.hpp"

namespace deepgemm {

#define DISPATCH_FP8_TYPES(scalar_type, FP8_TYPE, ...)              \
    [&]() {                                                         \
        if (scalar_type == at::ScalarType::Float8_e4m3fn) {         \
            using FP8_TYPE = at::Float8_e4m3fn;                     \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (scalar_type == at::ScalarType::Float8_e5m2) {    \
            using FP8_TYPE = at::Float8_e5m2;                       \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        TORCH_CHECK(false, "unsupported fp8 type\n");               \
    }()

void launch_paged_mqa_logits_fp8(const torch::Tensor& q, const torch::Tensor& kv_cache,
                                 const torch::Tensor& kv_cache_scales, const torch::Tensor& weights,
                                 const torch::Tensor& context_lens, const torch::Tensor& block_table,
                                 torch::Tensor& logits, int64_t aligned_max_context_len,
                                 int64_t launch_tokens, bool clean_logits) {
    const int batch_size = q.size(0), next_n = q.size(1), num_heads = q.size(2);
    const int num_kv_blocks = kv_cache.size(0), block_kv = kv_cache.size(1);
    const auto kv_cache_stride_bytes = kv_cache.stride(0), block_table_stride = block_table.stride(0);
    constexpr int num_math_warp_groups = 4, mmac_row_tile = 16, kTokensPerWarp = 64;
    constexpr int kTokensPerCta = num_math_warp_groups * kTokensPerWarp;
    // The regular native-FP8 kernel reuses the first 8 KiB for Q, metadata, scales, and wave-private KV tiles. Its
    // 1 KiB result scratch starts at offset 8 KiB, so no LDS above 9 KiB is addressable by this kernel.
    constexpr int fp8_regular_tile_smem_size = num_math_warp_groups * mmac_row_tile * 128 * sizeof(uint8_t);
    constexpr int fp8_regular_result_smem_size = kTokensPerCta * static_cast<int>(sizeof(float));
    constexpr int fp8_regular_smem_size = fp8_regular_tile_smem_size + fp8_regular_result_smem_size;
    // B4/B8 reuse their Q/weight/scale cache as a two-stage KV pipe after the first CTA barrier.
    // Four waves * two stages * 16 tokens * 128 FP8 bytes = 16 KiB, not two disjoint 16 KiB regions.
    constexpr int fp8_small_smem_size = num_math_warp_groups * 2 * mmac_row_tile * 128;
    dim3 grid(ceil_div(launch_tokens, int64_t{kTokensPerCta}), batch_size);
    const dim3 block(num_math_warp_groups * WARP_SIZE_GPU);
    const auto stream = at::hip::getCurrentHIPStream();
    DISPATCH_FP8_TYPES(q.scalar_type(), FP8_TYPE, [&] {
        DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
            DISPATCH_NEXT_N(next_n, NEXT_N, [&] {
                DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                    if constexpr (BLOCK_KV < 16) {
                        paged_mqa_logits_fp8_small_block<
                            FP8_TYPE, NEXT_N, NUM_HEADS, 128, BLOCK_KV, kTokensPerWarp, num_math_warp_groups>
                            <<<grid, block, fp8_small_smem_size, stream>>>(
                                reinterpret_cast<FP8_TYPE*>(q.data_ptr()),
                                reinterpret_cast<FP8_TYPE*>(kv_cache.data_ptr()),
                                kv_cache_scales.data_ptr<float>(), weights.data_ptr<float>(),
                                batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride,
                                context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                block_table.data_ptr<int32_t>(), clean_logits);
                    } else {
                        constexpr int split_kv = BLOCK_KV * 4;
                        paged_mqa_logits_fp8<
                            FP8_TYPE, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv,
                            num_math_warp_groups, kTokensPerWarp>
                            <<<paged_mqa_persistent_grid<NEXT_N>(grid, (paged_mqa_logits_fp8<
                            FP8_TYPE, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv,
                            num_math_warp_groups, kTokensPerWarp>), block.x, fp8_regular_smem_size, q.get_device(), clean_logits), block, fp8_regular_smem_size, stream>>>(
                                reinterpret_cast<FP8_TYPE*>(q.data_ptr()),
                                reinterpret_cast<FP8_TYPE*>(kv_cache.data_ptr()),
                                kv_cache_scales.data_ptr<float>(), weights.data_ptr<float>(),
                                batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride,
                                context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                block_table.data_ptr<int32_t>(), clean_logits);
                    }
                });
            });
        });
    });
}

using PagedFp4Fp8x8 = int __attribute__((ext_vector_type(2)));
struct PagedFp4Fp8 {
    using Fragment = PagedFp4Fp8x8;
    static constexpr int Steps = 4;
    static constexpr int ConvertSteps = 1;
    __device__ static int lo(unsigned x) {
        return ::min(::min(x & 255, (x >> 8) & 255), ::min((x >> 16) & 255, x >> 24));
    }
    __device__ static int hi(unsigned x) {
        return ::max(::max(x & 255, (x >> 8) & 255), ::max((x >> 16) & 255, x >> 24));
    }
    __device__ static int pivot(unsigned x) {
        return ::min(hi(x), lo(x) + 5);
    }
    // Fold each head's positive Q exponent into its weight once. The K
    // exponent is common to the head reduction and is restored only once.
    // Bound weights so this rearrangement cannot overflow its intermediates.
    __device__ static bool normal_weight(float w) {
        const float magnitude = fabsf(w);
        return magnitude == 0.f || (magnitude >= 0x1p-48f && magnitude <= 0x1p48f);
    }
    __device__ static float prepare_weight(float w, int q) {
        return ldexpf(w, q - 127);
    }
    __device__ static float finish(float value, int k) {
        return ldexpf(value, k - 127);
    }
    __device__ static bool normal(unsigned x) {
        return lo(x) >= 96 && hi(x) <= 160 && hi(x) - lo(x) <= 11;
    }
    __device__ static Fragment convert(PagedFp4Uint4 bits, int step, int e, int pivot) {
        const int delta = pivot - e;
        const unsigned word = bits[step];
        const unsigned hi = 0x4c484440u - unsigned(delta) * 0x08080808u;
        const unsigned lo = 0x3c383000u - unsigned(delta) * 0x08080800u;
        const unsigned even = __builtin_amdgcn_perm(hi, lo, word & 0x07070707u) | ((word & 0x08080808u) << 4);
        const unsigned odd = __builtin_amdgcn_perm(hi, lo, (word >> 4) & 0x07070707u) | (word & 0x80808080u);
        return {int(__builtin_amdgcn_perm(odd, even, 0x05010400u)), int(__builtin_amdgcn_perm(odd, even, 0x07030602u))};
    }
    __device__ static void convert_block(PagedFp4Uint4 bits, int step, int exp, int pivot, Fragment* out) {
        out[0] = convert(bits, step, exp, pivot);
    }
    __device__ static PagedFp4Float4 mma(Fragment k, Fragment q, PagedFp4Float4 c) {
        return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(k, q, c, false, false);
    }
};
void launch_paged_mqa_fp4_fp8(PagedMqaFp4Args a, int device, hipStream_t stream) {
    launch_paged_mqa_fp4<PagedFp4Fp8>(a, device, stream);
}

} // namespace deepgemm
