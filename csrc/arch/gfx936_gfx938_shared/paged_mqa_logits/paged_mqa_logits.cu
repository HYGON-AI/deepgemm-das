// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "paged_mqa_logits/cta_schedule.h"
#include "paged_mqa_logits/paged_mqa_logits.h"
#include "paged_mqa_logits_dispatch.hpp"
#include "paged_mqa_logits_impl.hpp"
#include "paged_mqa_logits_fp4_impl.hpp"
#include "paged_mqa_logits_int8_impl.hpp"

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "ATen/Dispatch.h"
#include <algorithm>


namespace deepgemm {

torch::Tensor paged_mqa_logits_native(const torch::Tensor& q,
                                      const torch::Tensor& fused_kv_cache,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& context_lens,
                                      const torch::Tensor& block_table,
                                      const int& max_context_len,
                                      const bool& clean_logits,
                                      std::optional<torch::Tensor> supplied_output) {

    // Declare these variables explicitly to avoid mis-binding in kernel launch lambdas
    int batch_size, next_n, num_heads, head_dim;
    int num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf;

    std::tie(batch_size, next_n, num_heads, head_dim) = get_shape<4>(q);
    std::tie(num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf) = get_shape<4>(fused_kv_cache);
    const auto& [batch_size_] = get_shape<1>(context_lens);
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [batch_size__, max_block_len] = get_shape<2>(block_table);
    const auto& kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const auto& block_table_stride = block_table.stride(0);

    TORCH_CHECK(batch_size == batch_size_ and batch_size == batch_size__);
    TORCH_CHECK(batch_size_next_n == batch_size * next_n);
    TORCH_CHECK(num_heads == num_heads_ and num_heads_kv == 1);

    TORCH_CHECK(next_n >= 1 and next_n <= 5);
    TORCH_CHECK(block_kv == 4 or block_kv == 8 or block_kv == 16 or block_kv == 32 or block_kv == 64);

    TORCH_CHECK(q.is_contiguous());
    TORCH_CHECK(kv_cache_stride_bytes % sizeof(float) == 0);
    TORCH_CHECK(fused_kv_cache.stride(1) == head_dim_with_sf);
    TORCH_CHECK(fused_kv_cache.stride(2) == head_dim_with_sf);
    TORCH_CHECK(fused_kv_cache.stride(3) == 1);
    TORCH_CHECK(weights.is_contiguous());
    TORCH_CHECK(context_lens.is_contiguous());
    TORCH_CHECK(block_table.stride(1) == 1);

    TORCH_CHECK(weights.scalar_type() == torch::kFloat);
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32);

    torch::Tensor kv_cache_scales;
    torch::Tensor kv_cache;
    const bool q_is_fp8 = q.scalar_type() == torch::kFloat8_e4m3fn || q.scalar_type() == torch::kFloat8_e5m2;
    const bool kv_is_fp8 = fused_kv_cache.scalar_type() == torch::kByte;
    const bool q_is_int8 = q.scalar_type() == torch::kInt8;
    const bool kv_is_int8 = fused_kv_cache.scalar_type() == torch::kInt8;
    const bool kv_is_quantized = kv_is_fp8 || kv_is_int8;

    if (kv_is_quantized) {
        TORCH_CHECK(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
        TORCH_CHECK(kv_is_int8 ? q_is_int8 : (q_is_fp8 || q.scalar_type() == torch::kBFloat16));
        const auto kv_dtype = kv_is_int8 ? torch::kInt8 : torch::kFloat8_e4m3fn;

        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(kv_dtype)
        );

        kv_cache_scales = torch::from_blob(
            reinterpret_cast<uint8_t*>(fused_kv_cache.data_ptr()) + block_kv * head_dim,
            {num_kv_blocks, block_kv},
            {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
    } else {
        TORCH_CHECK(!q_is_fp8 && !q_is_int8);
        TORCH_CHECK(fused_kv_cache.scalar_type() == torch::kFloat16 ||
                            fused_kv_cache.scalar_type() == torch::kBFloat16);
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(torch::kFloat16)
        );
    }

    constexpr int num_math_warp_groups = 4;
    constexpr int mmac_row_tile = 16;
    constexpr int kTokensPerWarp = 64;
    constexpr int kTokensPerCta = kTokensPerWarp * num_math_warp_groups;

    const bool use_bf16_q_fp8_kv = kv_is_fp8 && q.scalar_type() == torch::kBFloat16;
    // DeepGemm's supplied-output contract is contiguous with the exact requested width.
    // The regular kernels mask the final tile, so they can write it directly.
    // The small-block kernels retain their original padded-output allocation.
    TORCH_CHECK(!supplied_output || block_kv >= mmac_row_tile,
                "deep_gemm: supplied output requires BLOCK_KV >= 16");
    const int64_t output_alignment = block_kv < mmac_row_tile ? kTokensPerCta : block_kv;
    const int64_t aligned_max_context_len = supplied_output ? max_context_len :
        align(static_cast<int64_t>(max_context_len), output_alignment);

    // Both variants initialize their required positions in the kernel.
    torch::Tensor logits = supplied_output ? *supplied_output :
        torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
    if (!supplied_output) {
        logits = logits.slice(-1, 0, max_context_len);
    }

    // Shared memory required by the Q cache and KV pipeline.
    // for fp8: prefetch_seq_kv_stage = 2, k_tile = 64;
    // for fp16: prefetch_seq_kv_stage = 1, k_tile = 32;
    // smem_per_warp = prefetch_seq_kv_stage * head_dim * seq_kv_tile(16) * sizeof(T) = 32 * head_dim
    const int smem_kv_pipe_size = num_math_warp_groups * 32 * head_dim;
    const int smem_q_cache_size = num_heads * head_dim * static_cast<int>(q.element_size());
    // The BF16/FP16 kernels reuse the Q cache for weights when a full four-wave H64 Q tile fills LDS. Quantized-Q
    // kernels retain Q, weights, and scales together until all three have been consumed.
    const bool weights_reuse_q_cache = !q_is_fp8 && !q_is_int8 && num_heads >= mmac_row_tile * num_math_warp_groups;
    const int smem_weight_cache_size =
        weights_reuse_q_cache ? 0 : num_heads * static_cast<int>(sizeof(float));
    const int smem_scale_cache_size =
        (q_is_fp8 || q_is_int8) ? num_math_warp_groups * 64 * static_cast<int>(sizeof(float)) : 0;
    const int smem_cache_size = smem_q_cache_size + smem_weight_cache_size + smem_scale_cache_size;

    const int smem_size =
        (block_kv < 16 ? (smem_kv_pipe_size > smem_cache_size ? smem_kv_pipe_size : smem_cache_size)
                       : smem_kv_pipe_size);
    // The page-table shape is a fixed upper bound even when graph replay changes
    // device context lengths. Output stride/width and backend selection stay independent.
    const int64_t launch_tokens = clean_logits ? max_context_len :
        std::min<int64_t>(max_context_len, static_cast<int64_t>(max_block_len) * block_kv);
    const int max_chunk_size = ceil_div(launch_tokens, int64_t{kTokensPerCta});

    // Dispatch implementation
    dim3 grid = {(uint32_t)max_chunk_size, (uint32_t)batch_size, 1};
    const dim3 multi_next_grid = {(uint32_t)max_chunk_size, (uint32_t)batch_size, 1};
    const auto make_multi_next_legacy_grid = [&](int dispatched_next_n) {
        return dim3{(uint32_t)(max_chunk_size * dispatched_next_n), (uint32_t)batch_size, 1};
    };
    dim3 block = {(uint32_t)(num_math_warp_groups * WARP_SIZE_GPU), 1, 1};

    const auto stream = at::hip::getCurrentHIPStream();

    if (!kv_is_quantized) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                q.scalar_type(),
                "paged_mqa_float16",
                ([&] {
                    DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
                        DISPATCH_MULTI_NEXT_N(next_n, NEXT_N, [&] {
                            const dim3 legacy_grid = make_multi_next_legacy_grid(NEXT_N);
                            DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                                constexpr bool kPageMajor = BLOCK_KV >= 16;
                                constexpr bool kSmallHead = NUM_HEADS <= 16;
                                constexpr bool kUseRegKv =
                                    kPageMajor && kSmallHead && (NEXT_N == 3 || NEXT_N == 4);
                                constexpr bool kUseLdsKv =
                                    kPageMajor && NEXT_N >= 2 && (NEXT_N <= 3 || kSmallHead);
                                // The one-CTA multi-next launch regresses stably for
                                // N2/H64, so this new optimization keeps that shape
                                // on the pre-existing legacy path.
                                constexpr bool kN2WideHeadNeedsLegacy =
                                    NEXT_N == 2 && NUM_HEADS == 64;
                                constexpr bool kUseLdsKvMultiNext =
                                    kUseLdsKv && !kN2WideHeadNeedsLegacy;

                                if constexpr (kUseRegKv) {
                                    dim3 multi_next_block = {(uint32_t)(4 * WARP_SIZE_GPU), 1, 1};
                                    paged_mqa_logits_multi_next_reg_qw<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, true>
                                        <<<multi_next_grid, multi_next_block, 0, stream>>>(
                                            reinterpret_cast<scalar_t*>(q.data_ptr()),
                                            reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                            weights.data_ptr<float>(), batch_size,
                                            num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                            block_table_stride, context_lens.data_ptr<int32_t>(),
                                            logits.data_ptr<float>(),
                                            block_table.data_ptr<int32_t>(),
                                            clean_logits);
                                } else if constexpr (kUseLdsKvMultiNext) {
                                    dim3 multi_next_block = {(uint32_t)(4 * WARP_SIZE_GPU), 1, 1};
                                    constexpr int multi_next_smem_size = 4 * 16 * 128 * sizeof(scalar_t);
                                    paged_mqa_logits_multi_next_reg_qw<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV>
                                        <<<multi_next_grid, multi_next_block, multi_next_smem_size, stream>>>(
                                            reinterpret_cast<scalar_t*>(q.data_ptr()),
                                            reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                            weights.data_ptr<float>(), batch_size,
                                            num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                            block_table_stride, context_lens.data_ptr<int32_t>(),
                                            logits.data_ptr<float>(),
                                            block_table.data_ptr<int32_t>(),
                                            clean_logits);
                                } else if constexpr (BLOCK_KV < 16) {
                                    paged_mqa_logits_small_block<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV,
                                        kTokensPerWarp, num_math_warp_groups>
                                        <<<legacy_grid, block, smem_size, stream>>>(
                                        reinterpret_cast<scalar_t*>(q.data_ptr()),
                                        reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                        weights.data_ptr<float>(), batch_size,
                                        num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                        block_table_stride, context_lens.data_ptr<int32_t>(),
                                        logits.data_ptr<float>(),
                                        block_table.data_ptr<int32_t>(),
                                        clean_logits);
                                } else {
                                    constexpr int split_kv = BLOCK_KV * 4;
                                    paged_mqa_logits<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv,
                                        num_math_warp_groups, kTokensPerWarp>
                            <<<paged_mqa_persistent_grid<NEXT_N>(legacy_grid, (paged_mqa_logits<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv,
                                        num_math_warp_groups, kTokensPerWarp>), block.x, smem_size, q.get_device(), clean_logits), block, smem_size, stream>>>(
                                        reinterpret_cast<scalar_t*>(q.data_ptr()),
                                        reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                        weights.data_ptr<float>(), batch_size,
                                        num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                        block_table_stride, context_lens.data_ptr<int32_t>(),
                                        logits.data_ptr<float>(),
                                        block_table.data_ptr<int32_t>(),
                                        clean_logits);
                                }
                            });
                        });
                    });
                })
        );
    } else if (use_bf16_q_fp8_kv) {
        const int bf16_fp8_num_kv_stages = num_heads == 64 ? 2 : 1;
        const int bf16_fp8_raw_kv_smem_size = bf16_fp8_num_kv_stages * num_math_warp_groups * mmac_row_tile * head_dim;
        const int bf16_fp8_q_smem_size = num_heads * head_dim * static_cast<int>(sizeof(at::BFloat16));
        const int bf16_fp8_cache_smem_size = num_heads >= mmac_row_tile * num_math_warp_groups
                                                 ? bf16_fp8_q_smem_size
                                                 : bf16_fp8_q_smem_size + smem_weight_cache_size;
        const int bf16_fp8_smem_size =
            bf16_fp8_raw_kv_smem_size > bf16_fp8_cache_smem_size ? bf16_fp8_raw_kv_smem_size : bf16_fp8_cache_smem_size;

        DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
            DISPATCH_NEXT_N(next_n, NEXT_N, [&] {
                DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                    if constexpr (BLOCK_KV < 16) {
                        paged_mqa_logits_bf16_fp8_kv_small_block<NEXT_N, NUM_HEADS, BLOCK_KV,
                                                                 kTokensPerWarp, num_math_warp_groups>
                            <<<grid, block, bf16_fp8_smem_size, stream>>>(
                                reinterpret_cast<at::BFloat16*>(q.data_ptr()),
                                reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), kv_cache_stride_bytes, aligned_max_context_len,
                                block_table_stride, context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                block_table.data_ptr<int32_t>(), clean_logits);
                    } else {
                        // One CTA shares the decoded KV tile across all next queries,
                        // for both runtime clean modes. R=1 keeps its original LDS size;
                        // multi-next operands are loaded directly into registers.
                        const int regular_smem_size = NEXT_N == 1 ? bf16_fp8_smem_size : 0;
                        paged_mqa_logits_bf16_fp8_kv<NEXT_N, NUM_HEADS, BLOCK_KV, num_math_warp_groups,
                                                     kTokensPerWarp>
                            <<<paged_mqa_persistent_grid<NEXT_N>(multi_next_grid, (paged_mqa_logits_bf16_fp8_kv<NEXT_N, NUM_HEADS, BLOCK_KV, num_math_warp_groups,
                                                     kTokensPerWarp>), block.x, regular_smem_size, q.get_device(), clean_logits), block, regular_smem_size, stream>>>(
                                reinterpret_cast<at::BFloat16*>(q.data_ptr()),
                                reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), kv_cache_stride_bytes, aligned_max_context_len,
                                block_table_stride, context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                block_table.data_ptr<int32_t>(), clean_logits, batch_size);
                    }
                });
            });
        });
    } else if (kv_is_int8) {
        DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
            DISPATCH_MULTI_NEXT_N(next_n, NEXT_N, [&] {
                const dim3 legacy_grid = make_multi_next_legacy_grid(NEXT_N);
                DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                    // R=1 uses the original entry with runtime persistent scheduling. H8/H16
                    // multi-next cases use Register-QW: it loads every KV tile once
                    // and reuses it for every next query without LDS Q/weight reads.
                    // B4/B8 and H32/H64 retain their established kernels.
                    constexpr bool kUseRegisterQwMultiNextKernel =
                        NEXT_N >= 2 && BLOCK_KV >= 16 && NUM_HEADS <= 16;
                    if constexpr (kUseRegisterQwMultiNextKernel) {
                        dim3 multi_next_block = {(uint32_t)(4 * WARP_SIZE_GPU), 1, 1};
                        paged_mqa_logits_int8_multi_next_reg_qw<int8_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV>
                            <<<multi_next_grid, multi_next_block, 0, stream>>>(
                                q.data_ptr<int8_t>(), kv_cache.data_ptr<int8_t>(), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride, context_lens.data_ptr<int32_t>(),
                                logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), clean_logits);
                    } else if constexpr (BLOCK_KV < 16) {
                        paged_mqa_logits_int8_small_block<
                            int8_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, kTokensPerWarp, num_math_warp_groups>
                            <<<legacy_grid, block, smem_size, stream>>>(
                                q.data_ptr<int8_t>(), kv_cache.data_ptr<int8_t>(), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride, context_lens.data_ptr<int32_t>(),
                                logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), clean_logits);
                    } else {
                        constexpr int split_kv = BLOCK_KV * num_math_warp_groups;
                        paged_mqa_logits_int8<
                            int8_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv, num_math_warp_groups, kTokensPerWarp>
                            <<<paged_mqa_persistent_grid<NEXT_N>(legacy_grid, (paged_mqa_logits_int8<
                            int8_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv, num_math_warp_groups, kTokensPerWarp>), block.x, smem_size, q.get_device(), clean_logits), block, smem_size, stream>>>(
                                q.data_ptr<int8_t>(), kv_cache.data_ptr<int8_t>(), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride, context_lens.data_ptr<int32_t>(),
                                logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), clean_logits);
                    }
                });
            });
        });
    } else {
#ifdef DEEPGEMM_ENABLE_GFX938
        launch_paged_mqa_logits_fp8(q, kv_cache, kv_cache_scales, weights, context_lens,
                                    block_table, logits, aligned_max_context_len, launch_tokens, clean_logits);
#else
        TORCH_CHECK(false, "paged_mqa_logits: native FP8 kernels were not built for gfx938");
#endif
    }

    const hipError_t result = hipGetLastError();
    TORCH_CHECK(result == hipSuccess, "paged_mqa_logits native launch: ", hipGetErrorString(result));

    return logits;
}

// Shape-fixed scan: device lengths may change on every graph replay.
static __global__ void paged_mqa_fp4_metadata_kernel(const int* lens, int* schedule, int rows, int n, int ctas) {
    // Two routes per uint64 node retain the 64 KB LDS ceiling while avoiding
    // overflow when many long contexts contribute more than 2^32 splits.
    extern __shared__ uint64_t prefix[];
    const bool paired = n > 1 && rows / n >= 8;
    const int routes = paired ? rows / n * ((n + 1) / 2) : rows;
    const int pairs = (routes + 1) / 2;
    int size = 1;
    while (size < pairs)
        size <<= 1;
    const int tid = threadIdx.x;
    for (int i = tid; i < size; i += 256) {
        uint64_t count = 0;
        for (int route = 2 * i; route < ::min(2 * i + 2, routes); ++route)
            count += (uint64_t(paged_fp4_route_length(lens, route, n, paired)) + 255) / 256;
        prefix[i] = count;
    }
    __syncthreads();
    for (int step = 1; step < size; step <<= 1) {
        for (int i = (tid + 1) * 2 * step - 1; i < size; i += 512 * step)
            prefix[i] += prefix[i - step];
        __syncthreads();
    }
    const uint64_t total = prefix[size - 1];
    __syncthreads();
    if (tid == 0)
        prefix[size - 1] = 0;
    __syncthreads();
    for (int step = size / 2; step; step >>= 1) {
        for (int i = (tid + 1) * 2 * step - 1; i < size; i += 512 * step) {
            const uint64_t left = prefix[i - step];
            prefix[i - step] = prefix[i];
            prefix[i] += left;
        }
        __syncthreads();
    }
    for (int i = tid; i <= ctas; i += 256) {
        const uint64_t target = uint64_t(i) * (total / ctas) + ::min(uint64_t(i), total % ctas);
        int lo = 0, hi = pairs;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            const uint64_t end = mid + 1 < pairs ? prefix[mid + 1] : total;
            if (end <= target)
                lo = mid + 1;
            else
                hi = mid;
        }
        int route = routes;
        uint64_t offset = 0;
        if (lo < pairs) {
            route = 2 * lo;
            offset = target - prefix[lo];
            const uint64_t first = (uint64_t(paged_fp4_route_length(lens, route, n, paired)) + 255) / 256;
            if (offset >= first) {
                ++route;
                offset -= first;
            }
        }
        schedule[2 * i] = route;
        schedule[2 * i + 1] = offset;
    }
}

void launch_paged_mqa_fp4_metadata(const int* lens, int* schedule, int rows, int n, int ctas, hipStream_t stream) {
    const bool paired = n > 1 && rows / n >= 8;
    const int routes = paired ? rows / n * ((n + 1) / 2) : rows;
    int size = 1;
    while (size < (routes + 1) / 2)
        size <<= 1;
    paged_mqa_fp4_metadata_kernel<<<1, 256, size * sizeof(uint64_t), stream>>>(lens, schedule, rows, n, ctas);
}

#if defined(DEEPGEMM_ENABLE_GFX936)
void launch_paged_mqa_fp4_fp16(PagedMqaFp4Args a, int device, hipStream_t stream) {
    launch_paged_mqa_fp4<PagedFp4Fp16>(a, device, stream);
}
#endif

} // namespace deepgemm
