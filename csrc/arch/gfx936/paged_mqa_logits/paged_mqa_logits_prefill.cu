// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "paged_mqa_logits/paged_mqa_logits_prefill.h"
#include "paged_mqa_logits_prefill_int8_impl.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <limits>

namespace deepgemm {

#define DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, ...)               \
    [&]() {                                                         \
        if (num_heads == 8) {                                       \
            constexpr int NUM_HEADS = 8;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 16) {                               \
            constexpr int NUM_HEADS = 16;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 32) {                               \
            constexpr int NUM_HEADS = 32;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 64) {                               \
            constexpr int NUM_HEADS = 64;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        TORCH_CHECK(false, "unsupported head size\n");              \
    }()

#define DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, ...)                  \
    [&]() {                                                         \
        if (block_kv == 4) {                                        \
            constexpr int BLOCK_KV = 4;                             \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 8) {                                 \
            constexpr int BLOCK_KV = 8;                             \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 16) {                                \
            constexpr int BLOCK_KV = 16;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 32) {                                \
            constexpr int BLOCK_KV = 32;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 64) {                                \
            constexpr int BLOCK_KV = 64;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        TORCH_CHECK(false, "unsupported block_kv size\n");          \
    }()


torch::Tensor paged_mqa_logits_prefill_native(
    const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int64_t max_context_len, bool clean_logits) {
    TORCH_CHECK(q.is_cuda(), "prefix prefill requires GPU tensors");
    const c10::cuda::CUDAGuard guard(q.device());
    for (const auto* tensor : {&fused_kv_cache, &weights, &context_lens, &block_table}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->device() == q.device(),
                    "prefix prefill inputs must be on the same GPU");
    }
    TORCH_CHECK(q.dim() == 4 && fused_kv_cache.dim() == 4 && weights.dim() == 2 &&
                context_lens.dim() == 1 && block_table.dim() == 2, "invalid prefix prefill tensor rank");
    const bool fp8 = q.scalar_type() == at::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(q.scalar_type() == torch::kInt8 || fp8,
                "prefix prefill Q must be INT8 or E4M3FN FP8");
    TORCH_CHECK(fused_kv_cache.scalar_type() == (fp8 ? torch::kUInt8 : torch::kInt8),
                fp8 ? "FP8 prefix prefill requires packed uint8 KV cache"
                    : "INT8 prefix prefill requires packed INT8 KV cache");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32 && context_lens.scalar_type() == torch::kInt32 &&
                block_table.scalar_type() == torch::kInt32, "invalid prefix prefill metadata dtype");
    const int64_t tokens = q.size(0), heads = q.size(2), block_kv = fused_kv_cache.size(1);
    TORCH_CHECK(q.size(1) == 1 && q.size(3) == 128, "prefix prefill requires next_n=1 and head_dim=128");
    TORCH_CHECK(heads == 8 || heads == 16 || heads == 32 || heads == 64, "unsupported prefix prefill heads");
    TORCH_CHECK(block_kv == 4 || block_kv == 8 || block_kv == 16 || block_kv == 32 || block_kv == 64,
                "unsupported prefix prefill page size");
    TORCH_CHECK(fused_kv_cache.size(2) == 1 && fused_kv_cache.size(3) == 132,
                "packed prefix prefill cache must have shape [pages, block_kv, 1, 132]");
    TORCH_CHECK(weights.size(0) == tokens && weights.size(1) == heads &&
                context_lens.size(0) == tokens && block_table.size(0) == tokens,
                "inconsistent prefix prefill token/head dimensions");
    TORCH_CHECK(max_context_len >= 0 && max_context_len <= std::numeric_limits<int>::max() &&
                tokens <= std::numeric_limits<int>::max() &&
                fused_kv_cache.size(0) <= std::numeric_limits<int>::max(), "prefix prefill dimensions out of range");
    TORCH_CHECK(block_table.size(1) >= (max_context_len + block_kv - 1) / block_kv,
                "prefix prefill page table does not cover max_context_len");
    TORCH_CHECK(max_context_len == 0 || tokens == 0 || fused_kv_cache.size(0) > 0,
                "prefix prefill requires physical KV pages");
    TORCH_CHECK(q.is_contiguous() && fused_kv_cache.is_contiguous() && weights.is_contiguous() &&
                context_lens.is_contiguous() && block_table.stride(1) == 1,
                "prefix prefill requires contiguous Q/cache/weights/lengths and contiguous page-table columns");
    for (const auto* tensor : {&q, &fused_kv_cache, &weights}) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                    "prefix prefill Q/cache/weights pointers must be 16-byte aligned");
    }
    hipDeviceProp_t properties;
    const auto property_status = hipGetDeviceProperties(&properties, q.get_device());
    TORCH_CHECK(property_status == hipSuccess, hipGetErrorString(property_status));
    const std::string architecture(properties.gcnArchName);
    const auto arch = architecture.substr(0, architecture.find(':'));
    TORCH_CHECK(arch == "gfx936" || arch == "gfx938",
                "prefix prefill requires gfx936/gfx938");
    if (fp8) {
        TORCH_CHECK(arch == "gfx938", "FP8 prefix prefill requires gfx938");
#if !defined(DEEPGEMM_ENABLE_GFX938)
        TORCH_CHECK(false, "FP8 prefix prefill was not built for gfx938");
#endif
    }

    // No GPU metadata is read on the host. In particular, dispatch and grid
    // never depend on context_lens values, so in-place graph updates are safe.
    auto logits = torch::empty({tokens, max_context_len}, q.options().dtype(torch::kFloat32));
    if (tokens == 0 || max_context_len == 0) return logits;
    (void)clean_logits;  // Every output, including the causal tail, is written.
    if (fp8) {
#if defined(DEEPGEMM_ENABLE_GFX938)
        detail::launch_paged_mqa_logits_prefill_fp8(
            q, fused_kv_cache, weights, context_lens, block_table, logits, max_context_len);
#endif
    } else {
        const dim3 grid((max_context_len + 255) / 256, (tokens + 31) / 32);
        const auto stream = at::cuda::getCurrentCUDAStream();
        DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
            DISPATCH_NUM_HEADS(heads, NUM_HEADS, [&] {
                paged_mqa_logits_prefill_int8_kernel<NUM_HEADS, BLOCK_KV>
                    <<<grid, 256, 0, stream>>>(
                        q.data_ptr<int8_t>(), fused_kv_cache.data_ptr<int8_t>(), weights.data_ptr<float>(),
                        context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(), logits.data_ptr<float>(),
                        static_cast<int>(tokens), static_cast<int>(max_context_len),
                        static_cast<int>(fused_kv_cache.size(0)));
            });
        });
    }
    const auto launch_status = hipGetLastError();
    TORCH_CHECK(launch_status == hipSuccess, hipGetErrorString(launch_status));
    return logits;
}

}  // namespace deepgemm
