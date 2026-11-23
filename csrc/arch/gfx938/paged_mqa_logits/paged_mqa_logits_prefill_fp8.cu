// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "paged_mqa_logits/paged_mqa_logits_prefill.h"
#include "paged_mqa_logits_prefill_fp8_impl.hpp"

namespace deepgemm {
namespace detail {

#define DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, ...)                          \
    [&]() {                                                                    \
        if (num_heads == 8) {                                                  \
            constexpr int NUM_HEADS = 8;                                       \
            __VA_ARGS__();                                                     \
            return;                                                            \
        } else if (num_heads == 16) {                                          \
            constexpr int NUM_HEADS = 16;                                      \
            __VA_ARGS__();                                                     \
            return;                                                            \
        } else if (num_heads == 32) {                                          \
            constexpr int NUM_HEADS = 32;                                      \
            __VA_ARGS__();                                                     \
            return;                                                            \
        } else if (num_heads == 64) {                                          \
            constexpr int NUM_HEADS = 64;                                      \
            __VA_ARGS__();                                                     \
            return;                                                            \
        }                                                                      \
        TORCH_CHECK(false, "unsupported head size\n");                         \
    }()

#define DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, ...)                             \
    [&]() {                                                                    \
        if (block_kv == 4) {                                                   \
            constexpr int BLOCK_KV = 4;                                        \
            __VA_ARGS__();                                                     \
            return;                                                            \
        } else if (block_kv == 8) {                                            \
            constexpr int BLOCK_KV = 8;                                        \
            __VA_ARGS__();                                                     \
            return;                                                            \
        } else if (block_kv == 16) {                                           \
            constexpr int BLOCK_KV = 16;                                       \
            __VA_ARGS__();                                                     \
            return;                                                            \
        } else if (block_kv == 32) {                                           \
            constexpr int BLOCK_KV = 32;                                       \
            __VA_ARGS__();                                                     \
            return;                                                            \
        } else if (block_kv == 64) {                                           \
            constexpr int BLOCK_KV = 64;                                       \
            __VA_ARGS__();                                                     \
            return;                                                            \
        }                                                                      \
        TORCH_CHECK(false, "unsupported block_kv size\n");                     \
    }()

// Only launches the gfx938 FP8 kernel; the shared native entry owns dtype
// dispatch, input validation, device guard, output allocation and launch errors.
void launch_paged_mqa_logits_prefill_fp8(
    const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& block_table, torch::Tensor& logits,
    int64_t max_context_len) {
    const int64_t tokens = q.size(0), heads = q.size(2), block_kv = fused_kv_cache.size(1);
    const int tile_rows = block_kv >= 16 ? 64 : 32, tile_cols = block_kv >= 16 ? 256 : 128;
    const dim3 grid((max_context_len + tile_cols - 1) / tile_cols, (tokens + tile_rows - 1) / tile_rows);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
        DISPATCH_NUM_HEADS(heads, NUM_HEADS, [&] {
            const auto dispatch_tile = [&](auto block_tag) {
                constexpr int SELECTED_BLOCK_KV = decltype(block_tag)::value;
                const auto launch = [&](auto even_rows, auto even_cols) {
                    constexpr bool EVEN_ROWS = decltype(even_rows)::value;
                    constexpr bool EVEN_COLS = decltype(even_cols)::value;
                    paged_mqa_logits_prefill_fp8_kernel<NUM_HEADS, SELECTED_BLOCK_KV, EVEN_ROWS, EVEN_COLS>
                        <<<grid, 256, 0, stream>>>(
                            reinterpret_cast<const uint8_t*>(q.data_ptr()), fused_kv_cache.data_ptr<uint8_t>(),
                            weights.data_ptr<float>(), context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                            logits.data_ptr<float>(), static_cast<int>(tokens), static_cast<int>(max_context_len),
                            static_cast<int>(fused_kv_cache.size(0)));
                };
                if constexpr (SELECTED_BLOCK_KV >= 16) {
                    // Alignment depends only on metadata and remains fixed in a
                    // captured graph. Context/page-table values stay on device.
                    if (tokens % 64 == 0) {
                        if (max_context_len % 256 == 0) {
                            launch(std::true_type{}, std::true_type{});
                        } else {
                            launch(std::true_type{}, std::false_type{});
                        }
                    } else {
                        if (max_context_len % 256 == 0) {
                            launch(std::false_type{}, std::true_type{});
                        } else {
                            launch(std::false_type{}, std::false_type{});
                        }
                    }
                } else {
                    launch(std::false_type{}, std::false_type{});
                }
            };
            dispatch_tile(std::integral_constant<int, BLOCK_KV>{});
        });
    });
}

}  // namespace detail
}  // namespace deepgemm
