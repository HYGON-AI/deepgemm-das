// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace deepgemm {

torch::Tensor paged_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int64_t max_model_len,
    std::optional<torch::Tensor> out,
    bool clean_logits,
    const std::string& implementation,
    int64_t kernel_id,
    bool is_prefill);

std::vector<torch::Tensor> page_mqa_logits_sparse_mask(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int64_t max_model_len,
    bool clean_logits,
    int64_t num_warps,
    int64_t persistent_ctas);

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int64_t max_model_len,
    bool clean_logits,
    int64_t num_warps,
    int64_t persistent_ctas,
    int64_t group_size);

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped_int8(
    const torch::Tensor& q,
    const torch::Tensor& packed_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int64_t max_model_len,
    bool clean_logits,
    int64_t num_warps,
    int64_t persistent_ctas,
    int64_t group_size);

torch::Tensor get_paged_mqa_logits_fp4_metadata(
    const torch::Tensor& context_lens, int64_t block_kv, std::optional<int64_t> num_sms, int64_t rows_per_request);
torch::Tensor fp8_fp4_paged_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& q_sf,
    const torch::Tensor& kv,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const std::optional<torch::Tensor>& schedule_meta,
    int64_t max_context_len,
    int64_t table_block_size,
    const std::optional<torch::Tensor>& output,
    int64_t rows_per_request);

} // namespace deepgemm
