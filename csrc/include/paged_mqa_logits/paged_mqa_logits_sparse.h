// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/extension.h>

namespace deepgemm {

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_native(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas);

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped_native(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas,
    const int& group_size);

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped_int8_native(
    const torch::Tensor& q,
    const torch::Tensor& packed_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const int& max_context_len,
    const bool& clean_logits,
    const int& num_warps,
    const int& persistent_ctas,
    const int& group_size);

void launch_paged_mqa_logits_grouped_fp8(
    int group_size, const torch::Tensor& q, const torch::Tensor& kv_cache, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_table, torch::Tensor& logits,
    torch::Tensor& nonzero_mask, int persistent_ctas, hipStream_t stream);

}  // namespace deepgemm
