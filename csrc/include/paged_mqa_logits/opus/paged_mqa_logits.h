// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include <torch/extension.h>
#include <optional>

namespace deepgemm::opus {

torch::Tensor paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& cache,
                               const torch::Tensor& weights, const torch::Tensor& context,
                               const torch::Tensor& tables, std::optional<torch::Tensor> output,
                               int64_t max_len, int64_t kernel_id, bool clean_logits);

void launch_paged_mqa_logits_fp8(const torch::Tensor& q, const torch::Tensor& cache,
                                 const torch::Tensor& weights, const torch::Tensor& context,
                                 const torch::Tensor& tables, torch::Tensor& output,
                                 int max_len, int kernel_id, bool clean_logits);

} // namespace deepgemm::opus
