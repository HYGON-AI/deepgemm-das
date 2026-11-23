// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include <torch/extension.h>

namespace deepgemm {
// Non-paged MQA. Tensor validation and output ownership remain in each dtype adapter.
torch::Tensor mqa_logits(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
    const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
    const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
    const torch::optional<torch::Tensor> &out);
torch::Tensor mqa_logits_compact(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
    const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
    const torch::optional<torch::Tensor> &kv_scale, int64_t max_seqlen_k);
} // namespace deepgemm
