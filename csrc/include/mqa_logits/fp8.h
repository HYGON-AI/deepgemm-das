// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <torch/extension.h>

namespace deepgemm::mqa::fp8 {
// Internal Tensor adapter selected for FP8 Q: dense output, or compact
// output when max_seqlen_k is given (D_out is then unsupported).
torch::Tensor mqa_logits_fp8(const torch::Tensor &Q, const torch::Tensor &K,
                             const torch::Tensor &weights, const torch::Tensor &ks,
                             const torch::Tensor &ke, int m, int n, int h, int d,
                             const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
                             const torch::optional<torch::Tensor> &out,
                             const torch::optional<int64_t> &max_seqlen_k = torch::nullopt);
} // namespace deepgemm::mqa::fp8
