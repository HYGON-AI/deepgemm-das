// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <torch/extension.h>

namespace deepgemm::mqa::bf16 {
// Internal Tensor adapter; unsupported architectures raise a validation error.
// Without max_seqlen_k: dense [M,N] output (optional D_out, clean_logits).
// With max_seqlen_k=W: compact [M,align(W,256)] output returned as [:,:W];
// D_out must be absent and clean_logits is ignored.
torch::Tensor mqa_logits_bf16(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
    const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
    const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
    const torch::optional<torch::Tensor> &out,
    const torch::optional<int64_t> &max_seqlen_k = torch::nullopt);
} // namespace deepgemm::mqa::bf16
