// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Adapted from tensor_scores/csrc/tensor_ops/backends/score_api.cu.
#include "mqa_logits/dispatch.h"
#include "mqa_logits/bf16.h"
#include "mqa_logits/int8.h"
#include "mqa_logits/fp8.h"

namespace deepgemm {
torch::Tensor mqa_logits(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
    const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
    const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
    const torch::optional<torch::Tensor> &out) {
    TORCH_CHECK(q.defined() && k.defined(), "mqa_logits requires defined Q and K");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "mqa_logits Q/K dtype must match");
    if (q.scalar_type() == torch::kBFloat16) {
#if defined(DEEPGEMM_ENABLE_GFX936) || defined(DEEPGEMM_ENABLE_GFX938)
        return mqa::bf16::mqa_logits_bf16(q, k, weights, ks, ke, m, n, h, d, kv_scale, clean_logits, out);
#else
        TORCH_CHECK(false, "mqa_logits BF16 was not built; enable gfx936 or gfx938");
#endif
    }
    if (q.scalar_type() == torch::kInt8) {
#if defined(DEEPGEMM_ENABLE_GFX936) || defined(DEEPGEMM_ENABLE_GFX938)
        return mqa::int8::mqa_logits_int8(q, k, weights, ks, ke, m, n, h, d, kv_scale, clean_logits, out);
#else
        TORCH_CHECK(false, "mqa_logits INT8 was not built; enable gfx936 or gfx938");
#endif
    }
    if (q.scalar_type() == torch::kFloat8_e4m3fn) {
#if defined(DEEPGEMM_ENABLE_GFX938)
        return mqa::fp8::mqa_logits_fp8(q, k, weights, ks, ke, m, n, h, d, kv_scale, clean_logits, out);
#else
        TORCH_CHECK(false, "mqa_logits FP8 was not built; enable gfx938");
#endif
    }
    TORCH_CHECK(false, "mqa_logits supports BF16, INT8 and FP8 E4M3FN Q/K only");
}

torch::Tensor mqa_logits_compact(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
    const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
    const torch::optional<torch::Tensor> &kv_scale, int64_t max_seqlen_k) {
    TORCH_CHECK(q.defined() && k.defined(), "mqa_logits requires defined Q and K");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "mqa_logits Q/K dtype must match");
    if (q.scalar_type() == torch::kFloat8_e4m3fn) {
#if defined(DEEPGEMM_ENABLE_GFX938)
        return mqa::fp8::mqa_logits_fp8(q, k, weights, ks, ke, m, n, h, d, kv_scale, false, torch::nullopt, max_seqlen_k);
#else
        TORCH_CHECK(false, "mqa_logits FP8 was not built; enable gfx938");
#endif
    }
    if (q.scalar_type() == torch::kInt8) {
#if defined(DEEPGEMM_ENABLE_GFX936) || defined(DEEPGEMM_ENABLE_GFX938)
        return mqa::int8::mqa_logits_int8(q, k, weights, ks, ke, m, n, h, d, kv_scale, false, torch::nullopt, max_seqlen_k);
#else
        TORCH_CHECK(false, "mqa_logits INT8 was not built; enable gfx936 or gfx938");
#endif
    }
    if (q.scalar_type() == torch::kBFloat16) {
#if defined(DEEPGEMM_ENABLE_GFX936) || defined(DEEPGEMM_ENABLE_GFX938)
        return mqa::bf16::mqa_logits_bf16(q, k, weights, ks, ke, m, n, h, d, kv_scale, false, torch::nullopt, max_seqlen_k);
#else
        TORCH_CHECK(false, "mqa_logits BF16 was not built; enable gfx936 or gfx938");
#endif
    }
    TORCH_CHECK(false, "mqa_logits supports BF16, INT8 and FP8 E4M3FN Q/K only");
}
} // namespace deepgemm
