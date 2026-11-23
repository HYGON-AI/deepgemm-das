// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "paged_mqa_logits.h"

namespace deepgemm {

// Shared prefill entry: INT8 on gfx936/gfx938 and native E4M3FN FP8 on gfx938.
torch::Tensor paged_mqa_logits_prefill_native(
    const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    int64_t max_context_len, bool clean_logits);

namespace detail {

// Internal FP8 launcher. The shared native entry validates inputs, allocates
// logits and selects the dtype; launch geometry uses only tensor metadata.
void launch_paged_mqa_logits_prefill_fp8(
    const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& block_table, torch::Tensor& logits,
    int64_t max_context_len);

}  // namespace detail

}  // namespace deepgemm
