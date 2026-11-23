// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "paged_mqa_logits/paged_mqa_logits_sparse.h"
#include "paged_mqa_logits_grouped_native_fp8_impl.hpp"

namespace deepgemm {

void launch_paged_mqa_logits_grouped_fp8(
    int group_size, const torch::Tensor& q, const torch::Tensor& kv_cache, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_table, torch::Tensor& logits,
    torch::Tensor& nonzero_mask, int persistent_ctas, hipStream_t stream) {
    grouped_sparse_native_fp8::dispatch_grouped_rows(
        group_size, q, kv_cache, weights, context_lens, block_table, logits, nonzero_mask, persistent_ctas, stream);
}

}  // namespace deepgemm
