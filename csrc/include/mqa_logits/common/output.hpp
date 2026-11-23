// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "mqa_logits/common/params.hpp"
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <ATen/core/LegacyTypeDispatch.h>

namespace deepgemm::mqa::common {
inline int64_t checked_output_width(bool compressed,
                                    const torch::optional<int64_t> &max_seqlen_k,
                                    int64_t n) {
    if (!compressed) {
        TORCH_CHECK(!max_seqlen_k.has_value(),
                    "mqa_logits: max_seqlen_k requires compact output");
        return n;
    }
    TORCH_CHECK(max_seqlen_k.has_value(),
                "mqa_logits: compact output requires explicit max_seqlen_k");
    TORCH_CHECK(*max_seqlen_k > 0 &&
                    *max_seqlen_k <= std::numeric_limits<int32_t>::max(),
                "mqa_logits: max_seqlen_k must be in [1, INT32_MAX]");
    return *max_seqlen_k;
}

// PyTorch's caching allocator can return a 512-byte-offset block. Reserve
// 255 floats to align the returned view explicitly, without a second allocation.
inline torch::Tensor allocate_compact_output(const torch::Tensor &q, int64_t m, int64_t width) {
    const int64_t stride = (width + 255) / 256 * 256;
    TORCH_CHECK(m >= 0 && m <= (std::numeric_limits<int64_t>::max() / 4 - 255) / stride,
                "mqa_logits: compact output span exceeds signed 64-bit range");
    auto storage = torch::empty({m * stride + 255}, q.options().dtype(at::kFloat));
    const auto address = reinterpret_cast<uintptr_t>(storage.data_ptr());
    const int64_t offset = int64_t((1024 - (address & 1023)) & 1023) / 4;
    // This is fresh, private output storage, with no user aliases or history.
    // Skip autograd view-history construction only for this metadata operation;
    // do not use InferenceMode: the returned tensor keeps a normal version counter.
    TORCH_INTERNAL_ASSERT(!storage.requires_grad());
    at::AutoDispatchBelowADInplaceOrView view_guard;
    return storage.as_strided({m, width}, {stride, 1}, offset);
}
} // namespace deepgemm::mqa::common
