// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <torch/extension.h>
#include <optional>

namespace gfx928::deepgemm {
    void tf32_hc_prenorm_gemm(
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& D,
        const torch::Tensor& sqr_sum,
        const std::optional<int>& num_splits
    );
}

namespace gfx936::deepgemm {
    void tf32_hc_prenorm_gemm(
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& D,
        const torch::Tensor& sqr_sum,
        const std::optional<int>& num_splits
    );
}

namespace gfx938::deepgemm {
    void tf32_hc_prenorm_gemm(
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& D,
        const torch::Tensor& sqr_sum,
        const std::optional<int>& num_splits
    );
}

namespace gfx92a::deepgemm {
    void tf32_hc_prenorm_gemm(
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& D,
        const torch::Tensor& sqr_sum,
        const std::optional<int>& num_splits
    );
}
