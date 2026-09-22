#pragma once

#include <torch/extension.h>
namespace deepgemm {
    void tf32_hc_pernorm_gemm(
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& D,
        const torch::Tensor& sqr_sum,
        const std::optional<int>& num_splits
    );
}