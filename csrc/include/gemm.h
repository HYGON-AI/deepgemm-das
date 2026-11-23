#pragma once

#include <torch/extension.h>
#include <optional>
#include <string>

namespace gfx938::deepgemm {
void fp8_gemm(
                    torch::Tensor mat_a,
                    torch::Tensor mat_b,
                    torch::Tensor scale_a,
                    torch::Tensor scale_b,
                    torch::Tensor output,
                    const int64_t m,
                    const int64_t n,
                    const int64_t k,
                    const int64_t batch,
                    const std::string& transpose_flag,
                    const torch::Tensor& alpha,
                    const torch::Tensor& beta,
                    const std::optional<torch::Tensor>& bias);

}

namespace gfx92a::deepgemm {
void fp8_gemm(
                    torch::Tensor mat_a,
                    torch::Tensor mat_b,
                    torch::Tensor scale_a,
                    torch::Tensor scale_b,
                    torch::Tensor output,
                    const int64_t m,
                    const int64_t n,
                    const int64_t k,
                    const int64_t batch,
                    const std::string& transpose_flag,
                    const torch::Tensor& alpha,
                    const torch::Tensor& beta,
                    const std::optional<torch::Tensor>& bias);
}
