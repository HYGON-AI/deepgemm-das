#pragma once

#include <torch/extension.h>

namespace deepgemm {

torch::Tensor w8a16_linear(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale,
    int num_k_warps,
    int n_per_warp,
    int k_per_warp,
    int num_n_warps,
    int splitk,
    int block_m,
    int weight_layout);

torch::Tensor w8a16_linear_bias(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale,
    const torch::Tensor& bias,
    int num_k_warps,
    int n_per_warp,
    int k_per_warp,
    int num_n_warps,
    int splitk,
    int block_m,
    int weight_layout);

torch::Tensor w8a16_dequant_weight(
    const torch::Tensor& b,
    const torch::Tensor& scale);

torch::Tensor w8a16_dequant_weight_bf16(
    const torch::Tensor& b,
    const torch::Tensor& scale);

} // namespace deepgemm
