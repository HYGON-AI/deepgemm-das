#pragma once

#include <torch/all.h>

namespace deepgemm_wfp4a16_arch_dispatch {

void contiguous_wfp4a16_gemm_gfx936(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c);

void contiguous_wfp4a16_gemm_gfx938(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c);

void contiguous_wfp4a16_gemm_gfx92a(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c);

void masked_wfp4a16_gemm_gfx936(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts);

void masked_wfp4a16_gemm_gfx92a(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts);

void masked_wfp4a16_gemm_gfx938(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts);

}  // namespace deepgemm_wfp4a16_arch_dispatch
