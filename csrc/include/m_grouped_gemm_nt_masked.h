#pragma once

#include <torch/extension.h>

namespace deepgemm {

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_masked(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor a_scale, 
    torch::Tensor b_scale, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    int mode);
torch::Tensor m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    int expected_m_per_group,
    int mode);
torch::Tensor m_grouped_marlin_w8a8_gemm_nt_masked(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor a_scale, 
    torch::Tensor b_scale, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    int mode);
torch::Tensor m_grouped_marlin_w8a8_gemm_nt_compact_masked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    torch::Tensor expert_offsets,
    int expected_m_per_group);
torch::Tensor m_grouped_bf16_gemm_nt_masked(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    int mode);
torch::Tensor m_grouped_bf16_gemm_nt_masked_hipc(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output, 
    torch::Tensor masked_m, 
    int expected_m_per_group, 
    int mode);

    
void masked_fp8_gemm(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matri_a_scale,
    const torch::Tensor &matrixb_scale,
    const torch::Tensor &actual_tokens,
    torch::Tensor &matrix_c,
    int64_t max_tokens,
    int64_t experts,
    int64_t cu_s,
    bool b_block_wise,
    bool b_overlap,
    const c10::optional<torch::Tensor> &signal);
    
torch::Tensor group_gemm_fp8_masked(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matri_a_scale,
    const torch::Tensor &matrixb_scale,
    const torch::Tensor &actual_tokens,
    torch::Tensor &matrix_c,
    int64_t max_tokens,
    int64_t experts,
    int64_t cu_s,
    bool block_wise,
    bool b_overlap,
    const c10::optional<torch::Tensor> &signal);
void masked_fp8_gemm_tensor_wise(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matri_a_scale,
    const torch::Tensor &matrixb_scale,
    const torch::Tensor &actual_tokens,
    torch::Tensor &matrix_c,
    int64_t max_tokens,
    int64_t experts,
    int64_t cu_s,
    bool b_overlap,
    const c10::optional<torch::Tensor> &signal);
torch::Tensor group_gemm_fp8_masked_tensor_wise(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matri_a_scale,
    const torch::Tensor &matrixb_scale,
    const torch::Tensor &actual_tokens,
    torch::Tensor &matrix_c,
    int64_t max_tokens,
    int64_t experts,
    int64_t cu_s,
    bool b_overlap,
    const c10::optional<torch::Tensor> &signal);
}
