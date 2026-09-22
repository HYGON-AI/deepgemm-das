#pragma once

#include <torch/extension.h>

namespace deepgemm {
torch::Tensor m_grouped_w8a8_gemm_nt_nopad_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    torch::Tensor token_per_expert);

torch::Tensor m_grouped_w8a8_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);

torch::Tensor m_grouped_fp8_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);
}