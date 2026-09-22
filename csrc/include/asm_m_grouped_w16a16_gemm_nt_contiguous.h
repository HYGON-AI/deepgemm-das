#pragma once

#include <torch/extension.h>
namespace deepgemm {
torch::Tensor m_grouped_w16a16_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);
}