#pragma once

#include <torch/extension.h>

namespace deepgemm {

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);

}  // namespace deepgemm
