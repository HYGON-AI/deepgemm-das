// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <torch/extension.h>

#include "bm256_bn256_nloop4.inc.cuh"

namespace gfx936_gfx938_shared::m_grouped_int8_gemm_nt_contiguous_wasp_pair32 {

torch::Tensor m_grouped_int8_gemm_nt_contiguous_wasp_pair32 (torch::Tensor input,
                                                torch::Tensor b_qweight,
                                                torch::Tensor output,
                                                torch::Tensor a_scale,
                                                torch::Tensor b_scale,
                                                torch::Tensor m_indices,
                                                int64_t mode) {
  if (true) {
    return m_grouped_int8_gemm_nt_contiguous_wasp_pair32_bm256_bn256_nloop4::run(
        input, b_qweight, output, a_scale, b_scale, m_indices, 256, 256, 4);    
  }

  TORCH_CHECK(false, "unsupported mode=", mode);        
}

}  // namespace m_grouped_int8_gemm_nt_contiguous_wasp_pair32
