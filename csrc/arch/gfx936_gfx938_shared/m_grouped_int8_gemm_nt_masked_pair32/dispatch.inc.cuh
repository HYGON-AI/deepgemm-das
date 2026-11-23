// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <torch/extension.h>

#include "bm64_bn256_bk128.inc.cuh"
#include "bm128_bn256_bk128.inc.cuh"
#include "bm128_bn256_bk128_stable.inc.cuh"
#include "bm256_bn256_bk128.inc.cuh"

namespace gfx936_gfx938_shared::m_grouped_int8_gemm_nt_masked_pair32 {

torch::Tensor m_grouped_int8_gemm_nt_masked_pair32(torch::Tensor input,
                                                torch::Tensor b_qweight,
                                                torch::Tensor a_scale,
                                                torch::Tensor b_scale,
                                                torch::Tensor masked_m,
                                                torch::Tensor output,
                                                int64_t mode) {
  if (mode == 1000) {
    m_grouped_int8_gemm_nt_masked_pair32_bm64_bn256_bk128::run(
        input, b_qweight, a_scale, b_scale, masked_m, output);
    return output;
  } else if (mode == 1001) {
    m_grouped_int8_gemm_nt_masked_pair32_bm128_bn256_bk128::launch(
        input, b_qweight, a_scale, b_scale, masked_m, output);
    return output;
  } 
  else if (mode == 1003) {
    m_grouped_int8_gemm_nt_masked_pair32_bm128_bn256_bk128_stable::launch(
        input, b_qweight, a_scale, b_scale, masked_m, output);
    return output;
  }
  else if (mode == 1002) {
    m_grouped_int8_gemm_nt_masked_pair32_bm256_bn256_bk128::launch(
        input, b_qweight, a_scale, b_scale, masked_m, output);
    return output;
  }

  TORCH_CHECK(false, "unsupported mode=", mode);        
}

}  // namespace m_grouped_int8_gemm_nt_masked_pair32
