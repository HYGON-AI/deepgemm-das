// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <torch/extension.h>

#include "bm128_bn384_nloop4.inc.cuh"
#include "bm128_bn256_nloop4.inc.cuh"
#include "bm256_bn256_nloop4.inc.cuh"

namespace gfx938::m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2 {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2 (torch::Tensor input,
                                                torch::Tensor b_qweight,
                                                torch::Tensor output,
                                                torch::Tensor a_scale,
                                                torch::Tensor b_scale,
                                                torch::Tensor m_indices,
                                                int64_t mode) {
  // MODE 1000: persistent BM256 x BN256.
  if (mode == 1000) {
    return m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_bm256_bn256_nloop4_v2::run(
        input, b_qweight, output, a_scale, b_scale, m_indices, 256, 256, 4);
  }
  // MODE 1001: persistent BM128 x BN256.
  if (mode == 1001) {
    return m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_bm128_bn256_nloop4_v2::run(
        input, b_qweight, output, a_scale, b_scale, m_indices, 128, 256, 4);
  }
  // MODE 1002: persistent BM128 x BN384.
  if (mode == 1002) {
    return m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_bm128_bn384_nloop4_v2::run(
        input, b_qweight, output, a_scale, b_scale, m_indices, 128, 384, 4);
  }

  TORCH_CHECK(false, "unsupported mode=", mode);        
}

}  // namespace m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2
