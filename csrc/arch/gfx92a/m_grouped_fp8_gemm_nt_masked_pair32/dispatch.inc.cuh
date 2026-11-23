// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <tuple>

#include "bm64_bn256_bk128.inc.cuh"

namespace gfx92a::m_grouped_fp8_gemm_nt_masked_pair32 {

std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32 (torch::Tensor input,
                                                torch::Tensor b_qweight,
                                                torch::Tensor a_scale,
                                                torch::Tensor b_scale,
                                                torch::Tensor masked_m,
                                                torch::Tensor output,
                                                bool enable_overlap,
                                                const c10::optional<torch::Tensor>& signal,
                                                int64_t mode) {
  if (true) {    
    m_grouped_fp8_gemm_nt_masked_bm64_bn256_bk128::launch(input, b_qweight, a_scale, b_scale, masked_m, output);
    return std::tuple<int64_t, int64_t>(0, 0);;
  }

  TORCH_CHECK(false, "unsupported mode=", mode);        
}

}  // namespace m_grouped_fp8_gemm_nt_masked_pair32
