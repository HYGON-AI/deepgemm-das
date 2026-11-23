// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <tuple>

#include "bm64_bn256_bk128.inc.cuh"
#include "bm128_bn256_bk128.inc.cuh"
#include "bm128_bn256_bk128_stable.inc.cuh"
#include "bm256_bn256_bk128.inc.cuh"


template <typename T>
inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

namespace gfx938::m_grouped_fp8_gemm_nt_masked_pair32 {

std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32 (torch::Tensor input,
                                                torch::Tensor b_qweight,
                                                torch::Tensor a_scale,
                                                torch::Tensor b_scale,
                                                torch::Tensor masked_m,
                                                torch::Tensor output,
                                                bool enable_overlap,
                                                const c10::optional<torch::Tensor>& signal,
                                                int64_t mode) {

  const torch::Tensor signal_tensor = signal.has_value() ? *signal : torch::Tensor();
  int N = b_qweight.size(2) * 32;
  if (mode == 1000) {
    m_grouped_fp8_gemm_nt_masked_bm64_bn256_bk128::launch(input, b_qweight, a_scale, b_scale, masked_m, enable_overlap, signal_tensor, output);
    return std::tuple<int64_t, int64_t>(64, ceil_div(N, 256));  //block_m, ceil_div(N, block_n)
  }else if(mode == 1001) {
    m_grouped_fp8_gemm_nt_masked_bm128_bn256_bk128::launch(input, b_qweight, a_scale, b_scale, masked_m, enable_overlap, signal_tensor, output);
    return std::tuple<int64_t, int64_t>(128, ceil_div(N, 256)); //block_m, ceil_div(N, block_n)
  }else if(mode == 1003) {
    m_grouped_fp8_gemm_nt_masked_bm128_bn256_bk128_stable::launch(input, b_qweight, a_scale, b_scale, masked_m, enable_overlap, signal_tensor, output);
    return std::tuple<int64_t, int64_t>(128, ceil_div(N, 256)); //block_m, ceil_div(N, block_n)
  }else if(mode == 1002) {
    m_grouped_fp8_gemm_nt_masked_bm256_bn256_bk128::launch(input, b_qweight, a_scale, b_scale, masked_m, enable_overlap, signal_tensor, output);
    return std::tuple<int64_t, int64_t>(256, ceil_div(N, 256)); //block_m, ceil_div(N, block_n)
  }else{
    TORCH_CHECK(false, "unsupported mode=", mode);  
  }      
}

}  // namespace m_grouped_fp8_gemm_nt_masked_pair32
