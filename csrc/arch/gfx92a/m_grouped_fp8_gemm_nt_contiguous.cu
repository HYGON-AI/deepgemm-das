#include <torch/extension.h>

#include "m_grouped_fp8_gemm_nt_contiguous/dispatch.inc.cuh"

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
  m.def("m_grouped_fp8_gemm_nt_contiguous("
      "Tensor input, "
      "Tensor b_qweight, "
      "Tensor output, "
      "Tensor a_scale, "
      "Tensor b_scale, "
      "Tensor m_indices, "
      "int block_m, "
      "int block_n, int n_loop_num) -> Tensor");
  m.impl("m_grouped_fp8_gemm_nt_contiguous", torch::kCUDA,
         &m_grouped_fp8_contiguous::m_grouped_fp8_gemm_nt_contiguous);

  m.def(
      "m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise("
      "Tensor input, "
      "Tensor b_qweight, "
      "Tensor output, "
      "Tensor a_scale, "
      "Tensor b_scale, "
      "Tensor m_indices, "
      "int mode, "
      "int n_loop_num"
      ") -> Tensor");
  m.impl(
      "m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise",
      torch::kCUDA,
      &m_grouped_fp8_contiguous::m_grouped_fp8_gemm_nt_contiguous_tensor_wise);
}


