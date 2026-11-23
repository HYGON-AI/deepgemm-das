// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "kernel_registry_decls.h"

namespace deepgemm {
namespace dispatch {

torch::Tensor group_gemm_wfp4a16_contiguous(
    const torch::Tensor& matrix_a, const torch::Tensor& matrix_b,
    torch::Tensor& matrix_c, const torch::Tensor& matrixb_scale,
    const torch::Tensor& m_indices) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::group_gemm_wfp4a16_contiguous(matrix_a, matrix_b, matrix_c,
                                           matrixb_scale, m_indices);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::group_gemm_wfp4a16_contiguous(matrix_a, matrix_b, matrix_c,
                                           matrixb_scale, m_indices);
  }
#endif
  ::deepgemm::arch::unsupported("group_gemm_wfp4a16_contiguous");
}

torch::Tensor group_gemm_wfp4a16_masked(
    const torch::Tensor& matrix_a, const torch::Tensor& matrix_b,
    const torch::Tensor& matrixb_scale, const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c, int64_t max_tokens, int64_t experts) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::group_gemm_wfp4a16_masked(matrix_a, matrix_b, matrixb_scale,
                                       actual_tokens, matrix_c, max_tokens,
                                       experts);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::group_gemm_wfp4a16_masked(matrix_a, matrix_b, matrixb_scale,
                                       actual_tokens, matrix_c, max_tokens,
                                       experts);
  }
#endif
  ::deepgemm::arch::unsupported("group_gemm_wfp4a16_masked");
}

torch::Tensor group_gemm_w4a16_masked(
    const torch::Tensor& matrix_a, const torch::Tensor& matrix_b,
    const torch::Tensor& matrixb_scale, const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c, int64_t max_tokens, int64_t experts) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::group_gemm_w4a16_masked(matrix_a, matrix_b, matrixb_scale,
                                     actual_tokens, matrix_c, max_tokens,
                                     experts);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::group_gemm_w4a16_masked(matrix_a, matrix_b, matrixb_scale,
                                     actual_tokens, matrix_c, max_tokens,
                                     experts);
  }
#endif
  ::deepgemm::arch::unsupported("group_gemm_w4a16_masked");
}

torch::Tensor i8_contig_ll(torch::Tensor input, torch::Tensor b_qweight,
                           torch::Tensor output, torch::Tensor a_scale,
                           torch::Tensor b_scale, torch::Tensor m_indices,
                           int64_t block_m, int64_t block_n,
                           int64_t n_loop_num) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::i8_contig_ll_v92::i8_contig_ll(input, b_qweight, output, a_scale,
                                            b_scale, m_indices, block_m,
                                            block_n, n_loop_num);
  }
#endif
  ::deepgemm::arch::unsupported("i8_contig_ll");
}

}  // namespace dispatch
}  // namespace deepgemm
