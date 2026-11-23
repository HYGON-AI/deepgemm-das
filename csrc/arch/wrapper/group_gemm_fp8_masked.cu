// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "m_grouped_gemm_nt_masked.h"

namespace deepgemm {
namespace dispatch {

torch::Tensor group_gemm_fp8_masked(
    const torch::Tensor& matrix_a, const torch::Tensor& matrix_b,
    const torch::Tensor& matri_a_scale, const torch::Tensor& matrixb_scale,
    const torch::Tensor& actual_tokens, torch::Tensor& matrix_c,
    int64_t max_tokens, int64_t experts, int64_t cu_s, bool block_wise,
    bool b_overlap, const c10::optional<torch::Tensor>& signal,
    int64_t mode) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::deepgemm::group_gemm_fp8_masked(
        matrix_a, matrix_b, matri_a_scale, matrixb_scale, actual_tokens,
        matrix_c, max_tokens, experts, cu_s, block_wise, b_overlap, signal,
        mode);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::deepgemm::group_gemm_fp8_masked(
        matrix_a, matrix_b, matri_a_scale, matrixb_scale, actual_tokens,
        matrix_c, max_tokens, experts, cu_s, block_wise, b_overlap, signal,
        mode);
  }
#endif
  ::deepgemm::arch::unsupported("group_gemm_fp8_masked");
}

}  // namespace dispatch
}  // namespace deepgemm
