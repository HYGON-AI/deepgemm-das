// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

namespace gfx938 {
namespace m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2 {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int64_t mode);

}  // namespace m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2
}  // namespace gfx938

namespace deepgemm {
namespace dispatch {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int64_t mode) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx938::m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2::
        m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2(
            input, b_qweight, output, a_scale, b_scale, m_indices, mode);
  }
#endif
  ::deepgemm::arch::unsupported(
      "m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2");
}

}  // namespace dispatch
}  // namespace deepgemm
