// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "asm_m_grouped_w8a8_gemm_nt_contiguous.h"

namespace deepgemm {
namespace dispatch {

torch::Tensor m_grouped_w8a8_gemm_nt_nopad_contiguous(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    torch::Tensor token_per_expert) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::deepgemm::m_grouped_w8a8_gemm_nt_nopad_contiguous(
        input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::deepgemm::m_grouped_w8a8_gemm_nt_nopad_contiguous(
        input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_w8a8_gemm_nt_nopad_contiguous");
}

}  // namespace dispatch
}  // namespace deepgemm
