// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "m_grouped_gemm_nt_masked.h"

namespace deepgemm {
namespace dispatch {

torch::Tensor m_grouped_bf16_gemm_nt_masked(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor masked_m, int expected_m_per_group, int mode) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::deepgemm::m_grouped_bf16_gemm_nt_masked(
        input, b_qweight, output, masked_m, expected_m_per_group, mode);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::deepgemm::m_grouped_bf16_gemm_nt_masked(
        input, b_qweight, output, masked_m, expected_m_per_group, mode);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_bf16_gemm_nt_masked");
}

}  // namespace dispatch
}  // namespace deepgemm
