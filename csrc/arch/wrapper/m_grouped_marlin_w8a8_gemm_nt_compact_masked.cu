// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "m_grouped_gemm_nt_masked.h"

namespace deepgemm {
namespace dispatch {

torch::Tensor m_grouped_marlin_w8a8_gemm_nt_compact_masked(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor masked_m,
    torch::Tensor expert_offsets, int expected_m_per_group) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::deepgemm::m_grouped_marlin_w8a8_gemm_nt_compact_masked(
        input, b_qweight, output, a_scale, b_scale, masked_m, expert_offsets,
        expected_m_per_group);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::deepgemm::m_grouped_marlin_w8a8_gemm_nt_compact_masked(
        input, b_qweight, output, a_scale, b_scale, masked_m, expert_offsets,
        expected_m_per_group);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_marlin_w8a8_gemm_nt_compact_masked");
}

}  // namespace dispatch
}  // namespace deepgemm
