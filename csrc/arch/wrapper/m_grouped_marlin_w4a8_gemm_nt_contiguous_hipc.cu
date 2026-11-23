// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "m_grouped_w4a8_gemm_nt_contiguous.h"

namespace gfx92a {
namespace deepgemm {

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int mode);

}  // namespace deepgemm
}  // namespace gfx92a

namespace gfx936_gfx938_shared {
namespace deepgemm {

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int mode);

}  // namespace deepgemm
}  // namespace gfx936_gfx938_shared

namespace deepgemm {
namespace dispatch {

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int mode) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
        input, b_qweight, output, a_scale, b_scale, m_indices, mode);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936_gfx938_shared::deepgemm::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
        input, b_qweight, output, a_scale, b_scale, m_indices, mode);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx936_gfx938_shared::deepgemm::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
        input, b_qweight, output, a_scale, b_scale, m_indices, mode);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc");
}

}  // namespace dispatch
}  // namespace deepgemm
