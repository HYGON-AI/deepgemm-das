// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

namespace gfx92a {
namespace m_grouped_int8_gemm_nt_masked_pair32 {

torch::Tensor m_grouped_int8_gemm_nt_masked_pair32(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor a_scale,
    torch::Tensor b_scale, torch::Tensor masked_m, torch::Tensor output,
    int64_t mode);

}  // namespace m_grouped_int8_gemm_nt_masked_pair32
}  // namespace gfx92a

namespace gfx936_gfx938_shared {
namespace m_grouped_int8_gemm_nt_masked_pair32 {

torch::Tensor m_grouped_int8_gemm_nt_masked_pair32(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor a_scale,
    torch::Tensor b_scale, torch::Tensor masked_m, torch::Tensor output,
    int64_t mode);

}  // namespace m_grouped_int8_gemm_nt_masked_pair32
}  // namespace gfx936_gfx938_shared

namespace deepgemm {
namespace dispatch {

torch::Tensor m_grouped_int8_gemm_nt_masked_pair32(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor a_scale,
    torch::Tensor b_scale, torch::Tensor masked_m, torch::Tensor output,
    int64_t mode) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::m_grouped_int8_gemm_nt_masked_pair32::
        m_grouped_int8_gemm_nt_masked_pair32(input, b_qweight, a_scale,
                                             b_scale, masked_m, output, mode);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936_gfx938_shared::m_grouped_int8_gemm_nt_masked_pair32::
        m_grouped_int8_gemm_nt_masked_pair32(input, b_qweight, a_scale,
                                             b_scale, masked_m, output, mode);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx936_gfx938_shared::m_grouped_int8_gemm_nt_masked_pair32::
        m_grouped_int8_gemm_nt_masked_pair32(input, b_qweight, a_scale,
                                             b_scale, masked_m, output, mode);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_int8_gemm_nt_masked_pair32");
}

}  // namespace dispatch
}  // namespace deepgemm
