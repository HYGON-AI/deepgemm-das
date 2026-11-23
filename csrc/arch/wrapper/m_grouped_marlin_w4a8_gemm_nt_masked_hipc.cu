// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "m_grouped_gemm_nt_masked.h"

namespace gfx92a {
namespace deepgemm {

std::tuple<int64_t, int64_t> m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor masked_m,
    int expected_m_per_group, int mode, int persistent_multiplier,
    bool enable_overlap, const c10::optional<torch::Tensor>& signal);

}  // namespace deepgemm
}  // namespace gfx92a

namespace gfx936_gfx938_shared {
namespace deepgemm {

std::tuple<int64_t, int64_t> m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor masked_m,
    int expected_m_per_group, int mode, int persistent_multiplier,
    bool enable_overlap, const c10::optional<torch::Tensor>& signal);

}  // namespace deepgemm
}  // namespace gfx936_gfx938_shared

namespace deepgemm {
namespace dispatch {

std::tuple<int64_t, int64_t> m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor masked_m,
    int expected_m_per_group, int mode, int persistent_multiplier,
    bool enable_overlap, const c10::optional<torch::Tensor>& signal) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        expected_m_per_group, mode, persistent_multiplier, enable_overlap,
        signal);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936_gfx938_shared::deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        expected_m_per_group, mode, persistent_multiplier, enable_overlap,
        signal);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx936_gfx938_shared::deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        expected_m_per_group, mode, persistent_multiplier, enable_overlap,
        signal);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_marlin_w4a8_gemm_nt_masked_hipc");
}

}  // namespace dispatch
}  // namespace deepgemm
