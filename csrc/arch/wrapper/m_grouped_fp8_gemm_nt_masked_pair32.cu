// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

namespace gfx92a {
namespace m_grouped_fp8_gemm_nt_masked_pair32 {

std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor a_scale,
    torch::Tensor b_scale, torch::Tensor masked_m, torch::Tensor output,
    bool enable_overlap, const c10::optional<torch::Tensor>& signal,
    int64_t mode);

}  // namespace m_grouped_fp8_gemm_nt_masked_pair32
}  // namespace gfx92a

namespace gfx938 {
namespace m_grouped_fp8_gemm_nt_masked_pair32 {

std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor a_scale,
    torch::Tensor b_scale, torch::Tensor masked_m, torch::Tensor output,
    bool enable_overlap, const c10::optional<torch::Tensor>& signal,
    int64_t mode);

}  // namespace m_grouped_fp8_gemm_nt_masked_pair32
}  // namespace gfx938

namespace deepgemm {
namespace dispatch {

std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor a_scale,
    torch::Tensor b_scale, torch::Tensor masked_m, torch::Tensor output,
    bool enable_overlap, const c10::optional<torch::Tensor>& signal,
    int64_t mode) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::m_grouped_fp8_gemm_nt_masked_pair32::
        m_grouped_fp8_gemm_nt_masked_pair32(input, b_qweight, a_scale, b_scale,
                                            masked_m, output, enable_overlap,
                                            signal, mode);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx938::m_grouped_fp8_gemm_nt_masked_pair32::
        m_grouped_fp8_gemm_nt_masked_pair32(input, b_qweight, a_scale, b_scale,
                                            masked_m, output, enable_overlap,
                                            signal, mode);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_fp8_gemm_nt_masked_pair32");
}

}  // namespace dispatch
}  // namespace deepgemm
