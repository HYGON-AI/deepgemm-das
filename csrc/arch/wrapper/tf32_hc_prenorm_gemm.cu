// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include <c10/cuda/CUDAGuard.h>

#include "tf32_hc_prenorm_gemm.h"

namespace deepgemm {
namespace dispatch {

void tf32_hc_prenorm_gemm(const torch::Tensor& A, const torch::Tensor& B,
                          const torch::Tensor& D, const torch::Tensor& sqr_sum,
                          const std::optional<int>& num_splits) {
  TORCH_CHECK(A.is_cuda(), "A must be a GPU tensor");
  const c10::cuda::CUDAGuard device_guard(A.device());
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX928)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx928) {
    return ::gfx928::deepgemm::tf32_hc_prenorm_gemm(A, B, D, sqr_sum, num_splits);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936::deepgemm::tf32_hc_prenorm_gemm(A, B, D, sqr_sum, num_splits);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx938::deepgemm::tf32_hc_prenorm_gemm(A, B, D, sqr_sum, num_splits);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::tf32_hc_prenorm_gemm(A, B, D, sqr_sum, num_splits);
  }
#endif
  ::deepgemm::arch::unsupported("tf32_hc_prenorm_gemm");
}

}  // namespace dispatch
}  // namespace deepgemm
