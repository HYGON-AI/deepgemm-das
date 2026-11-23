// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "gemm.h"

#if defined(DEEPGEMM_ENABLE_GFX92A)
#include "../gfx92a/fp8_gemm_hipc.h"
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
#include "../gfx938/fp8_gemm_hipc.h"
#endif

namespace deepgemm {
namespace dispatch {

void fp8_gemm(torch::Tensor mat_a, torch::Tensor mat_b, torch::Tensor scale_a,
              torch::Tensor scale_b, torch::Tensor output, const int64_t m,
              const int64_t n, const int64_t k, const int64_t batch,
              const std::string& transpose_flag, const torch::Tensor& alpha,
              const torch::Tensor& beta,
              const std::optional<torch::Tensor>& bias) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx938::deepgemm::fp8_gemm(mat_a, mat_b, scale_a, scale_b, output, m, n, k,
                                batch, transpose_flag, alpha, beta, bias);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::fp8_gemm(mat_a, mat_b, scale_a, scale_b, output, m, n, k,
                                batch, transpose_flag, alpha, beta, bias);
  }
#endif
  ::deepgemm::arch::unsupported("fp8_gemm");
}

void fp8_gemm_hipc(torch::Tensor mat_a, torch::Tensor mat_b,
                   torch::Tensor scale_a, torch::Tensor scale_b,
                   torch::Tensor output, const int64_t m, const int64_t n,
                   const int64_t k, const int64_t batch,
                   const std::string& transpose_flag,
                   const torch::Tensor& alpha, const torch::Tensor& beta,
                   const std::optional<torch::Tensor>& bias) {
  const auto dg_arch = ::deepgemm::arch::current();
// #if defined(DEEPGEMM_ENABLE_GFX92A)
//   if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
//     return ::gfx92a::deepgemm::fp8_gemm_hipc(
//         mat_a, mat_b, scale_a, scale_b, output, m, n, k, batch,
//         transpose_flag, alpha, beta, bias);
//   }
// #endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx938::deepgemm::fp8_gemm_hipc(mat_a, mat_b, scale_a, scale_b, output, m,
                                     n, k, batch, transpose_flag, alpha, beta,
                                     bias);
  }
#endif
  ::deepgemm::arch::unsupported("fp8_gemm_hipc");
}

}  // namespace dispatch
}  // namespace deepgemm
