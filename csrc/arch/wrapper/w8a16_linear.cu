// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "moe_marlin/w8a16_linear.h"

namespace gfx92a {
namespace deepgemm {

torch::Tensor w8a16_linear(const torch::Tensor& a, const torch::Tensor& b,
                           const torch::Tensor& scale, int num_k_warps,
                           int n_per_warp, int k_per_warp, int num_n_warps,
                           int splitk, int block_m, int weight_layout);
torch::Tensor w8a16_linear_bias(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scale,
    const torch::Tensor& bias, int num_k_warps, int n_per_warp,
    int k_per_warp, int num_n_warps, int splitk, int block_m,
    int weight_layout);
torch::Tensor w8a16_dequant_weight(const torch::Tensor& b,
                                   const torch::Tensor& scale);
torch::Tensor w8a16_dequant_weight_bf16(const torch::Tensor& b,
                                        const torch::Tensor& scale);

}  // namespace deepgemm
}  // namespace gfx92a

namespace gfx936_gfx938_shared {
namespace deepgemm {

torch::Tensor w8a16_linear(const torch::Tensor& a, const torch::Tensor& b,
                           const torch::Tensor& scale, int num_k_warps,
                           int n_per_warp, int k_per_warp, int num_n_warps,
                           int splitk, int block_m, int weight_layout);
torch::Tensor w8a16_linear_bias(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scale,
    const torch::Tensor& bias, int num_k_warps, int n_per_warp,
    int k_per_warp, int num_n_warps, int splitk, int block_m,
    int weight_layout);
torch::Tensor w8a16_dequant_weight(const torch::Tensor& b,
                                   const torch::Tensor& scale);
torch::Tensor w8a16_dequant_weight_bf16(const torch::Tensor& b,
                                        const torch::Tensor& scale);

}  // namespace deepgemm
}  // namespace gfx936_gfx938_shared

namespace deepgemm {
namespace dispatch {

torch::Tensor w8a16_linear(const torch::Tensor& a, const torch::Tensor& b,
                           const torch::Tensor& scale, int num_k_warps,
                           int n_per_warp, int k_per_warp, int num_n_warps,
                           int splitk, int block_m, int weight_layout) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::w8a16_linear(
        a, b, scale, num_k_warps, n_per_warp, k_per_warp, num_n_warps, splitk,
        block_m, weight_layout);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_linear(
        a, b, scale, num_k_warps, n_per_warp, k_per_warp, num_n_warps, splitk,
        block_m, weight_layout);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_linear(
        a, b, scale, num_k_warps, n_per_warp, k_per_warp, num_n_warps, splitk,
        block_m, weight_layout);
  }
#endif
  ::deepgemm::arch::unsupported("w8a16_linear");
}

torch::Tensor w8a16_linear_bias(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scale,
    const torch::Tensor& bias, int num_k_warps, int n_per_warp,
    int k_per_warp, int num_n_warps, int splitk, int block_m,
    int weight_layout) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::w8a16_linear_bias(
        a, b, scale, bias, num_k_warps, n_per_warp, k_per_warp, num_n_warps,
        splitk, block_m, weight_layout);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_linear_bias(
        a, b, scale, bias, num_k_warps, n_per_warp, k_per_warp, num_n_warps,
        splitk, block_m, weight_layout);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_linear_bias(
        a, b, scale, bias, num_k_warps, n_per_warp, k_per_warp, num_n_warps,
        splitk, block_m, weight_layout);
  }
#endif
  ::deepgemm::arch::unsupported("w8a16_linear_bias");
}

torch::Tensor w8a16_dequant_weight(const torch::Tensor& b,
                                   const torch::Tensor& scale) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::w8a16_dequant_weight(b, scale);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_dequant_weight(b, scale);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_dequant_weight(b, scale);
  }
#endif
  ::deepgemm::arch::unsupported("w8a16_dequant_weight");
}

torch::Tensor w8a16_dequant_weight_bf16(const torch::Tensor& b,
                                        const torch::Tensor& scale) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX92A)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx92a) {
    return ::gfx92a::deepgemm::w8a16_dequant_weight_bf16(b, scale);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX936)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx936) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_dequant_weight_bf16(b,
                                                                       scale);
  }
#endif
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::gfx936_gfx938_shared::deepgemm::w8a16_dequant_weight_bf16(b,
                                                                       scale);
  }
#endif
  ::deepgemm::arch::unsupported("w8a16_dequant_weight_bf16");
}

}  // namespace dispatch
}  // namespace deepgemm
