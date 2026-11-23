// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "einsum.h"

namespace deepgemm {
namespace dispatch {
namespace einsum {

void einsum(const std::string& expr, const torch::Tensor& a,
            const torch::Tensor& b, const torch::Tensor& d,
            const std::optional<torch::Tensor>& c) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::deepgemm::einsum::einsum(expr, a, b, d, c);
  }
#endif
  ::deepgemm::arch::unsupported("einsum");
}

void fp8_einsum(const std::string& expr,
                const std::pair<torch::Tensor, torch::Tensor>& a,
                const std::pair<torch::Tensor, torch::Tensor>& b,
                const torch::Tensor& d,
                const std::optional<torch::Tensor>& c,
                const std::tuple<int, int, int>& recipe) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::deepgemm::einsum::fp8_einsum(expr, a, b, d, c, recipe);
  }
#endif
  ::deepgemm::arch::unsupported("fp8_einsum");
}

}  // namespace einsum
}  // namespace dispatch
}  // namespace deepgemm
