// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "kernel_registry_decls.h"

namespace deepgemm {
namespace dispatch {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_registry(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int64_t block_m, int64_t block_n, int64_t n_loop_num) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::m_grouped_fp8_contiguous::m_grouped_fp8_gemm_nt_contiguous(
        input, b_qweight, output, a_scale, b_scale, m_indices, block_m, block_n,
        n_loop_num);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_fp8_gemm_nt_contiguous");
}

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_marlin(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int64_t mode, int64_t n_loop_num) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::m_grouped_fp8_contig::m_grouped_fp8_gemm_nt_contiguous_marlin(
        input, b_qweight, output, a_scale, b_scale, m_indices, mode,
        n_loop_num);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_fp8_gemm_nt_contiguous_marlin");
}

torch::Tensor m_grouped_i8_gemm_nt_contiguous(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor m_indices,
    int64_t block_m, int64_t block_n, int64_t n_loop_num) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::m_grouped_i8_contiguous::m_grouped_i8_gemm_nt_contiguous(
        input, b_qweight, output, a_scale, b_scale, m_indices, block_m, block_n,
        n_loop_num);
  }
#endif
  ::deepgemm::arch::unsupported("m_grouped_i8_gemm_nt_contiguous");
}

torch::Tensor moe_marlin_w8a8_gemm2_prefill_down_fp8(
    torch::Tensor input, torch::Tensor b_qweight, torch::Tensor output,
    torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, int64_t top_k, int64_t real_topk) {
  const auto dg_arch = ::deepgemm::arch::current();
#if defined(DEEPGEMM_ENABLE_GFX938)
  if (dg_arch == ::deepgemm::arch::Arch::kGfx938) {
    return ::moe_marlin::moe_marlin_w8a8_gemm2_prefill_down_fp8(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad, top_k, real_topk);
  }
#endif
  ::deepgemm::arch::unsupported("moe_marlin_w8a8_gemm2_prefill_down_fp8");
}

}  // namespace dispatch
}  // namespace deepgemm
