// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Cross-library kernel symbol declarations for the split build
// (1 main library + per-arch kernel libraries gfx92a / gfx936 / gfx938).
//
// The main library references every op symbol (pybind bindings in
// csrc/common/deepgemm_ops.cpp) but contains no kernel definitions. Each
// per-arch kernel library must export the full symbol set; symbols whose real
// definition lives in another arch's source group are provided as stubs by
// csrc/common/kernel_stubs_<arch>.cpp. All signatures here are copied
// verbatim from the real definition sites so the C++ mangled names match
// across shared objects (resolved via dlopen(RTLD_GLOBAL)).
#pragma once

#include <torch/extension.h>

#include <tuple>

// csrc/arch/gfx938/m_grouped_fp8_gemm_nt_masked_pair32/dispatch.inc.cuh
// (gfx92a has an identical-signature copy under csrc/arch/gfx92a/...)
namespace m_grouped_fp8_gemm_nt_masked_pair32 {

std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    torch::Tensor output,
    bool enable_overlap,
    torch::Tensor signal,
    int64_t mode);

}  // namespace m_grouped_fp8_gemm_nt_masked_pair32

// csrc/arch/gfx938/m_grouped_fp8_gemm_nt_contiguous_wasp_pair32/dispatch.inc.cuh
// (gfx92a has an identical-signature copy under csrc/arch/gfx92a/...)
namespace m_grouped_fp8_gemm_nt_contiguous_wasp_pair32 {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_wasp_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode);

}  // namespace m_grouped_fp8_gemm_nt_contiguous_wasp_pair32

// csrc/arch/gfx936_gfx938_shared/m_grouped_int8_gemm_nt_masked_pair32/dispatch.inc.cuh
namespace m_grouped_int8_gemm_nt_masked_pair32 {

torch::Tensor m_grouped_int8_gemm_nt_masked_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    torch::Tensor output,
    int64_t mode);

}  // namespace m_grouped_int8_gemm_nt_masked_pair32

// csrc/arch/gfx936_gfx938_shared/m_grouped_int8_gemm_nt_contiguous_wasp_pair32/dispatch.inc.cuh
namespace m_grouped_int8_gemm_nt_contiguous_wasp_pair32 {

torch::Tensor m_grouped_int8_gemm_nt_contiguous_wasp_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode);

}  // namespace m_grouped_int8_gemm_nt_contiguous_wasp_pair32

// csrc/arch/gfx938/m_grouped_fp8_gemm_nt_contiguous/dispatch.inc.cuh
// (gfx92a provides a stub in csrc/common/kernel_stubs_gfx92a.cpp)
namespace m_grouped_fp8_contiguous {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t block_m,
    int64_t block_n,
    int64_t n_loop_num);

}  // namespace m_grouped_fp8_contiguous

// csrc/arch/gfx938/m_grouped_fp8_gemm_nt_contiguous.cu
namespace m_grouped_fp8_contig {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_marlin(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode,
    int64_t n_loop_num);

}  // namespace m_grouped_fp8_contig

// csrc/include/m_grouped_i8_gemm_nt_contiguous/dispatch.inc.cuh
// (definition compiled into the gfx938 kernel library via
// csrc/arch/gfx938/m_grouped_i8_gemm_nt_contiguous_optimized.cu)
namespace m_grouped_i8_contiguous {

torch::Tensor m_grouped_i8_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t block_m,
    int64_t block_n,
    int64_t n_loop_num);

}  // namespace m_grouped_i8_contiguous

// csrc/arch/gfx938/moe_w8a8_marlin_prefill_down_fp8.cu
namespace moe_marlin {

torch::Tensor moe_marlin_w8a8_gemm2_prefill_down_fp8(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t real_topk);

}  // namespace moe_marlin

// csrc/arch/gfx936_gfx938_shared/m_grouped_wfp4a16_nt_contiguous.cu (global scope)
torch::Tensor group_gemm_wfp4a16_contiguous(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    torch::Tensor& matrix_c,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& m_indices);

// csrc/arch/gfx936_gfx938_shared/m_grouped_wfp4a16_nt_masked.cu (global scope)
torch::Tensor group_gemm_wfp4a16_masked(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts);

// csrc/arch/gfx936_gfx938_shared/m_grouped_w4a16_nt_masked.cu (global scope)
torch::Tensor group_gemm_w4a16_masked(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts);

// csrc/arch/gfx936_gfx938_shared/m_grouped_w8a8_gemm_nt_contiguous_gfx936.cu
// (excluded from the gfx938 kernel library via SOURCE_EXCLUDED_ARCHS)
namespace i8_contig_ll_v92 {

torch::Tensor i8_contig_ll(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t block_m,
    int64_t block_n,
    int64_t n_loop_num);

}  // namespace i8_contig_ll_v92
