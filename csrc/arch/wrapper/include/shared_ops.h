// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Declarations of the architecture dispatch operators.
//
// Every operator registered with pybind is defined in csrc/arch/wrapper/*.cu
// inside namespace deepgemm::dispatch. At runtime it forwards to the
// implementation compiled for the detected GPU architecture
// (gfx92a / gfx936 / gfx938 / gfx936_gfx938_shared) and raises a
// "not supported on the current GPU architecture" error when the detected
// architecture has no implementation.
#pragma once

#include <torch/extension.h>

#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace deepgemm {
namespace dispatch {

// ---- grouped GEMM, contiguous -------------------------------------------------
torch::Tensor m_grouped_w8a8_gemm_nt_nopad_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    torch::Tensor token_per_expert);

torch::Tensor m_grouped_w8a8_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);

torch::Tensor m_grouped_w16a16_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);

torch::Tensor m_grouped_fp8_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode);

// ---- grouped GEMM, masked ----------------------------------------------------
torch::Tensor m_grouped_marlin_w4a8_gemm_nt_masked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    int expected_m_per_group,
    int mode);

std::tuple<int64_t, int64_t> m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    int expected_m_per_group,
    int mode,
    int persistent_multiplier = 0,
    bool enable_overlap = false,
    const c10::optional<torch::Tensor>& signal = c10::nullopt);

torch::Tensor m_grouped_marlin_w8a8_gemm_nt_masked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    int expected_m_per_group,
    int mode);

torch::Tensor m_grouped_marlin_w8a8_gemm_nt_compact_masked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    torch::Tensor expert_offsets,
    int expected_m_per_group);

torch::Tensor m_grouped_bf16_gemm_nt_masked(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor masked_m,
    int expected_m_per_group,
    int mode);

torch::Tensor group_gemm_fp8_masked(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matri_a_scale,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts,
    int64_t cu_s,
    bool block_wise,
    bool b_overlap,
    const c10::optional<torch::Tensor>& signal,
    int64_t mode);

// ---- WFP4A16 / W4A16 grouped GEMM -------------------------------------------
torch::Tensor group_gemm_wfp4a16_contiguous(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    torch::Tensor& matrix_c,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& m_indices);

torch::Tensor group_gemm_wfp4a16_masked(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts);

torch::Tensor group_gemm_w4a16_masked(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts);

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

// ---- pair32 / marlin registry operators -------------------------------------
std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    torch::Tensor output,
    bool enable_overlap,
    const c10::optional<torch::Tensor>& signal,
    int64_t mode);

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_wasp_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode);

std::tuple<int64_t, int64_t> m_grouped_fp8_gemm_nt_masked_pair32_v2(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    torch::Tensor output,
    bool enable_overlap,
    const c10::optional<torch::Tensor>& signal,
    int64_t mode);

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode);

torch::Tensor m_grouped_int8_gemm_nt_masked_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    torch::Tensor output,
    int64_t mode);

torch::Tensor m_grouped_int8_gemm_nt_contiguous_wasp_pair32(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode);

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_registry(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t block_m,
    int64_t block_n,
    int64_t n_loop_num);

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_marlin(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode,
    int64_t n_loop_num);

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

// ---- dense GEMM --------------------------------------------------------------
void fp8_gemm(torch::Tensor mat_a,
              torch::Tensor mat_b,
              torch::Tensor scale_a,
              torch::Tensor scale_b,
              torch::Tensor output,
              const int64_t m,
              const int64_t n,
              const int64_t k,
              const int64_t batch,
              const std::string& transpose_flag,
              const torch::Tensor& alpha,
              const torch::Tensor& beta,
              const std::optional<torch::Tensor>& bias);

void fp8_gemm_hipc(torch::Tensor mat_a,
                   torch::Tensor mat_b,
                   torch::Tensor scale_a,
                   torch::Tensor scale_b,
                   torch::Tensor output,
                   const int64_t m,
                   const int64_t n,
                   const int64_t k,
                   const int64_t batch,
                   const std::string& transpose_flag,
                   const torch::Tensor& alpha,
                   const torch::Tensor& beta,
                   const std::optional<torch::Tensor>& bias);

void tf32_hc_prenorm_gemm(const torch::Tensor& A,
                          const torch::Tensor& B,
                          const torch::Tensor& D,
                          const torch::Tensor& sqr_sum,
                          const std::optional<int>& num_splits);

// ---- W8A16 -------------------------------------------------------------------
torch::Tensor w8a16_linear(const torch::Tensor& a,
                           const torch::Tensor& b,
                           const torch::Tensor& scale,
                           int num_k_warps,
                           int n_per_warp,
                           int k_per_warp,
                           int num_n_warps,
                           int splitk,
                           int block_m,
                           int weight_layout);

torch::Tensor w8a16_linear_bias(const torch::Tensor& a,
                                const torch::Tensor& b,
                                const torch::Tensor& scale,
                                const torch::Tensor& bias,
                                int num_k_warps,
                                int n_per_warp,
                                int k_per_warp,
                                int num_n_warps,
                                int splitk,
                                int block_m,
                                int weight_layout);

torch::Tensor w8a16_dequant_weight(const torch::Tensor& b,
                                   const torch::Tensor& scale);

torch::Tensor w8a16_dequant_weight_bf16(const torch::Tensor& b,
                                        const torch::Tensor& scale);

// ---- einsum ------------------------------------------------------------------
namespace einsum {
void einsum(const std::string& expr,
            const torch::Tensor& a,
            const torch::Tensor& b,
            const torch::Tensor& d,
            const std::optional<torch::Tensor>& c = std::nullopt);

void fp8_einsum(const std::string& expr,
                const std::pair<torch::Tensor, torch::Tensor>& a,
                const std::pair<torch::Tensor, torch::Tensor>& b,
                const torch::Tensor& d,
                const std::optional<torch::Tensor>& c = std::nullopt,
                const std::tuple<int, int, int>& recipe = {1, 128, 128});
}  // namespace einsum

}  // namespace dispatch
}  // namespace deepgemm
