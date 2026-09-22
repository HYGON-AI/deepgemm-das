// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#if !defined(DEEPGEMM_BUILD_ONLY_W8A16) || defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
#include "m_grouped_gemm_nt_masked.h"
#endif

#ifdef DEEPGEMM_BUILD_ONLY_W4A8_MASKED
#include "m_grouped_w4a8_gemm_nt_contiguous.h"
#endif

#if !defined(DEEPGEMM_BUILD_ONLY_W8A16) && !defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
#include "asm_m_grouped_w8a8_gemm_nt_contiguous.h"
#include "asm_m_grouped_w16a16_gemm_nt_contiguous.h"
#include "m_grouped_w4a8_gemm_nt_contiguous.h"
#include "gemm.h"
#include "tf32_hc_pernorm_gemm.h"
#include "einsum.h"
#endif
#include "moe_marlin/w8a16_linear.h"
#include <torch/extension.h>

#include "deepgemm_ops.hpp"

#include <optional>
#include <string>
#include <tuple>
#include <utility>

#ifndef DEEPGEMM_ENABLE_GFX938
namespace deepgemm {
namespace {
[[noreturn]] void unsupported_unbuilt_op(const char* op_name) {
    TORCH_CHECK(false, op_name, " was not built for this DeepGEMM extension; "
                       "rebuild with PYTORCH_ROCM_ARCH including the matching source group");
}
}  // namespace

void fp8_gemm(
    torch::Tensor mat_a,
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
    const std::optional<torch::Tensor>& bias) {
    (void)mat_a; (void)mat_b; (void)scale_a; (void)scale_b; (void)output;
    (void)m; (void)n; (void)k; (void)batch; (void)transpose_flag;
    (void)alpha; (void)beta; (void)bias;
    unsupported_unbuilt_op("fp8_gemm");
}

void fp8_gemm_hipc(
    torch::Tensor mat_a,
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
    const std::optional<torch::Tensor>& bias) {
    (void)mat_a; (void)mat_b; (void)scale_a; (void)scale_b; (void)output;
    (void)m; (void)n; (void)k; (void)batch; (void)transpose_flag;
    (void)alpha; (void)beta; (void)bias;
    unsupported_unbuilt_op("fp8_gemm_hipc");
}

torch::Tensor m_grouped_fp8_gemm_nt_contiguous(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int mode) {
    (void)input; (void)b_qweight; (void)output; (void)a_scale;
    (void)b_scale; (void)m_indices; (void)mode;
    unsupported_unbuilt_op("m_grouped_fp8_gemm_nt_contiguous");
}

void tf32_hc_pernorm_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& D,
    const torch::Tensor& sqr_sum,
    const std::optional<int>& num_splits) {
    (void)A; (void)B; (void)D; (void)sqr_sum; (void)num_splits;
    unsupported_unbuilt_op("tf32_hc_pernorm_gemm");
}

namespace einsum {
void einsum(
    const std::string& expr,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c) {
    (void)expr; (void)a; (void)b; (void)d; (void)c;
    unsupported_unbuilt_op("einsum");
}

void fp8_einsum(
    const std::string& expr,
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c,
    const std::tuple<int, int, int>& recipe) {
    (void)expr; (void)a; (void)b; (void)d; (void)c; (void)recipe;
    unsupported_unbuilt_op("fp8_einsum");
}
}  // namespace einsum
}  // namespace deepgemm
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#if !defined(DEEPGEMM_BUILD_ONLY_W8A16) && !defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
    M_GROUP_GEMM_NT_CONTIGUOUS_PYBIND;
    M_GROUP_GEMM_NT_MASKED_PYBIND;
    GEMM_NT;
    FP8_GEMM_HIPC_PYBIND;
#elif defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
    M_GROUP_GEMM_NT_MASKED_W4A8_ONLY_PYBIND;
#endif
#ifndef DEEPGEMM_BUILD_ONLY_W4A8_MASKED
    W8A16_LINEAR_PYBIND;
#endif
#if !defined(DEEPGEMM_BUILD_ONLY_W8A16) && !defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
    TF32_HC_PERNORM_GEMM_PYBIND;
    EINSUM_PYBIND;
#endif
}
