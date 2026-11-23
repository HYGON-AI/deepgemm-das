// Low-latency grouped FP8 masked GEMM — fragment of ``deep_gemm`` (registered in csrc/common/deepgemm_ops.cpp).
#include "m_grouped_gemm_nt_masked.h"

namespace deepgemm{
    torch::Tensor group_gemm_fp8_masked(
        const torch::Tensor &matrix_a,
        const torch::Tensor &matrix_b,
        const torch::Tensor &matri_a_scale,
        const torch::Tensor &matrixb_scale,
        const torch::Tensor &actual_tokens,
        torch::Tensor &matrix_c,
        int64_t max_tokens,
        int64_t experts,
        int64_t cu_s,
        bool block_wise,
        bool b_overlap,
        const c10::optional<torch::Tensor> &signal,
        int64_t mode)
    {
        // DEEP_GEMM::FP8_GROUP_GEMM::MASKED::masked_fp8_gemm(
        //     matrix_a, matrix_b, matri_a_scale, matrixb_scale, actual_tokens, matrix_c,
        //     max_tokens, experts, cu_s, block_wise, b_overlap, signal, mode);
    deepgemm::masked_fp8_gemm(
        matrix_a, matrix_b, matri_a_scale, matrixb_scale, actual_tokens, matrix_c,
        max_tokens, experts, cu_s, block_wise, b_overlap, signal, mode);
    return matrix_c;
    }
}