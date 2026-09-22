#pragma once

#include <torch/extension.h>
namespace deepgemm {
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
                    const std::optional<torch::Tensor>& bias);

// fp8_e4m3 GEMM on the in-tree warp-specialized producer-consumer kernel
// (csrc/arch/gfx938/fp8_gemm_hipc.cu, kernel fp8_gemm_hipc_kernel_gfx938).
// Shares deepgemm::fp8_gemm's calling convention and mat_a<->mat_b swap
// (arg mat_b = A operand [M,K], arg mat_a = B operand [N,K]), so callers can
// switch engines by changing only the function name. Empty/undefined scales =
// identity (the kernel skips the multiply). Implemented as a thin forwarder
// over run_fp8_gemm_hipc (see fp8_gemm_hipc.cu).
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
                    const std::optional<torch::Tensor>& bias);
}