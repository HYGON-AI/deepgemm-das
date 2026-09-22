#include <c10/cuda/CUDAGuard.h>

#include "moe_marlin/w4a16_masked.h"

torch::Tensor group_gemm_w4a16_masked(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matrixb_scale,
    const torch::Tensor &actual_tokens,
    torch::Tensor &matrix_c,
    int64_t max_tokens,
    int64_t experts)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(matrix_a));
  TORCH_CHECK(matrix_b.device() == matrix_a.device(), "matrix_b must be on the same device as matrix_a");
  TORCH_CHECK(
      matrixb_scale.device() == matrix_a.device(),
      "matrix_b_scale must be on the same device as matrix_a");
  TORCH_CHECK(
      actual_tokens.device() == matrix_a.device(),
      "actual_tokens must be on the same device as matrix_a");
  TORCH_CHECK(matrix_c.device() == matrix_a.device(), "matrix_c must be on the same device as matrix_a");
  deepgemm_w4a16_masked::masked_w4a16_gemm(
      matrix_a, matrix_b, matrixb_scale, actual_tokens, matrix_c, max_tokens, experts);
  return matrix_c;
}

TORCH_LIBRARY_FRAGMENT(deep_gemm, m)
{
  m.def(
      "low_latency_w4a16_gemm("
      "Tensor matrix_a, "
      "Tensor matrix_b, "
      "Tensor matrix_b_scale, "
      "Tensor actual_tokens, "
      "Tensor! matrix_c, "
      "int max_tokens, "
      "int experts "
      ") -> (Tensor)"
  );
  m.impl("low_latency_w4a16_gemm", torch::kCUDA, &group_gemm_w4a16_masked);
}
