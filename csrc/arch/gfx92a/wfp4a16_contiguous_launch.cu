#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>

#include "moe_marlin/wfp4a16_contiguous_device.h"
#include "wfp4a16_arch_dispatch.hpp"

namespace deepgemm_wfp4a16_arch_dispatch {
namespace {

template <typename T, bool INTERLEAVE, bool DEQUANT_16BIT>
void launch_contiguous_explicit(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  const auto config = deepgemm_wfp4a16_contiguous::select_wfp4a16_contiguous_config(
      matrix_a.size(0), matrix_c.size(1), matrix_a.size(1));
  TORCH_CHECK(config.mode == 514,
              "WFP4A16 contiguous currently exposes AITER prefill/gemm1 mode 514 only");
  deepgemm_wfp4a16_contiguous::launch_wfp4a16_contiguous_optimized_explicit<
      T, INTERLEAVE, DEQUANT_16BIT>(
      matrix_a,
      matrix_b,
      matrix_b_scale,
      m_indices,
      matrix_c,
      matrix_a_ptr,
      matrix_c_ptr);
}

template <bool INTERLEAVE, bool DEQUANT_16BIT>
void contiguous_wfp4a16_gemm_explicit(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c) {
  TORCH_CHECK(matrix_a.is_contiguous(), "matrix_a must be contiguous");
  TORCH_CHECK(matrix_b.is_contiguous(), "matrix_b must be contiguous");
  TORCH_CHECK(matrix_b_scale.is_contiguous(), "matrix_b_scale must be contiguous");
  TORCH_CHECK(matrix_c.is_contiguous(), "matrix_c must be contiguous");
  TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
  TORCH_CHECK(m_indices.scalar_type() == at::kInt, "m_indices must be int32");
  TORCH_CHECK(matrix_a.scalar_type() == at::ScalarType::Half ||
                  matrix_a.scalar_type() == at::ScalarType::BFloat16,
              "wfp4a16 expects fp16/bf16 activations");
  TORCH_CHECK(matrix_a.scalar_type() == matrix_c.scalar_type(),
              "matrix_c dtype must match matrix_a dtype");
  TORCH_CHECK(matrix_b.scalar_type() == at::ScalarType::Byte,
              "wfp4a16 expects packed uint8 FP4 E2M1 weights");
  TORCH_CHECK(matrix_b_scale.scalar_type() == at::ScalarType::Byte,
              "wfp4a16 expects uint8 E8M0 scales");
  TORCH_CHECK(matrix_a.dim() == 2, "matrix_a must be [M,K]");
  TORCH_CHECK(matrix_b.dim() == 3, "matrix_b must be [E,N,K//2]");
  TORCH_CHECK(matrix_b_scale.dim() == 3, "matrix_b_scale must be [E,N,K//32]");
  TORCH_CHECK(matrix_c.dim() == 2, "matrix_c must be [M,N]");
  TORCH_CHECK(matrix_a.size(0) > 0, "M must be positive");
  TORCH_CHECK(matrix_b.size(0) > 0, "experts must be positive");
  TORCH_CHECK(matrix_c.size(0) == matrix_a.size(0));
  TORCH_CHECK(matrix_c.size(1) == matrix_b.size(1));
  TORCH_CHECK(m_indices.numel() == matrix_a.size(0),
              "m_indices length must match M");
  TORCH_CHECK(matrix_a.size(1) % 64 == 0,
              "AITER WFP4A16 prefill/gemm1 mode 514 requires K divisible by 64");
  TORCH_CHECK(matrix_b.size(2) == matrix_a.size(1) / 2);
  TORCH_CHECK(matrix_b_scale.size(0) == matrix_b.size(0));
  TORCH_CHECK(matrix_b_scale.size(1) == matrix_b.size(1));
  TORCH_CHECK(matrix_b_scale.size(2) == matrix_a.size(1) / 32);

  if (matrix_a.scalar_type() == at::ScalarType::Half) {
    launch_contiguous_explicit<__half, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a,
        matrix_b,
        matrix_b_scale,
        m_indices,
        matrix_c,
        reinterpret_cast<const __half*>(matrix_a.data_ptr<c10::Half>()),
        reinterpret_cast<__half*>(matrix_c.data_ptr<c10::Half>()));
  } else {
    launch_contiguous_explicit<__hip_bfloat16, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a,
        matrix_b,
        matrix_b_scale,
        m_indices,
        matrix_c,
        reinterpret_cast<const __hip_bfloat16*>(matrix_a.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__hip_bfloat16*>(matrix_c.data_ptr<c10::BFloat16>()));
  }
}

}  // namespace

void contiguous_wfp4a16_gemm_gfx92a(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c) {
  contiguous_wfp4a16_gemm_explicit<false, false>(
      matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c);
}

}  // namespace deepgemm_wfp4a16_arch_dispatch
