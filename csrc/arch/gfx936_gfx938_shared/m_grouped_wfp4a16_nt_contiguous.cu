#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <string>

#include "wfp4a16_arch_dispatch.hpp"

namespace {

std::string current_device_arch(const torch::Tensor& tensor) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(tensor));
  int device_id = 0;
  hipError_t err = hipGetDevice(&device_id);
  TORCH_CHECK(err == hipSuccess, "failed to get current HIP device: ", hipGetErrorString(err));

  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, device_id);
  TORCH_CHECK(err == hipSuccess, "failed to get HIP device properties: ", hipGetErrorString(err));

  std::string arch(prop.gcnArchName);
  const auto suffix_pos = arch.find(':');
  if (suffix_pos != std::string::npos) {
    arch.resize(suffix_pos);
  }
  return arch;
}

}  // namespace

torch::Tensor group_gemm_wfp4a16_contiguous(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    torch::Tensor& matrix_c,
    const torch::Tensor& matrixb_scale,
    const torch::Tensor& m_indices) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(matrix_a));
  TORCH_CHECK(matrix_b.device() == matrix_a.device(), "matrix_b must be on the same device as matrix_a");
  TORCH_CHECK(
      matrixb_scale.device() == matrix_a.device(),
      "matrix_b_scale must be on the same device as matrix_a");
  TORCH_CHECK(
      m_indices.device() == matrix_a.device(),
      "m_indices must be on the same device as matrix_a");
  TORCH_CHECK(matrix_c.device() == matrix_a.device(), "matrix_c must be on the same device as matrix_a");

  const std::string arch = current_device_arch(matrix_a);
  if (arch == "gfx938") {
#ifdef DEEPGEMM_ENABLE_GFX938
    deepgemm_wfp4a16_arch_dispatch::contiguous_wfp4a16_gemm_gfx938(
        matrix_a, matrix_b, matrixb_scale, m_indices, matrix_c);
    return matrix_c;
#else
    TORCH_CHECK(false, "WFP4A16 contiguous for gfx938 was not built in this extension");
#endif
  }
  if (arch == "gfx936") {
#ifdef DEEPGEMM_ENABLE_GFX936
    deepgemm_wfp4a16_arch_dispatch::contiguous_wfp4a16_gemm_gfx936(
        matrix_a, matrix_b, matrixb_scale, m_indices, matrix_c);
    return matrix_c;
#else
    TORCH_CHECK(false, "WFP4A16 contiguous for gfx936 was not built in this extension");
#endif
  }
    if (arch == "gfx92a") {
#ifdef DEEPGEMM_ENABLE_GFX92A
    deepgemm_wfp4a16_arch_dispatch::contiguous_wfp4a16_gemm_gfx92a(
        matrix_a, matrix_b, matrixb_scale, m_indices, matrix_c);
    return matrix_c;
#else
    TORCH_CHECK(false, "WFP4A16 contiguous for gfx92a was not built in this extension");
#endif
  }
  TORCH_CHECK(false, "WFP4A16 contiguous supports gfx936/gfx938 only, got ", arch);
  return matrix_c;
}

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
  m.def(
      "m_grouped_wfp4a16_gemm_nt_contiguous("
      "Tensor matrix_a, "
      "Tensor matrix_b, "
      "Tensor matrix_c, "
      "Tensor matrix_b_scale, "
      "Tensor m_indices "
      ") -> (Tensor)"
  );
  m.impl("m_grouped_wfp4a16_gemm_nt_contiguous", torch::kCUDA, &group_gemm_wfp4a16_contiguous);
}
