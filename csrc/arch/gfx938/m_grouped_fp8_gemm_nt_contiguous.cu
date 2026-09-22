// SPDX-License-Identifier: Apache-2.0
// DeepGEMM contiguous FP8 GEMM (marlin tile) — TORCH_LIBRARY_FRAGMENT in
// the ``deep_gemm`` namespace. Consumes ``m_indices`` directly (no Python-
// side routing build).
//
// Exposes deep_gemm::m_grouped_fp8_gemm_nt_contiguous_marlin
//   mode       : 0 -> auto by M, 1000/32 -> BM=32, 64 -> BM=64, 128 -> BM=128
//   n_loop_num : 0 = auto-pick largest valid in {4,2,1}; else must be 4/2/1

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "moe_marlin/moe_w8a8_gemm_params.h"
#include "moe_marlin/moe_w8a8_utils.h"
#include "moe_marlin/m_grouped_fp8_gemm_nt_contig_device.inc.h"
#include "moe_marlin/m_grouped_fp8_gemm_nt_contig_launch.inc.h"

namespace m_grouped_fp8_contig {

namespace {

constexpr int kBlockN = 128;

const char* as_char_ptr(const torch::Tensor& t) {
  TORCH_CHECK(t.is_contiguous(), "expected contiguous tensor for as_char_ptr");
  TORCH_CHECK(t.element_size() == 1, "expected 1-byte element (fp8_e4m3 / int8)");
  return reinterpret_cast<const char*>(t.data_ptr());
}

int pick_block_m(int size_m, int64_t mode) {
  if (mode == 0) {
    if (size_m % 64 == 0) {
      return 64;
    }
    if (size_m % 32 == 0) {
      return 32;
    }
    TORCH_CHECK(false, "m_grouped_fp8_gemm_nt_contiguous_marlin: size_m=", size_m,
                " is not divisible by 32/64/128 for auto BLOCK_M");
  }
  if (mode == 64) {
    return 64;
  }
  if (mode == 128) {
    return 128;
  }
  if (mode == 1000 || mode == 32) {
    return 32;
  }
  TORCH_CHECK(false, "m_grouped_fp8_gemm_nt_contiguous_marlin: unsupported mode ", mode,
              " (use 0/auto, 1000/32, 64, or 128)");
}

bool is_valid_n_loop(int size_n, int n_loop_num) {
  return n_loop_num > 0 && size_n % (kBlockN * n_loop_num) == 0;
}

int pick_n_loop_num(int size_n, int64_t requested) {
  if (requested != 0) {
    TORCH_CHECK(requested == 1 || requested == 2 || requested == 4,
                "n_loop_num must be 0 (auto), 1, 2, or 4; got ", requested);
    TORCH_CHECK(is_valid_n_loop(size_n, static_cast<int>(requested)),
                "size_n=", size_n, " is not divisible by BLOCK_N*n_loop_num=",
                kBlockN * requested);
    return static_cast<int>(requested);
  }
  // Prefer larger n_loop_num (fewer N-grid tiles) when shape allows.
  for (const int candidate : {4, 2, 1}) {
    if (is_valid_n_loop(size_n, candidate)) {
      return candidate;
    }
  }
  TORCH_CHECK(false,
              "m_grouped_fp8_gemm_nt_contiguous_marlin: size_n=", size_n,
              " must be divisible by at least one of ",
              kBlockN * 4, ", ", kBlockN * 2, ", or ", kBlockN);
}

template <int BLOCK_M, int N_LOOP_NUM, bool B_TENSOR_WISE>
void launch_tile(const LaunchParams& params, hipStream_t stream) {
  launch_m_grouped_fp8_gemm_nt_contig<
      BLOCK_M, kBlockN, 64, BLOCK_M, 32, 64, 2, N_LOOP_NUM, B_TENSOR_WISE, bhalf_t>(params, stream);
}

template <bool B_TENSOR_WISE>
void dispatch_launch_impl(int block_m, int n_loop_num, const LaunchParams& params, hipStream_t stream) {
  if (block_m == 32) {
    if (n_loop_num == 4) {
      launch_tile<32, 4, B_TENSOR_WISE>(params, stream);
    } else if (n_loop_num == 2) {
      launch_tile<32, 2, B_TENSOR_WISE>(params, stream);
    } else {
      launch_tile<32, 1, B_TENSOR_WISE>(params, stream);
    }
  } else if (block_m == 64) {
    if (n_loop_num == 4) {
      launch_tile<64, 4, B_TENSOR_WISE>(params, stream);
    } else if (n_loop_num == 2) {
      launch_tile<64, 2, B_TENSOR_WISE>(params, stream);
    } else {
      launch_tile<64, 1, B_TENSOR_WISE>(params, stream);
    }
  } else if (block_m == 128) {
    if (n_loop_num == 4) {
      launch_tile<128, 4, B_TENSOR_WISE>(params, stream);
    } else if (n_loop_num == 2) {
      launch_tile<128, 2, B_TENSOR_WISE>(params, stream);
    } else {
      launch_tile<128, 1, B_TENSOR_WISE>(params, stream);
    }
  } else {
    TORCH_CHECK(false, "unsupported BLOCK_M=", block_m);
  }
}

void dispatch_launch(int block_m, int n_loop_num, const LaunchParams& params, hipStream_t stream) {
  dispatch_launch_impl<false>(block_m, n_loop_num, params, stream);
}

void dispatch_launch_tensor_wise(int block_m, int n_loop_num, const LaunchParams& params, hipStream_t stream) {
  dispatch_launch_impl<true>(block_m, n_loop_num, params, stream);
}
} // namespace

torch::Tensor m_grouped_fp8_gemm_nt_contiguous_marlin(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode,
    int64_t n_loop_num) {
  const c10::OptionalDeviceGuard device_guard(device_of(input));

  TORCH_CHECK(input.is_cuda() && b_qweight.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(input.element_size() == 1, "input must be 1-byte (fp8_e4m3 or int8)");
  TORCH_CHECK(b_qweight.element_size() == 1, "weight must be 1-byte");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output bf16");
  TORCH_CHECK(
      a_scale.scalar_type() == at::kFloat && b_scale.scalar_type() == at::kFloat,
      "scales float32");
  TORCH_CHECK(m_indices.scalar_type() == at::kInt, "m_indices int32");
  TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
  TORCH_CHECK(m_indices.is_cuda(), "m_indices must be on CUDA");

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(output.size(-1));

  const int block_m = pick_block_m(size_m, mode);
  const int resolved_n_loop = pick_n_loop_num(size_n, n_loop_num);

  TORCH_CHECK(size_m % block_m == 0,
              "m_grouped_fp8_gemm_nt_contiguous_marlin: size_m=", size_m,
              " must be a multiple of BLOCK_M=", block_m);
  TORCH_CHECK(static_cast<int>(m_indices.numel()) == size_m,
              "m_indices length must equal size_m");
  TORCH_CHECK(size_k % 64 == 0,
              "m_grouped_fp8_gemm_nt_contiguous_marlin: size_k=", size_k,
              " must be a multiple of 64");

  TORCH_CHECK(a_scale.dim() == 1 || a_scale.dim() == 2,
              "a_scale must be 1-D [M] or 2-D [M, 1]");
  TORCH_CHECK(b_scale.dim() == 2 || b_scale.dim() == 3,
              "b_scale must be 2-D [E, N] or 3-D [E, N, 1]");

  const int stride_asm = static_cast<int>(a_scale.stride(0));
  const int stride_ask = a_scale.dim() == 2 ? static_cast<int>(a_scale.stride(1)) : 0;
  const int stride_bse = static_cast<int>(b_scale.stride(0));
  const int stride_bsn = static_cast<int>(b_scale.stride(1));
  const int stride_bsk = b_scale.dim() == 3 ? static_cast<int>(b_scale.stride(2)) : 0;

  LaunchParams params{
      as_char_ptr(input),
      as_char_ptr(b_qweight),
      output.data_ptr(),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      m_indices.data_ptr<int32_t>(),
      static_cast<uint32_t>(size_m),
      static_cast<uint32_t>(size_n),
      static_cast<uint32_t>(size_k),
      static_cast<uint32_t>(stride_asm),
      static_cast<uint32_t>(stride_ask),
      static_cast<uint32_t>(stride_bse),
      static_cast<uint32_t>(stride_bsn),
      static_cast<uint32_t>(stride_bsk),
  };

  const hipStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  dispatch_launch(block_m, resolved_n_loop, params, stream);
  return output;
}


torch::Tensor m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    int64_t mode,
    int64_t n_loop_num) {
  const c10::OptionalDeviceGuard device_guard(device_of(input));

  TORCH_CHECK(input.is_cuda() && b_qweight.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(input.element_size() == 1, "input must be 1-byte (fp8_e4m3 or int8)");
  TORCH_CHECK(b_qweight.element_size() == 1, "weight must be 1-byte");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output bf16");
  TORCH_CHECK(
      a_scale.scalar_type() == at::kFloat && b_scale.scalar_type() == at::kFloat,
      "scales float32");
  TORCH_CHECK(m_indices.scalar_type() == at::kInt, "m_indices int32");
  TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
  TORCH_CHECK(m_indices.is_cuda(), "m_indices must be on CUDA");

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(output.size(-1));

  const int block_m = pick_block_m(size_m, mode);
  const int resolved_n_loop = pick_n_loop_num(size_n, n_loop_num);

  TORCH_CHECK(size_m % block_m == 0,
              "m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise: size_m=", size_m,
              " must be a multiple of BLOCK_M=", block_m);
  TORCH_CHECK(static_cast<int>(m_indices.numel()) == size_m,
              "m_indices length must equal size_m");
  TORCH_CHECK(size_k % 64 == 0,
              "m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise: size_k=", size_k,
              " must be a multiple of 64");

  TORCH_CHECK(a_scale.dim() == 1 || a_scale.dim() == 2,
              "a_scale must be 1-D [M] or 2-D [M, 1]");
  TORCH_CHECK(b_scale.dim() == 1 || (b_scale.dim() == 2 && b_scale.size(1) == 1),
              "tensor-wise b_scale must be 1-D [E] or 2-D [E, 1]");

  const int stride_asm = static_cast<int>(a_scale.stride(0));
  const int stride_ask = a_scale.dim() == 2 ? static_cast<int>(a_scale.stride(1)) : 0;

  LaunchParams params{
      as_char_ptr(input),
      as_char_ptr(b_qweight),
      output.data_ptr(),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      m_indices.data_ptr<int32_t>(),
      static_cast<uint32_t>(size_m),
      static_cast<uint32_t>(size_n),
      static_cast<uint32_t>(size_k),
      static_cast<uint32_t>(stride_asm),
      static_cast<uint32_t>(stride_ask),
      static_cast<uint32_t>(size_n),
      1,
      0,
  };

  const hipStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  dispatch_launch_tensor_wise(block_m, resolved_n_loop, params, stream);
  return output;
}

} // namespace m_grouped_fp8_contig

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
  m.def(
      "m_grouped_fp8_gemm_nt_contiguous_marlin("
      "Tensor input, "
      "Tensor b_qweight, "
      "Tensor output, "
      "Tensor a_scale, "
      "Tensor b_scale, "
      "Tensor m_indices, "
      "int mode, "
      "int n_loop_num"
      ") -> Tensor");
  m.impl(
      "m_grouped_fp8_gemm_nt_contiguous_marlin",
      torch::kCUDA,
      &m_grouped_fp8_contig::m_grouped_fp8_gemm_nt_contiguous_marlin);

  m.def(
      "m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise("
      "Tensor input, "
      "Tensor b_qweight, "
      "Tensor output, "
      "Tensor a_scale, "
      "Tensor b_scale, "
      "Tensor m_indices, "
      "int mode, "
      "int n_loop_num"
      ") -> Tensor");
  m.impl(
      "m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise",
      torch::kCUDA,
      &m_grouped_fp8_contig::m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise);
}
