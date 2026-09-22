// SPDX-License-Identifier: Apache-2.0
// MoE Marlin W8A8 GEMM2 prefill-down FP8 — TORCH_LIBRARY_FRAGMENT in ``deep_gemm`` namespace.
//
// Exposes MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_FP8 (MoE Marlin routing API).

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "moe_marlin/moe_w8a8_gemm_params.h"
#include "moe_marlin/moe_w8a8_utils.h"
#include "moe_marlin/moe_marlin_prefill_down_fp8_device.inc.h"
#include "moe_marlin/moe_marlin_prefill_down_fp8_launch.inc.h"

namespace moe_marlin {

namespace {

const char* as_char_ptr(const torch::Tensor& t) {
  TORCH_CHECK(t.is_contiguous(), "expected contiguous tensor for as_char_ptr");
  TORCH_CHECK(t.element_size() == 1, "expected 1-byte element (fp8_e4m3 / int8 / uint8)");
  return reinterpret_cast<const char*>(t.data_ptr());
}

} // namespace

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
    int64_t real_topk) {
  const c10::OptionalDeviceGuard device_guard(device_of(input));

  TORCH_CHECK(input.is_cuda() && b_qweight.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(input.element_size() == 1, "input must be 1-byte (fp8_e4m3 or int8)");
  TORCH_CHECK(b_qweight.element_size() == 1, "weight must be 1-byte (fp8_e4m3 or int8)");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output bf16");
  TORCH_CHECK(
      a_scale.scalar_type() == at::kFloat && b_scale.scalar_type() == at::kFloat,
      "scales float32");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights float32");
  TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt, "sorted_token_ids int32");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "expert_ids int32");
  TORCH_CHECK(num_tokens_post_pad.scalar_type() == at::kInt, "num_tokens_post_pad int32");

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int size_n = static_cast<int>(output.size(-1));

  TORCH_CHECK(size_k % 64 == 0,
              "moe_marlin_prefill_down_fp8: size_k=", size_k,
              " must be a multiple of 64");

  // Default tile (mode 86): <32,128,64,32,32,64,2>, n_loop_num=4 → N % 512 == 0.
  constexpr int kBlockN = 128;
  constexpr int kNLoopNum = 4;
  TORCH_CHECK(size_n % (kBlockN * kNLoopNum) == 0,
              "moe_marlin_prefill_down_fp8: size_n=", size_n,
              " must be a multiple of ", kBlockN * kNLoopNum,
              " for tile <32,128,64,32,32,64,2>");

  TORCH_CHECK(a_scale.dim() == 1 || a_scale.dim() == 2,
              "a_scale must be 1-D [M] or 2-D [M, 1]");
  TORCH_CHECK(b_scale.dim() == 2 || b_scale.dim() == 3,
              "b_scale must be 2-D [E, N] or 3-D [E, N, 1]");
  const int stride_asm = static_cast<int>(a_scale.stride(0));
  const int stride_ask = a_scale.dim() == 2 ? static_cast<int>(a_scale.stride(1)) : 0;
  const int stride_bse = static_cast<int>(b_scale.stride(0));
  const int stride_bsn = static_cast<int>(b_scale.stride(1));
  const int stride_bsk = b_scale.dim() == 3 ? static_cast<int>(b_scale.stride(2)) : 0;

  const int64_t EM = sorted_token_ids.size(0);

  GemmParams<char, bhalf_t> params(
      as_char_ptr(input),
      as_char_ptr(b_qweight),
      reinterpret_cast<bhalf_t*>(output.data_ptr()),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      topk_weights.data_ptr<float>(),
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      0,
      num_tokens_post_pad.data_ptr<int32_t>(),
      static_cast<uint32_t>(size_m),
      static_cast<uint32_t>(size_n),
      static_cast<uint32_t>(size_k),
      static_cast<uint32_t>(stride_asm),
      static_cast<uint32_t>(stride_ask),
      static_cast<uint32_t>(stride_bse),
      static_cast<uint32_t>(stride_bsn),
      static_cast<uint32_t>(stride_bsk),
      static_cast<uint32_t>(EM),
      static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(real_topk),
      /*is_marlin=*/true);

  const hipStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  launch_moe_w8a8_second_stage_prefill_fp8<32, 128, 64, 32, 32, 64, 2>(params, stream);
  return output;
}

} // namespace moe_marlin

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
  m.def(
      "moe_w8a8_marlin_prefill_down_fp8("
      "Tensor input, "
      "Tensor b_qweight, "
      "Tensor output, "
      "Tensor a_scale, "
      "Tensor b_scale, "
      "Tensor topk_weights, "
      "Tensor sorted_token_ids, "
      "Tensor expert_ids, "
      "Tensor num_tokens_post_pad, "
      "int top_k, "
      "int real_topk"
      ") -> Tensor");
  m.impl(
      "moe_w8a8_marlin_prefill_down_fp8",
      torch::kCUDA,
      &moe_marlin::moe_marlin_w8a8_gemm2_prefill_down_fp8);
}


