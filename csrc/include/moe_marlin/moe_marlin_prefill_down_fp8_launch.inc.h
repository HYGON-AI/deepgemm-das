// SPDX-License-Identifier: Apache-2.0
// launch_moe_w8a8_second_stage_prefill_fp8 — sliced from moe/csrc_for_aiter/moe_w8a8_opt.h
#ifndef MOE_MARLIN_PREFILL_DOWN_FP8_LAUNCH_INC_H
#define MOE_MARLIN_PREFILL_DOWN_FP8_LAUNCH_INC_H

#include <algorithm>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "intrinsic_2.h"
#include "moe_w8a8_gemm_params.h"

// Default tile matches mode 86 in kernel_maps_gemm2_prefill_fp8:
//   launch_moe_w8a8_second_stage_prefill_fp8<32,128,64,32,32,64,2>.
template <int BLOCK_SIZE_M,
          int BLOCK_SIZE_N,
          int BLOCK_SIZE_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGES,
          typename T,
          typename T_hidden>
void launch_moe_w8a8_second_stage_prefill_fp8(const GemmParams<T, T_hidden>& params,
                                             hipStream_t stream) {
  const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = true;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  dim3 blockDim;
  dim3 gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  constexpr int n_loop_num = 4;
  gridDim.z = std::min(
      params.size_m * params.top_k,
      static_cast<uint32_t>(
          DIVIDE(static_cast<int>(params.sorted_token_lens), BLOCK_SIZE_M)));
  if (params.size_n % static_cast<uint32_t>(BLOCK_SIZE_N * n_loop_num) != 0) {
    return;
  }
  gridDim.y = DIVIDE(static_cast<int>(params.size_n),
                     BLOCK_SIZE_N * n_loop_num);
  gridDim.x = 1;

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;
  const int shared_mem_size = lds_size;

  if (params.is_marlin == false) {
    return;
  }

  MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_FP8<
      T_hidden,
      char,
      WARP_NUM,
      BLOCK_SIZE_M,
      BLOCK_SIZE_N,
      BLOCK_SIZE_K,
      WARP_M,
      WARP_N,
      WARP_K,
      GROUP_N,
      GROUP_K,
      STAGES,
      mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
      params.ptr_A,
      params.ptr_B0,
      params.ptr_C,
      params.ptr_A_scale,
      params.ptr_B_scale,
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids,
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.stride_asm,
      params.stride_ask,
      params.stride_bse,
      params.stride_bsn,
      params.stride_bsk,
      params.sorted_token_lens,
      params.top_k,
      params.real_topk);
}

#endif /* MOE_MARLIN_PREFILL_DOWN_FP8_LAUNCH_INC_H */

