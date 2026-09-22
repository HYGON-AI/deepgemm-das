// SPDX-License-Identifier: Apache-2.0
#ifndef M_GROUPED_FP8_GEMM_NT_CONTIG_LAUNCH_INC_H
#define M_GROUPED_FP8_GEMM_NT_CONTIG_LAUNCH_INC_H

#include <algorithm>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "intrinsic_2.h"
#include "m_grouped_fp8_gemm_nt_contig_device.inc.h"

namespace m_grouped_fp8_contig {

struct LaunchParams {
  const char* ptr_A;
  const char* ptr_B;
  void* ptr_C;
  float* ptr_A_scale;
  float* ptr_B_scale;
  const int32_t* m_indices;
  uint32_t size_m;
  uint32_t size_n;
  uint32_t size_k;
  uint32_t stride_asm;
  uint32_t stride_ask;
  uint32_t stride_bse;
  uint32_t stride_bsn;
  uint32_t stride_bsk;
};

// Mirrors the gridDim / blockDim of launch_moe_w8a8_second_stage_prefill_fp8.
// Default tile is the mode-86 marlin tile <32,128,64,32,32,64,2>.
template <int BLOCK_SIZE_M,
          int BLOCK_SIZE_N,
          int BLOCK_SIZE_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGES,
          int N_LOOP_NUM,
          bool B_TENSOR_WISE,
          typename T_hidden>
void launch_m_grouped_fp8_gemm_nt_contig(const LaunchParams& params,
                                        hipStream_t stream) {
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;
  constexpr int n_loop_num = N_LOOP_NUM;

  dim3 blockDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  if (params.size_n % static_cast<uint32_t>(BLOCK_SIZE_N * n_loop_num) != 0) {
    return;
  }

  dim3 gridDim;
  gridDim.x = 1;
  gridDim.y = DIVIDE(static_cast<int>(params.size_n),
                     BLOCK_SIZE_N * n_loop_num);
  gridDim.z = DIVIDE(static_cast<int>(params.size_m), BLOCK_SIZE_M);

  const int lds_size = BLOCK_SIZE_M * WARP_K * 2;
  const int shared_mem_size = lds_size;

  M_GROUPED_FP8_GEMM_NT_CONTIG_MARLIN<
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
      N_LOOP_NUM,
      B_TENSOR_WISE><<<gridDim, blockDim, shared_mem_size, stream>>>(
      params.ptr_A,
      params.ptr_B,
      reinterpret_cast<T_hidden*>(params.ptr_C),
      params.ptr_A_scale,
      params.ptr_B_scale,
      params.m_indices,
      params.size_m,
      params.size_n,
      params.size_k,
      params.stride_asm,
      params.stride_ask,
      params.stride_bse,
      params.stride_bsn,
      params.stride_bsk);
}

} // namespace m_grouped_fp8_contig

#endif /* M_GROUPED_FP8_GEMM_NT_CONTIG_LAUNCH_INC_H */
