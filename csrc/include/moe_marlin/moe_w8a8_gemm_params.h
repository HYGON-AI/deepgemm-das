// SPDX-License-Identifier: Apache-2.0
// Extracted GemmParams only (see moe/csrc_for_aiter/moe_w8a8_config.h).
#ifndef MOE_W8A8_GEMM_PARAMS_LL_H
#define MOE_W8A8_GEMM_PARAMS_LL_H

#include <cstdint>

#include "numeric_types.h"

template <typename T, typename T_hidden = bhalf_t>
struct GemmParams {
  GemmParams(const T* ptr_A, const T* ptr_B0, T_hidden* ptr_C, float* ptr_A_scale,
             float* ptr_B_scale, const float* topk_weights,
             const int32_t* sorted_token_ids, const int32_t* expert_ids,
             const int32_t num_tokens_post_pad,
             const int32_t* num_tokens_post_pad_ptr, uint32_t size_m,
             uint32_t size_n, uint32_t size_k, uint32_t stride_asm,
             uint32_t stride_ask, uint32_t stride_bse, uint32_t stride_bsn,
             uint32_t stride_bsk, uint32_t sorted_token_lens, uint32_t top_k,
             uint32_t real_topk, bool is_marlin)
      : ptr_A(ptr_A), ptr_B0(ptr_B0), ptr_C(ptr_C), ptr_A_scale(ptr_A_scale),
        ptr_B_scale(ptr_B_scale), topk_weights(topk_weights),
        sorted_token_ids(sorted_token_ids), expert_ids(expert_ids),
        num_tokens_post_pad(num_tokens_post_pad),
        num_tokens_post_pad_ptr(num_tokens_post_pad_ptr), size_m(size_m),
        size_n(size_n), size_k(size_k), stride_asm(stride_asm),
        stride_ask(stride_ask), stride_bse(stride_bse), stride_bsn(stride_bsn),
        stride_bsk(stride_bsk), sorted_token_lens(sorted_token_lens),
        top_k(top_k), real_topk(real_topk), is_marlin(is_marlin) {}

  const T* ptr_A;
  const T* ptr_B0;
  T_hidden* ptr_C;
  float* ptr_A_scale;
  float* ptr_B_scale;
  const float* topk_weights;
  const int32_t* sorted_token_ids;
  const int32_t* expert_ids;
  const int32_t num_tokens_post_pad;
  const int32_t* num_tokens_post_pad_ptr;
  uint32_t size_m;
  uint32_t size_n;
  uint32_t size_k;
  uint32_t stride_asm;
  uint32_t stride_ask;
  uint32_t stride_bse;
  uint32_t stride_bsn;
  uint32_t stride_bsk;
  uint32_t sorted_token_lens;
  uint32_t top_k;
  uint32_t real_topk;
  bool is_marlin;
};

#endif
