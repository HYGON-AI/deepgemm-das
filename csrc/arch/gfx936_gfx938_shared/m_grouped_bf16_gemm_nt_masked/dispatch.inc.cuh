// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "bm64_bn256_bk128_small_m.inc.cuh"
#include "bm64_bn256_bk128_prefetch.inc.cuh"

namespace deepgemm::grouped_bf16_gemm_nt_masked_bm64_bn256_bk128 {

constexpr int kAutoMode = 2000;
constexpr int kSmallMMode = 2001;
constexpr int kPrefetchMode = 2002;
constexpr int kMode = kAutoMode;
constexpr int kSmallMThreshold = 64;

inline bool is_supported_mode(int mode) {
  return mode == kAutoMode || mode == kSmallMMode || mode == kPrefetchMode;
}

inline torch::Tensor launch(torch::Tensor input,
                            torch::Tensor packed_weight,
                            torch::Tensor output,
                            torch::Tensor masked_m,
                            int expected_m_per_group,
                            int mode) {
  if (mode == kSmallMMode ||
      (mode == kAutoMode && expected_m_per_group <= kSmallMThreshold)) {
    return grouped_bf16_gemm_nt_masked_bm64_bn256_bk128_small_m::run(
        input, packed_weight, output, masked_m);
  }
  return grouped_bf16_gemm_nt_masked_bm64_bn256_bk128_prefetch::run(
      input, packed_weight, output, masked_m);
}

}  // namespace deepgemm::grouped_bf16_gemm_nt_masked_bm64_bn256_bk128
