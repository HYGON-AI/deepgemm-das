// SPDX-License-Identifier: MIT
// Single TORCH_LIBRARY root for namespace ``deep_gemm``. Other translation units must use
// TORCH_LIBRARY_FRAGMENT so PyTorch does not see multiple TORCH_LIBRARY(deep_gemm, ...).
#include <torch/extension.h>

TORCH_LIBRARY(deep_gemm, m) {
  // Operator schemas and CUDA implementations are registered via TORCH_LIBRARY_FRAGMENT
  // in low_latency_grouped_gemm.cu, moe_w8a8_marlin_prefill_down_fp8.cu, etc.
}

