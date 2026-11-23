// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#if !defined(DEEPGEMM_BUILD_ONLY_W8A16) || defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
#include "m_grouped_gemm_nt_masked.h"
#endif

#ifdef DEEPGEMM_BUILD_ONLY_W4A8_MASKED
#include "m_grouped_w4a8_gemm_nt_contiguous.h"
#endif

#if !defined(DEEPGEMM_BUILD_ONLY_W8A16) && !defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
#include "asm_m_grouped_w8a8_gemm_nt_contiguous.h"
#include "asm_m_grouped_w16a16_gemm_nt_contiguous.h"
#include "m_grouped_w4a8_gemm_nt_contiguous.h"
#include "gemm.h"
#include "tf32_hc_prenorm_gemm.h"
#include "einsum.h"
#endif
#include "moe_marlin/w8a16_linear.h"
#include <torch/extension.h>

#include "deepgemm_ops.hpp"
#include "mqa_logits/dispatch.h"
#include "shared_ops.h"
#include "kernel_registry_decls.h"
#include "paged_mqa_logits/dispatch.h"
#include "paged_mqa_logits/paged_sparse_mqa.h"
#include "mega_mhc.h"

#include <optional>
#include <string>
#include <tuple>
#include <utility>

// NOTE: this file only contains the pybind bindings. Every bound operator is
// declared in csrc/arch/wrapper/include/shared_ops.h and defined in
// csrc/arch/wrapper/*.cu, which dispatches at runtime to the implementation of
// the detected GPU architecture, or raises an error when the architecture does
// not support it.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#if defined(DEEPGEMM_BUILD_ONLY_W8A16)
    W8A16_LINEAR_PYBIND;
#elif defined(DEEPGEMM_BUILD_ONLY_W4A8_MASKED)
    M_GROUP_GEMM_NT_MASKED_W4A8_ONLY_PYBIND;
#else
    M_GROUP_GEMM_NT_CONTIGUOUS_PYBIND;
    M_GROUP_GEMM_NT_MASKED_PYBIND;
    GEMM_NT;
    FP8_GEMM_HIPC_PYBIND;
    W8A16_LINEAR_PYBIND;
    TF32_HC_PRENORM_GEMM_PYBIND;
    EINSUM_PYBIND;
    PAGED_MQA_LOGITS_PYBIND;
    MQA_LOGITS_PYBIND;
    PAGED_SPARSE_MQA_LOGITS_PYBIND;
    MEGA_MHC_PYBIND;
    KERNEL_REGISTRY_OPS_PYBIND;
#endif
}
