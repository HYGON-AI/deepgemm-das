// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <optional>

namespace deepgemm {

torch::Tensor paged_mqa_logits_native(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const int& max_context_len,
    const bool& clean_logits,
    std::optional<torch::Tensor> supplied_output = std::nullopt);

void launch_paged_mqa_logits_fp8(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& kv_cache_scales,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    torch::Tensor& logits,
    int64_t aligned_max_context_len,
    int64_t launch_tokens,
    bool clean_logits);

struct PagedMqaFp4Args {
    const uint8_t* q;
    const uint8_t* kv;
    const int* sf;
    const int* lens;
    const int* table;
    const int* schedule;
    const float* weights;
    float* output;
    uint8_t* exceptional;
    int64_t kv_stride;
    int64_t weight_stride;
    int64_t table_stride;
    int64_t output_stride;
    int rows;
    int heads;
    int width;
    int table_shift;
    int rows_per_request;
    int routes;
    int tiles;
    int schedule_ctas;
    bool paired;
    bool small_kernel;
};

void launch_paged_mqa_fp4_metadata(
    const int* lens, int* schedule, int rows, int rows_per_request, int ctas, hipStream_t stream);
void launch_paged_mqa_fp4_fp16(PagedMqaFp4Args args, int device, hipStream_t stream);
void launch_paged_mqa_fp4_fp8(PagedMqaFp4Args args, int device, hipStream_t stream);

} // namespace deepgemm
