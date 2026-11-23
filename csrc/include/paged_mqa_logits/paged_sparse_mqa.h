// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <optional>

namespace deepgemm {
constexpr int kPagedSparseMqaThreads = 256;
constexpr int kPagedSparseMqaTileTokens = 64;
constexpr int kPagedSparseMqaReuseTiles = 16;

struct PagedSparseMqaMapping {
    const int *lens, *table, *candidates;
    int64_t lens_stride, table_stride, candidate_stride, table_cols;
    int rows, capacity, page, block;
};

struct PagedSparseMqaArgs {
    PagedSparseMqaMapping mapping;
    const uint8_t *q, *kv;
    const uint32_t* scales;
    const float* weights;
    const int64_t* metadata;
    float* out;
    int64_t q_row, q_head, sf_row, sf_head, weight_row, weight_head, kv_stride, pages, out_row;
    int heads, tiles_per_row;
    bool mask_tail;
};

// Keep shape-derived addressing constants in the captured launch arguments.
// Non-power-of-two pages retain exact division; no layout is excluded.
struct PagedSparseMqaAddressArgs {
    PagedSparseMqaArgs base;
    int block_shift, page_shift;
};

struct PagedSparseMqaKernelArgs {
    PagedSparseMqaAddressArgs opt;
    uint8_t* exceptional;
};

void launch_paged_sparse_mqa_metadata(const PagedSparseMqaMapping& mapping, int64_t* metadata, hipStream_t stream);
void launch_paged_sparse_mqa_bf16(PagedSparseMqaKernelArgs args, bool fp4, int device, hipStream_t stream);
void launch_paged_sparse_mqa_fp8(PagedSparseMqaKernelArgs args, bool fp4, int device, hipStream_t stream);

// Opaque compact metadata. Valid sparse block indices are unique and strictly
// increasing per row. Regenerate after mapping changes, including graph replay.
// Metadata is private to HCU and not interchangeable with CUDA/LightOp.
torch::Tensor get_paged_sparse_mqa_logits_metadata(const torch::Tensor& context_lens, const torch::Tensor& block_table,
                                                   const torch::Tensor& indices, int64_t page_kv,
                                                   const torch::Tensor& candidates, at::ScalarType qk_dtype,
                                                   int64_t block_kv);

torch::Tensor fp8_fp4_paged_sparse_mqa_logits(const torch::Tensor& q, const torch::Tensor& q_scales,
                                              const torch::Tensor& kv, const torch::Tensor& weights,
                                              const std::optional<torch::Tensor>& metadata, int64_t capacity,
                                              int64_t block_kv, const std::optional<torch::Tensor>& context_lens,
                                              const std::optional<torch::Tensor>& candidates,
                                              const std::optional<torch::Tensor>& output,
                                              const std::optional<torch::Tensor>& block_table,
                                              const std::optional<bool>& mask_tail);

} // namespace deepgemm
