// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "paged_mqa_logits/paged_sparse_mqa.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/MemoryOverlap.h>
#include <climits>
#include <string>

namespace deepgemm {
static void sparse_mqa_check_tensor(const torch::Tensor& t, const torch::Tensor& ref, at::ScalarType type, int ndim,
                                    const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == ref.device(), name, ": expected same GPU device");
    TORCH_CHECK(t.scalar_type() == type && t.dim() == ndim, name, ": invalid dtype or rank");
    for (auto stride : t.strides())
        TORCH_CHECK(stride >= 0, name, ": negative stride");
}

static hipDeviceProp_t sparse_mqa_device_properties(int device) {
    hipDeviceProp_t p{};
    TORCH_CHECK(hipGetDeviceProperties(&p, device) == hipSuccess, "cannot query sparse MQA device");
    const std::string arch(p.gcnArchName);
    TORCH_CHECK(arch.rfind("gfx936", 0) == 0 || arch.rfind("gfx938", 0) == 0,
                "paged sparse MQA supports gfx936/gfx938");
    return p;
}

static void sparse_mqa_check_dimensions(int64_t rows, int64_t capacity, int64_t block, int64_t page) {
    TORCH_CHECK(rows >= 0 && rows <= INT_MAX, "invalid number of query rows");
    TORCH_CHECK(capacity > 0 && capacity <= 4096 && capacity % 4 == 0,
                "capacity must be a positive multiple of 4 <=4096");
    TORCH_CHECK((block == 8 || block == 16) && page > 0 && page <= INT_MAX && page % block == 0,
                "sparse_block_kv must be 8 or 16; page_kv must be a positive multiple");
}

static PagedSparseMqaMapping sparse_mqa_mapping(const torch::Tensor& lens, const torch::Tensor& table,
                                                const torch::Tensor& candidates, int64_t page, int64_t block) {
    sparse_mqa_check_tensor(lens, lens, at::kInt, 1, "context_lens");
    sparse_mqa_check_tensor(table, lens, at::kInt, 2, "block_table");
    sparse_mqa_check_tensor(candidates, lens, at::kInt, 2, "sparse_kv_block_indices");
    const auto rows = lens.size(0), capacity = candidates.size(1);
    sparse_mqa_check_dimensions(rows, capacity, block, page);
    TORCH_CHECK(table.size(0) == rows && candidates.size(0) == rows && table.stride(1) == 1 &&
                    candidates.stride(1) == 1,
                "invalid mapping shapes/strides");
    return {lens.data_ptr<int>(),
            table.data_ptr<int>(),
            candidates.data_ptr<int>(),
            lens.stride(0),
            table.stride(0),
            candidates.stride(0),
            table.size(1),
            int(rows),
            int(capacity),
            int(page),
            int(block)};
}

static bool sparse_mqa_fp4_type(at::ScalarType type) {
    TORCH_CHECK(type == at::kChar || type == at::kByte || type == at::ScalarType::Float8_e4m3fn,
                "qk dtype must be packed int8/uint8 MXFP4 or float8_e4m3fn MXFP8");
    return type != at::ScalarType::Float8_e4m3fn;
}

static torch::Tensor make_paged_sparse_mqa_plan(const PagedSparseMqaMapping& mapping, const torch::Tensor& lens) {
    const int chunks = (mapping.capacity * mapping.block + 1023) / 1024;
    auto metadata = torch::empty({(int64_t(mapping.rows) * (mapping.capacity + 1 + chunks) + 4) * 8},
                                 lens.options().dtype(at::kByte));
    if (mapping.rows) {
#if DEEPGEMM_HAS_GFX93X
        launch_paged_sparse_mqa_metadata(mapping, reinterpret_cast<int64_t*>(metadata.data_ptr()),
                                         at::hip::getCurrentHIPStream());
        TORCH_CHECK(hipGetLastError() == hipSuccess, "paged sparse metadata launch failed");
#else
        ::deepgemm::arch::unsupported("get_paged_sparse_mqa_logits_metadata");
#endif
    }
    return metadata;
}

torch::Tensor get_paged_sparse_mqa_logits_metadata(const torch::Tensor& lens, const torch::Tensor& table,
                                                   const torch::Tensor& indices, int64_t page,
                                                   const torch::Tensor& candidates, at::ScalarType dtype,
                                                   int64_t block) {
    auto mapping = sparse_mqa_mapping(lens, table, candidates, page, block);
    sparse_mqa_check_tensor(indices, lens, at::kInt, 1, "indices");
    TORCH_CHECK(indices.size(0) == mapping.rows, "indices must have one request id per row");
    sparse_mqa_fp4_type(dtype);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(lens.device());
    sparse_mqa_device_properties(lens.get_device());
    return make_paged_sparse_mqa_plan(mapping, lens);
}

torch::Tensor fp8_fp4_paged_sparse_mqa_logits(const torch::Tensor& q, const torch::Tensor& sf, const torch::Tensor& kv,
                                              const torch::Tensor& weights,
                                              const std::optional<torch::Tensor>& metadata, int64_t capacity,
                                              int64_t block, const std::optional<torch::Tensor>& lens,
                                              const std::optional<torch::Tensor>& candidates,
                                              const std::optional<torch::Tensor>& output,
                                              const std::optional<torch::Tensor>& table,
                                              const std::optional<bool>& mask_tail) {
    TORCH_CHECK(q.is_cuda() && (q.dim() == 3 || q.dim() == 4), "q must be GPU [R,1,H,D] or [R,H,D]");
    const bool fp4 = sparse_mqa_fp4_type(q.scalar_type());
    const int64_t rows = q.size(0), heads = q.size(-2), payload = fp4 ? 64 : 128;
    TORCH_CHECK(q.size(-1) == payload && q.stride(-1) == 1 && (q.dim() == 3 || q.size(1) == 1), "invalid q layout");
    TORCH_CHECK(heads >= 4 && heads <= 32 && heads % 4 == 0, "H must be 4..32 in steps of 4");
    TORCH_CHECK(sf.dim() == 2 || sf.dim() == 3, "q scales must be [R,H] or [R,1,H]");
    sparse_mqa_check_tensor(sf, q, at::kInt, sf.dim(), "q scales");
    TORCH_CHECK(sf.size(0) == rows && sf.size(-1) == heads && (sf.dim() == 2 || sf.size(1) == 1),
                "invalid q scales shape");
    sparse_mqa_check_tensor(weights, q, at::kFloat, 2, "weights");
    TORCH_CHECK(weights.size(0) == rows && weights.size(1) == heads, "weights must be [R,H]");
    sparse_mqa_check_tensor(kv, q, at::kByte, 4, "KV");
    const int64_t page = kv.size(1);
    sparse_mqa_check_dimensions(rows, capacity, block, page);
    TORCH_CHECK(kv.size(2) == 1 && kv.size(3) == payload + 4 && kv.stride(3) == 1 && kv.stride(1) == payload + 4 &&
                    kv.stride(0) >= page * (payload + 4),
                "KV must have page-local payload/scales storage with optional page padding");
    TORCH_CHECK(lens.has_value() == candidates.has_value(), "supply context_lens and candidates together");
    const bool create_metadata = !metadata.has_value(), tail = mask_tail.value_or(lens.has_value());
    TORCH_CHECK(!tail || lens.has_value(), "tail masking requires context_lens and candidates");
    TORCH_CHECK(create_metadata == table.has_value(), "block_table is required only for metadata=None");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(q.device());
    const auto props = sparse_mqa_device_properties(q.get_device());
    const bool native_fp8 = std::string(props.gcnArchName).rfind("gfx938", 0) == 0;
    torch::Tensor generated_metadata;
    PagedSparseMqaArgs a{};
    a.mapping.rows = rows;
    a.mapping.capacity = capacity;
    a.mapping.page = page;
    a.mapping.block = block;
    if (create_metadata) {
        TORCH_CHECK(lens.has_value(), "metadata=None requires context_lens and candidates");
        a.mapping = sparse_mqa_mapping(*lens, *table, *candidates, page, block);
        TORCH_CHECK(lens->device() == q.device() && a.mapping.rows == rows && a.mapping.capacity == capacity,
                    "mapping does not match q/capacity");
        generated_metadata = make_paged_sparse_mqa_plan(a.mapping, *lens);
        a.metadata = reinterpret_cast<const int64_t*>(generated_metadata.data_ptr());
    } else {
        sparse_mqa_check_tensor(*metadata, q, at::kByte, metadata->dim(), "HCU sparse metadata");
        constexpr int reuse = 16;
        const int64_t chunks =
            (capacity * block + kPagedSparseMqaTileTokens * reuse - 1) / (kPagedSparseMqaTileTokens * reuse);
        const bool compact_meta = metadata->dim() == 1 && metadata->numel() == (rows * (capacity + 1 + chunks) + 4) * 8;
        TORCH_CHECK(metadata->is_contiguous() && compact_meta, "invalid HCU sparse metadata shape or strides");
        a.metadata = reinterpret_cast<const int64_t*>(metadata->data_ptr());
    }
    if (lens) {
        sparse_mqa_check_tensor(*lens, q, at::kInt, 1, "context_lens");
        sparse_mqa_check_tensor(*candidates, q, at::kInt, 2, "candidates");
        TORCH_CHECK(lens->size(0) == rows && candidates->size(0) == rows && candidates->size(1) == capacity &&
                        candidates->stride(1) == 1,
                    "invalid tail input shape/stride");
        a.mapping.lens = lens->data_ptr<int>();
        a.mapping.lens_stride = lens->stride(0);
        a.mapping.candidates = candidates->data_ptr<int>();
        a.mapping.candidate_stride = candidates->stride(0);
    }
    auto out = output.value_or(torch::Tensor());
    if (!output)
        out = torch::empty({rows, capacity * block}, q.options().dtype(at::kFloat));
    sparse_mqa_check_tensor(out, q, at::kFloat, 2, "output");
    TORCH_CHECK(out.size(0) == rows && out.size(1) == capacity * block && out.stride(1) == 1 &&
                    out.stride(0) >= capacity * block,
                "output must be [R,M*block] with a nonoverlapping row stride");
    for (const auto& input : {q, sf, kv, weights})
        at::assert_no_overlap(out, input);
    if (lens) {
        at::assert_no_overlap(out, *lens);
        at::assert_no_overlap(out, *candidates);
    }
    if (table)
        at::assert_no_overlap(out, *table);
    if (metadata)
        at::assert_no_overlap(out, *metadata);
    if (!rows)
        return out;
    a.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
    a.kv = kv.data_ptr<uint8_t>();
    a.scales = reinterpret_cast<const uint32_t*>(sf.data_ptr());
    a.weights = weights.data_ptr<float>();
    a.out = out.data_ptr<float>();
    a.q_row = q.stride(0);
    a.q_head = q.stride(-2);
    a.sf_row = sf.stride(0);
    a.sf_head = sf.stride(-1);
    a.weight_row = weights.stride(0);
    a.weight_head = weights.stride(1);
    a.kv_stride = kv.stride(0);
    a.pages = kv.size(0);
    a.out_row = out.stride(0);
    a.heads = heads;
    a.tiles_per_row = (capacity * block + kPagedSparseMqaTileTokens - 1) / kPagedSparseMqaTileTokens;
    a.mask_tail = tail;
    TORCH_CHECK(uint64_t(rows) * a.tiles_per_row <= UINT_MAX, "sparse MQA grid is too large");
    const auto stream = at::hip::getCurrentHIPStream();
    torch::Tensor flags;
    if (!native_fp8 || fp4)
        flags = torch::empty({int64_t(rows) * a.tiles_per_row * 4}, q.options().dtype(at::kByte));
    const int page_shift = (page & (page - 1)) ? -1 : __builtin_ctz(unsigned(page));
    PagedSparseMqaKernelArgs args{{a, block == 8 ? 3 : 4, page_shift},
                                  flags.defined() ? flags.data_ptr<uint8_t>() : nullptr};
    if (native_fp8) {
#if defined(DEEPGEMM_ENABLE_GFX938)
        launch_paged_sparse_mqa_fp8(args, fp4, q.get_device(), stream);
#else
        TORCH_CHECK(false, "native FP8 requires a gfx938 build");
#endif
    } else {
#if DEEPGEMM_HAS_GFX93X
        launch_paged_sparse_mqa_bf16(args, fp4, q.get_device(), stream);
#else
        ::deepgemm::arch::unsupported("fp8_fp4_paged_sparse_mqa_logits");
#endif
    }
    TORCH_CHECK(hipGetLastError() == hipSuccess, "paged sparse scorer launch failed");
    return out;
}
} // namespace deepgemm
