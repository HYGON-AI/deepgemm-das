// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "arch_dispatch.h"

#include "paged_mqa_logits/dispatch.h"

#include "paged_mqa_logits/opus/paged_mqa_logits.h"
#include "paged_mqa_logits/paged_mqa_logits.h"
#include "paged_mqa_logits/paged_mqa_logits_prefill.h"
#include "paged_mqa_logits/paged_mqa_logits_sparse.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>

namespace deepgemm {
namespace {

enum class Backend { opus, native };

struct DeviceInfo {
    bool gfx936;
    bool gfx938;
    int compute_units;
};

const DeviceInfo& get_device_info(int device) {
    static thread_local int cached_device = -1;
    static thread_local DeviceInfo info{};
    if (cached_device != device) {
        hipDeviceProp_t prop{};
        TORCH_CHECK(hipGetDeviceProperties(&prop, device) == hipSuccess,
                    "paged_mqa_logits: cannot query HIP device");
        info = {std::strncmp(prop.gcnArchName, "gfx936", 6) == 0,
                std::strncmp(prop.gcnArchName, "gfx938", 6) == 0, prop.multiProcessorCount};
        cached_device = device;
    }
    return info;
}

bool is_one_of(int64_t value, std::initializer_list<int64_t> values) {
    for (const int64_t candidate : values) {
        if (value == candidate) return true;
    }
    return false;
}

// This is the only cross-implementation selection point. Device tensor contents,
// including context_lens and block_table, must not affect the selected backend.
Backend select_backend(const torch::Tensor& q,
                       const torch::Tensor& cache, int64_t max_model_len,
                       const std::string& implementation, int64_t kernel_id, bool gfx936,
                       bool opus_supported, bool native_supported) {
    std::string requested = implementation;
    const char* override = std::getenv("DEEPGEMM_PAGED_MQA_LOGITS_BACKEND");
    const bool overridden = override && *override;
    // The environment override takes precedence on both architectures, including
    // explicit implementation/kernel_id arguments. An unsupported override fails.
    if (overridden) {
        TORCH_CHECK(std::strcmp(override, "opus") == 0 || std::strcmp(override, "native") == 0,
                    "paged_mqa_logits: DEEPGEMM_PAGED_MQA_LOGITS_BACKEND must be opus or native");
        requested = override;
    }
    TORCH_CHECK(requested == "auto" || requested == "opus" || requested == "native",
                "paged_mqa_logits: implementation must be auto, opus, or native");
    if (requested == "opus" || (requested == "auto" && kernel_id != -1)) {
        TORCH_CHECK(opus_supported, "paged_mqa_logits: OPUS backend does not support these tensor metadata");
        return Backend::opus;
    }
    if (requested == "native") {
        TORCH_CHECK(native_supported, "paged_mqa_logits: native backend does not support these tensor metadata");
        TORCH_CHECK(overridden || kernel_id == -1,
                    "paged_mqa_logits: kernel_id applies only to the OPUS backend");
        return Backend::native;
    }
    TORCH_CHECK(opus_supported || native_supported,
                "paged_mqa_logits: neither backend supports these tensor metadata");
    if (!native_supported) return Backend::opus;
    if (!opus_supported) return Backend::native;

    if (!gfx936) return Backend::opus;
    // On gfx936, OPUS favors one query or a small single-request cache.
    // Multi-query requests with larger cache storage favor native KV reuse.
    // Use allocation metadata only: graph replay may change device lengths.
    const bool small_single = q.size(0) == 1 &&
        (q.size(1) == 1 || cache.size(0) * cache.size(1) <= max_model_len / 2);
    return small_single ? Backend::opus : Backend::native;
}

void print_debug_info(const torch::Tensor& q,
                      const torch::Tensor& fused_kv_cache,
                      const torch::Tensor& weights,
                      const torch::Tensor& context_lens,
                      const torch::Tensor& block_table,
                      int64_t max_model_len,
                      const std::optional<torch::Tensor>& out,
                      bool clean_logits,
                      const std::string& implementation,
                      int64_t kernel_id,
                      bool gfx936,
                      Backend backend,
                      const char* mode = "decode") {
    hipStreamCaptureStatus capture_status{};
    TORCH_CHECK(hipStreamIsCapturing(at::hip::getCurrentHIPStream(), &capture_status) == hipSuccess,
                "paged_mqa_logits: cannot query HIP stream capture status");
    const char* phase = capture_status == hipStreamCaptureStatusNone ? "eager" :
                        capture_status == hipStreamCaptureStatusActive ? "capture" : "capture_invalidated";
    std::ostringstream log;
    log << "[paged_mqa_logits] arch=" << (gfx936 ? "gfx936" : "gfx938")
        << " logical_device=" << static_cast<int>(q.get_device()) << " phase=" << phase << " mode=" << mode << '\n'
        << "  backend=" << (backend == Backend::opus ? "opus" : "native")
        << " requested_implementation=" << implementation << " requested_kernel_id=" << kernel_id << '\n';
    const auto print_tensor = [&log](const char* name, const torch::Tensor& tensor) {
        log << "  " << name << ": shape=" << tensor.sizes() << " dtype=" << tensor.scalar_type()
            << " stride=" << tensor.strides() << '\n';
    };
    print_tensor("q", q);
    print_tensor("fused_kv_cache", fused_kv_cache);
    print_tensor("weights", weights);
    print_tensor("context_lens", context_lens);
    print_tensor("block_table", block_table);
    if (out) {
        print_tensor("out", *out);
    } else {
        log << "  out: None\n";
    }
    // Debug-only blocking copy on the current stream. Never read device values
    // during capture; graph replay does not re-enter this host logging path.
    if (capture_status == hipStreamCaptureStatusNone) {
        const torch::Tensor lengths = context_lens.cpu();
        const int32_t* values = lengths.data_ptr<int32_t>();
        log << "  context_lens: values=[";
        for (int64_t i = 0; i < lengths.numel(); ++i) {
            if (i != 0) log << ", ";
            log << values[i];
        }
        if (lengths.numel() != 0) {
            const auto bounds = std::minmax_element(values, values + lengths.numel());
            log << "] min=" << *bounds.first << " max=" << *bounds.second << '\n';
        } else {
            log << "] min/max=<empty>\n";
        }
    } else {
        log << "  context_lens: values/min/max=<not read during capture>\n";
    }
    log << "  max_model_len=" << max_model_len << " clean_logits=" << (clean_logits ? "true" : "false") << '\n';
    std::fputs(log.str().c_str(), stderr);
}

}  // namespace

torch::Tensor paged_mqa_logits(const torch::Tensor& q,
                               const torch::Tensor& fused_kv_cache,
                               const torch::Tensor& weights,
                               const torch::Tensor& context_lens,
                               const torch::Tensor& block_table,
                               int64_t max_model_len,
                               std::optional<torch::Tensor> out,
                               bool clean_logits,
                               const std::string& implementation,
                               int64_t kernel_id,
                               bool is_prefill) {
    if (is_prefill) {
        TORCH_CHECK(q.is_cuda(), "paged_mqa_logits: prefix prefill requires GPU tensors");
        const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(q));
        const auto& device = get_device_info(q.get_device());
        select_backend(q, fused_kv_cache, max_model_len, implementation, kernel_id,
                       device.gfx936, false, device.gfx936 || device.gfx938);
        TORCH_CHECK(!out, "paged_mqa_logits: prefix prefill does not support supplied out");
#if defined(DEEPGEMM_ENABLE_GFX936) || defined(DEEPGEMM_ENABLE_GFX938)
        auto result = paged_mqa_logits_prefill_native(
            q, fused_kv_cache, weights, context_lens, block_table, max_model_len, clean_logits);
        const char* debug = std::getenv("DEEPGEMM_PAGED_MQA_LOGITS_DEBUG");
        if (debug && std::strcmp(debug, "1") == 0) {
            print_debug_info(q, fused_kv_cache, weights, context_lens, block_table, max_model_len,
                             out, clean_logits, implementation, kernel_id, device.gfx936, Backend::native, "prefill");
        }
        return result;
#else
        TORCH_CHECK(false, "paged_mqa_logits: prefix prefill was not built for gfx936/gfx938");
#endif
    }
    TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.size(0) > 0 && q.size(3) == 128,
                "paged_mqa_logits: Q must be GPU [B,R,H,128] with B>0");
    const int64_t batch = q.size(0), next_n = q.size(1), heads = q.size(2);
    TORCH_CHECK(batch * next_n <= 65535 && is_one_of(heads, {8, 16, 32, 64}),
                "paged_mqa_logits: B*R must fit uint16 and H must be 8/16/32/64");
    TORCH_CHECK(max_model_len > 0 && max_model_len <= INT_MAX,
                "paged_mqa_logits: max_model_len must be a positive int32 width");
    TORCH_CHECK(fused_kv_cache.dim() == 4 && fused_kv_cache.size(0) > 0 && fused_kv_cache.size(2) == 1,
                "paged_mqa_logits: cache must be [P,S,1,D+scale]");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == batch * next_n && weights.size(1) == heads &&
                weights.scalar_type() == at::kFloat,
                "paged_mqa_logits: weights must be FP32 [B*R,H]");
    TORCH_CHECK(context_lens.dim() == 1 && context_lens.size(0) == batch &&
                context_lens.scalar_type() == at::kInt,
                "paged_mqa_logits: context_lens must be int32 [B]");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == batch && block_table.size(1) > 0 &&
                block_table.scalar_type() == at::kInt,
                "paged_mqa_logits: block_table must be int32 [B,T]");
    for (const auto& tensor : {q, fused_kv_cache, weights, context_lens, block_table}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device() && !tensor.requires_grad(),
                    "paged_mqa_logits: inputs must be inference tensors on the same GPU");
    }
    TORCH_CHECK(q.is_contiguous() && weights.is_contiguous() && context_lens.is_contiguous(),
                "paged_mqa_logits: Q, weights, and context_lens must be contiguous");
    TORCH_CHECK(fused_kv_cache.stride(3) == 1 &&
                fused_kv_cache.stride(2) == fused_kv_cache.size(3) &&
                fused_kv_cache.stride(1) == fused_kv_cache.size(3) &&
                fused_kv_cache.stride(0) % sizeof(float) == 0 && block_table.stride(1) == 1,
                "paged_mqa_logits: unsupported fused cache or page-table strides");
    if (out) {
        TORCH_CHECK(out->is_cuda() && out->device() == q.device() && out->is_contiguous() &&
                    !out->requires_grad() && out->scalar_type() == at::kFloat && out->dim() == 2 &&
                    out->size(0) == batch * next_n && out->size(1) == max_model_len,
                    "paged_mqa_logits: out must be contiguous FP32 [B*R,max_model_len] on input GPU");
        for (const auto& tensor : {q, fused_kv_cache, weights, context_lens, block_table}) {
            TORCH_CHECK(!out->is_alias_of(tensor), "paged_mqa_logits: out must not alias inputs");
        }
    }

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(q));
    const auto& device = get_device_info(q.get_device());
    const bool gfx936 = device.gfx936;
    const bool gfx938 = device.gfx938;
    TORCH_CHECK(gfx936 || gfx938, "paged_mqa_logits: this build supports gfx936/gfx938 only");

    const int64_t page_size = fused_kv_cache.size(1);
    const auto q_dtype = q.scalar_type(), cache_dtype = fused_kv_cache.scalar_type();
    const bool fp8_cache = cache_dtype == at::kByte && fused_kv_cache.size(3) == 132;
    const bool int8_cache = cache_dtype == at::kChar && fused_kv_cache.size(3) == 132;
    const bool half_cache = (cache_dtype == at::kHalf || cache_dtype == at::kBFloat16) &&
                            fused_kv_cache.size(3) == 128;
    const bool fp8_q = q_dtype == at::ScalarType::Float8_e4m3fn;
    const bool opus_supported = fp8_q && fp8_cache && is_one_of(page_size, {1, 64}) &&
                                 is_one_of(next_n, {1, 2, 4}) && is_one_of(heads, {32, 64}) &&
                                 fused_kv_cache.size(0) <= INT_MAX / (page_size * 132) &&
                                 fused_kv_cache.is_contiguous() && block_table.is_contiguous();
    const bool native_supported = is_one_of(page_size, {4, 8, 16, 32, 64}) &&
                                   ((fp8_cache && (fp8_q || q_dtype == at::kBFloat16) &&
                                     is_one_of(next_n, {1, 2})) ||
                                    (int8_cache && q_dtype == at::kChar && next_n >= 1 && next_n <= 5) ||
                                    (half_cache && q_dtype == cache_dtype && next_n >= 1 && next_n <= 5));
    const Backend backend = select_backend(q, fused_kv_cache, max_model_len,
                                           implementation, kernel_id, gfx936,
                                           opus_supported, native_supported);
    const char* debug = std::getenv("DEEPGEMM_PAGED_MQA_LOGITS_DEBUG");
    if (debug && std::strcmp(debug, "1") == 0) {
        print_debug_info(q, fused_kv_cache, weights, context_lens, block_table, max_model_len,
                         out, clean_logits, implementation, kernel_id, gfx936, backend);
    }
    if (backend == Backend::opus) {
#if DEEPGEMM_HAS_GFX93X
        return opus::paged_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table,
                                       out, max_model_len, kernel_id, clean_logits);
#else
        ::deepgemm::arch::unsupported("paged_mqa_logits");
#endif
    }

    // gfx936 has no native FP8 MMAC. Match the upstream native public wrapper's
    // FP8-Q to BF16 conversion; gfx938 keeps FP8 Q for its native FP8 MMAC path.
    const torch::Tensor compute_q = gfx936 && fp8_q && fp8_cache ? q.to(at::kBFloat16) : q;
    const bool small_block = page_size < 16;
#if DEEPGEMM_HAS_GFX93X
    torch::Tensor result = paged_mqa_logits_native(
        compute_q, fused_kv_cache, weights, context_lens, block_table,
        static_cast<int>(max_model_len), clean_logits, small_block ? std::nullopt : out);
#else
    torch::Tensor result;
    ::deepgemm::arch::unsupported("paged_mqa_logits");
#endif
    if (out && small_block) {
        // The upstream small-block kernel writes into padded storage. Preserve
        // an exact-width supplied output without changing its boundary stores.
        out->copy_(result);
        return *out;
    }
    return result;
}

namespace {

std::vector<torch::Tensor> sparse_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& cache, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_table, int64_t max_model_len,
    bool clean_logits, int64_t num_warps, int64_t persistent_ctas, int64_t group_size,
    bool int8_kv, bool independent_rows) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.size(0) > 0 && q.size(0) <= INT_MAX,
                "sparse MQA: Q must be GPU [B,1,H,128] with positive int32 B");
    TORCH_CHECK(max_model_len > 0 && max_model_len <= INT_MAX && num_warps == 4 &&
                persistent_ctas >= 0 && persistent_ctas <= 4096 && is_one_of(group_size, {1, 3, 4, 5, 6}),
                "sparse MQA: invalid max_model_len, num_warps, persistent_ctas, or group_size");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(q));
    const auto& device = get_device_info(q.get_device());
    select_backend(q, cache, max_model_len, "auto", -1, device.gfx936, false, true);
    if (independent_rows) {
        TORCH_CHECK(cache.is_contiguous(), "sparse MQA: independent-row cache must be contiguous");
    }
    const bool legacy = independent_rows && device.gfx936 && device.compute_units == 80 &&
                        q.scalar_type() == at::kBFloat16 && cache.scalar_type() == at::kBFloat16 &&
                        is_one_of(q.size(0), {8, 40}) && q.size(2) == 64;
    const bool native_fp8 = !int8_kv && device.gfx938 && is_one_of(device.compute_units, {64, 72}) &&
                            q.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                            is_one_of(q.size(2), {8, 16, 32, 64}) &&
                            cache.scalar_type() == at::kByte;
    const bool fp8_q = q.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                       q.scalar_type() == at::ScalarType::Float8_e4m3fnuz;
    const torch::Tensor compute_q = cache.scalar_type() == at::kByte && fp8_q && !native_fp8
                                       ? q.to(at::kBFloat16) : q;
    // Tuned for 32 heads at this batch; measured slower for 8, 16 and 64 heads.
    if (native_fp8 && group_size == 6 && q.size(0) == 30 && q.size(2) == 32 && persistent_ctas == 0) {
        persistent_ctas = 128;
    }
    std::vector<torch::Tensor> result;
#if DEEPGEMM_HAS_GFX93X
    if (legacy) {
        result = page_mqa_logits_sparse_mask_native(compute_q, cache, weights, context_lens, block_table,
                                                    max_model_len, clean_logits, num_warps, persistent_ctas);
    } else if (int8_kv) {
        result = page_mqa_logits_sparse_mask_grouped_int8_native(
            compute_q, cache, weights, context_lens, block_table, max_model_len,
            clean_logits, num_warps, persistent_ctas, group_size);
    } else {
        result = page_mqa_logits_sparse_mask_grouped_native(
            compute_q, cache, weights, context_lens, block_table, max_model_len,
            clean_logits, num_warps, persistent_ctas, group_size);
    }
#else
    ::deepgemm::arch::unsupported("page_mqa_logits_sparse_mask");
#endif
    const char* debug = std::getenv("DEEPGEMM_PAGED_MQA_LOGITS_DEBUG");
    if (debug && std::strcmp(debug, "1") == 0) {
        print_debug_info(q, cache, weights, context_lens, block_table, max_model_len,
                         std::nullopt, clean_logits, "auto", -1, device.gfx936, Backend::native, "sparse");
    }
    return result;
}

}  // namespace

std::vector<torch::Tensor> page_mqa_logits_sparse_mask(
    const torch::Tensor& q, const torch::Tensor& fused_kv_cache, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_table, int64_t max_model_len,
    bool clean_logits, int64_t num_warps, int64_t persistent_ctas) {
    return sparse_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table, max_model_len,
                             clean_logits, num_warps, persistent_ctas, 1, false, true);
}

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped(
    const torch::Tensor& q, const torch::Tensor& fused_kv_cache, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_table, int64_t max_model_len,
    bool clean_logits, int64_t num_warps, int64_t persistent_ctas, int64_t group_size) {
    return sparse_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table, max_model_len,
                             clean_logits, num_warps, persistent_ctas, group_size, false, false);
}

std::vector<torch::Tensor> page_mqa_logits_sparse_mask_grouped_int8(
    const torch::Tensor& q, const torch::Tensor& packed_kv_cache, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_table, int64_t max_model_len,
    bool clean_logits, int64_t num_warps, int64_t persistent_ctas, int64_t group_size) {
    return sparse_mqa_logits(q, packed_kv_cache, weights, context_lens, block_table, max_model_len,
                             clean_logits, num_warps, persistent_ctas, group_size, true, false);
}

torch::Tensor
get_paged_mqa_logits_fp4_metadata(const torch::Tensor& lens, int64_t block, std::optional<int64_t> num_sms, int64_t n) {
    TORCH_CHECK(
        lens.is_cuda() && lens.scalar_type() == at::kInt && lens.is_contiguous() &&
            (lens.dim() == 1 || (lens.dim() == 2 && lens.size(1) == 1)),
        "context_lens must be contiguous int32 rows");
    const int64_t rows = lens.numel();
    TORCH_CHECK(
        block == 64 && rows > 0 && rows <= INT_MAX && n > 0 && n <= INT_MAX && rows % n == 0,
        "invalid page size or rows_per_request");
    const int64_t routes = n > 1 && rows / n >= 8 ? rows / n * ((n + 1) / 2) : rows;
    TORCH_CHECK(routes <= 16384, "route rows exceed 16384");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(lens.device());
    const auto& device = get_device_info(lens.get_device());
    TORCH_CHECK(device.gfx936 || device.gfx938, "paged FP4 requires gfx936/gfx938");
    TORCH_CHECK(!num_sms || *num_sms == device.compute_units, "num_sms must equal physical CU count when supplied");
    const int ctas = device.compute_units * 8;
    auto meta = torch::empty({ctas + 1, 2}, lens.options());
#if DEEPGEMM_HAS_GFX93X
    launch_paged_mqa_fp4_metadata(
        lens.data_ptr<int>(), meta.data_ptr<int>(), rows, n, ctas, at::hip::getCurrentHIPStream());
    TORCH_CHECK(hipGetLastError() == hipSuccess, "paged FP4 metadata launch failed");
#else
    ::deepgemm::arch::unsupported("get_paged_mqa_logits_fp4_metadata");
#endif
    return meta;
}

torch::Tensor fp8_fp4_paged_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& sf,
    const torch::Tensor& kv,
    const torch::Tensor& weights,
    const torch::Tensor& lens,
    const torch::Tensor& table,
    const std::optional<torch::Tensor>& meta,
    int64_t width,
    int64_t block,
    const std::optional<torch::Tensor>& output,
    int64_t n) {
    TORCH_CHECK(
        q.is_cuda() && (q.dim() == 3 || (q.dim() == 4 && q.size(1) == 1)) && q.size(-1) == 64 && q.is_contiguous() &&
            (q.scalar_type() == at::kChar || q.scalar_type() == at::kByte),
        "q must be contiguous packed E2M1 [R,H,64] or [R,1,H,64]");
    const int64_t rows = q.size(0), heads = q.size(-2);
    TORCH_CHECK((heads == 16 || heads == 32 || heads == 64) && rows <= 65535, "invalid heads or batch");
    for (const auto* t : {&sf, &kv, &weights, &lens, &table})
        TORCH_CHECK(t->is_cuda() && t->device() == q.device(), "all tensors must share Q's GPU");
    TORCH_CHECK(
        sf.scalar_type() == at::kInt && sf.is_contiguous() && sf.numel() == rows * heads,
        "q_sf must be packed int32 per head");
    TORCH_CHECK(
        kv.scalar_type() == at::kByte && ((kv.dim() == 2 && kv.size(1) == 4352 && kv.stride(1) == 1) ||
                                          (kv.dim() == 4 && kv.size(1) == 64 && kv.size(2) == 1 && kv.size(3) == 68 &&
                                           kv.stride(1) == 68 && kv.stride(3) == 1)),
        "KV must be packed uint8 [pages,64,1,68] or [pages,4352]");
    TORCH_CHECK(
        kv.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0,
        "Q and KV pages must be 16-byte aligned");
    TORCH_CHECK(
        weights.scalar_type() == at::kFloat && weights.dim() == 2 && weights.size(0) == rows &&
            weights.size(1) == heads && weights.stride(1) == 1,
        "weights must be FP32 [R,H]");
    TORCH_CHECK(
        lens.scalar_type() == at::kInt && lens.is_contiguous() && lens.numel() == rows,
        "context_lens must be int32 per row");
    TORCH_CHECK(
        table.scalar_type() == at::kInt && table.dim() == 2 && table.size(0) == rows && table.stride(1) == 1,
        "block_table must be int32 [R,blocks]");
    TORCH_CHECK(
        (block == 8 || block == 64) && width >= 0 && width <= INT_MAX - 256, "invalid table block or output width");
    TORCH_CHECK(
        n >= 1 && n <= INT_MAX && rows % n == 0 && (n == 1 || (block == 64 && heads <= 32)),
        "invalid rows_per_request");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(q.device());
    const auto& device = get_device_info(q.get_device());
    TORCH_CHECK(device.gfx936 || device.gfx938, "paged FP4 requires gfx936/gfx938");
    torch::Tensor out;
    if (output) {
        out = *output;
        TORCH_CHECK(
            out.is_cuda() && out.device() == q.device() && out.scalar_type() == at::kFloat && out.dim() == 2 &&
                out.size(0) == rows && out.size(1) >= width && out.stride(1) == 1 &&
                (rows <= 1 || out.stride(0) >= out.size(1)),
            "out must be FP32 [R,>=width] with nonoverlapping rows");
        at::assert_no_internal_overlap(out);
        for (const auto* t : {&q, &sf, &kv, &weights, &lens, &table})
            at::assert_no_overlap(out, *t);
    } else {
        out = torch::empty({rows, (width + 255) / 256 * 256}, weights.options()).slice(1, 0, width);
    }
    int ctas = device.compute_units * 8;
    if (meta) {
        TORCH_CHECK(
            meta->is_cuda() && meta->device() == q.device() && meta->scalar_type() == at::kInt && meta->dim() == 2 &&
                meta->size(0) >= 2 && meta->size(0) <= INT_MAX && meta->size(1) == 2 && meta->is_contiguous(),
            "schedule_meta must be contiguous int32 [ctas+1,2]");
        ctas = meta->size(0) - 1;
        at::assert_no_overlap(out, *meta);
        // ATen cannot decide some padded-row aliases. Metadata is contiguous,
        // so check each output row exactly without rejecting disjoint views.
        if (at::get_overlap_status(out, *meta) == at::MemOverlapStatus::TooHard) {
            const uintptr_t meta_begin = reinterpret_cast<uintptr_t>(meta->data_ptr());
            const uintptr_t meta_end = meta_begin + meta->nbytes();
            const uintptr_t out_begin = reinterpret_cast<uintptr_t>(out.data_ptr());
            for (int64_t row = 0; row < rows; ++row) {
                const uintptr_t begin = out_begin + row * out.stride(0) * sizeof(float);
                TORCH_CHECK(
                    begin + out.size(1) * sizeof(float) <= meta_begin || begin >= meta_end,
                    "out must not overlap schedule_meta");
            }
        }
    }
    // A partial table limits visibility even when the caller retains a wider
    // output buffer or replays larger context lengths.
    width = std::min(width, table.size(1) * block);
    if (rows == 0 || width == 0)
        return out;
    TORCH_CHECK(kv.size(0) > 0, "nonempty output requires KV pages");
    const int tiles = (width + 15) / 16;
    const bool paired = n > 1 && rows / n >= 8;
    const int routes = paired ? rows / n * ((n + 1) / 2) : rows;
    const bool small_kernel = uint64_t(routes) * ((width + 255) / 256) <= unsigned(device.compute_units);
    torch::Tensor flags;
    if (!small_kernel)
        flags = torch::empty({rows, tiles}, q.options().dtype(at::kByte));
    PagedMqaFp4Args a{
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        kv.data_ptr<uint8_t>(),
        sf.data_ptr<int>(),
        lens.data_ptr<int>(),
        table.data_ptr<int>(),
        meta ? meta->data_ptr<int>() : nullptr,
        weights.data_ptr<float>(),
        out.data_ptr<float>(),
        flags.defined() ? flags.data_ptr<uint8_t>() : nullptr,
        kv.stride(0),
        weights.stride(0),
        table.stride(0),
        out.stride(0),
        int(rows),
        int(heads),
        int(width),
        block == 64 ? 6 : 3,
        int(n),
        routes,
        tiles,
        ctas,
        paired,
        small_kernel};
    const auto stream = at::hip::getCurrentHIPStream();
    if (device.gfx938) {
#if defined(DEEPGEMM_ENABLE_GFX938)
        launch_paged_mqa_fp4_fp8(a, q.get_device(), stream);
#else
        TORCH_CHECK(false, "gfx938 FP4 kernel was not built");
#endif
    } else {
#if defined(DEEPGEMM_ENABLE_GFX936)
        launch_paged_mqa_fp4_fp16(a, q.get_device(), stream);
#else
        TORCH_CHECK(false, "gfx936 FP4 kernel was not built");
#endif
    }
    TORCH_CHECK(hipGetLastError() == hipSuccess, "paged FP4 compute launch failed");
    return out;
}

} // namespace deepgemm
