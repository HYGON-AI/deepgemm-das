// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "mqa_logits/int8.h"
#include "dispatch/config.hpp"
#include "mqa_logits/common/output.hpp"
#include "mqa_logits/common/memory.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <limits>

namespace deepgemm::mqa::int8 {
namespace {
using deepgemm::mqa::common::ByteRows;

void check_tensor(const torch::Tensor &t, const torch::Tensor &q, at::ScalarType dtype,
                  const char *name) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == q.device(), "mqa_logits INT8: ", name,
                " must be on Q's CUDA/HIP device");
    TORCH_CHECK(t.scalar_type() == dtype, "mqa_logits INT8: ", name, " must have dtype ", dtype,
                ", got ", t.scalar_type());
}

template <typename Pointer>
struct CheckedPointer {
    Pointer data;
    ByteRows rows;
};

// The public dtype dispatcher and entry checks establish the element sizes.
// Validate spans before obtaining pointers, then reuse them in launch params.
template <uint64_t Bytes, typename GetPointer>
auto checked_pointer(GetPointer get_pointer, int64_t rows, int64_t stride, int64_t width,
                     const char *name) {
    const uint64_t bytes = Bytes;
    const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / bytes;
    TORCH_CHECK(rows > 0 && width > 0 && stride >= width && uint64_t(width) <= limit &&
                    uint64_t(rows - 1) <= (limit - width) / uint64_t(stride),
                "mqa_logits INT8: ", name, " address span exceeds signed 64-bit range");
    auto pointer = get_pointer();
    const uint64_t address = reinterpret_cast<uintptr_t>(pointer);
    TORCH_CHECK((address & 3) == 0, "mqa_logits INT8: ", name,
                " requires 4-byte pointer alignment");
    const ByteRows result{address, uint64_t(rows), uint64_t(rows == 1 ? width : stride) * bytes,
                          uint64_t(width) * bytes};
    TORCH_CHECK(deepgemm::mqa::common::valid_rows(result), "mqa_logits INT8: ", name,
                " address span overflows");
    return CheckedPointer<decltype(pointer)>{pointer, result};
}

void check_disjoint(const ByteRows &out, const ByteRows &input, const char *name) {
    TORCH_CHECK(!deepgemm::mqa::common::rows_overlap(out, input), "mqa_logits INT8: D_out overlaps ",
                name);
}

struct CheckedInputs {
    MqaInt8Params params{};
    ByteRows q, k, weights, ks, ke, kv_scale;
};

// Called only after the output-mode-specific allocation and M/N empty return.
// Dense additionally uses these byte rows to check D_out against every input.
CheckedInputs check_inputs(const torch::Tensor &Q, const torch::Tensor &K,
                           const torch::Tensor &weights, const torch::Tensor &ks,
                           const torch::Tensor &ke, int m, int n, int h, int d,
                           const torch::Tensor &kv_scale) {
    TORCH_CHECK(Q.stride(2) == 1 && Q.stride(1) == 128 && Q.stride(0) == int64_t(h) * 128 &&
                    K.stride(1) == 1 && K.stride(0) == 128,
                "mqa_logits INT8: Q/K require contiguous D=128 layout; got strides ", Q.strides(),
                " and ", K.strides());
    const int64_t weights_stride_m = weights.stride(0);
    TORCH_CHECK(weights.stride(1) == 1 && weights_stride_m >= h,
                "mqa_logits INT8: weights need contiguous columns and nonoverlapping rows; ",
                "got strides ", weights.strides());
    const auto q = checked_pointer<1>(
        [&] { return static_cast<const uint8_t *>(Q.data_ptr()); }, m, int64_t(h) * 128,
        int64_t(h) * 128, "Q");
    const auto k = checked_pointer<1>(
        [&] { return static_cast<const uint8_t *>(K.data_ptr()); }, n, 128, 128, "K");
    const auto w = checked_pointer<4>(
        [&] { return weights.data_ptr<float>(); }, m, weights_stride_m, h, "weights");
    const auto ks_data = checked_pointer<4>([&] { return ks.data_ptr<int32_t>(); }, 1, m, m, "ks");
    const auto ke_data = checked_pointer<4>([&] { return ke.data_ptr<int32_t>(); }, 1, m, m, "ke");
    const auto scale = checked_pointer<4>(
        [&] { return kv_scale.data_ptr<float>(); }, 1, n, n, "kv_scale");

    CheckedInputs inputs{{}, q.rows, k.rows, w.rows, ks_data.rows, ke_data.rows, scale.rows};
    auto &p = inputs.params;
    p.q = q.data;
    p.kv = k.data;
    p.weights = w.data;
    p.kv_scale = scale.data;
    p.m = m;
    p.n = n;
    p.h = h;
    p.d = d;
    p.q_stride_m = Q.stride(0);
    p.q_stride_h = Q.stride(1);
    p.kv_stride_n = K.stride(0);
    p.weights_stride_m = weights_stride_m;
    p.ks = ks_data.data;
    p.ke = ke_data.data;
    return inputs;
}

// Dense [M,N] output (or D_out) after the shared checks.
template <class DeviceProps>
torch::Tensor compute_dense_output(const torch::Tensor &Q, const torch::Tensor &K,
                                   const torch::Tensor &weights,
                                   const torch::Tensor &ks, const torch::Tensor &ke,
                                   int m, int n, int h, int d,
                                   const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
                                   const torch::optional<torch::Tensor> &out,
                                   const DeviceProps *prop) {
    torch::Tensor result;
    if (out.has_value() && out->defined()) {
        check_tensor(*out, Q, at::kFloat, "D_out");
        TORCH_CHECK(out->dim() == 2 && out->size(0) >= m && out->size(1) >= n,
                    "mqa_logits INT8: D_out must cover [M,N]");
        result =
            (out->size(0) == m && out->size(1) == n) ? *out : out->slice(0, 0, m).slice(1, 0, n);
    } else {
        result = torch::empty({m, n}, Q.options().dtype(at::kFloat));
    }
    if (!m || !n)
        return result;
    auto inputs = check_inputs(Q, K, weights, ks, ke, m, n, h, d, *kv_scale);
    const int64_t out_stride_m = result.stride(0);
    TORCH_CHECK(result.stride(1) == 1 && out_stride_m >= n,
                "mqa_logits INT8: D_out needs contiguous columns and nonoverlapping rows; ",
                "got strides ", result.strides());
    const auto output = checked_pointer<4>(
        [&] { return result.data_ptr<float>(); }, m, out_stride_m, n, "D_out");
    check_disjoint(output.rows, inputs.q, "Q");
    check_disjoint(output.rows, inputs.k, "K");
    check_disjoint(output.rows, inputs.weights, "weights");
    check_disjoint(output.rows, inputs.ks, "ks");
    check_disjoint(output.rows, inputs.ke, "ke");
    check_disjoint(output.rows, inputs.kv_scale, "kv_scale");
    auto &p = inputs.params;
    p.out = output.data;
    p.out_stride_m = out_stride_m;
    p.clean_logits = clean_logits;
    const auto stream = at::cuda::getCurrentCUDAStream(Q.get_device());
    auto status = launch(p, resolve(p), stream, prop->maxGridSize[0], prop->maxGridSize[1]);
    TORCH_CHECK(status == hipSuccess,
                "mqa_logits INT8 launch/configuration failed: ", hipGetErrorString(status));
    return result;
}

// Compact [M,align(W,256)] output after the shared checks; returns [:,:W].
template <class DeviceProps>
torch::Tensor compute_packed_output(const torch::Tensor &Q, const torch::Tensor &K,
                                    const torch::Tensor &weights,
                                    const torch::Tensor &ks, const torch::Tensor &ke,
                                    int m, int n, int h, int d,
                                    const torch::optional<torch::Tensor> &kv_scale,
                                    int64_t max_seqlen_k, const DeviceProps *prop) {
    const int64_t width = deepgemm::mqa::common::checked_output_width(true, max_seqlen_k, n);
    auto result = deepgemm::mqa::common::allocate_compact_output(Q, m, width);
    if (!m || !n)
        return result;
    auto inputs = check_inputs(Q, K, weights, ks, ke, m, n, h, d, *kv_scale);
    auto &p = inputs.params;
    p.out = result.data_ptr<float>();
    p.out_stride_m = result.stride(0);
    p.clean_logits = 0;
    p.output_compressed = 1;
    p.output_width = uint32_t(width);
    const auto stream = at::cuda::getCurrentCUDAStream(Q.get_device());
    auto status = launch(p, resolve_compact(p), stream, prop->maxGridSize[0], prop->maxGridSize[1]);
    TORCH_CHECK(status == hipSuccess,
                "mqa_logits INT8 launch/configuration failed: ", hipGetErrorString(status));
    return result;
}
} // namespace

// INT8 Tensor adapter for dense and compact output. Shared checks run once;
// max_seqlen_k selects compact output (no D_out), otherwise dense output.
torch::Tensor mqa_logits_int8(const torch::Tensor &Q, const torch::Tensor &K,
                              const torch::Tensor &weights,
                              const torch::Tensor &ks, const torch::Tensor &ke,
                              int m, int n, int h, int d,
                              const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
                              const torch::optional<torch::Tensor> &out,
                              const torch::optional<int64_t> &max_seqlen_k) {
    TORCH_CHECK(Q.defined() && Q.is_cuda(), "mqa_logits INT8 requires CUDA/HIP Q");
    // The public dtype dispatcher selects this adapter only for INT8 Q.
    check_tensor(K, Q, at::kChar, "K");
    check_tensor(weights, Q, at::kFloat, "weights");
    check_tensor(ks, Q, at::kInt, "ks");
    check_tensor(ke, Q, at::kInt, "ke");
    TORCH_CHECK(m >= 0 && n >= 0 && h >= 1 && h <= 128 && !(h & (h - 1)) && d == 128,
                "mqa_logits INT8 requires M,N>=0, D=128 and power-of-two H in [1,128]");
    TORCH_CHECK(Q.dim() == 3 && Q.size(0) == m && Q.size(1) == h && Q.size(2) == d &&
                    K.dim() == 2 && K.size(0) == n && K.size(1) == d,
                "mqa_logits INT8: explicit M/N/H/D must match Q and K shapes");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == m && weights.size(1) == h,
                "mqa_logits INT8: weights must have shape [M,H]");
    TORCH_CHECK(ks.dim() == 1 && ke.dim() == 1 && ks.numel() >= m && ke.numel() >= m &&
                    ks.is_contiguous() && ke.is_contiguous(),
                "mqa_logits INT8: ks/ke must be contiguous int32 vectors with at least M elements");
    TORCH_CHECK(kv_scale.has_value() && kv_scale->defined(), "mqa_logits INT8 requires kv_scale");
    check_tensor(*kv_scale, Q, at::kFloat, "kv_scale");
    TORCH_CHECK(kv_scale->is_contiguous() && kv_scale->numel() >= n,
                "mqa_logits INT8: kv_scale must be contiguous with at least N elements");
    const c10::cuda::CUDAGuard guard(Q.device());
    if (max_seqlen_k.has_value()) {
        TORCH_CHECK(!(out.has_value() && out->defined()),
                    "mqa_logits INT8: max_seqlen_k selects compact output; D_out is unsupported");
    }
    // Both output modes use ATen's cached device properties.
    const auto *prop = at::cuda::getDeviceProperties(Q.get_device());
    TORCH_CHECK((std::strncmp(prop->gcnArchName, "gfx936", 6) == 0 ||
                 std::strncmp(prop->gcnArchName, "gfx938", 6) == 0) && prop->warpSize == 64,
                "mqa_logits INT8 requires gfx936/gfx938 with wave64");
    if (!max_seqlen_k.has_value())
        return compute_dense_output(Q, K, weights, ks, ke, m, n, h, d, kv_scale, clean_logits, out, prop);
    return compute_packed_output(Q, K, weights, ks, ke, m, n, h, d, kv_scale, *max_seqlen_k, prop);
}
} // namespace deepgemm::mqa::int8
