// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace deepgemm::mqa::common {
struct MqaParams {
    // Preserve the audited device argument offsets. No public/versioned ABI.
    uint32_t output_compressed = 0;
    uint32_t output_width = 0;
    uint32_t reserved_header[2]{};
    const uint8_t *q;
    const uint8_t *kv;
    const float *weights;
    const float *kv_scale; /* per-KV-token scale; nullptr selects unit scale */
    float *out;
    int64_t m, n, h, d;
    int64_t q_stride_m, q_stride_h, kv_stride_n;
    int64_t weights_stride_m, out_stride_m;
    /* Optional contiguous signed int32 [M] bounds, clamped to [0,N] on
     * device. Null ks/ke mean 0/N. ke <= ks denotes an empty window. */
    const int32_t *ks;
    const int32_t *ke;
    /* Dense kernels: 0 only defines valid window values; 1 writes -inf outside.
     * Compact kernels use 0; only the valid prefix is defined. */
    uint32_t clean_logits = 0;
    uint32_t bounds_reserved;
};

static_assert(std::is_standard_layout_v<MqaParams>);
static_assert(offsetof(MqaParams, output_compressed) == 0);
static_assert(offsetof(MqaParams, output_width) == 4);
static_assert(offsetof(MqaParams, q) == 16);
static_assert(offsetof(MqaParams, ks) == 128);
static_assert(sizeof(MqaParams) == 152);
} // namespace deepgemm::mqa::common

namespace deepgemm::mqa::fp8 {
using MqaFp8Params = common::MqaParams;
} // namespace deepgemm::mqa::fp8

namespace deepgemm::mqa::int8 {
using MqaInt8Params = common::MqaParams;
} // namespace deepgemm::mqa::int8

namespace deepgemm::mqa::bf16 {
using MqaBf16Params = common::MqaParams;
} // namespace deepgemm::mqa::bf16
