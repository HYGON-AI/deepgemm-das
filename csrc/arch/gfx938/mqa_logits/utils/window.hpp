// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "mqa_logits/common/window.hpp"
#include "mqa_logits/common/epilogue.hpp"
#include "mqa_logits/common/params.hpp"

namespace deepgemm::mqa {
namespace fp8 {
using common::F2;
using common::F4;
using common::output_vector_aligned;
using common::store1;
using common::store4;
using common::store_logits4;
using common::tail_rows;

using common::load_window;
using common::TileWindow;
using common::Window;

using common::classify_window;
using common::small_window;

template <int AlignN>
__device__ __forceinline__ void fill_empty_tile(const MqaFp8Params &p, int64_t m0, int64_t n0,
                                                int valid_m, int valid_n, int wave, int lane) {
    common::fill_empty_tile<AlignN>(p.out, p.out_stride_m, m0, n0, valid_m, valid_n, wave, lane);
}

__device__ __forceinline__ void empty_output(const MqaFp8Params &p, int64_t m0, int64_t n0, int vm,
                                             int vn) {
    for (int i = threadIdx.x; i < vm * vn; i += 256)
        p.out[(m0 + i / vn) * p.out_stride_m + n0 + i % vn] = -__builtin_inff();
}

} // namespace fp8
} // namespace deepgemm::mqa
