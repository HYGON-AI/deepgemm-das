// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "arch.hpp"

namespace deepgemm::mqa::fp8 {
using ScalarBufferLoad = arch::BufferLoadAccessor<4>;
// The three-argument overload adds the head offset while preserving masked rows.
// finish_weight waits for pending memory operations at the head boundary.
using WeightBufferLoad = ScalarBufferLoad;

// Assemble four adjacent scales using the widest fully valid buffer requests.
// Each issued request is either entirely valid or uses the OOB sentinel.
template <int AlignN, bool EvenN>
__device__ __forceinline__ common::F4 load_scale_buffer(common::I4 descriptor, int col,
                                                           int valid_n) {
    static_assert(AlignN == 4 || AlignN == 2 || AlignN == 1);
    common::F4 value;
#pragma unroll
    for (int j = 0; j < 4; j += AlignN) {
        uint32_t offset = uint32_t(col + j) * 4;
        if constexpr (!EvenN)
            offset = col + j < valid_n ? offset : ScalarBufferLoad::kOOBOffset;
        if constexpr (AlignN == 4) {
            value = arch::BufferLoadAccessor<16>::load(
                descriptor, offset);
        } else if constexpr (AlignN == 2) {
            const common::F2 pair =
                arch::BufferLoadAccessor<8>::load(
                    descriptor, offset);
            value[j] = pair[0];
            value[j + 1] = pair[1];
        } else {
            value[j] = ScalarBufferLoad::load(descriptor, offset);
        }
    }
    return value;
}
} // namespace deepgemm::mqa::fp8
