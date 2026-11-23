// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "arch.hpp"

namespace deepgemm::mqa::fp8 {
using common::I2;
using common::I4;
using common::F2;
using common::F4;
using arch::wait_vm;
using arch::wait_vm_barrier;
using arch::barrier;
using arch::finish_kv_half;
using arch::finish_weight;
using arch::finish_scales;
using MLS64x32B8 = arch::MLSAccessor<arch::MLSKind::B8_64x32_T1_F1>;
using MLS128x16B8 = arch::MLSAccessor<arch::MLSKind::B8_128x16_T1_F0>;
using MLS64x16B8 = arch::MLSAccessor<arch::MLSKind::B8_64x16_T1_F0>;
using arch::shuffle_weight;
using arch::relu;

__device__ __forceinline__ F2 pk_add(F2 a, F2 b) {
    return __builtin_hcu_pk_add_f32(a, b);
}

__device__ __forceinline__ F2 pk_fma(F2 a, F2 b, F2 c) {
    return __builtin_hcu_pk_fma_f32(a, b, c);
}

__device__ __forceinline__ F4 pk_mul(F4 a, F4 b) {
    F4 result;
    result.xy = __builtin_hcu_pk_mul_f32(a.xy, b.xy);
    result.zw = __builtin_hcu_pk_mul_f32(a.zw, b.zw);
    return result;
}

// Keep a scheduling boundary before the paired MMAC results are consumed.
// gfx938 packed FMA has no op_sel broadcast: both weight lanes are explicit.
__device__ __forceinline__ F4 accumulate_relu_ready(F4 sum, F4 dot, F2 weight) {
    // The paired-dot caller issues four independent MMACs before consuming
    // its first dot, then four independent maxes before consuming the second.
    __builtin_hcu_sched_barrier(0);
    F4 activated;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        activated[j] = relu(dot[j]);
    sum.xy = pk_fma(activated.xy, weight, sum.xy);
    sum.zw = pk_fma(activated.zw, weight, sum.zw);
    return sum;
}

// Shared by the dense and compact QH kernels.
__device__ __forceinline__ F4 qh_accumulate(F4 sum, F4 dot, F4 weight) {
    __builtin_hcu_sched_barrier(0);
    F4 activated;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        activated[j] = relu(dot[j]);
    sum.xy = pk_fma(activated.xy, weight.xy, sum.xy);
    sum.zw = pk_fma(activated.zw, weight.zw, sum.zw);
    return sum;
}

template <int Pair> __device__ __forceinline__ F4 mmac(I4 a, I4 b, F4 c) {
    static_assert(Pair == 0 || Pair == 1);
    const I2 av = {a[2 * Pair], a[2 * Pair + 1]};
    const I2 bv = {b[2 * Pair], b[2 * Pair + 1]};
    return arch::mmac<arch::MMADataType::FP8E4M3FN, true, false,
                          arch::MMAEmitMode::Builtin>(av, bv, c);
}

__device__ __forceinline__ F4 dot128(I4 a0, I4 a1, I4 b0, I4 b1, F4 zero) {
    F4 d = mmac<0>(a0, b0, zero);
    d = mmac<1>(a0, b0, d);
    d = mmac<0>(a1, b1, d);
    return mmac<1>(a1, b1, d);
}
} // namespace deepgemm::mqa::fp8
