// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// Warp-level reduction primitives for HCU gfx938 (64-lane wavefront).
// Ported from V1 csrc/kernels/utils.cuh warp_reduce family
// (implementation identical to DeepEP utils.cuh — __shfl_xor butterfly).

#include <hip/hip_runtime.h>
#include "params.h"

namespace deep_gemm::mega::hcu {

static constexpr uint64_t kFullWarpMask = ~0ull;

template <typename T>
__device__ __forceinline__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
    return __shfl_xor(val, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T shfl_sync(const T val, int srcLane = 0, int width = kWarpSize) {
    return __shfl(val, srcLane, width);
}

__device__ __forceinline__ int any_sync(uint64_t mask, int predicate) {
    uint64_t bits = __ballot(predicate);
    return (bits & mask) > 0;
}

__device__ __forceinline__ int all_sync(uint64_t mask, int predicate) {
    uint64_t bits = __ballot(predicate);
    return (~bits & mask) == 0;
}

__device__ __forceinline__ void syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

__device__ __forceinline__ int lane_id() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0));
}

// Reduction operators
template <typename T>
struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};

// Unified warp reduction for 64-lane wavefront.
// Uses __shfl_xor butterfly across all 6 power-of-2 strides (1,2,4,8,16,32).
template <typename T, typename Op>
__device__ __forceinline__ T warp_reduce(T value, Op op) {
    value = op(value, shfl_xor(value, 1));
    value = op(value, shfl_xor(value, 2));
    value = op(value, shfl_xor(value, 4));
    value = op(value, shfl_xor(value, 8));
    value = op(value, shfl_xor(value, 16));
    value = op(value, shfl_xor(value, 32));
    return value;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
    return warp_reduce<T>(value, ReduceSum<T>{});
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T value) {
    return warp_reduce<T>(value, ReduceMax<T>{});
}

template <typename T>
__device__ __forceinline__ T warp_reduce_min(T value) {
    return warp_reduce<T>(value, ReduceMin<T>{});
}

// Broadcast from lane 0
template <typename T>
__device__ __forceinline__ T broadcast_lane0(T value) {
    int32_t v = __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int32_t, value));
    return __builtin_bit_cast(T, v);
}

// Find n-th set bit in mask (like CUDA __fns)
// Returns the lane index of the (n+1)-th set bit in mask, starting from start_lane.
__device__ __forceinline__ int fns(uint32_t mask, int start_lane, int n) {
    // HCU doesn't have a direct __fns equivalent; emulate with ballot + popc
    // Find the n-th set bit at or after start_lane
    uint32_t shifted = mask >> start_lane;
    for (int i = 0; i < n; ++i) {
        if (shifted == 0) return -1;
        shifted &= (shifted - 1);  // clear lowest set bit
    }
    if (shifted == 0) return -1;
    int pos = __ffs(shifted) - 1;
    return pos + start_lane;
}

} // namespace deep_gemm::mega::hcu
