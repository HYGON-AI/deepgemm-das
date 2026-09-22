#pragma once

#include "hip/hip_runtime.h"

#include <torch/all.h>
#include <torch/extension.h>
#include "intrinsic.h"
#include "exception.h"
#ifndef LIGHTOP_DEVICE_ASSERT
#define LIGHTOP_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef LIGHTOP_STATIC_ASSERT
#define LIGHTOP_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#define WARP_SIZE_GPU 64

namespace deepgemm {

__forceinline__ __device__ uintx4 make_rscr_matrix_load(const void* ptr, const int strides) {
    uintx4 rsrc;
    *(uint64_t*)&rsrc = reinterpret_cast<uint64_t>(ptr);
    rsrc[2] = strides;
    rsrc[3] = 0;
    return rsrc;
}

template<typename T, class A, class B>
constexpr T min(A const& a, B const& b) {
    return static_cast<T>(b < a ? b : a);
}

template<typename T, class A, class B>
constexpr T max(A const& a, B const& b) {
    return static_cast<T>(b > a ? b : a);
}

template<int min_value, int max_value>
__forceinline__ __device__ int inline_min_max(int source){
    return max<int>(min_value, min<int>(max_value, source));
}

template <typename T>
__device__ __host__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ T align(T a, T b) {
    return ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template<typename T>
__forceinline__ __device__ void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ uint32_t get_lane_idx() {
    return threadIdx.x % WARP_SIZE_GPU;
}

template<typename T>
__device__  __forceinline__ T ld_shared(const T* ptr) {
    return *ptr;
}

template<typename T>
__device__  __forceinline__ void st_shared(const T* ptr, T val) {
    return *ptr = val;
}

} // namespace deepgemm

#include "tensor_utils.h"
