#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// namespace DEEP_GEMM
// {
//   namespace FP8_GROUP_GEMM
//   {
namespace deepgemm
{
    namespace MASKED_UTILS
    {
      template <typename scalar_t, size_t len>
      using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

      using vec4_bf16 = vec<unsigned short, 4>;

      union v_type
      {
        vec<int8_t, 8> int8_arr[2];
        vec<int32_t, 4> int_arr[1];
      };

      static __device__ vec<int, 4> mmac_(const vec<int8_t, 8> &v1, const vec<int8_t, 8> &v2, vec<int, 4> &v3)
      {
#if defined(__gfx938__)
        v3 = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(v1, v2, v3, 1, 0, 0);
#elif defined(__gfx936__) || defined(__gfx92a__) || defined(__gfx928__)
        v3 = __builtin_hcu_mmac_i32_16x16x32_i8(v1, v2, v3);
#endif
        return v3;
      }

      static __device__ vec<float, 4> mmac_(const vec<int8_t, 8> &v1, const vec<int8_t, 8> &v2, vec<float, 4> &v3)
      {
#if defined(__gfx938__)
        // v3 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8(v1, v2, v3);
        v3 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(v1, v2, v3, 1 /* if interleave */, 0 /* if trans*/);
#elif defined(__gfx92a__)
        v3 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8(v1, v2, v3);
#endif
        return v3;
      }

      template <typename T>
      __device__ __host__ T ceil_div(T a, T b)
      {
        return (a + b - 1) / b;
      }

      __device__ __forceinline__ int atomic_add_release_global(const int *ptr, int value)
      {
        int ret;
#ifdef USE_ROCM
        ret = __hip_atomic_fetch_add(const_cast<int *>(ptr), value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
        asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
#endif
        return ret;
      }

#define MOE_LL_K_SWITCH(k, ...)                                             \
  [&] {                                                                     \
    if (k == 1024) {                                                        \
      constexpr static int K = 1024;                                        \
      return __VA_ARGS__();                                                 \
    } else if (k == 1536) {                                                 \
      constexpr static int K = 1536;                                        \
      return __VA_ARGS__();                                                 \
    } else if (k == 2048) {                                                 \
      constexpr static int K = 2048;                                        \
      return __VA_ARGS__();                                                 \
    } else if (k == 3072) {                                                 \
      constexpr static int K = 3072;                                        \
      return __VA_ARGS__();                                                 \
    } else if (k == 4096) {                                                 \
      constexpr static int K = 4096;                                        \
      return __VA_ARGS__();                                                 \
    } else if (k == 5120) {                                                 \
      constexpr static int K = 5120;                                        \
      return __VA_ARGS__();                                                 \
    } else if (k == 6144) {                                                 \
      constexpr static int K = 6144;                                        \
      return __VA_ARGS__();                                                 \
    } else if (k == 7168) {                                                 \
      constexpr static int K = 7168;                                        \
      return __VA_ARGS__();                                                 \
    } else {                                                                \
      std::cout << "unsupported low latency k " << k << std::endl;          \
    }                                                                       \
  }();

#define MOE_LL_N_SWITCH(n, ...)                                             \
  [&] {                                                                     \
    if (n == 2048) {                                                        \
      constexpr static int N = 2048;                                        \
      return __VA_ARGS__();                                                 \
    } else if (n == 3072) {                                                 \
      constexpr static int N = 3072;                                        \
      return __VA_ARGS__();                                                 \
    } else if (n == 4096) {                                                 \
      constexpr static int N = 4096;                                        \
      return __VA_ARGS__();                                                 \
    } else if (n == 4608) {                                                 \
      constexpr static int N = 4608;                                        \
      return __VA_ARGS__();                                                 \
    } else if (n == 5120) {                                                 \
      constexpr static int N = 5120;                                        \
      return __VA_ARGS__();                                                 \
    } else if (n == 6144) {                                                 \
      constexpr static int N = 6144;                                        \
      return __VA_ARGS__();                                                 \
    } else if (n == 7168) {                                                 \
      constexpr static int N = 7168;                                        \
      return __VA_ARGS__();                                                 \
    } else {                                                                \
      std::cout << "unsupported low latency n " << n << std::endl;          \
    }                                                                       \
  }();

#define MOE_LL_TYPE_SWITCH(type, ...)                                       \
  [&] {                                                                     \
    if (type == at::kChar) {                                                \
      using Acc_Type = int32_t;                                             \
      return __VA_ARGS__();                                                 \
    } else if (type == at::kFloat8_e4m3fn) {                                \
      using Acc_Type = float;                                               \
      return __VA_ARGS__();                                                 \
    } else {                                                                \
      std::cout << "unsupported low latency matrix_a type " << std::endl;   \
    }                                                                       \
  }();

#define MOE_LL_OVERLAP_SWITCH(overlap, ...)                                 \
  [&] {                                                                     \
    if (overlap == true) {                                                  \
      constexpr static bool SBO = true;                                     \
      return __VA_ARGS__();                                                 \
    } else {                                                                \
      constexpr static bool SBO = false;                                    \
      return __VA_ARGS__();                                                 \
    }                                                                       \
  }();

#define MOE_LL_CU_SWITCH(cu, ...)                                           \
  [&] {                                                                     \
    if (cu == 64) {                                                         \
      constexpr static int CUs = 64;                                        \
      return __VA_ARGS__();                                                 \
    } else if (cu == 72) {                                                  \
      constexpr static int CUs = 72;                                        \
      return __VA_ARGS__();                                                 \
    } else if (cu == 128) {                                                 \
      constexpr static int CUs = 128;                                       \
      return __VA_ARGS__();                                                 \
    } else if (cu == 144) {                                                 \
      constexpr static int CUs = 144;                                       \
      return __VA_ARGS__();                                                 \
    } else if (cu == 192) {                                                 \
      constexpr static int CUs = 192;                                       \
      return __VA_ARGS__();                                                 \
    } else if (cu == 216) {                                                 \
      constexpr static int CUs = 216;                                       \
      return __VA_ARGS__();                                                 \
    } else if (cu == 256) {                                                 \
      constexpr static int CUs = 256;                                       \
      return __VA_ARGS__();                                                 \
    } else if (cu == 288) {                                                 \
      constexpr static int CUs = 288;                                       \
      return __VA_ARGS__();                                                 \
    } else {                                                                \
      std::cout << "unsupported low latency cu " << cu << std::endl;        \
    }                                                                       \
  }();

    } // end namespace MASKED_UTILS

} // end namespace DEEP_GEMM


