#pragma once

#include <ATen/hip/HIPContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <type_traits>

using half_t = _Float16;
using half4_t = __attribute__((__vector_size__(4 * sizeof(half_t)))) half_t;
using bf16_t = __hip_bfloat16;
using bf164_t = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using float4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using int4_reg_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using vec4_uint = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

template <typename Element, int Elems>
struct w16_vec {
  using type = __attribute__((__vector_size__(Elems * sizeof(Element)))) Element;
};

template <int Elems>
struct w16_vec<bf16_t, Elems> {
  using type = __attribute__((__vector_size__(Elems * sizeof(uint16_t)))) uint16_t;
};

template <typename Element>
using w16_vec4_t = typename w16_vec<Element, 4>::type;

template <typename Element>
using w16_vec8_t = typename w16_vec<Element, 8>::type;

template <typename Element>
__device__ __forceinline__ Element float_to_w16(float value) {
  if constexpr (std::is_same_v<Element, bf16_t>) {
    return __float2bfloat16(value);
  } else {
    return static_cast<Element>(value);
  }
}

template<bool Sync>
__forceinline__ __device__ void wait_buffer_data_arrived(const int wait_count = 0) {
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (Sync) asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier\n" :: "n"(wait_count) : "memory");
    else asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"(wait_count) : "memory");
    __builtin_amdgcn_sched_barrier(0);
    return;
}

template<bool Sync>
__forceinline__ __device__ void wait_lds_data_arrived(const int wait_count = 0) {
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (Sync) asm volatile("s_waitcnt lgkmcnt(%0)\n\ts_barrier\n" :: "n"(wait_count) : "memory");
    else asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(wait_count) : "memory");
    __builtin_amdgcn_sched_barrier(0);
    return;
}

template<typename VEC>
__forceinline__ __device__ void inlineasm_ds_read_b128(int lds_offset, VEC& data) {
    asm volatile("ds_read_b128 %0, %1\n"
        : "=v"(data)
        : "v"(lds_offset)
        :);
}

template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void inline_buffer_load_dwordx4_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

  int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n"
               :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
               :);
}

template<typename T>
__device__ __forceinline__ vec4_uint prepare_for_buffer_load(const T* ptr) {
    vec4_uint res;
    *(uint64_t*)&res = reinterpret_cast<uint64_t>(ptr);
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    return res;
}

template <typename T, typename Vec>
__device__ __forceinline__ void buffer_load_dwordx4_reg(const T* ptr, Vec& dst, int offset_bytes) {
  const char* base = reinterpret_cast<const char*>(ptr);
  dst = *reinterpret_cast<const Vec*>(base + offset_bytes);
}

#define DG_VMCNT_WAIT_0()                 \
  do {                                    \
    __builtin_hcu_sched_barrier(0);    \
    asm volatile("s_waitcnt vmcnt(0)\n\t" \
                 ::: "memory");          \
    __builtin_hcu_sched_barrier(0);    \
  } while (0)

__device__ __forceinline__ half4_t dequant_int8x4(int32_t packed, half_t scale) {
  half4_t out;
  out[0] = static_cast<half_t>((int8_t)(packed & 0xFF)) * scale;
  out[1] = static_cast<half_t>((int8_t)((packed >> 8) & 0xFF)) * scale;
  out[2] = static_cast<half_t>((int8_t)((packed >> 16) & 0xFF)) * scale;
  out[3] = static_cast<half_t>((int8_t)((packed >> 24) & 0xFF)) * scale;
  return out;
}

__device__ __forceinline__ half4_t cast_int8x4_to_fp16(int32_t packed) {
  half4_t out;
  out[0] = static_cast<half_t>((int8_t)(packed & 0xFF));
  out[1] = static_cast<half_t>((int8_t)((packed >> 8) & 0xFF));
  out[2] = static_cast<half_t>((int8_t)((packed >> 16) & 0xFF));
  out[3] = static_cast<half_t>((int8_t)((packed >> 24) & 0xFF));
  return out;
}

__device__ __forceinline__ bf164_t cast_int8x4_to_bf16(uint32_t packed_in) {
  uint32_t packed = packed_in ^ 0x80808080u;
  float fp32_intermediates[4];
  uint32_t* fp32_bits = reinterpret_cast<uint32_t*>(fp32_intermediates);
  static constexpr uint32_t fp32_base = 0x4B000000;
  static constexpr float signed_i8_offset = 8388736.f; // 2^23 + 128

  fp32_bits[0] = __byte_perm(packed, fp32_base, 0x7650);
  fp32_bits[1] = __byte_perm(packed, fp32_base, 0x7651);
  fp32_bits[2] = __byte_perm(packed, fp32_base, 0x7652);
  fp32_bits[3] = __byte_perm(packed, fp32_base, 0x7653);

  fp32_intermediates[0] -= signed_i8_offset;
  fp32_intermediates[1] -= signed_i8_offset;
  fp32_intermediates[2] -= signed_i8_offset;
  fp32_intermediates[3] -= signed_i8_offset;

  bf164_t out;
  uint32_t* out32 = reinterpret_cast<uint32_t*>(&out);
  out32[0] = __byte_perm(fp32_bits[0], fp32_bits[1], 0x7632);
  out32[1] = __byte_perm(fp32_bits[2], fp32_bits[3], 0x7632);
  return out;
}

template <typename Element>
__device__ __forceinline__ w16_vec4_t<Element> dequant_int8x4_w16(int32_t packed, half_t scale) {
  if constexpr (std::is_same_v<Element, bf16_t>) {
    w16_vec4_t<Element> out;
    bf16_t* vals = reinterpret_cast<bf16_t*>(&out);
    const float sc = static_cast<float>(scale);
    vals[0] = __float2bfloat16(static_cast<float>((int8_t)( packed        & 0xFF)) * sc);
    vals[1] = __float2bfloat16(static_cast<float>((int8_t)((packed >>  8) & 0xFF)) * sc);
    vals[2] = __float2bfloat16(static_cast<float>((int8_t)((packed >> 16) & 0xFF)) * sc);
    vals[3] = __float2bfloat16(static_cast<float>((int8_t)((packed >> 24) & 0xFF)) * sc);
    return out;
  } else {
    return dequant_int8x4(packed, scale);
  }
}

template <typename Element>
__device__ __forceinline__ w16_vec4_t<Element> dequant_int8x4_w16_float_scale(int32_t packed, float scale) {
  if constexpr (std::is_same_v<Element, bf16_t>) {
    w16_vec4_t<Element> out;
    bf16_t* vals = reinterpret_cast<bf16_t*>(&out);
    vals[0] = __float2bfloat16(static_cast<float>((int8_t)( packed        & 0xFF)) * scale);
    vals[1] = __float2bfloat16(static_cast<float>((int8_t)((packed >>  8) & 0xFF)) * scale);
    vals[2] = __float2bfloat16(static_cast<float>((int8_t)((packed >> 16) & 0xFF)) * scale);
    vals[3] = __float2bfloat16(static_cast<float>((int8_t)((packed >> 24) & 0xFF)) * scale);
    return out;
  } else {
    half4_t out;
    out[0] = static_cast<half_t>(static_cast<float>((int8_t)( packed        & 0xFF)) * scale);
    out[1] = static_cast<half_t>(static_cast<float>((int8_t)((packed >>  8) & 0xFF)) * scale);
    out[2] = static_cast<half_t>(static_cast<float>((int8_t)((packed >> 16) & 0xFF)) * scale);
    out[3] = static_cast<half_t>(static_cast<float>((int8_t)((packed >> 24) & 0xFF)) * scale);
    return out;
  }
}

template <typename Element>
__device__ __forceinline__ void dequant_w8a16_to_reg(
    uint32_t packed0,
    uint32_t packed1,
    w16_vec4_t<Element>& dst0,
    w16_vec4_t<Element>& dst1) {
  if constexpr (std::is_same_v<Element, bf16_t>) {
    dst0 = cast_int8x4_to_bf16(packed0);
    dst1 = cast_int8x4_to_bf16(packed1);
  } else {
    dst0 = cast_int8x4_to_fp16(static_cast<int32_t>(packed0));
    dst1 = cast_int8x4_to_fp16(static_cast<int32_t>(packed1));
  }
}

__device__ __forceinline__ void dequant_w8a16_to_reg(
    uint32_t packed0,
    uint32_t packed1,
    half4_t& dst0,
    half4_t& dst1) {
  dequant_w8a16_to_reg<half_t>(packed0, packed1, dst0, dst1);
}

__device__ __forceinline__ int marlin_w8a16_offset_u32(int n, int k_byte, int N) {
  const int n_block = n >> 4;
  const int n_inner = n & 15;
  const int k_u32 = k_byte >> 2;
  const int k32 = k_u32 >> 3;
  const int k_u32_inner = k_u32 & 7;
  const int k128 = k32 >> 2;
  const int k_sub = k32 & 3;
  return (((k128 * (N >> 4) + n_block) * 4 + k_sub) * 16 + n_inner) * 8 + k_u32_inner;
}

template <typename B_VEC, bool MARLIN_LAYOUT>
__device__ __forceinline__ B_VEC load_w8a16_weight_vec(
    const int8_t* __restrict__ b,
    int N,
    int K,
    int n,
    int k_byte) {
  if constexpr (!MARLIN_LAYOUT) {
    return *(const B_VEC*)(b + (size_t)n * K + k_byte);
  } else {
    constexpr int kWords = sizeof(B_VEC) / sizeof(int32_t);
    B_VEC out;
    int32_t* out_words = reinterpret_cast<int32_t*>(&out);
    const int32_t* b_words = reinterpret_cast<const int32_t*>(b);
    #pragma unroll
    for (int i = 0; i < kWords; i++) {
      out_words[i] = b_words[marlin_w8a16_offset_u32(n, k_byte + i * 4, N)];
    }
    return out;
  }
}

inline __device__ void builtin_amdgcn_mmac(
    const half4_t& reg_a,
    const half4_t& reg_b,
    float4_t& reg_c) {
#if defined(__gfx936__) || defined(__gfx92a__) || defined(__gfx928__)
  reg_c = __builtin_hcu_mmac_f32_16x16x16_f16(reg_a, reg_b, reg_c);
#elif defined(__gfx938__)
  reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a, reg_b, reg_c, false, false);
#endif
}

template <typename Element>
inline __device__ void builtin_amdgcn_mmac_w16(
    const w16_vec4_t<Element>& reg_a,
    const w16_vec4_t<Element>& reg_b,
    float4_t& reg_c) {
  if constexpr (std::is_same_v<Element, bf16_t>) {
#if defined(__gfx936__) || defined(__gfx92a__) || defined(__gfx928__) || defined(__gfx938__)
    reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16(reg_a, reg_b, reg_c);
#endif
  } else {
    builtin_amdgcn_mmac(reg_a, reg_b, reg_c);
  }
}

#define NUM_K_WARPS_SWITCH(num_k_warps, ...) \
  [&] {                                      \
    if (num_k_warps == 1) {                  \
      constexpr static int NUM_K_WARPS = 1;  \
      return __VA_ARGS__();                  \
    } else if (num_k_warps == 2) {           \
      constexpr static int NUM_K_WARPS = 2;  \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr static int NUM_K_WARPS = 4;  \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define NPerWarp_SWITCH(n_per_warp, ...)    \
  [&] {                                     \
    if (n_per_warp == 32) {                 \
      constexpr static int NPerWarp = 32;   \
      return __VA_ARGS__();                 \
    } else {                                \
      constexpr static int NPerWarp = 64;   \
      return __VA_ARGS__();                 \
    }                                       \
  }()

#define KPerWarp_SWITCH(k_per_warp, ...)     \
  [&] {                                      \
    if (k_per_warp == 64) {                  \
      constexpr static int KPerWarp = 64;    \
      return __VA_ARGS__();                  \
    } else if (k_per_warp == 128) {          \
      constexpr static int KPerWarp = 128;   \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr static int KPerWarp = 256;   \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define NUM_N_WARPS_SWITCH(num_n_warps, ...) \
  [&] {                                      \
    if (num_n_warps == 2) {                  \
      constexpr static int NUM_N_WARPS = 2;  \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr static int NUM_N_WARPS = 4;  \
      return __VA_ARGS__();                  \
    }                                        \
  }()
