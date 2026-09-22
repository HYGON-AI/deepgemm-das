#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

#include "moe_wna16_utils.h"
#include "intrinsic.h"

namespace deepgemm_w4a16_masked {

template <typename T>
__device__ __forceinline__ float to_float(T x) {
  return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<__hip_bfloat16>(__hip_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ __hip_bfloat16 from_float<__hip_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <typename scalar_t2>
__device__ __forceinline__ void dequant_w4_zp8(int q, scalar_t2* res) {
  dequant<scalar_t2, 4>(q, res);
}

template <>
__device__ __forceinline__ void dequant_w4_zp8<half2>(int q, half2* res) {
  dequant<half2, 4>(q, res);
  static constexpr uint32_t ZP = 0x48004800;  // +8.0 fp16x2
  half2 zp = *reinterpret_cast<const half2*>(&ZP);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    res[i] = __hsub2(res[i], zp);
  }
}

template <>
__device__ __forceinline__ void dequant_w4_zp8<__hip_bfloat162>(
    int q,
    __hip_bfloat162* res) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  static constexpr uint32_t ADD = 0xC308C308;  // -136.0 bf16x2 = -128 - zp(8)
  using __bf16x2 = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;
  __bf16x2 add_ = *reinterpret_cast<const __bf16x2*>(&ADD);
  __bf16x2 lo0_ = *reinterpret_cast<__bf16x2*>(&lo0);
  __bf16x2 hi0_ = *reinterpret_cast<__bf16x2*>(&hi0);
  __bf16x2 lo1_ = *reinterpret_cast<__bf16x2*>(&lo1);
  __bf16x2 hi1_ = *reinterpret_cast<__bf16x2*>(&hi1);

  __bf16x2 tmp0 = (__bf16x2)(lo0_ + add_);
  __bf16x2 tmp1 = (__bf16x2)(hi0_ + add_);
  __bf16x2 tmp2 = (__bf16x2)(lo1_ + add_);
  __bf16x2 tmp3 = (__bf16x2)(hi1_ + add_);
  res[0] = *reinterpret_cast<__hip_bfloat162*>(&tmp0);
  res[1] = *reinterpret_cast<__hip_bfloat162*>(&tmp1);
  res[2] = *reinterpret_cast<__hip_bfloat162*>(&tmp2);
  res[3] = *reinterpret_cast<__hip_bfloat162*>(&tmp3);
}

template <typename scalar_t2>
__device__ __forceinline__ void dequant_w4_scaled(
    int q,
    scalar_t2 scale_f2_vec,
    scalar_t2* res) {
  dequant<scalar_t2, 4>(q, res);
  scalar_t2 qzero_f2_vec;
  if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>) {
    qzero_f2_vec = *reinterpret_cast<const scalar_t2*>(&(const uint32_t&)0x41004100u);
  } else {
    qzero_f2_vec = *reinterpret_cast<const scalar_t2*>(&(const uint32_t&)0x44004400u);
  }
  const scalar_t2 bias = -qzero_f2_vec * scale_f2_vec;
  #pragma unroll
  for (int pack_id = 0; pack_id < 4; ++pack_id) {
    res[pack_id] = res[pack_id] * scale_f2_vec + bias;
  }
}

template <>
__device__ __forceinline__ void dequant_w4_scaled<__hip_bfloat162>(
    int q,
    __hip_bfloat162 scale_f2_vec,
    __hip_bfloat162* res) {
  dequant_w4_zp8<__hip_bfloat162>(q, res);
  #pragma unroll
  for (int pack_id = 0; pack_id < 4; ++pack_id) {
    res[pack_id] = res[pack_id] * scale_f2_vec;
  }
}

__device__ __forceinline__ float cvt_f32_ubyte(uint32_t x, int byte_id) {
  float out;
  switch (byte_id) {
    case 0:
      asm volatile("v_cvt_f32_ubyte0 %0, %1\n" : "=v"(out) : "v"(x));
      break;
    case 1:
      asm volatile("v_cvt_f32_ubyte1 %0, %1\n" : "=v"(out) : "v"(x));
      break;
    case 2:
      asm volatile("v_cvt_f32_ubyte2 %0, %1\n" : "=v"(out) : "v"(x));
      break;
    default:
      asm volatile("v_cvt_f32_ubyte3 %0, %1\n" : "=v"(out) : "v"(x));
      break;
  }
  return out;
}

__device__ __forceinline__ uint32_t pack_bf16x2_from_f32(float lo, float hi) {
#if defined(__gfx938__) && !defined(DEEPGEMM_W4A16_DISABLE_CVT_PK_BF16)
  using bf16x2_t = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;
  const bf16x2_t packed = __builtin_hcu_cvt_pk_bf16_f32(lo, hi, false);
  return *reinterpret_cast<const uint32_t*>(&packed);
#else
  const __hip_bfloat16 lo_bf16 = __float2bfloat16(lo);
  const __hip_bfloat16 hi_bf16 = __float2bfloat16(hi);
  const uint32_t lo_raw = *reinterpret_cast<const uint16_t*>(&lo_bf16);
  const uint32_t hi_raw = *reinterpret_cast<const uint16_t*>(&hi_bf16);
  return lo_raw | (hi_raw << 16);
#endif
}

__device__ __forceinline__ uint64_t pack_bf16x4_from_f32(
    float v0,
    float v1,
    float v2,
    float v3) {
  const uint32_t lo = pack_bf16x2_from_f32(v0, v1);
  const uint32_t hi = pack_bf16x2_from_f32(v2, v3);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

__device__ __forceinline__ int32_t ds_bpermute_b32(int byte_offset, int32_t src) {
  int32_t out;
  asm volatile("ds_bpermute_b32 %0, %1, %2"
               : "=v"(out)
               : "v"(byte_offset), "v"(src));
  return out;
}

__device__ __forceinline__ float bpermute_float(float x, int src_lane) {
  const int32_t bits = __builtin_bit_cast(int32_t, x);
  const int32_t out = ds_bpermute_b32(src_lane << 2, bits);
  return __builtin_bit_cast(float, out);
}

__device__ __forceinline__ uint32_t bf16_trunc_bits_from_f32(float x) {
  return (*reinterpret_cast<const uint32_t*>(&x)) >> 16;
}

__device__ __forceinline__ uint32_t pack_bf16x2_trunc_from_f32(float lo, float hi) {
#if defined(__gfx938__) && !defined(DEEPGEMM_W4A16_DISABLE_CVT_PK_BF16)
  uint32_t out;
  asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(out) : "v"(lo), "v"(hi));
  return out;
#else
  return bf16_trunc_bits_from_f32(lo) | (bf16_trunc_bits_from_f32(hi) << 16);
#endif
}

template <typename scalar_t2>
__device__ __forceinline__ void dequant_w4_scaled_pk_f32(
    int q,
    scalar_t2 scale_f2_vec,
    scalar_t2* res) {
  dequant_w4_scaled<scalar_t2>(q, scale_f2_vec, res);
}

template <>
__device__ __forceinline__ void dequant_w4_scaled_pk_f32<__hip_bfloat162>(
    int q,
    __hip_bfloat162 scale_f2_vec,
    __hip_bfloat162* res) {
  const uint32_t lo_nibbles = static_cast<uint32_t>(q) & 0x0f0f0f0fu;
  const uint32_t hi_nibbles = (static_cast<uint32_t>(q) >> 4) & 0x0f0f0f0fu;
  const float scale = __bfloat1622float2(scale_f2_vec).x;
  const float bias = -8.0f * scale;
  __float2 scale_pair = {scale, scale};
  __float2 bias_pair = {bias, bias};

  __float2 q01 = {cvt_f32_ubyte(lo_nibbles, 0), cvt_f32_ubyte(hi_nibbles, 0)};
  __float2 q23 = {cvt_f32_ubyte(lo_nibbles, 1), cvt_f32_ubyte(hi_nibbles, 1)};
  __float2 q45 = {cvt_f32_ubyte(lo_nibbles, 2), cvt_f32_ubyte(hi_nibbles, 2)};
  __float2 q67 = {cvt_f32_ubyte(lo_nibbles, 3), cvt_f32_ubyte(hi_nibbles, 3)};
  q01 = inlineasm_fa_v_pk_fma_f32(q01, scale_pair, bias_pair);
  q23 = inlineasm_fa_v_pk_fma_f32(q23, scale_pair, bias_pair);
  q45 = inlineasm_fa_v_pk_fma_f32(q45, scale_pair, bias_pair);
  q67 = inlineasm_fa_v_pk_fma_f32(q67, scale_pair, bias_pair);

  uint32_t packed0 = pack_bf16x2_trunc_from_f32(q01.x, q45.x);
  uint32_t packed1 = pack_bf16x2_trunc_from_f32(q01.y, q45.y);
  uint32_t packed2 = pack_bf16x2_trunc_from_f32(q23.x, q67.x);
  uint32_t packed3 = pack_bf16x2_trunc_from_f32(q23.y, q67.y);
  res[0] = *reinterpret_cast<__hip_bfloat162*>(&packed0);
  res[1] = *reinterpret_cast<__hip_bfloat162*>(&packed1);
  res[2] = *reinterpret_cast<__hip_bfloat162*>(&packed2);
  res[3] = *reinterpret_cast<__hip_bfloat162*>(&packed3);
}

template <
    typename T,
    int BLOCK_M,
    int BLOCK_N,
    int NUM_SM,
    bool SCALE_TILED = false>
__global__ void w4a16_masked_scalar_kernel(
    const T* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const void* __restrict__ matrix_b_scale,
    const int32_t* __restrict__ actual_m,
    T* __restrict__ matrix_c,
    int size_m,
    int size_n,
    int size_k,
    int experts) {
  constexpr int GROUP_WISE_K = 32;
  const int tid = threadIdx.x;
  const int elems_per_tile = BLOCK_M * BLOCK_N;
  const int num_n_tiles = (size_n + BLOCK_N - 1) / BLOCK_N;
  const T* __restrict__ matrix_b_scale_t =
      reinterpret_cast<const T*>(matrix_b_scale);

  uint32_t tile_idx = blockIdx.x;
  int last_problem_end = 0;

  for (int expert = 0; expert < experts; ++expert) {
    const int m = actual_m[expert];
    const int num_m_tiles = (m + BLOCK_M - 1) / BLOCK_M;
    const int expert_tiles = num_m_tiles * num_n_tiles;

    while (tile_idx >= static_cast<uint32_t>(last_problem_end) &&
           tile_idx < static_cast<uint32_t>(last_problem_end + expert_tiles)) {
      const int local_tile = static_cast<int>(tile_idx) - last_problem_end;
      const int tile_m_idx = local_tile / num_n_tiles;
      const int tile_n_idx = local_tile - tile_m_idx * num_n_tiles;
      const int m_base = tile_m_idx * BLOCK_M;
      const int n_base = tile_n_idx * BLOCK_N;

      for (int elem = tid; elem < elems_per_tile; elem += blockDim.x) {
        const int local_m = elem / BLOCK_N;
        const int local_n = elem - local_m * BLOCK_N;
        const int m_idx = m_base + local_m;
        const int n_idx = n_base + local_n;
        if (m_idx >= m || n_idx >= size_n) continue;

        float acc = 0.0f;
        const uint64_t a_base =
            (static_cast<uint64_t>(expert) * size_m + m_idx) * size_k;
        const uint64_t b_base =
            static_cast<uint64_t>(expert) * (size_k / 128) * (size_n / 16) * 4 * 16 * 4;
        const uint64_t s_base = SCALE_TILED
            ? static_cast<uint64_t>(expert) * (size_k / 128) * (size_n / 16) * 4 * 16
            : static_cast<uint64_t>(expert) * size_n * (size_k / GROUP_WISE_K);

        for (int k = 0; k < size_k; ++k) {
          const int k128 = k / 128;
          const int k_in_128 = k - k128 * 128;
          const int k32 = k_in_128 / 32;
          const int k_in_32 = k_in_128 - k32 * 32;
          const int pack = k_in_32 / 8;
          const int nibble = k_in_32 - pack * 8;
          const int n16 = n_idx / 16;
          const int n_in_16 = n_idx - n16 * 16;

          const uint64_t b_offset =
              b_base +
              static_cast<uint64_t>(k128) * (size_n / 16) * 4 * 16 * 4 +
              static_cast<uint64_t>(n16) * 4 * 16 * 4 +
              k32 * 16 * 4 +
              n_in_16 * 4 +
              pack;

          const uint32_t packed = matrix_b[b_offset];
          const int q = static_cast<int>((packed >> (nibble * 4)) & 0xFu);
          const uint64_t scale_offset = SCALE_TILED
              ? s_base + static_cast<uint64_t>(k128) * (size_n / 16) * 4 * 16 +
                    static_cast<uint64_t>(n16) * 4 * 16 + k32 * 16 + n_in_16
              : s_base + static_cast<uint64_t>(n_idx) * (size_k / GROUP_WISE_K) +
                    k / GROUP_WISE_K;
          const float scale = to_float(matrix_b_scale_t[scale_offset]);
          const float a = to_float(matrix_a[a_base + k]);
          acc += a * ((static_cast<float>(q) - 8.0f) * scale);
        }

        matrix_c[(static_cast<uint64_t>(expert) * size_m + m_idx) * size_n + n_idx] =
            from_float<T>(acc);
      }

      tile_idx += gridDim.x;
    }

    last_problem_end += expert_tiles;
  }
}

template <
    typename T,
    int NUM_WARPS,
    int NUM_SM,
    int N_REUSE = 1,
    bool USE_A_LDS = true,
    bool DENSE_M16 = false,
    bool CHECK_N_TILE = true,
    bool EXACT_M16 = false,
    bool SCALE_TILED = false,
    bool B_LDS_BUFFER_LOAD = false,
    bool B_REG_BUFFER_LOAD = false,
    bool A_REG_BUFFER_LOAD = false,
    bool PK_DEQUANT = false,
    bool PACK_BF16_STORE = false,
    bool PIPE_DEQUANT_MFMA = false>
__global__ void __launch_bounds__(NUM_WARPS * 64, 1) w4a16_masked_mfma_kernel(
    const T* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const void* __restrict__ matrix_b_scale,
    const int32_t* __restrict__ actual_m,
    T* __restrict__ matrix_c,
    int size_m,
    int size_n,
    int size_k,
    int experts) {
  using Dtype = ScalarType<T>;
  using scalar_t2 = typename ScalarType<T>::scalar_t2;

  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  constexpr int GROUP_WISE_K = 32;
  constexpr int WARP_K = 128;
  constexpr int MFMA_K = 32;
  constexpr int B_LDS_DWORDS_PER_WARP = 64 * 4;
  constexpr int B_LDS_STAGE_DWORDS = N_REUSE * NUM_WARPS * B_LDS_DWORDS_PER_WARP;
  static_assert(!B_LDS_BUFFER_LOAD || !USE_A_LDS, "B_LDS_BUFFER_LOAD is direct-A only");

  const int tid = threadIdx.x;
  const int warp_id = tid / 64;
  const int lane_id = tid & 63;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  const int n_tiles = size_n / TILE_N;
  const int n_groups = (n_tiles + NUM_WARPS * N_REUSE - 1) / (NUM_WARPS * N_REUSE);
  extern __shared__ __align__(16) unsigned char shared_storage[];
  T* a_smem = reinterpret_cast<T*>(shared_storage);
  uint32_t* b_smem = reinterpret_cast<uint32_t*>(shared_storage);
  const T* __restrict__ matrix_b_scale_t =
      reinterpret_cast<const T*>(matrix_b_scale);
  vec<uint, 4> b_resource;
  vec<uint, 4> scale_resource;
  if constexpr (B_LDS_BUFFER_LOAD || B_REG_BUFFER_LOAD) {
    b_resource = tcp_cache_swizzle_func<64, uint32_t>(matrix_b);
    scale_resource = tcp_cache_swizzle_func<64, T>(matrix_b_scale_t);
  }

  int expert_id = -1;
  int expert_start = 0;
  int expert_end = 0;

  for (uint32_t tile_idx = blockIdx.x;; tile_idx += gridDim.x) {
    int local_tile = 0;
    if constexpr (DENSE_M16) {
      expert_id = static_cast<int>(tile_idx / n_groups);
      if (expert_id >= experts) return;
      local_tile = static_cast<int>(tile_idx) - expert_id * n_groups;
    } else {
      while (tile_idx >= static_cast<uint32_t>(expert_end)) {
        ++expert_id;
        if (expert_id >= experts) return;
        expert_start = expert_end;
        const int expert_m_tiles = (actual_m[expert_id] + TILE_M - 1) / TILE_M;
        expert_end += expert_m_tiles * n_groups;
      }
      local_tile = static_cast<int>(tile_idx) - expert_start;
    }

    const int expert_m = EXACT_M16 ? TILE_M : actual_m[expert_id];
    const int tile_m_idx = EXACT_M16 ? 0 : local_tile / n_groups;
    const int n_group_idx = local_tile - tile_m_idx * n_groups;
    const int n_tile_idx_base = (n_group_idx * NUM_WARPS + warp_id) * N_REUSE;
    int n_tile_idx_arr[N_REUSE];
    int n_base_arr[N_REUSE];
    bool valid_n_tile_arr[N_REUSE];
    #pragma unroll
    for (int nr = 0; nr < N_REUSE; ++nr) {
      n_tile_idx_arr[nr] = n_tile_idx_base + nr;
      n_base_arr[nr] = n_tile_idx_arr[nr] * TILE_N;
      valid_n_tile_arr[nr] = !CHECK_N_TILE || n_tile_idx_arr[nr] < n_tiles;
    }

    const int m_base = tile_m_idx * TILE_M;
    const int block_actual_m = EXACT_M16
        ? TILE_M
        : (expert_m - m_base > TILE_M ? TILE_M : expert_m - m_base);
    if (block_actual_m <= 0) continue;

    union_vec_opt_w4a16_A<T, WARP_K / 4> A_reg;
    union_vec_opt_w4a16<uint32_t, WARP_K / 32> B_int_reg;
    reg_bf16_fp16<scalar_t2> B_reg[PIPE_DEQUANT_MFMA ? 1 : WARP_K / 32];
    floatx4 C_reg[N_REUSE];
    #pragma unroll
    for (int nr = 0; nr < N_REUSE; ++nr) {
      C_reg[nr] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    const uint64_t a_base =
        (static_cast<uint64_t>(expert_id) * size_m + m_base) * size_k;
    const T* __restrict__ a_expert_ptr =
        matrix_a + static_cast<uint64_t>(expert_id) * size_m * size_k;
    const uint64_t b_base =
        static_cast<uint64_t>(expert_id) * (size_k / 128) * (size_n / 16) * 4 * 16 * 4;
    const uint64_t s_base = SCALE_TILED
        ? static_cast<uint64_t>(expert_id) * (size_k / WARP_K) * (size_n / TILE_N) *
              (WARP_K / GROUP_WISE_K) * TILE_N
        : static_cast<uint64_t>(expert_id) * size_n * (size_k / GROUP_WISE_K);
    const int warp_id_s = __builtin_hcu_readfirstlane(warp_id);

    if constexpr (B_LDS_BUFFER_LOAD) {
      #pragma unroll
      for (int nr = 0; nr < N_REUSE; ++nr) {
        if (valid_n_tile_arr[nr]) {
          const uint64_t b_tile_base =
              b_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) * 4 * 16 * 4 +
              static_cast<uint64_t>(col_id) * 16 * 4 + static_cast<uint64_t>(row_id) * 4;
          const int lds_offset = __builtin_hcu_readfirstlane(
              nr * NUM_WARPS * B_LDS_DWORDS_PER_WARP +
              warp_id_s * B_LDS_DWORDS_PER_WARP);
          inline_buffer_load_dwordx4_lds<uint32_t, 2>(
              b_smem,
              b_resource,
              lds_offset,
              0,
              static_cast<int>(b_tile_base));
        }
      }
    }

    for (int k128 = 0; k128 < size_k / 128; ++k128) {
      if constexpr (USE_A_LDS) {
        for (int a_idx = tid; a_idx < TILE_M * WARP_K; a_idx += blockDim.x) {
          const int a_m = a_idx / WARP_K;
          const int a_k = a_idx - a_m * WARP_K;
          if (a_m < block_actual_m) {
            a_smem[a_idx] =
                matrix_a[a_base + static_cast<uint64_t>(a_m) * size_k + k128 * WARP_K + a_k];
          } else {
            a_smem[a_idx] = from_float<T>(0.0f);
          }
        }
        __syncthreads();
      }

      #pragma unroll
      for (int k_tile = 0; k_tile < WARP_K / GROUP_WISE_K; ++k_tile) {
        const int k_base = col_id * 32 + k_tile * 8;
        if constexpr (USE_A_LDS) {
          A_reg.input_half8[k_tile] =
              *reinterpret_cast<const vec8_Element<T>*>(&a_smem[row_id * WARP_K + k_base]);
        } else {
          if (row_id < block_actual_m) {
            if constexpr (B_LDS_BUFFER_LOAD || A_REG_BUFFER_LOAD) {
              const int a_offset = static_cast<int>(
                  (static_cast<uint64_t>(m_base + row_id) * size_k +
                   k128 * WARP_K + k_base) *
                  sizeof(T));
              buffer_load_reg_dwordx4_half8(
                  a_expert_ptr, A_reg.input_half8[k_tile], 0, a_offset);
            } else {
              A_reg.input_half8[k_tile] =
                  *reinterpret_cast<const vec8_Element<T>*>(
                      &matrix_a[a_base + static_cast<uint64_t>(row_id) * size_k +
                                k128 * WARP_K + k_base]);
            }
          } else {
            A_reg.input_half8[k_tile] = {};
          }
        }
      }
      scalar_t2 scale_f2_vec_arr[N_REUSE];
      if constexpr (B_LDS_BUFFER_LOAD) {
        #pragma unroll
        for (int nr = 0; nr < N_REUSE; ++nr) {
          scale_f2_vec_arr[nr] = Dtype::num2num2(from_float<T>(0.0f));
        }

        #pragma unroll
        for (int nr = 0; nr < N_REUSE; ++nr) {
          if (valid_n_tile_arr[nr]) {
            T scale_num;
            const uint64_t s_tile_base = SCALE_TILED
                ? s_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) *
                               (WARP_K / GROUP_WISE_K) * TILE_N +
                      static_cast<uint64_t>(col_id) * TILE_N + row_id
                : s_base + static_cast<uint64_t>(n_base_arr[nr] + row_id) *
                               (size_k / GROUP_WISE_K) +
                      col_id;
            const uint64_t scale_offset = SCALE_TILED
                ? s_tile_base + static_cast<uint64_t>(k128) *
                                     (size_n / TILE_N) *
                                     (WARP_K / GROUP_WISE_K) * TILE_N
                : s_tile_base + k128 * 4;
            inline_buffer_load_ushort<T, 1>(
                scale_num, static_cast<int>(scale_offset), scale_resource, 0);
            scale_f2_vec_arr[nr] = Dtype::num2num2(scale_num);
          }
        }

        const int next_k128 = k128 + 1;
        if (next_k128 < size_k / 128) {
          const int next_stage_base = (next_k128 & 1) * B_LDS_STAGE_DWORDS;
          #pragma unroll
          for (int nr = 0; nr < N_REUSE; ++nr) {
            if (valid_n_tile_arr[nr]) {
              const uint64_t b_tile_base =
                  b_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) * 4 * 16 * 4 +
                  static_cast<uint64_t>(col_id) * 16 * 4 + static_cast<uint64_t>(row_id) * 4;
              const int lds_offset = __builtin_hcu_readfirstlane(
                  next_stage_base + nr * NUM_WARPS * B_LDS_DWORDS_PER_WARP +
                  warp_id_s * B_LDS_DWORDS_PER_WARP);
              const uint64_t b_offset =
                  b_tile_base +
                  static_cast<uint64_t>(next_k128) * (size_n / 16) * 4 * 16 * 4;
              inline_buffer_load_dwordx4_lds<uint32_t, 2>(
                  b_smem,
                  b_resource,
                  lds_offset,
                  0,
                  static_cast<int>(b_offset));
            }
          }

          if constexpr (N_REUSE == 1) {
            if (valid_n_tile_arr[0]) {
              vmcnt_wait(1);
            } else {
              vmcnt_wait(0);
            }
          } else if constexpr (N_REUSE == 2) {
            if (valid_n_tile_arr[1]) {
              vmcnt_wait(2);
            } else if (valid_n_tile_arr[0]) {
              vmcnt_wait(1);
            } else {
              vmcnt_wait(0);
            }
          } else {
            vmcnt_wait(4);
          }
        } else {
          vmcnt_wait(0);
        }
      }

      if constexpr (
          B_REG_BUFFER_LOAD && !B_LDS_BUFFER_LOAD && !SCALE_TILED &&
          !PIPE_DEQUANT_MFMA && N_REUSE == 2) {
        union_vec_opt_w4a16<uint32_t, WARP_K / 32> B_int_reg_pair[2];
        T scale_num_pair[2];
        #pragma unroll
        for (int nr = 0; nr < 2; ++nr) {
          B_int_reg_pair[nr].int4_array[0] = {};
          if (valid_n_tile_arr[nr]) {
            const uint64_t b_tile_base =
                b_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) * 4 * 16 * 4 +
                static_cast<uint64_t>(col_id) * 16 * 4 + static_cast<uint64_t>(row_id) * 4;
            const uint64_t b_offset =
                b_tile_base + static_cast<uint64_t>(k128) * (size_n / 16) * 4 * 16 * 4;
            const uint64_t s_tile_base =
                s_base + static_cast<uint64_t>(n_base_arr[nr] + row_id) *
                             (size_k / GROUP_WISE_K) +
                col_id;
            inline_buffer_load_dwordx4<vec<uint32_t, 4>, 2>(
                B_int_reg_pair[nr].int4_array[0],
                b_resource,
                0,
                static_cast<int>(b_offset));
            inline_buffer_load_ushort<T, 1>(
                scale_num_pair[nr],
                static_cast<int>(s_tile_base + k128 * 4),
                scale_resource,
                0);
          }
        }
        vmcnt_only_wait(0);

        #pragma unroll
        for (int nr = 0; nr < 2; ++nr) {
          scalar_t2 scale_f2_vec = Dtype::num2num2(from_float<T>(0.0f));
          if (valid_n_tile_arr[nr]) {
            scale_f2_vec = Dtype::num2num2(scale_num_pair[nr]);
          }
          #pragma unroll
          for (int j = 0; j < WARP_K / GROUP_WISE_K; ++j) {
            if constexpr (PK_DEQUANT) {
              dequant_w4_scaled_pk_f32<scalar_t2>(
                  B_int_reg_pair[nr].uint_array[j], scale_f2_vec, B_reg[j].bf162);
            } else {
              dequant_w4_scaled<scalar_t2>(
                  B_int_reg_pair[nr].uint_array[j], scale_f2_vec, B_reg[j].bf162);
            }
          }

          if (valid_n_tile_arr[nr]) {
            #pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; ++k_tile) {
              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 0],
                      A_reg.input_half[k_tile * 8 + 4],
                      A_reg.input_half[k_tile * 8 + 1],
                      A_reg.input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[0][0],
                      B_reg[k_tile].vec2_i16[0][1],
                      B_reg[k_tile].vec2_i16[1][0],
                      B_reg[k_tile].vec2_i16[1][1]},
                  C_reg[nr]);

              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 2],
                      A_reg.input_half[k_tile * 8 + 6],
                      A_reg.input_half[k_tile * 8 + 3],
                      A_reg.input_half[k_tile * 8 + 7]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[2][0],
                      B_reg[k_tile].vec2_i16[2][1],
                      B_reg[k_tile].vec2_i16[3][0],
                      B_reg[k_tile].vec2_i16[3][1]},
                  C_reg[nr]);
            }
          }
        }
      } else {
      #pragma unroll
      for (int nr = 0; nr < N_REUSE; ++nr) {
        scalar_t2 scale_f2_vec = Dtype::num2num2(from_float<T>(0.0f));
        if (valid_n_tile_arr[nr]) {
          if constexpr (B_LDS_BUFFER_LOAD) {
            const int lds_offset =
                (k128 & 1) * B_LDS_STAGE_DWORDS + nr * blockDim.x * 4 + tid * 4;
            B_int_reg.int4_array[0] =
                *reinterpret_cast<const vec<uint32_t, 4>*>(&b_smem[lds_offset]);
          } else {
            const uint64_t b_tile_base =
                b_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) * 4 * 16 * 4 +
                static_cast<uint64_t>(col_id) * 16 * 4 + static_cast<uint64_t>(row_id) * 4;
            const uint64_t b_offset =
                b_tile_base + static_cast<uint64_t>(k128) * (size_n / 16) * 4 * 16 * 4;
            T scale_num;
            const uint64_t s_tile_base = SCALE_TILED
                ? s_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) *
                               (WARP_K / GROUP_WISE_K) * TILE_N +
                      static_cast<uint64_t>(col_id) * TILE_N + row_id
                : s_base + static_cast<uint64_t>(n_base_arr[nr] + row_id) *
                               (size_k / GROUP_WISE_K) +
                      col_id;
            if constexpr (B_REG_BUFFER_LOAD) {
              inline_buffer_load_dwordx4<vec<uint32_t, 4>, 2>(
                  B_int_reg.int4_array[0],
                  b_resource,
                  0,
                  static_cast<int>(b_offset));
              const uint64_t scale_offset = SCALE_TILED
                  ? s_tile_base + static_cast<uint64_t>(k128) *
                                       (size_n / TILE_N) *
                                       (WARP_K / GROUP_WISE_K) * TILE_N
                  : s_tile_base + k128 * 4;
              inline_buffer_load_ushort<T, 1>(
                  scale_num, static_cast<int>(scale_offset), scale_resource, 0);
              vmcnt_only_wait(0);
            } else {
              B_int_reg.int4_array[0] =
                  *reinterpret_cast<const vec<uint32_t, 4>*>(&matrix_b[b_offset]);
              scale_num = SCALE_TILED
                  ? matrix_b_scale_t[
                        s_tile_base + static_cast<uint64_t>(k128) *
                                         (size_n / TILE_N) *
                                         (WARP_K / GROUP_WISE_K) * TILE_N]
                  : matrix_b_scale_t[s_tile_base + k128 * 4];
            }
            scale_f2_vec = Dtype::num2num2(scale_num);
          }
          if constexpr (B_LDS_BUFFER_LOAD) {
            scale_f2_vec = scale_f2_vec_arr[nr];
          }
        } else {
          B_int_reg.int4_array[0] = {};
        }
        if constexpr (PIPE_DEQUANT_MFMA) {
          #pragma unroll
          for (int k_tile = 0; k_tile < WARP_K / MFMA_K; ++k_tile) {
            if constexpr (PK_DEQUANT) {
              dequant_w4_scaled_pk_f32<scalar_t2>(
                  B_int_reg.uint_array[k_tile], scale_f2_vec, B_reg[0].bf162);
            } else {
              dequant_w4_scaled<scalar_t2>(
                  B_int_reg.uint_array[k_tile], scale_f2_vec, B_reg[0].bf162);
            }

            if (valid_n_tile_arr[nr]) {
              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 0],
                      A_reg.input_half[k_tile * 8 + 4],
                      A_reg.input_half[k_tile * 8 + 1],
                      A_reg.input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[0].vec2_i16[0][0],
                      B_reg[0].vec2_i16[0][1],
                      B_reg[0].vec2_i16[1][0],
                      B_reg[0].vec2_i16[1][1]},
                  C_reg[nr]);

              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 2],
                      A_reg.input_half[k_tile * 8 + 6],
                      A_reg.input_half[k_tile * 8 + 3],
                      A_reg.input_half[k_tile * 8 + 7]},
                  vec4_Element<T>{
                      B_reg[0].vec2_i16[2][0],
                      B_reg[0].vec2_i16[2][1],
                      B_reg[0].vec2_i16[3][0],
                      B_reg[0].vec2_i16[3][1]},
                  C_reg[nr]);
            }
          }
        } else {
          #pragma unroll
          for (int j = 0; j < WARP_K / GROUP_WISE_K; ++j) {
            if constexpr (PK_DEQUANT) {
              dequant_w4_scaled_pk_f32<scalar_t2>(
                  B_int_reg.uint_array[j], scale_f2_vec, B_reg[j].bf162);
            } else {
              dequant_w4_scaled<scalar_t2>(
                  B_int_reg.uint_array[j], scale_f2_vec, B_reg[j].bf162);
            }
          }

          if (valid_n_tile_arr[nr]) {
            #pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; ++k_tile) {
              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 0],
                      A_reg.input_half[k_tile * 8 + 4],
                      A_reg.input_half[k_tile * 8 + 1],
                      A_reg.input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[0][0],
                      B_reg[k_tile].vec2_i16[0][1],
                      B_reg[k_tile].vec2_i16[1][0],
                      B_reg[k_tile].vec2_i16[1][1]},
                  C_reg[nr]);

              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 2],
                      A_reg.input_half[k_tile * 8 + 6],
                      A_reg.input_half[k_tile * 8 + 3],
                      A_reg.input_half[k_tile * 8 + 7]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[2][0],
                      B_reg[k_tile].vec2_i16[2][1],
                      B_reg[k_tile].vec2_i16[3][0],
                      B_reg[k_tile].vec2_i16[3][1]},
                  C_reg[nr]);
            }
          }
        }
      }
      }
      if constexpr (USE_A_LDS) {
        __syncthreads();
      }
    }

    if (row_id < block_actual_m) {
      #pragma unroll
      for (int nr = 0; nr < N_REUSE; ++nr) {
        if (valid_n_tile_arr[nr]) {
          const uint64_t c_offset =
              (static_cast<uint64_t>(expert_id) * size_m + m_base + row_id) * size_n +
              n_base_arr[nr] + col_id;
          if constexpr (PACK_BF16_STORE && std::is_same_v<T, __hip_bfloat16>) {
            #pragma unroll
            for (int reg_id = 0; reg_id < 4; ++reg_id) {
              const int src_lane_base = row_id;
              const float v0 = bpermute_float(C_reg[nr][reg_id], src_lane_base + 0 * 16);
              const float v1 = bpermute_float(C_reg[nr][reg_id], src_lane_base + 1 * 16);
              const float v2 = bpermute_float(C_reg[nr][reg_id], src_lane_base + 2 * 16);
              const float v3 = bpermute_float(C_reg[nr][reg_id], src_lane_base + 3 * 16);
              if (col_id == 0) {
                const uint64_t packed = pack_bf16x4_from_f32(v0, v1, v2, v3);
                *reinterpret_cast<uint64_t*>(&matrix_c[c_offset + reg_id * 4]) = packed;
              }
            }
          } else {
            #pragma unroll
            for (int reg_id = 0; reg_id < 4; ++reg_id) {
              matrix_c[c_offset + reg_id * 4] = Dtype::float2num(C_reg[nr][reg_id]);
            }
          }
        }
      }
    }
  }
}

template <typename T, int M_REUSE, int NUM_WARPS, int NUM_SM>
__global__ void __launch_bounds__(NUM_WARPS * 64, 1)
w4a16_masked_mfma_mreuse_kernel(
    const T* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const void* __restrict__ matrix_b_scale,
    const int32_t* __restrict__ actual_m,
    T* __restrict__ matrix_c,
    int size_m,
    int size_n,
    int size_k,
    int experts) {
  using Dtype = ScalarType<T>;
  using scalar_t2 = typename ScalarType<T>::scalar_t2;

  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  constexpr int GROUP_WISE_K = 32;
  constexpr int WARP_K = 128;
  constexpr int MFMA_K = 32;

  const int tid = threadIdx.x;
  const int warp_id = tid / 64;
  const int lane_id = tid & 63;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  const int n_tiles = size_n / TILE_N;
  const int n_groups = (n_tiles + NUM_WARPS - 1) / NUM_WARPS;

  extern __shared__ __align__(16) unsigned char shared_storage[];
  T* a_smem = reinterpret_cast<T*>(shared_storage);
  const T* __restrict__ matrix_b_scale_t =
      reinterpret_cast<const T*>(matrix_b_scale);

  int expert_id = -1;
  int expert_start = 0;
  int expert_end = 0;

  for (uint32_t tile_idx = blockIdx.x;; tile_idx += gridDim.x) {
    while (tile_idx >= static_cast<uint32_t>(expert_end)) {
      ++expert_id;
      if (expert_id >= experts) return;
      expert_start = expert_end;
      const int expert_m_groups =
          (actual_m[expert_id] + M_REUSE * TILE_M - 1) / (M_REUSE * TILE_M);
      expert_end += expert_m_groups * n_groups;
    }

    const int local_tile = static_cast<int>(tile_idx) - expert_start;

    const int expert_m = actual_m[expert_id];
    const int tile_m_group_idx = local_tile / n_groups;
    const int n_group_idx = local_tile - tile_m_group_idx * n_groups;
    const int n_tile_idx = n_group_idx * NUM_WARPS + warp_id;
    const bool valid_n_tile = n_tile_idx < n_tiles;

    const int m_group_base = tile_m_group_idx * M_REUSE * TILE_M;
    const int n_base = n_tile_idx * TILE_N;
    const int valid_m_tiles =
        (expert_m - m_group_base + TILE_M - 1) / TILE_M < M_REUSE
            ? (expert_m - m_group_base + TILE_M - 1) / TILE_M
            : M_REUSE;

    union_vec_opt_w4a16_A<T, WARP_K / 4> A_reg[M_REUSE];
    union_vec_opt_w4a16<uint32_t, WARP_K / 32> B_int_reg;
    reg_bf16_fp16<scalar_t2> B_reg[WARP_K / 32];
    floatx4 C_reg[M_REUSE];
    #pragma unroll
    for (int m_tile = 0; m_tile < M_REUSE; ++m_tile) {
      C_reg[m_tile] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    const uint64_t a_group_base =
        (static_cast<uint64_t>(expert_id) * size_m + m_group_base) * size_k;
    const uint64_t b_base =
        static_cast<uint64_t>(expert_id) * (size_k / 128) * (size_n / 16) * 4 * 16 * 4;
    const uint64_t s_base =
        static_cast<uint64_t>(expert_id) * size_n * (size_k / GROUP_WISE_K);

    #pragma unroll
    for (int k128 = 0; k128 < size_k / 128; ++k128) {
      #pragma unroll
      for (int a_idx = tid; a_idx < valid_m_tiles * TILE_M * WARP_K; a_idx += blockDim.x) {
        const int local_m = a_idx / WARP_K;
        const int a_k = a_idx - local_m * WARP_K;
        const int global_m = m_group_base + local_m;
        if (global_m < expert_m) {
          a_smem[a_idx] =
              matrix_a[a_group_base + static_cast<uint64_t>(local_m) * size_k +
                       k128 * WARP_K + a_k];
        } else {
          a_smem[a_idx] = from_float<T>(0.0f);
        }
      }
      __syncthreads();

      #pragma unroll
      for (int m_tile = 0; m_tile < M_REUSE; ++m_tile) {
        if (m_tile < valid_m_tiles) {
          #pragma unroll
          for (int k_tile = 0; k_tile < WARP_K / GROUP_WISE_K; ++k_tile) {
            const int k_base = col_id * 32 + k_tile * 8;
            const int a_offset = (m_tile * TILE_M + row_id) * WARP_K + k_base;
            A_reg[m_tile].input_half8[k_tile] =
                *reinterpret_cast<const vec8_Element<T>*>(&a_smem[a_offset]);
          }
        }
      }

      scalar_t2 scale_f2_vec = Dtype::num2num2(from_float<T>(0.0f));
      if (valid_n_tile) {
        const uint64_t b_offset =
            b_base +
            static_cast<uint64_t>(k128) * (size_n / 16) * 4 * 16 * 4 +
            static_cast<uint64_t>(n_tile_idx) * 4 * 16 * 4 +
            static_cast<uint64_t>(col_id) * 16 * 4 +
            static_cast<uint64_t>(row_id) * 4;
        B_int_reg.int4_array[0] =
            *reinterpret_cast<const vec<uint32_t, 4>*>(&matrix_b[b_offset]);

        const T scale_num = matrix_b_scale_t[
            s_base + static_cast<uint64_t>(n_base + row_id) * (size_k / GROUP_WISE_K) +
            k128 * 4 + col_id];
        scale_f2_vec = Dtype::num2num2(scale_num);
      } else {
        B_int_reg.int4_array[0] = {};
      }
      #pragma unroll
      for (int j = 0; j < WARP_K / GROUP_WISE_K; ++j) {
        dequant_w4_scaled<scalar_t2>(
            B_int_reg.uint_array[j], scale_f2_vec, B_reg[j].bf162);
      }

      if (valid_n_tile) {
        #pragma unroll
        for (int m_tile = 0; m_tile < M_REUSE; ++m_tile) {
          if (m_tile < valid_m_tiles) {
            #pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; ++k_tile) {
              C_reg[m_tile] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg[m_tile].input_half[k_tile * 8 + 0],
                      A_reg[m_tile].input_half[k_tile * 8 + 4],
                      A_reg[m_tile].input_half[k_tile * 8 + 1],
                      A_reg[m_tile].input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[0][0],
                      B_reg[k_tile].vec2_i16[0][1],
                      B_reg[k_tile].vec2_i16[1][0],
                      B_reg[k_tile].vec2_i16[1][1]},
                  C_reg[m_tile]);

              C_reg[m_tile] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg[m_tile].input_half[k_tile * 8 + 2],
                      A_reg[m_tile].input_half[k_tile * 8 + 6],
                      A_reg[m_tile].input_half[k_tile * 8 + 3],
                      A_reg[m_tile].input_half[k_tile * 8 + 7]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[2][0],
                      B_reg[k_tile].vec2_i16[2][1],
                      B_reg[k_tile].vec2_i16[3][0],
                      B_reg[k_tile].vec2_i16[3][1]},
                  C_reg[m_tile]);
            }
          }
        }
      }
      __syncthreads();
    }

    if (valid_n_tile) {
      #pragma unroll
      for (int m_tile = 0; m_tile < M_REUSE; ++m_tile) {
        if (m_tile < valid_m_tiles) {
          const int m_base = m_group_base + m_tile * TILE_M;
          const int block_actual_m = expert_m - m_base > TILE_M ? TILE_M : expert_m - m_base;
          if (row_id < block_actual_m) {
            #pragma unroll
            for (int reg_id = 0; reg_id < 4; ++reg_id) {
              const int n_idx = n_base + reg_id * 4 + col_id;
              matrix_c[(static_cast<uint64_t>(expert_id) * size_m + m_base + row_id) * size_n + n_idx] =
                  Dtype::float2num(C_reg[m_tile][reg_id]);
            }
          }
        }
      }
    }
  }
}

template <
    typename T,
    int NUM_WARPS,
    int NUM_SM,
    int N_REUSE,
    bool CHECK_N_TILE,
    bool SCALE_TILED,
    bool B_LDS_BUFFER_LOAD = false,
    bool B_REG_BUFFER_LOAD = false,
    bool A_REG_BUFFER_LOAD = false,
    bool DENSE_M16 = false,
    bool EXACT_M16 = false,
    bool PK_DEQUANT = false,
    bool PACK_BF16_STORE = false,
    bool PIPE_DEQUANT_MFMA = false>
void launch_w4a16_direct_a_kernel(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int size_m,
    int size_n,
    int size_k,
    int experts,
    int grid_x,
    hipStream_t stream) {
  w4a16_masked_mfma_kernel<
      T,
      NUM_WARPS,
      NUM_SM,
      N_REUSE,
      false,
      DENSE_M16,
      CHECK_N_TILE,
      EXACT_M16,
      SCALE_TILED,
      B_LDS_BUFFER_LOAD,
      B_REG_BUFFER_LOAD,
      A_REG_BUFFER_LOAD,
      PK_DEQUANT,
      PACK_BF16_STORE,
      PIPE_DEQUANT_MFMA>
      <<<dim3(grid_x),
         dim3(NUM_WARPS * 64),
         B_LDS_BUFFER_LOAD ? 2 * N_REUSE * NUM_WARPS * 64 * 4 * sizeof(uint32_t) : 0,
         stream>>>(
          reinterpret_cast<const T*>(matrix_a.data_ptr()),
          reinterpret_cast<const uint32_t*>(matrix_b.data_ptr()),
          matrix_b_scale.data_ptr(),
          reinterpret_cast<const int32_t*>(actual_tokens.data_ptr()),
          reinterpret_cast<T*>(matrix_c.data_ptr()),
          size_m,
          size_n,
          size_k,
          experts);
}

template <
    typename T,
    int NUM_WARPS,
    int NUM_SM,
    bool B_LDS_BUFFER_LOAD = false,
    bool B_REG_BUFFER_LOAD = false,
    bool A_REG_BUFFER_LOAD = false,
    bool DENSE_M16 = false,
    bool EXACT_M16 = false,
    bool PK_DEQUANT = false,
    bool PACK_BF16_STORE = false,
    bool PIPE_DEQUANT_MFMA = false>
void launch_w4a16_direct_a(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int size_m,
    int size_n,
    int size_k,
    int experts,
    int grid_x,
    hipStream_t stream,
    int nreuse_value,
    bool use_tiled_scale) {
  const bool full_n_groups = ((size_n / 16) % NUM_WARPS) == 0;
  const bool full_n_groups2 = ((size_n / 16) % (NUM_WARPS * 2)) == 0;
  const bool full_n_groups4 = ((size_n / 16) % (NUM_WARPS * 4)) == 0;

  if (nreuse_value == 4) {
    if (use_tiled_scale) {
      if (full_n_groups4) {
        launch_w4a16_direct_a_kernel<
            T, NUM_WARPS, NUM_SM, 4, false, true, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
            matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
            size_m, size_n, size_k, experts, grid_x, stream);
      } else {
        launch_w4a16_direct_a_kernel<
            T, NUM_WARPS, NUM_SM, 4, true, true, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
            matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
            size_m, size_n, size_k, experts, grid_x, stream);
      }
    } else if (full_n_groups4) {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS, NUM_SM, 4, false, false, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream);
    } else {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS, NUM_SM, 4, true, false, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream);
    }
  } else if (nreuse_value == 2) {
    if (use_tiled_scale) {
      if (full_n_groups2) {
        launch_w4a16_direct_a_kernel<
            T, NUM_WARPS, NUM_SM, 2, false, true, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
            matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
            size_m, size_n, size_k, experts, grid_x, stream);
      } else {
        launch_w4a16_direct_a_kernel<
            T, NUM_WARPS, NUM_SM, 2, true, true, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
            matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
            size_m, size_n, size_k, experts, grid_x, stream);
      }
    } else if (full_n_groups2) {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS, NUM_SM, 2, false, false, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream);
    } else {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS, NUM_SM, 2, true, false, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream);
    }
  } else if (use_tiled_scale) {
    if (full_n_groups) {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS, NUM_SM, 1, false, true, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream);
    } else {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS, NUM_SM, 1, true, true, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream);
    }
  } else if (full_n_groups) {
    launch_w4a16_direct_a_kernel<
        T, NUM_WARPS, NUM_SM, 1, false, false, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
        size_m, size_n, size_k, experts, grid_x, stream);
  } else {
    launch_w4a16_direct_a_kernel<
        T, NUM_WARPS, NUM_SM, 1, true, false, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, PK_DEQUANT, PACK_BF16_STORE, PIPE_DEQUANT_MFMA>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
        size_m, size_n, size_k, experts, grid_x, stream);
  }
}

template <
    typename T,
    int NUM_WARPS,
    int NUM_SM,
    bool B_LDS_BUFFER_LOAD = false,
    bool B_REG_BUFFER_LOAD = false,
    bool A_REG_BUFFER_LOAD = false,
    bool DENSE_M16 = false,
    bool EXACT_M16 = false,
    bool PIPE_DEQUANT_MFMA = false>
void launch_w4a16_direct_a_dispatch(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int size_m,
    int size_n,
    int size_k,
    int experts,
    int grid_x,
    hipStream_t stream,
    int nreuse_value,
    bool use_tiled_scale,
    bool use_pk_dequant,
    bool use_pack_bf16_store) {
  if (use_pk_dequant) {
    if (use_pack_bf16_store) {
      launch_w4a16_direct_a<
          T, NUM_WARPS, NUM_SM, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, true, true, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream, nreuse_value, use_tiled_scale);
    } else {
      launch_w4a16_direct_a<
          T, NUM_WARPS, NUM_SM, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, true, false, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream, nreuse_value, use_tiled_scale);
    }
  } else {
    if (use_pack_bf16_store) {
      launch_w4a16_direct_a<
          T, NUM_WARPS, NUM_SM, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, false, true, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream, nreuse_value, use_tiled_scale);
    } else {
      launch_w4a16_direct_a<
          T, NUM_WARPS, NUM_SM, B_LDS_BUFFER_LOAD, B_REG_BUFFER_LOAD, A_REG_BUFFER_LOAD, DENSE_M16, EXACT_M16, false, false, PIPE_DEQUANT_MFMA>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, experts, grid_x, stream, nreuse_value, use_tiled_scale);
    }
  }
}

template <typename T>
void launch_w4a16_masked(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts) {
  constexpr int NUM_SM = 360;
  constexpr int NUM_WARPS8 = 8;
  constexpr int NUM_WARPS4 = 4;
  constexpr int TILE_N = 16;

  const int size_m = matrix_a.size(1);
  const int size_n = matrix_c.size(2);
  const int size_k = matrix_a.size(2);
  const int n_tiles = size_n / TILE_N;
  const int expected_m = static_cast<int>(max_tokens);

  TORCH_CHECK(
      matrix_b_scale.dim() == 3,
      "default W4A16 masked path expects plain scale layout [E,N,K//32]");

  const int small_expected_m_switch = size_k <= 2048 ? 1 : 4;
  const bool use_small_expected_m_path = expected_m <= small_expected_m_switch;
  const char* grid_env = std::getenv("DEEPGEMM_W4A16_GRID");
  const int grid_x = grid_env != nullptr ? std::atoi(grid_env) : NUM_SM;
  TORCH_CHECK(grid_x > 0, "DEEPGEMM_W4A16_GRID must be positive, got ", grid_x);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  // Keep only the production default instantiations. The older experimental
  // launch fanout (scalar, LDS, m-reuse, tiled scale, PK off, dense multi-M,
  // etc.) triggered gfx936 compiler register-allocation crashes while
  // compiling unused paths.
  if (use_small_expected_m_path) {
    const bool full_n_groups2 = (n_tiles % (NUM_WARPS8 * 2)) == 0;
    if (full_n_groups2) {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS8, NUM_SM, 2, false, false, false, true, true, false, false, true, false, false>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, static_cast<int>(experts), grid_x, stream);
    } else {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS8, NUM_SM, 2, true, false, false, true, true, false, false, true, false, false>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, static_cast<int>(experts), grid_x, stream);
    }
  } else {
    const bool full_n_groups4 = (n_tiles % (NUM_WARPS4 * 4)) == 0;
    if (full_n_groups4) {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS4, NUM_SM, 4, false, false, false, true, true, false, false, true, false, false>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, static_cast<int>(experts), grid_x, stream);
    } else {
      launch_w4a16_direct_a_kernel<
          T, NUM_WARPS4, NUM_SM, 4, true, false, false, true, true, false, false, true, false, false>(
          matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c,
          size_m, size_n, size_k, static_cast<int>(experts), grid_x, stream);
    }
  }
}

inline void masked_w4a16_gemm(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts) {
  TORCH_CHECK(matrix_a.is_contiguous());
  TORCH_CHECK(matrix_b.is_contiguous());
  TORCH_CHECK(matrix_b_scale.is_contiguous());
  TORCH_CHECK(matrix_c.is_contiguous());
  TORCH_CHECK(actual_tokens.is_contiguous());
  TORCH_CHECK(actual_tokens.scalar_type() == at::kInt);
  TORCH_CHECK(matrix_a.scalar_type() == at::ScalarType::Half ||
              matrix_a.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(matrix_a.scalar_type() == matrix_b_scale.scalar_type());
  TORCH_CHECK(matrix_a.scalar_type() == matrix_c.scalar_type());
  TORCH_CHECK(matrix_b.scalar_type() == at::ScalarType::UInt32 ||
              matrix_b.scalar_type() == at::ScalarType::Int,
              "w4a16 expects packed uint32/int32 weight tensor, got ",
              matrix_b.scalar_type());
  TORCH_CHECK(matrix_a.dim() == 3 && matrix_c.dim() == 3);
  TORCH_CHECK(matrix_b.dim() == 6);
  TORCH_CHECK(
      matrix_b_scale.dim() == 3,
      "matrix_b_scale must be [E,N,K//32], got dim ",
      matrix_b_scale.dim());
  TORCH_CHECK(matrix_a.size(0) >= experts);
  TORCH_CHECK(matrix_b.size(0) == experts);
  TORCH_CHECK(matrix_b_scale.size(0) == experts);
  TORCH_CHECK(matrix_c.size(0) == experts);
  TORCH_CHECK(matrix_a.size(1) == matrix_c.size(1));
  TORCH_CHECK(matrix_a.size(2) % 128 == 0);
  TORCH_CHECK(matrix_c.size(2) % 16 == 0);
  TORCH_CHECK(matrix_b_scale.size(1) == matrix_c.size(2));
  TORCH_CHECK(matrix_b_scale.size(2) == matrix_a.size(2) / 32);

  if (matrix_a.scalar_type() == at::ScalarType::Half) {
    launch_w4a16_masked<__half>(
        matrix_a,
        matrix_b,
        matrix_b_scale,
        actual_tokens,
        matrix_c,
        max_tokens,
        experts);
  } else {
    launch_w4a16_masked<__hip_bfloat16>(
        matrix_a,
        matrix_b,
        matrix_b_scale,
        actual_tokens,
        matrix_c,
        max_tokens,
        experts);
  }
}

}  // namespace deepgemm_w4a16_masked
