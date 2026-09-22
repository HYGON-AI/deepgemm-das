#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <type_traits>

#include "moe_marlin/moe_wna16_utils.h"
#include "moe_marlin/intrinsic.h"

namespace deepgemm_wfp4a16_masked {

struct Wfp4a16MaskedConfig {
  int mode;
  int block_m;
  int block_n;
  int block_k;
};

inline Wfp4a16MaskedConfig select_wfp4a16_masked_config(
    int64_t max_tokens,
    int64_t size_n,
    int64_t size_k) {
  (void)max_tokens;
  (void)size_n;
  (void)size_k;
  return {514, 128, 128, 64};
}

inline bool wfp4a16_optimized_supported(int64_t size_k) {
  return size_k % 64 == 0;
}

template <typename Element>
__device__ __forceinline__ uint16_t fp4_e2m1_e8m0_to_bits(
    uint32_t code,
    uint8_t scale) {
  const uint32_t mag = code & 0x7u;
  const uint32_t sign = (code & 0x8u) ? 0x8000u : 0u;
  if (mag == 0u) {
    return static_cast<uint16_t>(sign);
  }

  const int exp_delta = static_cast<int>(mag >> 1) - 1;
  const bool has_half_mant = (mag & 1u) && mag != 1u;
  if constexpr (std::is_same_v<Element, __hip_bfloat16>) {
    const int exp = static_cast<int>(scale) + exp_delta;
    if (exp <= 0) {
      return static_cast<uint16_t>(sign);
    }
    if (exp >= 255) {
      return static_cast<uint16_t>(sign | 0x7f80u);
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exp) << 7) | (has_half_mant ? 0x40u : 0u));
  } else {
    const int exp = static_cast<int>(scale) - 112 + exp_delta;
    if (exp <= 0) {
      return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exp) << 10) | (has_half_mant ? 0x200u : 0u));
  }
}

template <typename Element, typename scalar_t2>
__device__ __forceinline__ void dequant_fp4_e2m1_e8m0(
    uint32_t q,
    uint8_t scale,
    scalar_t2* res) {
  uint32_t* out = reinterpret_cast<uint32_t*>(res);
  #pragma unroll
  for (int pack_id = 0; pack_id < 4; ++pack_id) {
    const uint32_t code0 = (q >> (pack_id * 4)) & 0x0fu;
    const uint32_t code1 = (q >> (pack_id * 4 + 16)) & 0x0fu;
    const uint16_t v0 = fp4_e2m1_e8m0_to_bits<Element>(code0, scale);
    const uint16_t v1 = fp4_e2m1_e8m0_to_bits<Element>(code1, scale);
    out[pack_id] = static_cast<uint32_t>(v0) | (static_cast<uint32_t>(v1) << 16);
  }
}

__device__ __forceinline__ float e8m0_to_float_fast(uint8_t scale) {
  union {
    uint32_t u;
    float f;
  } cvt;
  cvt.u = static_cast<uint32_t>(scale) << 23;
  return cvt.f;
}

__device__ __forceinline__ uint32_t pack_bf16_pair_from_f32(float lo, float hi) {
#if defined(__gfx938__)
  uint32_t packed;
  asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2\n\t"
               : "=v"(packed)
               : "v"(lo), "v"(hi));
  return packed;
#else
  const __hip_bfloat16 lo_b = __float2bfloat16(lo);
  const __hip_bfloat16 hi_b = __float2bfloat16(hi);
  uint32_t packed = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo_b));
  packed |= static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi_b)) << 16;
  return packed;
#endif
}

template <typename Element, typename scalar_t2>
__device__ __forceinline__ void dequant_fp4_e2m1_e8m0_lut(
    uint32_t q,
    uint8_t scale,
    const float* fp4_lut,
    scalar_t2* res) {
  if constexpr (!std::is_same_v<Element, __hip_bfloat16>) {
    dequant_fp4_e2m1_e8m0<Element, scalar_t2>(q, scale, res);
  } else {
    uint32_t* out = reinterpret_cast<uint32_t*>(res);
    const float scale_f = e8m0_to_float_fast(scale);
    #pragma unroll
    for (int pack_id = 0; pack_id < 4; ++pack_id) {
      const uint32_t code0 = (q >> (pack_id * 4)) & 0x0fu;
      const uint32_t code1 = (q >> (pack_id * 4 + 16)) & 0x0fu;
      const float v0 = fp4_lut[code0] * scale_f;
      const float v1 = fp4_lut[code1] * scale_f;
      out[pack_id] = pack_bf16_pair_from_f32(v0, v1);
    }
  }
}

template <typename Element>
__device__ __forceinline__ floatx4 wfp4a16_mmac_normal(
    const vec4_Element<Element>& a,
    const vec4_Element<Element>& b,
    const floatx4& c) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx928__)
  return mmac<Element, float>(a, b, c);
#else
  return {0, 0, 0, 0};
#endif
}

template <typename Element>
__device__ __forceinline__ floatx4 wfp4a16_mmac_interleave(
    const vec4_Element<Element>& a,
    const vec4_Element<Element>& b,
    const floatx4& c) {
#if defined(__gfx938__)
  constexpr bool lit = true;
  constexpr bool lts = false;
  if constexpr (std::is_same_v<Element, __hip_bfloat16>) {
    return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, lit, lts);
  } else {
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a, b, c, lit, lts);
  }
#else
  return {0, 0, 0, 0};
#endif
}

template <typename Element, bool INTERLEAVE>
__device__ __forceinline__ floatx4 wfp4a16_mmac(
    const vec4_Element<Element>& a,
    const vec4_Element<Element>& b,
    const floatx4& c) {
  if constexpr (INTERLEAVE) {
    return wfp4a16_mmac_interleave<Element>(a, b, c);
  } else {
    return wfp4a16_mmac_normal<Element>(a, b, c);
  }
}

__device__ __forceinline__ int wfp4a16_min_int(int a, int b) {
  return a < b ? a : b;
}

template <typename Element>
__device__ __forceinline__ void store_vec8_with_n_guard(
    Element* ptr,
    const vec8_Element<Element>& val,
    int valid_elems) {
  if (valid_elems >= 8) {
    *reinterpret_cast<vec8_Element<Element>*>(ptr) = val;
  } else {
    uint16_t* out = reinterpret_cast<uint16_t*>(ptr);
    const uint16_t* in = reinterpret_cast<const uint16_t*>(&val);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < valid_elems) {
        out[i] = in[i];
      }
    }
  }
}

template<
    int WARP_NUM,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_K,
    typename Element,
    typename scalar_t2,
    bool INTERLEAVE,
    bool DEQUANT_16BIT,
    int32_t PACK_SCALES,
    bool FP4_A_LOAD_BY_WARP = true>
__forceinline__ __device__ void gemm_nt_wfp4a16_prefill_stage1_b128_k64_masked(
    const Element* __restrict__ input_ptr,
    const uint32_t* __restrict__ weight_ptr,
    const uint8_t* __restrict__ weight_scale_u8_ptr,
    int actual_m,
    int size_m,
    int size_n,
    floatx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int size_k,
    int n_base,
    int m_base,
    int warp_m_id) {
  (void)size_m;
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;

  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int warp_n_num = BLOCK_N / WARP_N;
  constexpr int COMPUTE_K = BLOCK_K;
  constexpr int warp_k_id = 0;
  const int warp_n_id = warp_id % warp_n_num;

  extern __shared__ float lds_buffer[];
  Element* lds_A = reinterpret_cast<Element*>(lds_buffer);
  Element* lds_B = reinterpret_cast<Element*>(lds_buffer);
  constexpr int32_t FP4_LUT_BYTES = 64;
  constexpr int32_t FP4_LUT_FLOATS = FP4_LUT_BYTES / sizeof(float);
  constexpr int32_t FP4_LUT_OFFSET_FLOATS = (16 * 1024) / sizeof(float);
  float* fp4_lut = lds_buffer + FP4_LUT_OFFSET_FLOATS;

  constexpr int32_t BLOCK_SIZE = WARP_NUM * 64;
  constexpr int32_t ELEMENTS_PER_LD_A = 128 / (sizeof(Element) * 8);
  constexpr int32_t G2R_A_COLS = COMPUTE_K / ELEMENTS_PER_LD_A;
  constexpr int32_t G2R_A_ROWS = BLOCK_SIZE / G2R_A_COLS;
  constexpr int32_t FP4_A_ROWS_PER_WARP = 16;
  constexpr int32_t FP4_A_ROWS_PER_LD_ROUND = 64 / G2R_A_COLS;
  constexpr int32_t FP4_A_LD_ROUNDS_PER_GROUP =
      (FP4_A_ROWS_PER_WARP + FP4_A_ROWS_PER_LD_ROUND - 1) /
      FP4_A_ROWS_PER_LD_ROUND;
  constexpr int32_t FP4_A_LOAD_GROUPS =
      (BLOCK_M + FP4_A_ROWS_PER_WARP - 1) / FP4_A_ROWS_PER_WARP;
  constexpr int32_t FP4_A_LOAD_WARPS = WARP_NUM;
  constexpr int32_t FP4_A_GROUPS_PER_WARP =
      (FP4_A_LOAD_GROUPS + FP4_A_LOAD_WARPS - 1) / FP4_A_LOAD_WARPS;
  constexpr int32_t LEGACY_G2S_LDA_COUNT = BLOCK_M / G2R_A_ROWS;
  constexpr int32_t FP4_G2S_LDA_COUNT =
      FP4_A_GROUPS_PER_WARP * FP4_A_LD_ROUNDS_PER_GROUP;
  constexpr int32_t G2S_LDA_COUNT =
      FP4_A_LOAD_BY_WARP ? FP4_G2S_LDA_COUNT : LEGACY_G2S_LDA_COUNT;
  static_assert(!FP4_A_LOAD_BY_WARP ||
                    FP4_A_ROWS_PER_WARP % FP4_A_ROWS_PER_LD_ROUND == 0,
                "FP4 A load rows must be divisible by per-round rows");

  int32_t g2r_a_col_id = threadIdx.x % G2R_A_COLS;
  int32_t g2r_a_row_id = threadIdx.x / G2R_A_COLS;

  constexpr int32_t ELEMENTS_PER_LD_B = 128 / 4;
  constexpr int32_t G2R_B_COLS = COMPUTE_K / ELEMENTS_PER_LD_B;
  constexpr int32_t G2R_B_THREAD_SIZE = BLOCK_SIZE;
  constexpr int32_t G2R_B_ROWS = G2R_B_THREAD_SIZE / G2R_B_COLS;
  static_assert(G2R_B_ROWS == BLOCK_N, "unexpected B load rows");
  int32_t g2r_b_tid = threadIdx.x;
  int32_t g2r_b_col_id = g2r_b_tid % G2R_B_COLS;
  int32_t g2r_b_row_id = g2r_b_tid / G2R_B_COLS;
  constexpr int32_t ELEMENTS_PER_32BIT = 32 / 4;
  constexpr int32_t INT32S_PER_128BIT = 128 / 32;
  constexpr int32_t R2S_B_COLS = COMPUTE_K / ELEMENTS_PER_LD_A;

  vec8_Element<Element> a_reg[WARP_M / MFMA_M][COMPUTE_K / 32];

  if (threadIdx.x < FP4_LUT_FLOATS) {
    constexpr float fp4_base[FP4_LUT_FLOATS] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    fp4_lut[threadIdx.x] = fp4_base[threadIdx.x];
  }
  __syncthreads();

  for (int it_out = 0; it_out < size_k; it_out += BLOCK_K * PACK_SCALES) {
    uint8_t scale_u8_reg[PACK_SCALES];
    #pragma unroll
    for (int i = 0; i < PACK_SCALES; ++i) {
      const int scale_col = it_out / GROUP_K + i * (BLOCK_K / GROUP_K) +
                            warp_k_id * (COMPUTE_K / GROUP_K) + g2r_b_col_id;
      const int n_global = n_base + g2r_b_row_id;
      if (n_global < size_n && scale_col < size_k / GROUP_K) {
        scale_u8_reg[i] = weight_scale_u8_ptr[
            static_cast<uint64_t>(g2r_b_row_id) * (size_k / GROUP_K) + scale_col];
      } else {
        scale_u8_reg[i] = 0;
      }
    }

    #pragma unroll
    for (int it_in = 0; it_in < PACK_SCALES * BLOCK_K; it_in += BLOCK_K) {
      const int k_start = it_out + it_in + warp_k_id * COMPUTE_K;
      vec8_Element<Element> a_val[G2S_LDA_COUNT];
      int32_t a_lds_row[G2S_LDA_COUNT] = {0};

      #pragma unroll
      for (int i = 0; i < G2S_LDA_COUNT; ++i) {
        if constexpr (FP4_A_LOAD_BY_WARP) {
          constexpr int32_t A_ROWS_PER_WARP = 16;
          constexpr int32_t A_ROWS_PER_ROUND = 64 / G2R_A_COLS;
          constexpr int32_t A_ROUNDS_PER_GROUP = A_ROWS_PER_WARP / A_ROWS_PER_ROUND;
          constexpr int32_t A_LOAD_GROUPS =
              (BLOCK_M + A_ROWS_PER_WARP - 1) / A_ROWS_PER_WARP;
          constexpr int32_t A_LOAD_WARPS = WARP_NUM;
          constexpr int32_t A_GROUPS_PER_WARP =
              (A_LOAD_GROUPS + A_LOAD_WARPS - 1) / A_LOAD_WARPS;
          static_assert(G2S_LDA_COUNT == A_GROUPS_PER_WARP * A_ROUNDS_PER_GROUP,
                        "unexpected FP4 A load count");
          const int32_t group_iter = i / A_ROUNDS_PER_GROUP;
          const int32_t round_iter = i % A_ROUNDS_PER_GROUP;
          const int32_t a_group = warp_id + group_iter * A_LOAD_WARPS;
          const int32_t basic_row_a =
              a_group * A_ROWS_PER_WARP + round_iter * A_ROWS_PER_ROUND +
              (lane_id / G2R_A_COLS);
          const int32_t basic_col_a = g2r_a_col_id;
          const int32_t global_m = m_base + basic_row_a;
          a_lds_row[i] = basic_row_a;
          if (a_group < A_LOAD_GROUPS && basic_row_a < BLOCK_M &&
              global_m < actual_m && k_start + basic_col_a * ELEMENTS_PER_LD_A +
                                      ELEMENTS_PER_LD_A <= size_k) {
            a_val[i] = *reinterpret_cast<const vec8_Element<Element>*>(
                &input_ptr[static_cast<uint64_t>(global_m) * size_k + k_start +
                           basic_col_a * ELEMENTS_PER_LD_A]);
          } else {
            a_val[i] = {};
          }
        } else {
          const int32_t basic_row_a = g2r_a_row_id + i * G2R_A_ROWS;
          const int32_t basic_col_a = g2r_a_col_id;
          const int32_t global_m = m_base + basic_row_a;
          a_lds_row[i] = basic_row_a;
          if (basic_row_a < BLOCK_M && global_m < actual_m &&
              k_start + basic_col_a * ELEMENTS_PER_LD_A + ELEMENTS_PER_LD_A <= size_k) {
            a_val[i] = *reinterpret_cast<const vec8_Element<Element>*>(
                &input_ptr[static_cast<uint64_t>(global_m) * size_k + k_start +
                           basic_col_a * ELEMENTS_PER_LD_A]);
          } else {
            a_val[i] = {};
          }
        }
      }

      __builtin_hcu_sched_barrier(0);

      const int global_k_b = k_start + g2r_b_col_id * ELEMENTS_PER_LD_B;
      union_vec_opt_w4a16<uint32_t, INT32S_PER_128BIT> b_val;
      const int n_global_b = n_base + g2r_b_row_id;
      if (n_global_b < size_n && global_k_b + ELEMENTS_PER_LD_B <= size_k) {
        const uint64_t b_offset =
            static_cast<uint64_t>(g2r_b_row_id) * (size_k / ELEMENTS_PER_32BIT) +
            global_k_b / ELEMENTS_PER_32BIT;
        b_val.int4_array[0] =
            *reinterpret_cast<const vec<uint32_t, INT32S_PER_128BIT>*>(&weight_ptr[b_offset]);
      } else {
        b_val.int4_array[0] = {};
      }

      __builtin_hcu_sched_barrier(0);

      #pragma unroll
      for (int i = 0; i < G2S_LDA_COUNT; ++i) {
        const int32_t basic_row_a = a_lds_row[i];
        const int32_t basic_col_a = g2r_a_col_id;
        if (!FP4_A_LOAD_BY_WARP || basic_row_a < BLOCK_M) {
          const int swizzled_col_a = (basic_row_a + basic_col_a) % G2R_A_COLS;
          if constexpr (!DEQUANT_16BIT) {
            *reinterpret_cast<vec8_Element<Element>*>(
                &lds_A[basic_row_a * COMPUTE_K +
                       swizzled_col_a * ELEMENTS_PER_LD_A]) = a_val[i];
          } else {
            vec8_Element<Element> tmp;
            tmp[0] = a_val[i][0];
            tmp[1] = a_val[i][4];
            tmp[2] = a_val[i][1];
            tmp[3] = a_val[i][5];
            tmp[4] = a_val[i][2];
            tmp[5] = a_val[i][6];
            tmp[6] = a_val[i][3];
            tmp[7] = a_val[i][7];
            *reinterpret_cast<vec8_Element<Element>*>(
                &lds_A[basic_row_a * COMPUTE_K +
                       swizzled_col_a * ELEMENTS_PER_LD_A]) = tmp;
          }
        }
      }

      __syncthreads();

      #pragma unroll
      for (int k_step = 0; k_step < COMPUTE_K; k_step += 32) {
        #pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; ++m_tile) {
          const int m_idx = warp_m_id * WARP_M + m_tile * MFMA_M + row_id;
          const int k_idx = k_step / 8 + col_id;
          const int swizzled_col_a = (m_idx + k_idx) % G2R_A_COLS;
          a_reg[m_tile][k_step / 32] = *reinterpret_cast<vec8_Element<Element>*>(
              &lds_A[m_idx * COMPUTE_K + swizzled_col_a * ELEMENTS_PER_LD_A]);
        }
      }

      __syncthreads();

      const int32_t scale_iter = it_in / BLOCK_K;
      reg_bf16_fp16<scalar_t2> deq_res[INT32S_PER_128BIT];
      #pragma unroll
      for (int i = 0; i < INT32S_PER_128BIT; ++i) {
        dequant_fp4_e2m1_e8m0_lut<Element, scalar_t2>(
            b_val.uint_array[i], scale_u8_reg[scale_iter], fp4_lut, deq_res[i].bf162);
        const vec8_Element<Element>* b_dequant =
            reinterpret_cast<const vec8_Element<Element>*>(deq_res[i].bf162);
        vec8_Element<Element> b_plain;
        b_plain[0] = (*b_dequant)[0];
        b_plain[1] = (*b_dequant)[2];
        b_plain[2] = (*b_dequant)[4];
        b_plain[3] = (*b_dequant)[6];
        b_plain[4] = (*b_dequant)[1];
        b_plain[5] = (*b_dequant)[3];
        b_plain[6] = (*b_dequant)[5];
        b_plain[7] = (*b_dequant)[7];
        const int c = g2r_b_col_id * (ELEMENTS_PER_LD_B / ELEMENTS_PER_32BIT) + i;
        const int r = g2r_b_row_id;
        const int swizzled_col_b = (r + c) % R2S_B_COLS;
        *reinterpret_cast<vec8_Element<Element>*>(
            &lds_B[r * COMPUTE_K + swizzled_col_b * ELEMENTS_PER_32BIT]) = b_plain;
      }

      __syncthreads();

      #pragma unroll
      for (int k_step = 0; k_step < COMPUTE_K; k_step += 32) {
        vec8_Element<Element> b_reg[WARP_N / MFMA_N];
        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; ++n_tile) {
          const int n_idx = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
          const int k_idx = k_step / 8 + col_id;
          const int swizzled_col_b = (n_idx + k_idx) % R2S_B_COLS;
          b_reg[n_tile] = *reinterpret_cast<vec8_Element<Element>*>(
              &lds_B[n_idx * COMPUTE_K + swizzled_col_b * ELEMENTS_PER_LD_A]);
        }

        #pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; ++m_tile) {
          #pragma unroll
          for (int n_tile = 0; n_tile < WARP_N / MFMA_N; ++n_tile) {
            vec4_Element<Element>* a_vec4 =
                reinterpret_cast<vec4_Element<Element>*>(&a_reg[m_tile][k_step / 32]);
            vec4_Element<Element>* b_vec4 =
                reinterpret_cast<vec4_Element<Element>*>(&b_reg[n_tile]);
            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] =
                wfp4a16_mmac<Element, INTERLEAVE>(
                    a_vec4[0], b_vec4[0],
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] =
                wfp4a16_mmac<Element, INTERLEAVE>(
                    a_vec4[1], b_vec4[1],
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
          }
        }
      }

      __syncthreads();
    }
  }
}

template<
    typename Element,
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M,
    uint16_t BLOCK_SIZE_N,
    uint16_t BLOCK_SIZE_K,
    uint16_t WARP_M,
    uint16_t WARP_N,
    uint16_t WARP_K,
    uint16_t GROUP_N,
    uint16_t GROUP_K,
    int STAGES,
    bool mul_topk_weight,
    bool INTERLEAVE,
    bool DEQUANT_16BIT,
    int32_t PACK_SCALES>
__global__ void __launch_bounds__(512, 1) GEMM_NT_1STAGE_WFP4A16_MASKED_PREFILL(
    const Element* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const uint8_t* __restrict__ weight_scale_u8,
    const int32_t* __restrict__ actual_tokens,
    Element* __restrict__ output,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t experts) {
  static_assert(STAGES == 1, "WFP4A16 prefill/gemm1 migration uses one stage");
  static_assert(!mul_topk_weight, "gemm1 does not multiply topk weights");
  static_assert(GROUP_N == 1, "AITER WFP4A16 prefill/gemm1 uses GROUP_N=1");
  static_assert(GROUP_K == 32, "AITER WFP4A16 prefill/gemm1 uses GROUP_K=32");
  static_assert(BLOCK_SIZE_M == 128 && BLOCK_SIZE_N == 128 && BLOCK_SIZE_K == 64,
                "AITER WFP4A16 prefill/gemm1 mode 514 is BM128 BN128 BK64");
  static_assert(WARP_M == 64 && WARP_N == 64 && WARP_K == 64,
                "AITER WFP4A16 prefill/gemm1 mode 514 is WM64 WN64 WK64");

  const int expert = blockIdx.z;
  const int bidy = blockIdx.x;
  const int bidx = blockIdx.y;
  if (expert >= static_cast<int>(experts)) {
    return;
  }

  const int m_base = bidx * BLOCK_SIZE_M;
  const int n_base = bidy * BLOCK_SIZE_N;
  const int actual_m = actual_tokens[expert];
  if (m_base >= actual_m) {
    return;
  }

  using Dtype = ScalarType<Element>;
  using scalar_t2 = typename ScalarType<Element>::scalar_t2;

  const uint64_t expert_offset_elements = static_cast<uint64_t>(size_n) * size_k * expert;
  const uint64_t block_n_offset_elements = static_cast<uint64_t>(bidy) * BLOCK_SIZE_N * size_k;
  const uint64_t total_elements_offset = expert_offset_elements + block_n_offset_elements;
  const uint64_t qweight_offset = total_elements_offset / 8;
  const uint64_t weight_scale_offset = total_elements_offset / GROUP_K;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;

  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_m_id = warp_id / (warp_n_num * warp_k_num);
  int warp_nk_id = warp_id % (warp_n_num * warp_k_num);
  int warp_k_id = warp_nk_id % warp_k_num;
  int warp_n_id = warp_nk_id / warp_k_num;

  extern __shared__ uint16_t output_lds_tmp[];
  Element* output_lds = reinterpret_cast<Element*>(output_lds_tmp);

  const Element* g_input = input + static_cast<uint64_t>(expert) * size_m * size_k;
  Element* g_output = output + (static_cast<uint64_t>(expert) * size_m * size_n) +
                      bidy * BLOCK_SIZE_N;
  const uint32_t* g_qweight = qweight + qweight_offset;
  const uint8_t* g_weight_scale_u8 = weight_scale_u8 + weight_scale_offset;

  floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0};

  gemm_nt_wfp4a16_prefill_stage1_b128_k64_masked<
      WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N,
      WARP_K, GROUP_K, Element, scalar_t2, INTERLEAVE, DEQUANT_16BIT,
      PACK_SCALES, true>(
      g_input, g_qweight, g_weight_scale_u8, actual_m, size_m, size_n, C_reg,
      warp_id, size_k, n_base, m_base, warp_m_id);

  if constexpr (INTERLEAVE) {
    constexpr int32_t WARP_M_TILES = WARP_M / mfma_m;
    constexpr int32_t HALF_TILES_M = WARP_M_TILES / 2;
    constexpr int32_t WARP_M_HALF = HALF_TILES_M * mfma_m;
    constexpr int32_t HALF_M = BLOCK_SIZE_M / 2;

    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
      int32_t start_tile_m = phase * HALF_TILES_M;
      int32_t end_tile_m = (phase + 1) * HALF_TILES_M;
      if (warp_k_id == 0 || warp_k_num == 1) {
        for (int min_tile_m = start_tile_m; min_tile_m < end_tile_m; ++min_tile_m) {
          int32_t min_tile_m_local = min_tile_m - start_tile_m;
          int32_t r_lds = warp_m_id * WARP_M_HALF + min_tile_m_local * mfma_m +
                          (lane_id % 16);
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; ++min_tile_n) {
            constexpr int32_t SWZ_COLS = BLOCK_SIZE_N / 4;
            constexpr int32_t SWZ_ELEMENT = 2;
            int32_t c = (warp_n_id * WARP_N + min_tile_n * mfma_n) / 4 +
                        (lane_id / 16);
            int32_t c_swz = (r_lds * SWZ_ELEMENT + c) % SWZ_COLS;
            #pragma unroll
            for (int reg_id = 0; reg_id < 4; ++reg_id) {
              int32_t index = r_lds * BLOCK_SIZE_N + c_swz * 4 + reg_id;
              output_lds[index] = Dtype::float2num(
                  C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
            }
          }
        }
      }
      __syncthreads();

      constexpr int32_t N_thread = BLOCK_SIZE_N / 8;
      int32_t m_idx_local = threadIdx.x / N_thread;
      int32_t n_idx = threadIdx.x % N_thread;
      for (; m_idx_local < HALF_M; m_idx_local += (WARP_NUM * 64) / N_thread) {
        int32_t warp_idx = m_idx_local / WARP_M_HALF;
        int32_t offset = m_idx_local % WARP_M_HALF;
        int32_t m_idx_global = warp_idx * WARP_M + phase * WARP_M_HALF + offset;
        int32_t global_m = m_base + m_idx_global;
        int32_t global_n = n_idx * 8;
        int32_t n_swz = (m_idx_local + n_idx) % N_thread;
        if (global_m < actual_m && n_base + global_n < static_cast<int>(size_n)) {
          vec8_Element<Element> tmp = *reinterpret_cast<vec8_Element<Element>*>(
              &output_lds[m_idx_local * BLOCK_SIZE_N + n_swz * 8]);
          const int valid_n = wfp4a16_min_int(8, static_cast<int>(size_n) - (n_base + global_n));
          store_vec8_with_n_guard<Element>(
              &g_output[static_cast<uint64_t>(global_m) * size_n + global_n],
              tmp,
              valid_n);
        }
      }
      __syncthreads();
    }
  } else {
    constexpr int32_t WARP_M_TILES = WARP_M / mfma_m;
    constexpr int32_t HALF_TILES_M = WARP_M_TILES / 2;
    constexpr int32_t WARP_M_HALF = HALF_TILES_M * mfma_m;
    constexpr int32_t HALF_M = BLOCK_SIZE_M / 2;
    constexpr int32_t SWZ_ATOM = 2 * 4;

    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
      int32_t start_tile_m = phase * HALF_TILES_M;
      int32_t end_tile_m = (phase + 1) * HALF_TILES_M;
      if (warp_k_id == 0 || warp_k_num == 1) {
        for (int min_tile_m = start_tile_m; min_tile_m < end_tile_m; ++min_tile_m) {
          int32_t min_tile_m_local = min_tile_m - start_tile_m;
          for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; ++min_tile_n) {
            #pragma unroll
            for (int32_t r_o = 0; r_o < 2; ++r_o) {
              #pragma unroll
              for (int32_t r_i = 0; r_i < 2; ++r_i) {
                int32_t row_lds = warp_m_id * WARP_M_HALF +
                                  min_tile_m_local * mfma_m + (lane_id & 15);
                int32_t col = warp_n_id * WARP_N + min_tile_n * mfma_n + r_o * 8 +
                              (lane_id / 16) * 2 + r_i;
                int32_t col_swz = (row_lds * SWZ_ATOM + col) % BLOCK_SIZE_N;
                int32_t index = row_lds * BLOCK_SIZE_N + col_swz;
                int32_t reg_id = r_o * 2 + r_i;
                output_lds[index] = Dtype::float2num(
                    C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id]);
              }
            }
          }
        }
      }

      __syncthreads();

      constexpr int N_thread = BLOCK_SIZE_N / 8;
      int32_t m_idx_local = threadIdx.x / N_thread;
      int32_t n_idx = threadIdx.x % N_thread;
      for (; m_idx_local < HALF_M; m_idx_local += (WARP_NUM * 64) / N_thread) {
        int32_t warp_idx = m_idx_local / WARP_M_HALF;
        int32_t offset = m_idx_local % WARP_M_HALF;
        int32_t m_idx_global = warp_idx * WARP_M + phase * WARP_M_HALF + offset;
        int32_t global_m = m_base + m_idx_global;
        int32_t global_n = n_idx * 8;
        if (global_m < actual_m && n_base + global_n < static_cast<int>(size_n)) {
          int32_t lds_offset =
              m_idx_local * BLOCK_SIZE_N +
              (m_idx_local * SWZ_ATOM + n_idx * 8) % BLOCK_SIZE_N;
          vec8_Element<Element> tmp =
              *reinterpret_cast<vec8_Element<Element>*>(&output_lds[lds_offset]);
          vec8_Element<Element> tmp1;
          tmp1[0] = tmp[0];
          tmp1[4] = tmp[1];
          tmp1[1] = tmp[2];
          tmp1[5] = tmp[3];
          tmp1[2] = tmp[4];
          tmp1[6] = tmp[5];
          tmp1[3] = tmp[6];
          tmp1[7] = tmp[7];
          const int valid_n = wfp4a16_min_int(8, static_cast<int>(size_n) - (n_base + global_n));
          store_vec8_with_n_guard<Element>(
              &g_output[static_cast<uint64_t>(global_m) * size_n + global_n],
              tmp1,
              valid_n);
        }
      }

      if (phase == 0) {
        __syncthreads();
      }
    }
  }
}

template <typename T, int32_t PACK_SCALES, bool INTERLEAVE, bool DEQUANT_16BIT>
void launch_wfp4a16_masked_prefill_gemm1_kernel(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  constexpr int BLOCK_SIZE_M = 128;
  constexpr int BLOCK_SIZE_N = 128;
  constexpr int BLOCK_SIZE_K = 64;
  constexpr int WARP_M = 64;
  constexpr int WARP_N = 64;
  constexpr int WARP_K = 64;
  constexpr int WARP_NUM =
      (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N) *
      (BLOCK_SIZE_K / WARP_K);
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 32;
  constexpr int STAGES = 1;
  constexpr bool MUL_TOPK_WEIGHT = false;

  const int size_m = matrix_a.size(1);
  const int size_n = matrix_c.size(2);
  const int size_k = matrix_a.size(2);
  const dim3 block(WARP_NUM * 64);
  const dim3 grid(
      (size_n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
      (static_cast<int>(max_tokens) + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
      static_cast<unsigned int>(experts));
  const int shared_mem_size = 16 * 1024 + 64;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  GEMM_NT_1STAGE_WFP4A16_MASKED_PREFILL<
      T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N,
      WARP_K, GROUP_N, GROUP_K, STAGES, MUL_TOPK_WEIGHT, INTERLEAVE,
      DEQUANT_16BIT, PACK_SCALES><<<grid, block, shared_mem_size, stream>>>(
      matrix_a_ptr,
      reinterpret_cast<const uint32_t*>(matrix_b.data_ptr<uint8_t>()),
      matrix_b_scale.data_ptr<uint8_t>(),
      actual_tokens.data_ptr<int32_t>(),
      matrix_c_ptr,
      size_m,
      size_n,
      size_k,
      static_cast<uint32_t>(experts));
}

template <typename T, int32_t PACK_SCALES>
void launch_wfp4a16_masked_prefill_gemm1_arch(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
#if defined(__gfx938__)
  launch_wfp4a16_masked_prefill_gemm1_kernel<T, PACK_SCALES, true, true>(
      matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
      experts, matrix_a_ptr, matrix_c_ptr);
#else
  launch_wfp4a16_masked_prefill_gemm1_kernel<T, PACK_SCALES, false, false>(
      matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
      experts, matrix_a_ptr, matrix_c_ptr);
#endif
}

template <typename T, int32_t PACK_SCALES, bool INTERLEAVE, bool DEQUANT_16BIT>
void launch_wfp4a16_masked_prefill_gemm1_explicit(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  launch_wfp4a16_masked_prefill_gemm1_kernel<
      T, PACK_SCALES, INTERLEAVE, DEQUANT_16BIT>(
      matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
      experts, matrix_a_ptr, matrix_c_ptr);
}

template <typename T>
void launch_wfp4a16_masked_optimized(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  TORCH_CHECK(wfp4a16_optimized_supported(matrix_a.size(2)),
              "AITER WFP4A16 prefill/gemm1 mode 514 requires K divisible by 64");
  const int64_t size_k = matrix_a.size(2);
  if (size_k == 192) {
    launch_wfp4a16_masked_prefill_gemm1_arch<T, 3>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else if (size_k == 256) {
    launch_wfp4a16_masked_prefill_gemm1_arch<T, 4>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else if (size_k == 384) {
    launch_wfp4a16_masked_prefill_gemm1_arch<T, 6>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else if (size_k == 768) {
    launch_wfp4a16_masked_prefill_gemm1_arch<T, 6>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else {
    launch_wfp4a16_masked_prefill_gemm1_arch<T, 8>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  }
}


template <typename T, bool INTERLEAVE, bool DEQUANT_16BIT>
void launch_wfp4a16_masked_optimized_explicit(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    int64_t max_tokens,
    int64_t experts,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  TORCH_CHECK(wfp4a16_optimized_supported(matrix_a.size(2)),
              "AITER WFP4A16 prefill/gemm1 mode 514 requires K divisible by 64");
  const int64_t size_k = matrix_a.size(2);
  if (size_k == 192) {
    launch_wfp4a16_masked_prefill_gemm1_explicit<T, 3, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else if (size_k == 256) {
    launch_wfp4a16_masked_prefill_gemm1_explicit<T, 4, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else if (size_k == 384) {
    launch_wfp4a16_masked_prefill_gemm1_explicit<T, 6, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else if (size_k == 768) {
    launch_wfp4a16_masked_prefill_gemm1_explicit<T, 6, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  } else {
    launch_wfp4a16_masked_prefill_gemm1_explicit<T, 8, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens,
        experts, matrix_a_ptr, matrix_c_ptr);
  }
}

}  // namespace deepgemm_wfp4a16_masked
