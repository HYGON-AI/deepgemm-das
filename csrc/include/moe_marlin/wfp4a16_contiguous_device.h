#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>

#include "moe_marlin/wfp4a16_masked_device.h"

namespace deepgemm_wfp4a16_contiguous {

struct Wfp4a16ContiguousConfig {
  int mode;
  int block_m;
  int block_n;
  int block_k;
};

inline Wfp4a16ContiguousConfig select_wfp4a16_contiguous_config(
    int64_t size_m,
    int64_t size_n,
    int64_t size_k) {
  (void)size_m;
  (void)size_n;
  (void)size_k;
  return {514, 128, 128, 64};
}

inline bool wfp4a16_contiguous_supported(int64_t size_k) {
  return size_k % 64 == 0;
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
__global__ void __launch_bounds__(512, 1) GEMM_NT_1STAGE_WFP4A16_CONTIGUOUS_PREFILL(
    const Element* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const uint8_t* __restrict__ weight_scale_u8,
    const int32_t* __restrict__ m_indices,
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

  const int bidy = blockIdx.x;
  const int bidx = blockIdx.y;
  const int m_base = bidx * BLOCK_SIZE_M;
  const int n_base = bidy * BLOCK_SIZE_N;
  if (m_base >= static_cast<int>(size_m)) {
    return;
  }

  const int expert = m_indices[m_base];
  if (expert < 0 || expert >= static_cast<int>(experts)) {
    return;
  }
  const int expert_s = __builtin_hcu_readfirstlane(expert);

  using Dtype = ScalarType<Element>;
  using scalar_t2 = typename ScalarType<Element>::scalar_t2;

  const uint64_t expert_offset_elements = static_cast<uint64_t>(size_n) * size_k * expert_s;
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

  const Element* g_input = input;
  Element* g_output = output + bidy * BLOCK_SIZE_N;
  const uint32_t* g_qweight = qweight + qweight_offset;
  const uint8_t* g_weight_scale_u8 = weight_scale_u8 + weight_scale_offset;

  floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0};

  deepgemm_wfp4a16_masked::gemm_nt_wfp4a16_prefill_stage1_b128_k64_masked<
      WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N,
      WARP_K, GROUP_K, Element, scalar_t2, INTERLEAVE, DEQUANT_16BIT,
      PACK_SCALES, true>(
      g_input, g_qweight, g_weight_scale_u8, static_cast<int>(size_m), size_m,
      size_n, C_reg, warp_id, size_k, n_base, m_base, warp_m_id);

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
        if (global_m < static_cast<int>(size_m) && m_indices[global_m] >= 0 &&
            n_base + global_n < static_cast<int>(size_n)) {
          vec8_Element<Element> tmp = *reinterpret_cast<vec8_Element<Element>*>(
              &output_lds[m_idx_local * BLOCK_SIZE_N + n_swz * 8]);
          const int valid_n = deepgemm_wfp4a16_masked::wfp4a16_min_int(
              8, static_cast<int>(size_n) - (n_base + global_n));
          deepgemm_wfp4a16_masked::store_vec8_with_n_guard<Element>(
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
        if (global_m < static_cast<int>(size_m) && m_indices[global_m] >= 0 &&
            n_base + global_n < static_cast<int>(size_n)) {
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
          const int valid_n = deepgemm_wfp4a16_masked::wfp4a16_min_int(
              8, static_cast<int>(size_n) - (n_base + global_n));
          deepgemm_wfp4a16_masked::store_vec8_with_n_guard<Element>(
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
void launch_wfp4a16_contiguous_prefill_gemm1_kernel(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c,
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

  const int size_m = matrix_a.size(0);
  const int size_n = matrix_c.size(1);
  const int size_k = matrix_a.size(1);
  const int experts = matrix_b.size(0);
  const dim3 block(WARP_NUM * 64);
  const dim3 grid(
      (size_n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
      (size_m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
      1);
  const int shared_mem_size = 16 * 1024 + 64;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  GEMM_NT_1STAGE_WFP4A16_CONTIGUOUS_PREFILL<
      T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N,
      WARP_K, GROUP_N, GROUP_K, STAGES, MUL_TOPK_WEIGHT, INTERLEAVE,
      DEQUANT_16BIT, PACK_SCALES><<<grid, block, shared_mem_size, stream>>>(
      matrix_a_ptr,
      reinterpret_cast<const uint32_t*>(matrix_b.data_ptr<uint8_t>()),
      matrix_b_scale.data_ptr<uint8_t>(),
      m_indices.data_ptr<int32_t>(),
      matrix_c_ptr,
      size_m,
      size_n,
      size_k,
      static_cast<uint32_t>(experts));
}

template <typename T, int32_t PACK_SCALES>
void launch_wfp4a16_contiguous_prefill_gemm1_arch(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
#if defined(__gfx938__)
  launch_wfp4a16_contiguous_prefill_gemm1_kernel<T, PACK_SCALES, true, true>(
      matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
      matrix_c_ptr);
#else
  launch_wfp4a16_contiguous_prefill_gemm1_kernel<T, PACK_SCALES, false, false>(
      matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
      matrix_c_ptr);
#endif
}

template <typename T, int32_t PACK_SCALES, bool INTERLEAVE, bool DEQUANT_16BIT>
void launch_wfp4a16_contiguous_prefill_gemm1_explicit(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  launch_wfp4a16_contiguous_prefill_gemm1_kernel<
      T, PACK_SCALES, INTERLEAVE, DEQUANT_16BIT>(
      matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
      matrix_c_ptr);
}

template <typename T>
void launch_wfp4a16_contiguous_optimized(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  TORCH_CHECK(wfp4a16_contiguous_supported(matrix_a.size(1)),
              "AITER WFP4A16 prefill/gemm1 mode 514 requires K divisible by 64");
  const int64_t size_k = matrix_a.size(1);
  if (size_k == 192) {
    launch_wfp4a16_contiguous_prefill_gemm1_arch<T, 3>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else if (size_k == 256) {
    launch_wfp4a16_contiguous_prefill_gemm1_arch<T, 4>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else if (size_k == 384) {
    launch_wfp4a16_contiguous_prefill_gemm1_arch<T, 6>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else if (size_k == 768) {
    launch_wfp4a16_contiguous_prefill_gemm1_arch<T, 6>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else {
    launch_wfp4a16_contiguous_prefill_gemm1_arch<T, 8>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  }
}


template <typename T, bool INTERLEAVE, bool DEQUANT_16BIT>
void launch_wfp4a16_contiguous_optimized_explicit(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& matrix_c,
    const T* matrix_a_ptr,
    T* matrix_c_ptr) {
  TORCH_CHECK(wfp4a16_contiguous_supported(matrix_a.size(1)),
              "AITER WFP4A16 prefill/gemm1 mode 514 requires K divisible by 64");
  const int64_t size_k = matrix_a.size(1);
  if (size_k == 192) {
    launch_wfp4a16_contiguous_prefill_gemm1_explicit<T, 3, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else if (size_k == 256) {
    launch_wfp4a16_contiguous_prefill_gemm1_explicit<T, 4, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else if (size_k == 384) {
    launch_wfp4a16_contiguous_prefill_gemm1_explicit<T, 6, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else if (size_k == 768) {
    launch_wfp4a16_contiguous_prefill_gemm1_explicit<T, 6, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  } else {
    launch_wfp4a16_contiguous_prefill_gemm1_explicit<T, 8, INTERLEAVE, DEQUANT_16BIT>(
        matrix_a, matrix_b, matrix_b_scale, m_indices, matrix_c, matrix_a_ptr,
        matrix_c_ptr);
  }
}

}  // namespace deepgemm_wfp4a16_contiguous
