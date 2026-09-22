#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include "low_latency_fp8_masked_utils.h"

#include "m_grouped_gemm_nt_masked.h"

#include <cstdint>

// template <typename T>
// __forceinline__ __device__ void buffer_load_reg_dwordx4___(const T *ptr, vec<int, 4> &rsrc, const int vindex, int offset)
// { // const int offset

//   intx4 global_ptr;
//   *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
//   global_ptr[1] += 0x40800000;
//   global_ptr[2] = 0x80000000;
//   global_ptr[3] = 0x00020000;

//   asm volatile(
//       "s_nop 4 \n\t"
//       "buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n\t"
//       : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));

//   return;
// }

/*
namespace DEEP_GEMM -> FP8_GROUP_GEMM -> MASKED/CONTIGUOUS
*/
namespace deepgemm
{
      using bhalf_t = __hip_bfloat16;
      using namespace MASKED_UTILS;
      using uintx4 = vec<uint32_t, 4>;

      __device__ __forceinline__ uintx4 make_no_swizzle_buffer_resource(
          const void *__restrict__ ptr,
          uint32_t range_bytes)
      {
        uintx4 global_ptr;
        *reinterpret_cast<uint64_t *>(&global_ptr) = reinterpret_cast<uint64_t>(ptr);
        global_ptr[2] = range_bytes;
        global_ptr[3] = 0x00020000;
        return global_ptr;
      }

      __device__ __forceinline__ uintx4 force_buffer_resource_sgpr(uintx4 resource)
      {
        uintx4 sgpr_resource;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
          sgpr_resource[i] = __builtin_hcu_readfirstlane(resource[i]);
        }
        return sgpr_resource;
      }

      __device__ __forceinline__ vec<int32_t, 4> buffer_load_dwordx4_to_vgpr_i32(
          const uintx4 &global_ptr,
          int byte_offset)
      {
        uintx4 data_words;
        asm volatile("buffer_load_dwordx4 %0,%1,%2,0 offen offset:0\n"
                     : "=v"(data_words)
                     : "v"(byte_offset), "s"(global_ptr)
                     : "memory");
        return *reinterpret_cast<vec<int32_t, 4> *>(&data_words);
      }

      __device__ __forceinline__ void wait_vmcnt_3()
      {
        asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
      }

      __device__ __forceinline__ void wait_vmcnt_4()
      {
        asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
      }

      __device__ __forceinline__ void wait_vmcnt_5()
      {
        asm volatile("s_waitcnt vmcnt(5)" ::: "memory");
      }

      __device__ __forceinline__ void wait_vmcnt_6()
      {
        asm volatile("s_waitcnt vmcnt(6)" ::: "memory");
      }

      __device__ __forceinline__ void wait_vmcnt_7()
      {
        asm volatile("s_waitcnt vmcnt(7)" ::: "memory");
      }

      __device__ __forceinline__ void wait_vmcnt_8()
      {
        asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
      }


      __device__ __forceinline__ void wait_vmcnt_0()
      {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }

      template <int M, int N, int K, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int NUM_WARPS, int K_SCALE_RANGE, int N_SCALE_RANGE, int CUs>
      __global__ __launch_bounds__(256, 1) void kGroupGemm_aggressive(const char *__restrict__ matrixA,
                                                                      const char *__restrict__ matrixB,
                                                                      const float *__restrict__ matrixA_scale,
                                                                      const float *__restrict__ matrixB_scale,
                                                                      const int32_t *__restrict__ actual_M,
                                                                      const int M_,
                                                                      const int num_expert,
                                                                      bhalf_t *__restrict__ MatrixC)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        constexpr int32_t N_TILES = N / BLOCK_N;
        constexpr int32_t K_ITERS = K / BLOCK_K;
        constexpr int32_t M_TILES = (M / BLOCK_M);
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t stage = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];

        constexpr int32_t EXPERT_TILES = M_TILES * N_TILES;
        const int32_t total_tiles = EXPERT_TILES * num_expert;
        const int32_t tile_iter_count = ceil_div(total_tiles, CUs);
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = true;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < num_expert)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int32_t it_tile = 0; it_tile < tile_iter_count; it_tile++)
        {
          int32_t cur_tile = it_tile * CUs + tile_id;
          if (cur_tile >= total_tiles)
          {
            return;
          }

          int32_t e = cur_tile / EXPERT_TILES;

          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          const float *cur_matrix_as = matrixA_scale + e * M;
          const float *cur_matrix_bs = matrixB_scale + e * N;
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);
          v_type rA[stage];
          v_type rB[stage][N_REPEAT];
          vec<int32_t, 4> rC[N_REPEAT] = {0};
          int tile_m = cur_tile / N_TILES;
          int tile_n = cur_tile % N_TILES;
          // offset
          const int32_t warp_offset_M = 0;
          const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
          // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
          const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
          const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
          // base ptr
          const char *cur_M_base = cur_matrixA + tile_m * BLOCK_M * K;
          const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
          const char *cur_M_warp_base = cur_M_base;
          const char *cur_N_warp_base = cur_N_base + warp_offset_N;
          bhalf_t *cur_matrixC_warp_base = cur_matrixC + tile_m * BLOCK_M * N + tile_n * BLOCK_N + warp_idx * MMA_N;
          int32_t sa_offset = e * M + tile_m * BLOCK_M + ld_row;
          float input_scale = matrixA_scale[sa_offset];
          // int32_t sb_offset = e * N + tile_n * BLOCK_N + ld_row;
          vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
          for (int i = 0; i < N_REPEAT; ++i)
          {
            const vec<float, MMA_C_REGS> *p_ws = reinterpret_cast<const vec<float, MMA_C_REGS> *>(matrixB_scale + e * N + tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
            weight_scale[i] = p_ws[0];
          }
          // loop
          for (int k = 0; k < K_ITERS; k++)
          {
            const vec<int32_t, 4> *addr_a = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % stage].int_arr[0] = addr_a[0];
            for (int r = 0; r < 4; ++r)
            {
              rA[k % stage].int_arr[0][r] = __shfl(rA[k % stage].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n)
            {
              const vec<int32_t, 4> *addr_b = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
              rB[k % stage][n].int_arr[0] = addr_b[0];
            }
            for (int n = 0; n < N_REPEAT; ++n)
            {
              rC[n] = mmac(rA[k % stage].int8_arr[0], rB[k % stage % stage][n].int8_arr[0], rC[n]);
              rC[n] = mmac(rA[k % stage].int8_arr[1], rB[k % stage % stage][n].int8_arr[1], rC[n]);
            }
          }
          // dequant

          for (int n = 0; n < N_REPEAT; n++)
          {
            vec4_bf16 st;
            for (int32_t i = 0; i < MMA_C_REGS; i++)
            {
              float r_st = float(rC[n][i]) * input_scale * weight_scale[n][i];

              bhalf_t tmp = bhalf_t(r_st);
              st[i] = reinterpret_cast<unsigned short &>(tmp);
            }
            int32_t st_lane_offset = ld_row * N + ld_col * MMA_C_REGS;
            vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
            p_st[0] = st;
          }
        }
      }

      template <class Acc_Type, int N, int K, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int NUM_WARPS, int K_SCALE_RANGE, int N_SCALE_RANGE, int CUs, int ACTIVE_M_SUBTILES, bool B_TENSOR_WISE = false>
      __global__ __launch_bounds__(256, 1) void kGroupGemm(const char *__restrict__ matrixA,
                                                           const char *__restrict__ matrixB,
                                                           const float *__restrict__ matrixA_scale,
                                                           const float *__restrict__ matrixB_scale,
                                                           const int32_t *__restrict__ actual_M,
                                                           const int M,
                                                           const int num_expert,
                                                           bhalf_t *__restrict__ MatrixC)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        constexpr int N_TILES = N / BLOCK_N;
        constexpr int K_ITERS = K / BLOCK_K;
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t PIPE_STAGE = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = false;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < num_expert)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int e = 0; e < num_expert; ++e)
        {
          // TODO: opt actual M by shfl
          // int32_t cur_tokens = actual_M[e];
          // int32_t cur_tokens = __shfl(local_tokens, e);
          int32_t cur_tokens = actual_M[e];
          int m_tiles = ceil_div(cur_tokens, BLOCK_M);
          int32_t cur_expert_tiles = m_tiles * N_TILES;
          // expert base ptr
          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          // float *cur_matrixC = cur_matrixC + e * M * N;
          // abort core dump
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          // scale base ptr
          const float *cur_matrix_as = matrixA_scale + e * M;
          float tensor_weight_scale = 1.0f;
          if constexpr (B_TENSOR_WISE)
          {
            tensor_weight_scale = matrixB_scale[e];
          }
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

          while ((tile_id >= last_expert_end) && (tile_id < last_expert_end + cur_expert_tiles))
          {

            int local_tile_id = tile_id - last_expert_end;
            int tile_m = local_tile_id / N_TILES;
            int tile_n = local_tile_id % N_TILES;
            constexpr int32_t M_SUBTILES = BLOCK_M / 16;
            static_assert(ACTIVE_M_SUBTILES >= 1 && ACTIVE_M_SUBTILES <= M_SUBTILES, "invalid ACTIVE_M_SUBTILES");
            const int32_t m_block_start = tile_m * BLOCK_M;
            int32_t valid_rows_in_tile = cur_tokens - m_block_start;
            valid_rows_in_tile = valid_rows_in_tile <= 0 ? 0 : (valid_rows_in_tile >= BLOCK_M ? BLOCK_M : valid_rows_in_tile);
            constexpr int32_t active_m_subtiles = ACTIVE_M_SUBTILES;

            v_type rA[PIPE_STAGE][M_SUBTILES];
            v_type rB[PIPE_STAGE][N_REPEAT];
            vec<Acc_Type, 4> rC[M_SUBTILES][N_REPEAT] = {0};
            float input_scale[M_SUBTILES];

            // offset
            const int32_t warp_offset_M = 0;
            const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
            // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
            const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
            const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
            // base ptr
            const char *cur_M_base = cur_matrixA + m_block_start * K;
#if defined(__gfx938__)
            const uintx4 matrixA_resource =
                force_buffer_resource_sgpr(
                    make_no_swizzle_buffer_resource(
                        cur_M_base, static_cast<uint32_t>(BLOCK_M * K + 16)));
            const uintx4 matrixB_resource =
                force_buffer_resource_sgpr(
                    make_no_swizzle_buffer_resource(
                        cur_matrixB, static_cast<uint32_t>(N * K)));
#else
            const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
            const char *cur_M_warp_base = cur_M_base;
            const char *cur_N_warp_base = cur_N_base + warp_offset_N;
#endif
            bhalf_t *cur_matrixC_warp_base = cur_matrixC + m_block_start * N + tile_n * BLOCK_N + warp_idx * MMA_N;
#pragma unroll
            for (int m_loop = 0; m_loop < M_SUBTILES; ++m_loop)
            {
              int32_t sa_offset = e * M + m_block_start + m_loop * 16 + ld_row;
              input_scale[m_loop] = matrixA_scale[sa_offset];
            }
            // int32_t sb_offset = e * N + tile_n * BLOCK_N + ld_row;
            vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
            if constexpr (!B_TENSOR_WISE)
            {
              for (int i = 0; i < N_REPEAT; ++i)
              {
                const vec<float, MMA_C_REGS> *p_ws = reinterpret_cast<const vec<float, MMA_C_REGS> *>(matrixB_scale + e * N + tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
                weight_scale[i] = p_ws[0];
              }
            }

// loop
#pragma unroll
            for (int k = 0; k < (PIPE_STAGE - 1); k++)
            {
#pragma unroll
              for (int m_loop = 0; m_loop < M_SUBTILES; ++m_loop)
              {
                if (m_loop < active_m_subtiles)
                {
#if defined(__gfx938__)
                  const int a_byte_offset = m_loop * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_;
                  rA[k % PIPE_STAGE][m_loop].int_arr[0] =
                      buffer_load_dwordx4_to_vgpr_i32(matrixA_resource, a_byte_offset);
#else
                  const vec<int32_t, 4> *addr_a_next =
                      reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + m_loop * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
                  rA[k % PIPE_STAGE][m_loop].int_arr[0] = addr_a_next[0];
#endif
                }
              }
#pragma unroll
              for (int n = 0; n < N_REPEAT; ++n)
              {
#if defined(__gfx938__)
                const int b_byte_offset =
                    tile_n * LD_ELEMENTS_PER_BLOCK + warp_offset_N +
                    k * LD_N_ITERK_STRIDE + lane_offset_N +
                    n * NUM_WARPS * LD_ELEMENTS_PER_WARP;
                rB[k % PIPE_STAGE][n].int_arr[0] =
                    buffer_load_dwordx4_to_vgpr_i32(matrixB_resource, b_byte_offset);
#else
                const vec<int32_t, 4> *addr_b_next =
                    reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                rB[k % PIPE_STAGE][n].int_arr[0] = addr_b_next[0];
#endif
              }
            }
#pragma unroll
            for (int k = (PIPE_STAGE - 1); k < K_ITERS; k++)
            {
#pragma unroll
              for (int m_loop = 0; m_loop < M_SUBTILES; ++m_loop)
              {
                if (m_loop < active_m_subtiles)
                {
#if defined(__gfx938__)
                  const int a_byte_offset = m_loop * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_;
                  rA[k % PIPE_STAGE][m_loop].int_arr[0] =
                      buffer_load_dwordx4_to_vgpr_i32(matrixA_resource, a_byte_offset);
#else
                  const vec<int32_t, 4> *addr_a_next =
                      reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + m_loop * 16 * K + k * LD_M_ITERK_STRIDE + lane_offset_M_);
                  rA[k % PIPE_STAGE][m_loop].int_arr[0] = addr_a_next[0];
#endif
                }
              }
#pragma unroll
              for (int n = 0; n < N_REPEAT; ++n)
              {
#if defined(__gfx938__)
                const int b_byte_offset =
                    tile_n * LD_ELEMENTS_PER_BLOCK + warp_offset_N +
                    k * LD_N_ITERK_STRIDE + lane_offset_N +
                    n * NUM_WARPS * LD_ELEMENTS_PER_WARP;
                rB[k % PIPE_STAGE][n].int_arr[0] =
                    buffer_load_dwordx4_to_vgpr_i32(matrixB_resource, b_byte_offset);
#else
                const vec<int32_t, 4> *addr_b_next =
                    reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                rB[k % PIPE_STAGE][n].int_arr[0] = addr_b_next[0];
#endif
              }
#if defined(__gfx938__)
              constexpr int32_t VMEM_LOADS_PER_STAGE = active_m_subtiles + N_REPEAT;
              if constexpr (VMEM_LOADS_PER_STAGE <= 3)
                wait_vmcnt_3();
              else if constexpr (VMEM_LOADS_PER_STAGE == 4)
                wait_vmcnt_4();
              else if constexpr (VMEM_LOADS_PER_STAGE == 5)
                wait_vmcnt_5();
              else if constexpr (VMEM_LOADS_PER_STAGE == 6)
                wait_vmcnt_6();
              else if constexpr (VMEM_LOADS_PER_STAGE == 7)
                wait_vmcnt_7();
              else
                wait_vmcnt_8();
#endif
              __builtin_hcu_sched_barrier(0);
#pragma unroll
              for (int m_loop = 0; m_loop < M_SUBTILES; ++m_loop)
              {
                if (m_loop < active_m_subtiles)
                {
#pragma unroll
                  for (int32_t r = 0; r < 4; ++r)
                  {
                    rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int_arr[0][r] = __shfl(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int_arr[0][r], shfl_src_lane);
                  }
#pragma unroll
                  for (int n = 0; n < N_REPEAT; ++n)
                  {
                    rC[m_loop][n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int8_arr[0], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[0], rC[m_loop][n]);
                    rC[m_loop][n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int8_arr[1], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[1], rC[m_loop][n]);
                  }
                }
              }
            }
// epilogue
#pragma unroll
            for (int k = K_ITERS; k < K_ITERS + (PIPE_STAGE - 1); k++)
            {

#if defined(__gfx938__)
              wait_vmcnt_0();
#endif
#pragma unroll
              for (int m_loop = 0; m_loop < M_SUBTILES; ++m_loop)
              {
                if (m_loop < active_m_subtiles)
                {
#pragma unroll
                  for (int32_t r = 0; r < 4; ++r)
                  {
                    rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int_arr[0][r] = __shfl(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int_arr[0][r], shfl_src_lane);
                  }
#pragma unroll
                  for (int n = 0; n < N_REPEAT; ++n)
                  {
                    rC[m_loop][n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int8_arr[0], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[0], rC[m_loop][n]);
                    rC[m_loop][n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][m_loop].int8_arr[1], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[1], rC[m_loop][n]);
                  }
                }
              }
            }
            // dequant
#pragma unroll
            for (int m_loop = 0; m_loop < M_SUBTILES; ++m_loop)
            {
              if (m_loop < active_m_subtiles)
              {
#pragma unroll
                for (int n = 0; n < N_REPEAT; n++)
                {
                  vec4_bf16 st;
#pragma unroll
                  for (int32_t i = 0; i < MMA_C_REGS; i++)
                  {
                    float r_st;
                    if constexpr (B_TENSOR_WISE)
                    {
                      r_st = float(rC[m_loop][n][i]) * input_scale[m_loop] * tensor_weight_scale;
                    }
                    else
                    {
                      r_st = float(rC[m_loop][n][i]) * input_scale[m_loop] * weight_scale[n][i];
                    }

                    bhalf_t tmp = bhalf_t(r_st);
                    st[i] = reinterpret_cast<unsigned short &>(tmp);
                  }
                  if (m_loop * 16 + ld_row < valid_rows_in_tile)
                  {
                    int32_t st_lane_offset = (m_loop * 16 + ld_row) * N + ld_col * MMA_C_REGS;
                    vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
                    p_st[0] = st;
                  }
                }
              }
            }

            tile_id += CUs;
          }
          last_expert_end += cur_expert_tiles;
        }
      }

      template <class Acc_Type,
                int N,
                int K,
                int BLOCK_M,
                int BLOCK_N,
                int BLOCK_K /*each iter load stride*/,
                int WARP_M,
                int WARP_N,
                int NUM_WARPS,
                int K_SCALE_RANGE,
                int N_SCALE_RANGE,
                int CUs>
      __global__ __launch_bounds__(256, 1) void kGroupGemmBlockWise(const char *__restrict__ matrixA,
                                                                    const char *__restrict__ matrixB,
                                                                    const float *__restrict__ matrixA_scale,
                                                                    const float *__restrict__ matrixB_scale,
                                                                    const int32_t *__restrict__ actual_M,
                                                                    const int M,
                                                                    const int num_expert,
                                                                    bhalf_t *__restrict__ MatrixC)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        static_assert(BLOCK_N >= N_SCALE_RANGE, "BLOCK_N should greater equal than N_SCALE_RANGE");
        static_assert(BLOCK_K <= K_SCALE_RANGE, "BLOCK_K should less equal than K_SCALE_RANGE");
        constexpr int N_TILES = N / BLOCK_N;
        constexpr int32_t K_OUT_ITERS = K / K_SCALE_RANGE;
        constexpr int32_t K_IN_ITERS = K_SCALE_RANGE / BLOCK_K;
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t stage = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = false;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < num_expert)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int e = 0; e < num_expert; ++e)
        {
          // TODO: opt actual M by shfl
          // int32_t cur_tokens = actual_M[e];
          // int32_t cur_tokens = __shfl(local_tokens, e);
          int32_t cur_tokens = actual_M[e];
          int m_tiles = ceil_div(cur_tokens, BLOCK_M);
          int32_t cur_expert_tiles = m_tiles * N_TILES;
          // expert base ptr
          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          // float *cur_matrixC = cur_matrixC + e * M * N;
          // abort core dump
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          // scale base ptr
          const float *cur_matrix_as = matrixA_scale + e * M * (K / K_SCALE_RANGE);
          const float *cur_matrix_bs = matrixB_scale + e * (N / N_SCALE_RANGE) * (K / K_SCALE_RANGE);
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

          while ((tile_id >= last_expert_end) && (tile_id < last_expert_end + cur_expert_tiles))
          {

            v_type rA[stage];
            v_type rB[stage][N_REPEAT];
            vec<Acc_Type, 4> rD[N_REPEAT] = {0};
            int local_tile_id = tile_id - last_expert_end;
            int tile_m = local_tile_id / N_TILES;
            int tile_n = local_tile_id % N_TILES;
            // offset
            const int32_t warp_offset_M = 0;
            const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
            // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
            const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
            const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
            // base ptr
            const char *cur_M_base = cur_matrixA + tile_m * BLOCK_M * K;
            const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
            const char *cur_M_warp_base = cur_M_base;
            const char *cur_N_warp_base = cur_N_base + warp_offset_N;
            bhalf_t *cur_matrixC_warp_base = cur_matrixC + tile_m * BLOCK_M * N + tile_n * BLOCK_N + warp_idx * MMA_N;
            int32_t sa_row = tile_m * BLOCK_M + ld_row;

// iter over block-wise dequant
#pragma unroll 8
            for (int32_t k_o = 0; k_o < K_OUT_ITERS; k_o++)
            {
              vec<Acc_Type, 4> rC[N_REPEAT] = {0};
              // TODO: add load matrix A & B scaler factor
              int32_t sa_col = k_o;
              float token_scale = cur_matrix_as[sa_row * K_OUT_ITERS + sa_col];
              float weight_scale[N_REPEAT];
              for (int i = 0; i < N_REPEAT; i += 2)
              {
                // int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_col = k_o;
                const float *p_sw = cur_matrix_bs + sw_row * K_OUT_ITERS + sw_col;
                weight_scale[i] = p_sw[0];
                weight_scale[i + 1] = weight_scale[i];
              }

              // mmac with fp8
              for (int32_t k_in = 0; k_in < K_IN_ITERS; k_in++)
              {
                int32_t k = k_o * K_IN_ITERS + k_in;
                const vec<int32_t, 4> *addr_a = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
                rA[k % stage].int_arr[0] = addr_a[0];

                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  const vec<int32_t, 4> *addr_b = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                  rB[k % stage][n].int_arr[0] = addr_b[0];
                }
                for (int32_t r = 0; r < 4; ++r)
                {
                  rA[k % stage].int_arr[0][r] = __shfl(rA[k % stage].int_arr[0][r], shfl_src_lane);
                }
                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  rC[n] = mmac_(rA[k % stage].int8_arr[0], rB[k % stage][n].int8_arr[0], rC[n]);
                  rC[n] = mmac_(rA[k % stage].int8_arr[1], rB[k % stage][n].int8_arr[1], rC[n]);
                }
              }
              // dequant mma result

              // update rD
              for (int n = 0; n < N_REPEAT; ++n)
              {
                for (int r = 0; r < 4; ++r)
                {
                  rD[n][r] = rD[n][r] + rC[n][r] * token_scale * weight_scale[n];
                }
              }
            }
            // dequant

            for (int n = 0; n < N_REPEAT; n++)
            {
              vec4_bf16 st;
              for (int32_t i = 0; i < MMA_C_REGS; i++)
              {
                bhalf_t tmp = bhalf_t(rD[n][i]);
                st[i] = reinterpret_cast<unsigned short &>(tmp);
              }
              int32_t st_lane_offset = ld_row * N + ld_col * MMA_C_REGS;
              vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
              p_st[0] = st;
            }

            tile_id += CUs;
          }
          last_expert_end += cur_expert_tiles;
        }
      }

      template <class Acc_Type,
                int N,
                int K,
                int BLOCK_M,
                int BLOCK_N,
                int BLOCK_K /*each iter load stride*/,
                int WARP_M,
                int WARP_N,
                int NUM_WARPS,
                int K_SCALE_RANGE,
                int N_SCALE_RANGE,
                int CUs,
                bool SBO>
      __global__ __launch_bounds__(256, 1) void kGroupGemmBlockWise_(const char *__restrict__ matrixA,
                                                                     const char *__restrict__ matrixB,
                                                                     const float *__restrict__ matrixA_scale,
                                                                     const float *__restrict__ matrixB_scale,
                                                                     const int32_t *__restrict__ actual_M,
                                                                     const int M,
                                                                     const int num_expert,
                                                                     bhalf_t *__restrict__ MatrixC,
                                                                     int32_t *__restrict__ signal)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        static_assert(BLOCK_N >= N_SCALE_RANGE, "BLOCK_N should greater equal than N_SCALE_RANGE");
        static_assert(BLOCK_K <= K_SCALE_RANGE, "BLOCK_K should less equal than K_SCALE_RANGE");
        constexpr int N_TILES = N / BLOCK_N;
        constexpr int32_t K_OUT_ITERS = K / K_SCALE_RANGE;
        constexpr int32_t K_IN_ITERS = K_SCALE_RANGE / BLOCK_K;
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t stage = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = false;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < num_expert)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int e = 0; e < num_expert; ++e)
        {
          // TODO: opt actual M by shfl
          // int32_t cur_tokens = actual_M[e];
          // int32_t cur_tokens = __shfl(local_tokens, e);
          int32_t cur_tokens = actual_M[e];
          int m_tiles = ceil_div(cur_tokens, BLOCK_M);
          int32_t cur_expert_tiles = m_tiles * N_TILES;
          // expert base ptr
          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          // float *cur_matrixC = cur_matrixC + e * M * N;
          // abort core dump
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          // scale base ptr
          const float *cur_matrix_as = matrixA_scale + e * M * (K / K_SCALE_RANGE);
          const float *cur_matrix_bs = matrixB_scale + e * (N / N_SCALE_RANGE) * (K / K_SCALE_RANGE);
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

          while ((tile_id >= last_expert_end) && (tile_id < last_expert_end + cur_expert_tiles))
          {

            v_type rA[K_IN_ITERS];
            v_type rB[K_IN_ITERS][N_REPEAT];
            vec<Acc_Type, 4> rD[N_REPEAT] = {0};
            int local_tile_id = tile_id - last_expert_end;
            int tile_m = local_tile_id / N_TILES;
            int tile_n = local_tile_id % N_TILES;
            // offset
            const int32_t warp_offset_M = 0;
            const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
            // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
            const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
            const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
            // base ptr
            const char *cur_M_base = cur_matrixA + tile_m * BLOCK_M * K;
            const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
            const char *cur_M_warp_base = cur_M_base;
            const char *cur_N_warp_base = cur_N_base + warp_offset_N;
            bhalf_t *cur_matrixC_warp_base = cur_matrixC + tile_m * BLOCK_M * N + tile_n * BLOCK_N + warp_idx * MMA_N;
            int32_t sa_row = tile_m * BLOCK_M + ld_row;

// iter over block-wise dequant
#pragma unroll 8
            for (int32_t k_o = 0; k_o < K_OUT_ITERS; k_o++)
            {
              if (k_o % 2 == 0)
              {
                //__builtin_hcu_sched_barrier(0);
              }
              vec<Acc_Type, 4> rC[N_REPEAT] = {0};
              // TODO: add load matrix A & B scaler factor
              int32_t sa_col = k_o;
              float token_scale = cur_matrix_as[sa_row * K_OUT_ITERS + sa_col];
              float weight_scale[N_REPEAT];
              for (int i = 0; i < N_REPEAT; i += 2)
              {
                // int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_col = k_o;
                const float *p_sw = cur_matrix_bs + sw_row * K_OUT_ITERS + sw_col;
                weight_scale[i] = p_sw[0];
                weight_scale[i + 1] = weight_scale[i];
              }

              // mmac with fp8
              for (int32_t k_in = 0; k_in < K_IN_ITERS; k_in++)
              {
                int32_t k = k_o * K_IN_ITERS + k_in;
                const vec<int32_t, 4> *addr_a = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
                rA[k_in].int_arr[0] = addr_a[0];

                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  const vec<int32_t, 4> *addr_b = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                  rB[k_in][n].int_arr[0] = addr_b[0];
                }
                for (int32_t r = 0; r < 4; ++r)
                {
                  rA[k_in].int_arr[0][r] = __shfl(rA[k_in].int_arr[0][r], shfl_src_lane);
                }
              }
              __builtin_hcu_sched_barrier(0);
              for (int32_t k_in = 0; k_in < K_IN_ITERS; k_in++)
              {
                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  rC[n] = mmac_(rA[k_in].int8_arr[0], rB[k_in][n].int8_arr[0], rC[n]);
                  rC[n] = mmac_(rA[k_in].int8_arr[1], rB[k_in][n].int8_arr[1], rC[n]);
                }
              }
              // dequant mma result

              // update rD
              for (int n = 0; n < N_REPEAT; ++n)
              {
                for (int r = 0; r < 4; ++r)
                {
                  rD[n][r] = rD[n][r] + rC[n][r] * token_scale * weight_scale[n];
                }
              }
            }
            // dequant

            for (int n = 0; n < N_REPEAT; n++)
            {
              vec4_bf16 st;
              for (int32_t i = 0; i < MMA_C_REGS; i++)
              {
                bhalf_t tmp = bhalf_t(rD[n][i]);
                st[i] = reinterpret_cast<unsigned short &>(tmp);
              }
              int32_t st_lane_offset = ld_row * N + ld_col * MMA_C_REGS;
              vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
              p_st[0] = st;
            }

            if constexpr (SBO)
            {
              asm volatile("s_waitcnt vmcnt(0)");
              if (threadIdx.x == 0)
              {
                atomic_add_release_global(signal + e * ceil_div(M, BLOCK_M) + tile_m, 1);
              }
            }

            tile_id += CUs;
          }
          last_expert_end += cur_expert_tiles;
        }
      }

      template <bool B_TENSOR_WISE>
      void masked_fp8_gemm_impl(
          const torch::Tensor &matrix_a,
          const torch::Tensor &matrix_b,
          const torch::Tensor &matri_a_scale,
          const torch::Tensor &matrixb_scale,
          const torch::Tensor &actual_tokens,
          torch::Tensor &matrix_c,
          int64_t max_tokens,
          int64_t experts,
          int64_t cu_s,
          bool b_block_wise,
          bool b_overlap,
          const c10::optional<torch::Tensor> &signal)
      {
        // check basic info
        TORCH_CHECK(matrix_a.dim() == 3, "expect tensor matrix_a is 3 which is ", matrix_a.dim());
        TORCH_CHECK(matrix_b.dim() == 6, "expect tensor matrix_b is 6 which is ", matrix_b.dim());
        TORCH_CHECK(actual_tokens.dim() == 1, "expect tensor actual_tokens is 1 which is ", actual_tokens.dim());
        TORCH_CHECK(matrix_c.dim() == 3, "expect tensor matrix_c is 3 which is ", matrix_c.dim());
        TORCH_CHECK(matrix_a.is_cuda() && matrix_b.is_cuda() && matri_a_scale.is_cuda() &&
                        matrixb_scale.is_cuda() && actual_tokens.is_cuda() && matrix_c.is_cuda(),
                    "low_latency_grouped_gemm expects CUDA tensors");
        TORCH_CHECK(matrix_a.is_contiguous(), "matrix_a must be contiguous");
        TORCH_CHECK(matrix_b.is_contiguous(), "matrix_b must be contiguous");
        TORCH_CHECK(matrix_c.is_contiguous(), "matrix_c must be contiguous");
        TORCH_CHECK(matri_a_scale.is_contiguous(), "matrix_a_scale must be contiguous");
        TORCH_CHECK(matrixb_scale.is_contiguous(), "matrix_b_scale must be contiguous");
        TORCH_CHECK(actual_tokens.is_contiguous(), "actual_tokens must be contiguous");
        TORCH_CHECK(actual_tokens.scalar_type() == at::ScalarType::Int, "actual_tokens must be int32");
        TORCH_CHECK(matri_a_scale.scalar_type() == at::ScalarType::Float, "matrix_a_scale must be float32");
        TORCH_CHECK(matrixb_scale.scalar_type() == at::ScalarType::Float, "matrix_b_scale must be float32");
        TORCH_CHECK(matrix_c.scalar_type() == at::ScalarType::BFloat16, "matrix_c must be bfloat16");
        TORCH_CHECK(matrix_a.scalar_type() == at::ScalarType::Char || matrix_a.scalar_type() == at::kFloat8_e4m3fn,
                    "matrix_a must be int8 or float8_e4m3fn");
        TORCH_CHECK(matrix_b.scalar_type() == matrix_a.scalar_type(), "matrix_b dtype must match matrix_a dtype");
        TORCH_CHECK(max_tokens > 0, "max_tokens must be positive");
        TORCH_CHECK(experts > 0, "experts must be positive");
        TORCH_CHECK(cu_s > 0, "cu_s must be positive");
        TORCH_CHECK(matrix_a.size(0) == experts, "matrix_a expert dim mismatch, got ", matrix_a.size(0), " expected ", experts);
        TORCH_CHECK(matrix_b.size(0) == experts, "matrix_b expert dim mismatch, got ", matrix_b.size(0), " expected ", experts);
        TORCH_CHECK(matrix_c.size(0) == experts, "matrix_c expert dim mismatch, got ", matrix_c.size(0), " expected ", experts);
        TORCH_CHECK(actual_tokens.numel() == experts,
                    "actual_tokens length mismatch, got ", actual_tokens.numel(), " expected ", experts);
        TORCH_CHECK(matrix_c.size(1) == matrix_a.size(1),
                    "matrix_c M mismatch, got ", matrix_c.size(1), " expected ", matrix_a.size(1));

        int32_t n = matrix_c.size(2);
        int32_t k = matrix_a.size(2);
        int32_t n_out = matrix_b.size(2);
        int32_t n_in = matrix_b.size(4);
        int32_t k_out = matrix_b.size(1);
        int32_t k_mid = matrix_b.size(3);
        int32_t k_in = matrix_b.size(5);
        int32_t b_k = k_out * k_mid * k_in;
        int32_t b_n = n_out * n_in;
        torch::ScalarType type_a = matrix_a.scalar_type();
        TORCH_CHECK(b_k == k, "matrix_b k miss match, k_out ", k_out, "k_mid ", k_mid, "k_in ", k_in, "k ", k);
        TORCH_CHECK(b_n == n, "matrix_b n miss match, n_out ", n_out, "n_in ", n_in, "n ", n);

        if constexpr (B_TENSOR_WISE)
        {
          TORCH_CHECK(!b_block_wise, "tensor-wise FP8 masked GEMM does not support block_wise");
          TORCH_CHECK(matri_a_scale.dim() == 2, "tensor-wise matrix_a_scale must be [E,M]");
          TORCH_CHECK(matrixb_scale.dim() == 1 || (matrixb_scale.dim() == 2 && matrixb_scale.size(1) == 1),
                      "tensor-wise matrix_b_scale must be [E] or [E,1]");
          TORCH_CHECK(matri_a_scale.size(0) == experts, "matrix_a_scale expert dim mismatch");
          TORCH_CHECK(matri_a_scale.size(1) == matrix_a.size(1), "matrix_a_scale M dim mismatch");
          TORCH_CHECK(matrixb_scale.size(0) == experts, "matrix_b_scale expert dim mismatch");
        }
        else if (b_block_wise)
        {
          TORCH_CHECK(matri_a_scale.dim() == 3, "block-wise matrix_a_scale must be [E,M,K/128]");
          TORCH_CHECK(matrixb_scale.dim() == 3, "block-wise matrix_b_scale must be [E,N/128,K/128]");
          TORCH_CHECK(matri_a_scale.size(0) == experts, "matrix_a_scale expert dim mismatch");
          TORCH_CHECK(matri_a_scale.size(1) == matrix_a.size(1), "matrix_a_scale M dim mismatch");
          TORCH_CHECK(matri_a_scale.size(2) == k / 128, "matrix_a_scale K dim mismatch");
          TORCH_CHECK(matrixb_scale.size(0) == experts, "matrix_b_scale expert dim mismatch");
          TORCH_CHECK(matrixb_scale.size(1) == n / 128, "matrix_b_scale N dim mismatch");
          TORCH_CHECK(matrixb_scale.size(2) == k / 128, "matrix_b_scale K dim mismatch");
        }
        else
        {
          TORCH_CHECK(matri_a_scale.dim() == 2, "channel-wise matrix_a_scale must be [E,M]");
          TORCH_CHECK(matrixb_scale.dim() == 2, "channel-wise matrix_b_scale must be [E,N]");
          TORCH_CHECK(matri_a_scale.size(0) == experts, "matrix_a_scale expert dim mismatch");
          TORCH_CHECK(matri_a_scale.size(1) == matrix_a.size(1), "matrix_a_scale M dim mismatch");
          TORCH_CHECK(matrixb_scale.size(0) == experts, "matrix_b_scale expert dim mismatch");
          TORCH_CHECK(matrixb_scale.size(1) == n, "matrix_b_scale N dim mismatch");
        }
        // get device ptr
        const char *matrixA = reinterpret_cast<const char *>(matrix_a.data_ptr());
        const char *matrixB = reinterpret_cast<const char *>(matrix_b.data_ptr());
        const float *matrixA_scale = reinterpret_cast<const float *>(matri_a_scale.data_ptr());
        const float *matrixB_scale = reinterpret_cast<const float *>(matrixb_scale.data_ptr());
        const int32_t *actual_M = reinterpret_cast<const int32_t *>(actual_tokens.data_ptr());
        bhalf_t *d_MatrixC = reinterpret_cast<bhalf_t *>(matrix_c.data_ptr());
        int32_t *d_signal = nullptr;
        if (b_overlap)
        {
          TORCH_CHECK(signal.has_value(), "tensor signal empty");
          d_signal = reinterpret_cast<int32_t *>(signal.value().data_ptr());
        }

        int32_t M = matrix_a.size(1);
        int32_t num_expert = experts;
        int32_t cu = cu_s;
        constexpr int32_t NUM_WARPS = 4;
        constexpr int32_t WARP_SIZE = 64;
        // constexpr int32_t CUs = 80 * 4;
        dim3 block(NUM_WARPS * WARP_SIZE, 1, 1);
        TORCH_CHECK(!b_overlap, "low_latency_grouped_gemm does not support overlap");
        const hipStream_t stream = at::cuda::getCurrentHIPStream();
        MOE_LL_OVERLAP_SWITCH(b_overlap, [&]
                              { MOE_LL_TYPE_SWITCH(type_a, [&]
                                                   { MOE_LL_CU_SWITCH(cu, [&]
                                                                      { MOE_LL_N_SWITCH(n, [&]
                                                                                                          { MOE_LL_K_SWITCH(k, [&]
                                                                                                                            { dim3 grid(CUs, 1, 1);
                                                                                                TORCH_CHECK(!b_block_wise, "low_latency_grouped_gemm only supports channel-wise non-block-wise scales");
                                                                                                  if (max_tokens <= 48) {
                                                                                                    kGroupGemm<Acc_Type, N, K, 16, 256, 64, 16, 16, NUM_WARPS, 128, 128, CUs, 1, B_TENSOR_WISE><<<grid, block, 0, stream>>>(matrixA, matrixB, matrixA_scale, matrixB_scale, actual_M, M, num_expert, d_MatrixC);
                                                                                                  } else {
                                                                                                    kGroupGemm<Acc_Type, N, K, 64, 128, 64, 16, 16, NUM_WARPS, 128, 128, CUs, 4, B_TENSOR_WISE><<<grid, block, 0, stream>>>(matrixA, matrixB, matrixA_scale, matrixB_scale, actual_M, M, num_expert, d_MatrixC);
                                                                                                  } }) }) }) }) })
      }

      void masked_fp8_gemm(
          const torch::Tensor &matrix_a,
          const torch::Tensor &matrix_b,
          const torch::Tensor &matri_a_scale,
          const torch::Tensor &matrixb_scale,
          const torch::Tensor &actual_tokens,
          torch::Tensor &matrix_c,
          int64_t max_tokens,
          int64_t experts,
          int64_t cu_s,
          bool b_block_wise,
          bool b_overlap,
          const c10::optional<torch::Tensor> &signal)
      {
        masked_fp8_gemm_impl<false>(matrix_a, matrix_b, matri_a_scale, matrixb_scale,
                                    actual_tokens, matrix_c, max_tokens, experts, cu_s,
                                    b_block_wise, b_overlap, signal);
      }

      void masked_fp8_gemm_tensor_wise(
          const torch::Tensor &matrix_a,
          const torch::Tensor &matrix_b,
          const torch::Tensor &matri_a_scale,
          const torch::Tensor &matrixb_scale,
          const torch::Tensor &actual_tokens,
          torch::Tensor &matrix_c,
          int64_t max_tokens,
          int64_t experts,
          int64_t cu_s,
          bool b_overlap,
          const c10::optional<torch::Tensor> &signal)
      {
        masked_fp8_gemm_impl<true>(matrix_a, matrix_b, matri_a_scale, matrixb_scale,
                                   actual_tokens, matrix_c, max_tokens, experts, cu_s,
                                   false, b_overlap, signal);
      }
} // end namespace deepgemm
