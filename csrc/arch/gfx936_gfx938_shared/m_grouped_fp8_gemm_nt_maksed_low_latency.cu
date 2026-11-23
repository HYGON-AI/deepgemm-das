#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include "low_latency_fp8_masked_utils.h"

#include "m_grouped_gemm_nt_masked.h"

#include <cstdint>

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

      template <class Acc_Type, int N, int K, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int NUM_WARPS, int K_SCALE_RANGE, int N_SCALE_RANGE, int ACTIVE_M_SUBTILES, bool B_TENSOR_WISE = false>
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
        const int grid_x = gridDim.x;
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
          const int64_t expert_index = static_cast<int64_t>(e);
          const char *cur_matrixA = matrixA + expert_index * M * K;
          const char *cur_matrixB = matrixB + expert_index * N * K;
          // float *cur_matrixC = cur_matrixC + e * M * N;
          // abort core dump
          bhalf_t *cur_matrixC = MatrixC + expert_index * M * N;
          // scale base ptr
          const float *cur_matrix_as = matrixA_scale + expert_index * M;
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

            static_assert(PIPE_STAGE == 2, "scalarized double buffer assumes PIPE_STAGE == 2");
            static_assert((K_ITERS % 2) == 0, "supported low-latency K values use an even number of K tiles");

            v_type rA0_m0;
            v_type rA0_m1;
            v_type rA0_m2;
            v_type rA0_m3;
            v_type rA1_m0;
            v_type rA1_m1;
            v_type rA1_m2;
            v_type rA1_m3;
            v_type rB0_n0;
            v_type rB0_n1;
            v_type rB0_n2;
            v_type rB0_n3;
            v_type rB1_n0;
            v_type rB1_n1;
            v_type rB1_n2;
            v_type rB1_n3;
            vec<Acc_Type, 4> rC[M_SUBTILES][N_REPEAT] = {0};
            float input_scale[M_SUBTILES];

#define DG_A_REG(STAGE, MIDX) rA##STAGE##_m##MIDX
#define DG_B_REG(STAGE, NIDX) rB##STAGE##_n##NIDX
#if defined(__gfx938__)
#define DG_LOAD_A_REG(STAGE, MIDX, KITER)                                                                                 \
            do                                                                                                            \
            {                                                                                                             \
              if constexpr (active_m_subtiles > (MIDX))                                                                   \
              {                                                                                                           \
                constexpr int32_t m_const = (MIDX);                                                                       \
                const int a_byte_offset = m_const * 16 * K + (KITER)*LD_M_ITERK_STRIDE + lane_offset_M_;                  \
                DG_A_REG(STAGE, MIDX).int_arr[0] =                                                                        \
                    buffer_load_dwordx4_to_vgpr_i32(matrixA_resource, a_byte_offset);                                     \
              }                                                                                                           \
            } while (0)
#define DG_LOAD_B_REG(STAGE, NIDX, KITER)                                                                                 \
            do                                                                                                            \
            {                                                                                                             \
              if constexpr (N_REPEAT > (NIDX))                                                                            \
              {                                                                                                           \
                constexpr int32_t n_const = (NIDX);                                                                       \
                const int b_byte_offset =                                                                                 \
                    tile_n * LD_ELEMENTS_PER_BLOCK + warp_offset_N +                                                      \
                    (KITER)*LD_N_ITERK_STRIDE + lane_offset_N +                                                           \
                    n_const * NUM_WARPS * LD_ELEMENTS_PER_WARP;                                                           \
                DG_B_REG(STAGE, NIDX).int_arr[0] =                                                                        \
                    buffer_load_dwordx4_to_vgpr_i32(matrixB_resource, b_byte_offset);                                     \
              }                                                                                                           \
            } while (0)
#else
#define DG_LOAD_A_REG(STAGE, MIDX, KITER)                                                                                 \
            do                                                                                                            \
            {                                                                                                             \
              if constexpr (active_m_subtiles > (MIDX))                                                                   \
              {                                                                                                           \
                constexpr int32_t m_const = (MIDX);                                                                       \
                const vec<int32_t, 4> *addr_a_next =                                                                       \
                    reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + m_const * 16 * K +                       \
                                                              (KITER)*LD_M_ITERK_STRIDE + lane_offset_M_);                \
                DG_A_REG(STAGE, MIDX).int_arr[0] = addr_a_next[0];                                                        \
              }                                                                                                           \
            } while (0)
#define DG_LOAD_B_REG(STAGE, NIDX, KITER)                                                                                 \
            do                                                                                                            \
            {                                                                                                             \
              if constexpr (N_REPEAT > (NIDX))                                                                            \
              {                                                                                                           \
                constexpr int32_t n_const = (NIDX);                                                                       \
                const vec<int32_t, 4> *addr_b_next =                                                                       \
                    reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base +                                           \
                                                              (KITER)*LD_N_ITERK_STRIDE + lane_offset_N +                 \
                                                              n_const * NUM_WARPS * LD_ELEMENTS_PER_WARP);                \
                DG_B_REG(STAGE, NIDX).int_arr[0] = addr_b_next[0];                                                        \
              }                                                                                                           \
            } while (0)
#endif
#define DG_LOAD_STAGE_REG(STAGE, KITER)                                                                                   \
            do                                                                                                            \
            {                                                                                                             \
              DG_LOAD_A_REG(STAGE, 0, KITER);                                                                             \
              DG_LOAD_A_REG(STAGE, 1, KITER);                                                                             \
              DG_LOAD_A_REG(STAGE, 2, KITER);                                                                             \
              DG_LOAD_A_REG(STAGE, 3, KITER);                                                                             \
              DG_LOAD_B_REG(STAGE, 0, KITER);                                                                             \
              DG_LOAD_B_REG(STAGE, 1, KITER);                                                                             \
              DG_LOAD_B_REG(STAGE, 2, KITER);                                                                             \
              DG_LOAD_B_REG(STAGE, 3, KITER);                                                                             \
            } while (0)
#define DG_WAIT_PREV_STAGE_REG()                                                                                          \
            do                                                                                                            \
            {                                                                                                             \
              constexpr int32_t VMEM_LOADS_PER_STAGE = active_m_subtiles + N_REPEAT;                                      \
              if constexpr (VMEM_LOADS_PER_STAGE <= 3)                                                                    \
                wait_vmcnt_3();                                                                                           \
              else if constexpr (VMEM_LOADS_PER_STAGE == 4)                                                               \
                wait_vmcnt_4();                                                                                           \
              else if constexpr (VMEM_LOADS_PER_STAGE == 5)                                                               \
                wait_vmcnt_5();                                                                                           \
              else if constexpr (VMEM_LOADS_PER_STAGE == 6)                                                               \
                wait_vmcnt_6();                                                                                           \
              else if constexpr (VMEM_LOADS_PER_STAGE == 7)                                                               \
                wait_vmcnt_7();                                                                                           \
              else                                                                                                        \
                wait_vmcnt_8();                                                                                           \
            } while (0)
#define DG_SHFL_A_REG(STAGE, MIDX)                                                                                        \
            do                                                                                                            \
            {                                                                                                             \
              DG_A_REG(STAGE, MIDX).int_arr[0][0] = __shfl(DG_A_REG(STAGE, MIDX).int_arr[0][0], shfl_src_lane);          \
              DG_A_REG(STAGE, MIDX).int_arr[0][1] = __shfl(DG_A_REG(STAGE, MIDX).int_arr[0][1], shfl_src_lane);          \
              DG_A_REG(STAGE, MIDX).int_arr[0][2] = __shfl(DG_A_REG(STAGE, MIDX).int_arr[0][2], shfl_src_lane);          \
              DG_A_REG(STAGE, MIDX).int_arr[0][3] = __shfl(DG_A_REG(STAGE, MIDX).int_arr[0][3], shfl_src_lane);          \
            } while (0)
#define DG_COMPUTE_MN_REG(STAGE, MIDX, NIDX)                                                                              \
            do                                                                                                            \
            {                                                                                                             \
              if constexpr (active_m_subtiles > (MIDX) && N_REPEAT > (NIDX))                                              \
              {                                                                                                           \
                rC[MIDX][NIDX] = mmac_(DG_A_REG(STAGE, MIDX).int8_arr[0], DG_B_REG(STAGE, NIDX).int8_arr[0], rC[MIDX][NIDX]); \
                rC[MIDX][NIDX] = mmac_(DG_A_REG(STAGE, MIDX).int8_arr[1], DG_B_REG(STAGE, NIDX).int8_arr[1], rC[MIDX][NIDX]); \
              }                                                                                                           \
            } while (0)
#define DG_COMPUTE_M_REG(STAGE, MIDX)                                                                                     \
            do                                                                                                            \
            {                                                                                                             \
              if constexpr (active_m_subtiles > (MIDX))                                                                   \
              {                                                                                                           \
                DG_SHFL_A_REG(STAGE, MIDX);                                                                               \
                DG_COMPUTE_MN_REG(STAGE, MIDX, 0);                                                                        \
                DG_COMPUTE_MN_REG(STAGE, MIDX, 1);                                                                        \
                DG_COMPUTE_MN_REG(STAGE, MIDX, 2);                                                                        \
                DG_COMPUTE_MN_REG(STAGE, MIDX, 3);                                                                        \
              }                                                                                                           \
            } while (0)
#define DG_COMPUTE_STAGE_REG(STAGE)                                                                                       \
            do                                                                                                            \
            {                                                                                                             \
              DG_COMPUTE_M_REG(STAGE, 0);                                                                                 \
              DG_COMPUTE_M_REG(STAGE, 1);                                                                                 \
              DG_COMPUTE_M_REG(STAGE, 2);                                                                                 \
              DG_COMPUTE_M_REG(STAGE, 3);                                                                                 \
            } while (0)

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
              int64_t sa_offset = expert_index * M + m_block_start + m_loop * 16 + ld_row;
              input_scale[m_loop] = matrixA_scale[sa_offset];
            }
            // int32_t sb_offset = e * N + tile_n * BLOCK_N + ld_row;
            vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
            if constexpr (!B_TENSOR_WISE)
            {
              for (int i = 0; i < N_REPEAT; ++i)
              {
                const vec<float, MMA_C_REGS> *p_ws = reinterpret_cast<const vec<float, MMA_C_REGS> *>(matrixB_scale + expert_index * N + tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
                weight_scale[i] = p_ws[0];
              }
            }

            DG_LOAD_STAGE_REG(0, 0);
#pragma unroll
            for (int k = 1; k < K_ITERS; ++k)
            {
              if ((k & 1) != 0)
              {
                DG_LOAD_STAGE_REG(1, k);
#if defined(__gfx938__)
                DG_WAIT_PREV_STAGE_REG();
#endif
                __builtin_hcu_sched_barrier(0);
                DG_COMPUTE_STAGE_REG(0);
              }
              else
              {
                DG_LOAD_STAGE_REG(0, k);
#if defined(__gfx938__)
                DG_WAIT_PREV_STAGE_REG();
#endif
                __builtin_hcu_sched_barrier(0);
                DG_COMPUTE_STAGE_REG(1);
              }
            }
#if defined(__gfx938__)
            wait_vmcnt_0();
#endif
            __builtin_hcu_sched_barrier(0);
            DG_COMPUTE_STAGE_REG(1);

#undef DG_COMPUTE_STAGE_REG
#undef DG_COMPUTE_M_REG
#undef DG_COMPUTE_MN_REG
#undef DG_SHFL_A_REG
#undef DG_WAIT_PREV_STAGE_REG
#undef DG_LOAD_STAGE_REG
#undef DG_LOAD_B_REG
#undef DG_LOAD_A_REG
#undef DG_B_REG
#undef DG_A_REG
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

            tile_id += grid_x;
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
          const c10::optional<torch::Tensor> &signal,
          int64_t mode)
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
          TORCH_CHECK(matri_a_scale.dim() == 2 || (matri_a_scale.dim() == 3 && matri_a_scale.size(2) == 1), 
                      "tensor-wise matrix_a_scale must be [E,M] or [E,M,1]");
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
          TORCH_CHECK(matri_a_scale.dim() == 2 ||
                          (matri_a_scale.dim() == 3 && matri_a_scale.size(2) == 1),
                      "channel-wise matrix_a_scale must be [E,M] or [E,M,1]");
          TORCH_CHECK(matrixb_scale.dim() == 2 ||
                          (matrixb_scale.dim() == 3 && matrixb_scale.size(2) == 1),
                      "channel-wise matrix_b_scale must be [E,N] or [E,N,1]");
          if (matri_a_scale.dim() == 3) {
            TORCH_CHECK(matri_a_scale.is_contiguous(),
                        "channel-wise matrix_a_scale with shape [E,M,1] must be contiguous");
          }
          if (matrixb_scale.dim() == 3) {
            TORCH_CHECK(matrixb_scale.is_contiguous(),
                        "channel-wise matrix_b_scale with shape [E,N,1] must be contiguous");
          }
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
        (void)signal;

        int32_t M = matrix_a.size(1);
        int32_t num_expert = experts;
        int32_t cu = cu_s;
        constexpr int32_t NUM_WARPS = 4;
        constexpr int32_t WARP_SIZE = 64;
        dim3 block(NUM_WARPS * WARP_SIZE, 1, 1);
        TORCH_CHECK(!b_overlap, "low_latency_grouped_gemm does not support overlap");
        const hipStream_t stream = at::cuda::getCurrentHIPStream();
        MOE_LL_TYPE_SWITCH(type_a, [&] {
          MOE_LL_N_SWITCH(n, [&] {
            MOE_LL_K_SWITCH(k, [&] {
              dim3 grid(cu, 1, 1);
              TORCH_CHECK(!b_block_wise, "low_latency_grouped_gemm only supports channel-wise non-block-wise scales");
              if(mode == 1100){
                // GLM5 M32 path: reuse the packed FP8 weight fragment across two M=16 subtiles.
                kGroupGemm<Acc_Type, N, K, 32, 256, 64, 16, 16, NUM_WARPS, 128, 128, 2, B_TENSOR_WISE><<<grid, block, 0, stream>>>(matrixA, matrixB, matrixA_scale, matrixB_scale, actual_M, M, num_expert, d_MatrixC);
              } else if(mode == 1101){
                kGroupGemm<Acc_Type, N, K, 16, 256, 64, 16, 16, NUM_WARPS, 128, 128, 1, B_TENSOR_WISE><<<grid, block, 0, stream>>>(matrixA, matrixB, matrixA_scale, matrixB_scale, actual_M, M, num_expert, d_MatrixC);
              } else if(mode == 1102){
                kGroupGemm<Acc_Type, N, K, 64, 128, 64, 16, 16, NUM_WARPS, 128, 128, 4, B_TENSOR_WISE><<<grid, block, 0, stream>>>(matrixA, matrixB, matrixA_scale, matrixB_scale, actual_M, M, num_expert, d_MatrixC);
              } else{
                TORCH_CHECK(false, mode, " mode error.");
              }
            });
          });
        });
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
          const c10::optional<torch::Tensor> &signal,
          int64_t mode)
      {
        masked_fp8_gemm_impl<false>(matrix_a, matrix_b, matri_a_scale, matrixb_scale,
                                    actual_tokens, matrix_c, max_tokens, experts, cu_s,
                                    b_block_wise, b_overlap, signal, mode);
      }
} // end namespace deepgemm
