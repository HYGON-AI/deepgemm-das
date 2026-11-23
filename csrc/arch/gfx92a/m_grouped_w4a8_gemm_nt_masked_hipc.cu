// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <cstdlib>
#include <numeric>
#include <tuple>
#include <utility>
#include <type_traits>
#include <vector>

#include "m_grouped_gemm_nt_masked.h"
#include "hipc_w4a8/w4a8_intrinsics.h"

#include "m_grouped_w4a8_gemm_nt_masked_pair32/bm128_bn256_bk128.inc.cuh"

namespace gfx92a {

using half_t = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define DG_HIP_CHECK(call)                                                   \
  do {                                                                       \
    hipError_t err = (call);                                                  \
    TORCH_CHECK(                                                              \
        err == hipSuccess, #call " failed: ", hipGetErrorString(err));       \
  } while (0)

template <typename T, typename T_hidden = bhalf_t>
struct GemmParams_w4a8 {
    GemmParams_w4a8(const T* ptr_A,
               const T* ptr_B0,
               T_hidden* ptr_C,
               float* ptr_A_scale,
               float* ptr_B_scale,
               const int32_t* masked_m,
               uint32_t experts,
               uint32_t size_m,
               uint32_t size_n, 
               uint32_t size_k, 
               uint32_t stride_asm,
               uint32_t stride_ask,
               uint32_t stride_bse,
               uint32_t stride_bsn, 
               uint32_t stride_bsk,
               uint32_t blocks_per_expert,
               int persistent_multiplier = 0,
               bool enable_overlap = false,
               int32_t* signal = nullptr
            )
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
          ptr_B_scale(ptr_B_scale),
          masked_m(masked_m),
          experts(experts),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          stride_asm(stride_asm),
          stride_ask(stride_ask),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk),
          blocks_per_expert(blocks_per_expert),
          persistent_multiplier(persistent_multiplier),
          enable_overlap(enable_overlap),
          signal(signal)
         {}

    const T* ptr_A;
    const T* ptr_B0;
    T_hidden* ptr_C;
    float* ptr_A_scale;
    float* ptr_B_scale;
    const int32_t* masked_m;
    uint32_t experts;
    uint32_t size_m;
    uint32_t size_n;
    uint32_t size_k;
    uint32_t stride_asm;
    uint32_t stride_ask;
    uint32_t stride_bse;
    uint32_t stride_bsn;
    uint32_t stride_bsk;
    uint32_t blocks_per_expert;
    int persistent_multiplier;
    bool enable_overlap;
    int32_t* signal;
};

template<bool Is_store_A, 
    int A_prefetch_level, 
    int B_prefetch_level, 
    int WARP_NUM,
    int BLOCK_M, 
    int BLOCK_N, 
    int BLOCK_K, 
    int WARP_M,  
    int WARP_N,  
    int WARP_K,  
    int STAGE,  
    int GROUP_N,
    int GROUP_K,
    int SIZE_K,
    typename Element, 
    typename ElementAccum = int32_t,
    int N_LOOP_NUM = 4,
    bool ABC_ORDER = false,
    bool LDS_ONLY_REUSE = false>
__forceinline__ __device__ void  gemm_nt_marlin_prefill_w4a8(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
    int valid_m_for_load,
    union_vec_opt<Element,WARP_K/4> A_reg[][STAGE],
    union_vec_opt<Element,WARP_K/4> B_reg[][2][STAGE],
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    const int32_t expert_id,
    const int bidx

    ) {

        constexpr int n_loop_num = N_LOOP_NUM;
        
    const int size_k = (SIZE_K == 0) ? seqlen_A_stride : SIZE_K;
    const int k_loop_count = size_k / 128;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2 ) * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        int s_index = (warp_n_id  * (WARP_N / 32)  + n_tile ) * 32 * 32;
        int index = s_index + row_id  * 2 * 8  + col_id * 32 * 8;

            g_row_B[n_tile ] = index ;

    }

    int A_index = warp_id;
    

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){

           

            int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index*4;
            int row = std::min(s_offset + col_id, valid_m_for_load - 1);
            g_row_A[m_tile] = (expert_id * max_n_len_offset + row) * size_k + row_id * 4;
            

    }


#if defined(__gfx938__)
    constexpr bool matrix_a = (WARP_M == 64 || WARP_M == 128) && WARP_NUM == 4 &&
        BLOCK_K == 64 && WARP_K == 64 && (N_LOOP_NUM == 2 || N_LOOP_NUM == 4);
#else
    constexpr bool matrix_a = false;
#endif
    constexpr bool cache_a_descriptor = matrix_a && WARP_M == 64 && WARP_N == 32 && N_LOOP_NUM == 2 && STAGE == 2;
    using ADesc = unsigned __attribute__((ext_vector_type(4)));
    ADesc a_descriptor[DIVIDE(WARP_M, 64)];
    int a_lds_offset[DIVIDE(WARP_M, 64)];
    if constexpr (cache_a_descriptor) {
        #pragma unroll
        for (int tile = 0; tile < WARP_M / 64; ++tile) {
            const int first_row = bidx * BLOCK_M + tile * 64 + (warp_id & 1) * 32;
            const int rows = std::max(0, std::min(32, valid_m_for_load - first_row));
            const uint64_t address = reinterpret_cast<uint64_t>(input_ptr +
                (expert_id * max_n_len_offset + std::min(first_row, valid_m_for_load - 1)) * size_k +
                (warp_id / 2) * 64);
            a_descriptor[tile] = ADesc{
                static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(address))),
                static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(address >> 32))),
                static_cast<unsigned>(__builtin_amdgcn_readfirstlane(size_k)),
                static_cast<unsigned>(__builtin_amdgcn_readfirstlane((32 - rows) << 8))};
            a_lds_offset[tile] = __builtin_amdgcn_readfirstlane(static_cast<int>(
                reinterpret_cast<uintptr_t>(A_lds) + tile * 4096 + (warp_id & 1) * 2048 +
                (warp_id / 2) * WARP_M * 64));
        }
    }
    auto load_a = [&](int next_k) {
        if constexpr (cache_a_descriptor) {
            #pragma unroll
            for (int tile = 0; tile < WARP_M / 64; ++tile) {
                ADesc descriptor = a_descriptor[tile];
                const uint64_t address = (static_cast<uint64_t>(descriptor[1]) << 32 | descriptor[0]) + next_k;
                descriptor[0] = static_cast<unsigned>(address);
                descriptor[1] = static_cast<unsigned>(address >> 32);
                asm volatile("s_nop 4\n\t"
                             "matrix_load_64x32_b8 %0, %1, moffset:0 lds\n\t"
                             : : "s"(descriptor), "s"(a_lds_offset[tile]) : "memory");
            }
        } else if constexpr (matrix_a) {
            using U4 = unsigned __attribute__((ext_vector_type(4)));
            #pragma unroll
            for (int tile=0; tile<WARP_M/64; ++tile) {
                const int first_row = bidx * BLOCK_M + tile * 64 + (warp_id & 1) * 32;
                const int rows = std::max(0, std::min(32, valid_m_for_load - first_row));
                U4 descriptor = {};
                const uint64_t address = reinterpret_cast<uint64_t>(input_ptr +
                    (expert_id * max_n_len_offset + std::min(first_row, valid_m_for_load - 1)) *
                        size_k + next_k + (warp_id / 2) * 64);
                descriptor[0] = static_cast<uint32_t>(address);
                descriptor[1] = static_cast<uint32_t>(address >> 32);
                descriptor[2] = size_k;
                descriptor[3] = (32 - rows) << 8;
                const U4 uniform = {
                    static_cast<unsigned>(__builtin_amdgcn_readfirstlane(descriptor[0])),
                    static_cast<unsigned>(__builtin_amdgcn_readfirstlane(descriptor[1])),
                    static_cast<unsigned>(__builtin_amdgcn_readfirstlane(descriptor[2])),
                    static_cast<unsigned>(__builtin_amdgcn_readfirstlane(descriptor[3]))};
                const int offset = __builtin_amdgcn_readfirstlane(static_cast<int>(
                    (uintptr_t)A_lds + tile * 4096 + (warp_id & 1) * 2048 + (warp_id / 2) * WARP_M * 64));
                // VMEM completion is handled by the existing A/B pipeline waits.
                asm volatile("s_nop 4\n\t"
                             "matrix_load_64x32_b8 %0, %1, moffset:0 lds\n\t"
                             : : "s"(uniform), "s"(offset) : "memory");
            }

        } else {
            #pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                #pragma unroll
                for (int mt = 0; mt < WARP_M / MFMA_M; ++mt) {
                    inline_buffer_load_dword_lds(A_lds, g_input,
                        (mt * 16 * WARP_K + A_index * 4 * WARP_K + stage * WARP_M * WARP_K) / 4,
                        (next_k + warp_k_id * WARP_K + stage * stage_offset) / 4, g_row_A[mt] / 4);
                }
            }
        }
    };
    constexpr bool prefetch_b_next = cache_a_descriptor;
    union {
        vec<int, 4> v;
        uint32_t i[4];
    } weight_reg_tmp[ WARP_N / 32][2][STAGE];
    auto prefetch_first_b = [&](int next_k) {
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            #pragma unroll
            for (int nt = 0; nt < WARP_N / 32; ++nt) {
                #pragma unroll
                for (int kt = 0; kt < WARP_K / READ_K; ++kt) {
                    buffer_load_reg_dwordx4_w4a8(weight_ptr, weight_reg_tmp[nt][0][stage].v,
                        warp_k_id * (WARP_K / 2) * size_n + next_k / 2 * size_n +
                        stage * stage_offset_b + kt * 32 * size_n, g_row_B[nt]);
                }
            }
        }
    };
    int kloop = 0;
    int i = 0;
    load_a(0);
    if constexpr (prefetch_b_next) prefetch_first_b(0);

auto compute_kloop = [&](auto prefetch_next_constant) {
    constexpr bool PREFETCH_NEXT = decltype(prefetch_next_constant)::value;

    if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);
    // The first B wait also completes the older A loads before reading LDS.
    if constexpr (STAGE != 2) {
        vmcnt_wait(0);
    }

    int k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (128 / 2) * size_n;
    

    
    

    
    

        

        

    

    

   

    

    

    

if constexpr (STAGE == 2){

const Element* cur_weight_ptr = weight_ptr ;

int i =0;        
int stage_b_flag =0;
int n_loop =0;

if constexpr (!prefetch_b_next) {
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);
#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ] );
        
        }

}

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

i = 1;

k_start_b += stage_offset_b;

      
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile ]   );
            
        }

}
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

}
vmcnt_wait(2 *( WARP_N / 32 ) );

	    #pragma unroll
	    for(int i =0;i<2;i++){
	        #pragma unroll
	        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
	            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                // MLS swizzles each pair of rows across the two 32-byte K halves.
                int s_index = m_tile * (16) * 64 + + (i) * WARP_M / 16 * (16) * WARP_K;
                A_reg[m_tile][i].int4_array[k_tile] = *( vec<Element, 16> *) (&A_lds[ s_index + row_id * 64 + (col_id * 16 ^ (matrix_a ? ((row_id & 2) << 4) : 0)) ]);

            }
	        } 
	    }
		k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (WARP_K * 2 / 2) * size_n;

n_loop ++;

for(n_loop;n_loop < n_loop_num;n_loop++)
{

stage_b_flag ^= 1;
cur_weight_ptr  +=   BLOCK_N * 32;
int i =0;

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ]);
        

        }

}
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

i = 1;

k_start_b += stage_offset_b;
        
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);
#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ] );
        
        }

}
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

   

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);
vmcnt_only_wait( 3* ( WARP_N / 32 ));

    
n_loop--;
i = 0;
stage_b_flag ^= 1;

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

    

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            
            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }  
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            } 
        }
    }

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

#pragma unroll 
for(int m_tile = 0; m_tile <  WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                       
                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

vmcnt_only_wait( 2* ( WARP_N / 32 ));

i = 1;

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

    

            

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            
            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }  
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            } 
        }
    }

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

#pragma unroll
for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                
                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}

    
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);
stage_b_flag ^= 1;
n_loop ++;

k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (WARP_K * 2 / 2) * size_n;
   
}

    n_loop--;
    // Finish shared A reads while register-only B loads continue.
    if constexpr (cache_a_descriptor && LDS_ONLY_REUSE) {
        __builtin_hcu_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n\t" ::: "memory");
        __builtin_hcu_sched_barrier(0);
    } else {
        __syncthreads();
    }
    if constexpr (PREFETCH_NEXT) {
        load_a((kloop + 1) * 128);
        if constexpr (prefetch_b_next) prefetch_first_b((kloop + 1) * 128);
    }
    vmcnt_only_wait( (prefetch_b_next && PREFETCH_NEXT ? 2 : 0) + ( WARP_N / 32 ) + (matrix_a ? WARP_M / 64 : 2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)) );

if constexpr (!PREFETCH_NEXT) {
    vmcnt_wait(  ( WARP_N / 32 ) );
}

i = 0;

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

    

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            
            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }  
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            } 
            
        }
    }

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

#pragma unroll
for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

    // Only register B operands are consumed here; LDS reuse is fenced above.
    vmcnt_only_wait((prefetch_b_next && PREFETCH_NEXT ? 2 : 0) + (matrix_a ? WARP_M / 64 : 2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));

    if constexpr (!PREFETCH_NEXT) {
        vmcnt_wait(  0 );
    }
    

i = 1;

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

    

            
     
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            
            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }  
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            } 
        }
    }

  

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

#pragma unroll
for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]), 
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}
if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

if constexpr (!(matrix_a && WARP_M == 128 && std::is_same_v<ElementAccum, bhalf_t>))
        __builtin_hcu_sched_barrier(0);

    }
else if constexpr (STAGE == 1){
;
}

};

for (; kloop < k_loop_count - 1; kloop++) {
    compute_kloop(std::true_type{});
}
if (k_loop_count > 0) {
    compute_kloop(std::false_type{});
}
    }

template<
  typename scalar_t,
  typename Element, 
  int WARP_NUM,
  int BLOCK_SIZE_M, 
  int BLOCK_SIZE_N, 
  int BLOCK_SIZE_K, 
  int WARP_M, 
  int WARP_N, 
  int WARP_K, 
  int GROUP_N,
  int GROUP_K,
  int STAGES,
  int N_LOOP_NUM = 4,
  bool DIRECT_DWORDX4 = false,
  bool PERSISTENT = false,
  bool LDS_ONLY_REUSE = false>
__global__ void __launch_bounds__(256,1) DEEPGEMM_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_MASKED(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  scalar_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ masked_m,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn, 
  uint32_t stride_bsk,
  uint32_t experts,
  uint32_t blocks_per_expert,
  bool enable_overlap,
  int32_t* signal
) {

  int bidy = 0;
  uint32_t expert_id = 0;
  int block_in_expert = 0;
  int block_m_base = 0;
  int valid_m = 0;
  int scan_task = blockIdx.x;
  int scan_n_blocks = 0;
  int scan_expert = 0;
  int scan_last_expert_end = 0;
  bool static_done = false;
  // signal 槽位按 64 行粒度组织（E * ceil(M/64)）；block_m<64 时
  // 一个槽位聚合 64/block_m 个 m tile 的完成计数。
  const int m_bands_per_expert = DIVIDE(size_m, 64);
  constexpr int kTilesPerBand = 64 / BLOCK_SIZE_M;

  while (true) {
    if constexpr (PERSISTENT) {
      if (scan_n_blocks == 0) {
        scan_n_blocks = DIVIDE(size_n, BLOCK_SIZE_N * N_LOOP_NUM);
      }
      bool found_task = false;
      #pragma unroll 1
      for (; scan_expert < int(experts); scan_expert++) {
        const int cur_valid_m = masked_m[scan_expert];
        const int m_blocks =
            cur_valid_m > 0 ? DIVIDE(cur_valid_m, BLOCK_SIZE_M) : 0;
        const int cur_expert_tiles = m_blocks * scan_n_blocks;
        while (scan_task >= scan_last_expert_end &&
               scan_task < scan_last_expert_end + cur_expert_tiles) {
          const int local_tile = scan_task - scan_last_expert_end;
          block_in_expert = local_tile / scan_n_blocks;
          bidy = local_tile - block_in_expert * scan_n_blocks;
          expert_id = scan_expert;
          block_m_base = block_in_expert * BLOCK_SIZE_M;
          valid_m = cur_valid_m;
          scan_task += int(gridDim.x);
          found_task = true;
          break;
        }
        if (found_task) break;
        scan_last_expert_end += cur_expert_tiles;
      }
      if (!found_task) break;
    } else {
      if (static_done) break;
      static_done = true;
      const int bidx = blockIdx.z;
      bidy = blockIdx.y;
      expert_id = bidx / blocks_per_expert;
      block_in_expert = bidx - expert_id * blocks_per_expert;
      block_m_base = block_in_expert * BLOCK_SIZE_M;
      valid_m = masked_m[expert_id];
    }

  if (block_m_base >= valid_m) return;

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id;
  

  auto g_input = input  ;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[];
  Element* input_lds = (Element*)&(smem);
  Element* qweight_lds = input_lds;

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

  

  

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
    for(int i = 0 ;i< 4;i++){
      int row = block_m_base + idx * mfma_m + col_id + i * 4;
      int token_index_safe = std::min(row, valid_m - 1);
      float input_scale_value = *(input_scale + (expert_id * size_m + token_index_safe) * stride_asm);
      weight_dot_a_scale[idx][i] =   input_scale_value;
  
    }
  }

  

  constexpr  int  n_loop_num = N_LOOP_NUM;

  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num  ;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num ;

  const uint64_t weight_scale_offset = stride_bse  * expert_id + bidy * BLOCK_SIZE_N * n_loop_num ;
  scalar_t* g_output;
  g_output = output + expert_id * size_m * size_n + output_offset;

constexpr int store_size = WARP_M  / 4;
  int32_t row_store[store_size];

  {

  for(int min_tile_m = 0 ; min_tile_m < WARP_M / mfma_m; min_tile_m ++){
  for(int i = 0; i < 4 ; i++){
    int m_idx = min_tile_m * 16 + i * 4 + col_id ;
    int it =  m_idx / 4;

    row_store[it] = block_m_base + m_idx;

  }
  }
  }

  

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
    
  float b_scale_dwordx4[n_loop_num][8];
  constexpr bool defer_b_scale = DIRECT_DWORDX4 && !PERSISTENT &&
                                 std::is_same_v<scalar_t, bhalf_t>;
  auto load_b_scale = [&]() {
    #pragma unroll
    for(int n_loop=0;n_loop < n_loop_num;n_loop++){
      const int scale_base = weight_scale_offset + BLOCK_SIZE_N * n_loop +
                             warp_id * WARP_N + col_id * 8;
      #pragma unroll
      for(int i = 0; i < 8; i++){
        b_scale_dwordx4[n_loop][i] = weight_scale[scale_base + i];
      }
    }
  };
  if constexpr (DIRECT_DWORDX4 && !defer_b_scale) {
    load_b_scale();
  }

   

  

    
    intx4 C_reg[n_loop_num][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};
    __builtin_hcu_sched_barrier(0);
      

      constexpr int SIZE_K = 0;
      gemm_nt_marlin_prefill_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, N_LOOP_NUM, DIRECT_DWORDX4, LDS_ONLY_REUSE>
      (g_input, g_qweight, input_lds, qweight_lds, input_scale, g_weight_scale, size_m, valid_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, expert_id, block_in_expert);
    __builtin_hcu_sched_barrier(0);
    if constexpr (defer_b_scale) {
      // Keep the epilogue scales out of registers during the K loop.
      asm volatile("" ::: "memory");
      load_b_scale();
    }
    

    

  

    const bool full_direct_block = block_m_base + BLOCK_SIZE_M <= valid_m;

    if constexpr (DIRECT_DWORDX4) {
      auto store_direct_row = [&](int n_loop, int min_tile_m, int row) {
        const float a_scale_value =
            input_scale[(expert_id * size_m + row) * stride_asm];
        vec_element_8<scalar_t> packed_store_value;
        const int tile_idx = min_tile_m * (WARP_N / mfma_n);
        #pragma unroll
        for(int reg_id =0; reg_id < 4 ; reg_id ++){
#if defined(__gfx938__)
          if constexpr (WARP_M == 128 && std::is_same_v<scalar_t, bhalf_t>) {
            const float lo = static_cast<float>(C_reg[n_loop][tile_idx][reg_id]) *
                             a_scale_value * b_scale_dwordx4[n_loop][reg_id * 2];
            const float hi = static_cast<float>(C_reg[n_loop][tile_idx + 1][reg_id]) *
                             a_scale_value * b_scale_dwordx4[n_loop][reg_id * 2 + 1];
            auto pair = __builtin_hcu_cvt_pk_bf16_f32(lo, hi, false);
            reinterpret_cast<uint32_t*>(&packed_store_value.data)[reg_id] =
                *reinterpret_cast<uint32_t*>(&pair);
          } else
#endif
          {
          #pragma unroll
          for (int pair_id = 0; pair_id < 2; pair_id++) {
            const int acc = C_reg[n_loop][tile_idx + pair_id][reg_id];
            const float scaled = static_cast<float>(acc) *
                                 a_scale_value *
                                 b_scale_dwordx4[n_loop][reg_id * 2 + pair_id];
            packed_store_value.data[reg_id * 2 + pair_id] =
                b32_to_b16<scalar_t>(scaled);
          }
          }
        }
        const int n_index = BLOCK_SIZE_N * n_loop + warp_id * WARP_N + col_id * 8;
        *(vec_element_8<scalar_t>*)(&g_output[row * size_n + n_index]) =
            packed_store_value;
      };

      if (full_direct_block) {
        #pragma unroll
        for(int n_loop=0;n_loop < n_loop_num;n_loop++){
          #pragma unroll
          for(int min_tile_m =0 ; min_tile_m < (WARP_M / mfma_m) ; min_tile_m ++){
            const int row = block_m_base + min_tile_m * mfma_m + row_id;
            store_direct_row(n_loop, min_tile_m, row);
          }
        }
      } else {
        #pragma unroll
        for(int n_loop=0;n_loop < n_loop_num;n_loop++){
          #pragma unroll
          for(int min_tile_m =0 ; min_tile_m < (WARP_M / mfma_m) ; min_tile_m ++){
            const int row = block_m_base + min_tile_m * mfma_m + row_id;
            if (row < valid_m) {
              store_direct_row(n_loop, min_tile_m, row);
            }
          }
        }
      }
    } else {
      float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id*WARP_N;
      float b_scale[n_loop_num][(WARP_N / mfma_n) ];
      #pragma unroll
      for(int n_loop=0;n_loop < n_loop_num;n_loop++){
        #pragma unroll
        for(int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++){
          #pragma unroll
          for(int i = 0 ;i < 32 / mfma_n ;i++){
            b_scale[n_loop][min_tile_n * 2 + i ] =
                b_scale_ptr[BLOCK_SIZE_N* n_loop + min_tile_n * 32 + i + row_id * 2];
          }
        }
      }
      #pragma unroll
      for(int n_loop=0;n_loop < n_loop_num;n_loop++){
        #pragma unroll
        for(int min_tile_m =0 ; min_tile_m < (WARP_M / mfma_m) ; min_tile_m ++){
          
            
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < (WARP_N / 32) ; min_tile_n ++ ){
                int index =  row_id * 2 + min_tile_n * 32   + warp_id * WARP_N + BLOCK_SIZE_N* n_loop;
                const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2;
                if (full_direct_block) {
                  #pragma unroll
                  for(int reg_id =0; reg_id < 4 ; reg_id ++){
                    int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    scalar_t store_pair[2];
                    #pragma unroll
                    for (int pair_id = 0; pair_id < 2; pair_id++) {
                      const int acc = C_reg[n_loop][tile_idx + pair_id][reg_id];
                      const float scaled = static_cast<float>(acc) *
                                           weight_dot_a_scale[min_tile_m][reg_id] *
                                           b_scale[n_loop][min_tile_n * 2 + pair_id];
                      store_pair[pair_id] = b32_to_b16<scalar_t>(scaled);
                    }
                    *(vec_element_2<scalar_t>*)(&g_output[row_store[it] * size_n + index]) =
                        *(vec_element_2<scalar_t>*)(&store_pair[0]);
                  }
                } else {
                  #pragma unroll
                  for(int reg_id =0; reg_id < 4 ; reg_id ++){
                    int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    bool valid_store = row_store[it] < valid_m;
                    if(valid_store){
                      scalar_t store_pair[2];
                      #pragma unroll
                      for (int pair_id = 0; pair_id < 2; pair_id++) {
                        const int acc = C_reg[n_loop][tile_idx + pair_id][reg_id];
                        const float scaled = static_cast<float>(acc) *
                                             weight_dot_a_scale[min_tile_m][reg_id] *
                                             b_scale[n_loop][min_tile_n * 2 + pair_id];
                        store_pair[pair_id] = b32_to_b16<scalar_t>(scaled);
                      }
                      *(vec_element_2<scalar_t>*)(&g_output[row_store[it] * size_n + index]) =
                          *(vec_element_2<scalar_t>*)(&store_pair[0]);
                    }
                  }
                }
            }
        }
      }
    }

    if (enable_overlap) {
      vmcnt_only_wait(0);
      __syncthreads();
      if (threadIdx.x == 0) {
        constexpr int kNLoopBlockN = BLOCK_SIZE_N * N_LOOP_NUM;
        int* slot = signal + expert_id * m_bands_per_expert
                            + block_in_expert / kTilesPerBand;
        // 该 64 行带内实际存在的 tile 数：除末带外恒为 kTilesPerBand。
        const int expert_tiles = DIVIDE(valid_m, BLOCK_SIZE_M);
        const int band_id = block_in_expert / kTilesPerBand;
        const int band_tiles = (expert_tiles - band_id * kTilesPerBand) < kTilesPerBand
                                   ? (expert_tiles - band_id * kTilesPerBand)
                                   : kTilesPerBand;
        const int old = atomicAdd(slot, 1);
        if (old + 1 == band_tiles * DIVIDE(size_n, kNLoopBlockN)) {
          // 末带不满 64 行（valid_m 非 64 倍数）时补齐差额，使每个活跃槽位
          // 的终值统一为 (64/block_m) * ceil_div(N, block_n)。
          atomicAdd(slot, (kTilesPerBand - band_tiles) * DIVIDE(size_n, kNLoopBlockN));
        }
      }
    }
  }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden, int N_LOOP_NUM, bool DIRECT_DWORDX4, bool PERSISTENT = false, bool LDS_ONLY_REUSE = false>
void launch_w4a8_hipc_masked_prefill_kernel(
    const GemmParams_w4a8<T,T_hidden>& params) {
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    constexpr int n_loop_num = N_LOOP_NUM;
    dim3 blockDim(WARP_NUM * 64, 1, 1);
    dim3 gridDim;
    if constexpr (PERSISTENT) {
      int device = 0;
      hipDeviceProp_t device_properties;
      DG_HIP_CHECK(hipGetDevice(&device));
      DG_HIP_CHECK(hipGetDeviceProperties(&device_properties, device));
      int hcu_cus = device_properties.multiProcessorCount;
      if (const char* env = std::getenv("DEEPGEMM_GPU_CUS")) {
        const int env_cus = std::strtol(env, nullptr, 10);
        if (env_cus > 0) hcu_cus = env_cus;
      }
      int persistent_multiplier =
          params.persistent_multiplier > 0 ? params.persistent_multiplier : 2;
      if (const char* env = std::getenv("DEEPGEMM_W4A8_MASKED_PERSISTENT_MULT")) {
        const int env_multiplier = std::strtol(env, nullptr, 10);
        if (env_multiplier > 0) persistent_multiplier = env_multiplier;
      }
      const int persistent_thread_blocks = hcu_cus * persistent_multiplier;
      gridDim = dim3(persistent_thread_blocks, 1, 1);
    } else {
      gridDim = dim3(1, DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num),
                     params.experts * params.blocks_per_expert);
    }
    const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    DEEPGEMM_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_MASKED<T_hidden,char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
    GROUP_N, GROUP_K, STAGES, N_LOOP_NUM, DIRECT_DWORDX4, PERSISTENT, LDS_ONLY_REUSE><<<gridDim, blockDim, shared_mem_size, stream>>>(
    params.ptr_A,
    params.ptr_B0,
    params.ptr_C,
    params.ptr_A_scale,
    params.ptr_B_scale,
    params.masked_m,
    params.size_m,
    params.size_n,
    params.size_k,
    params.stride_asm,
    params.stride_ask,
    params.stride_bse,
    params.stride_bsn,
    params.stride_bsk,
    params.experts,
    params.blocks_per_expert,
    params.enable_overlap,
    params.signal);
}

namespace deepgemm {

constexpr int kW4A8HipcBlockM = 64;

template <int BLOCK_SIZE_M, int N_LOOP_NUM, typename scalar_t, bool LDS_ONLY_REUSE = false>
void launch_w4a8_hipc_masked_persistent(
    const torch::Tensor& input,
    const torch::Tensor& b_qweight,
    torch::Tensor& output,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    const torch::Tensor& masked_m,
    int experts,
    int size_m,
    int size_n,
    int size_k,
    int persistent_multiplier,
    bool enable_overlap,
    int32_t* signal) {
  const int persistent_blocks_per_expert = DIVIDE(size_m, BLOCK_SIZE_M);
  GemmParams_w4a8<char, scalar_t> persistent_params(
      reinterpret_cast<const char*>(input.data_ptr<int8_t>()),
      reinterpret_cast<const char*>(b_qweight.data_ptr<int8_t>()),
      reinterpret_cast<scalar_t*>(output.data_ptr()),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      masked_m.data_ptr<int32_t>(),
      experts,
      size_m,
      size_n,
      size_k,
      1,
      1,
      b_scale.stride(0),
      b_scale.stride(1),
      b_scale.stride(2),
      persistent_blocks_per_expert,
      persistent_multiplier,
      enable_overlap,
      signal);
  launch_w4a8_hipc_masked_prefill_kernel<
      BLOCK_SIZE_M, 128, 64, BLOCK_SIZE_M, 32, 64, 2,
      char, scalar_t, N_LOOP_NUM, true, true, LDS_ONLY_REUSE>(
          persistent_params);
}

template <typename scalar_t>
std::pair<int, int> launch_w4a8_hipc_masked_prefill(
    const torch::Tensor& input,
    const torch::Tensor& b_qweight,
    torch::Tensor& output,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    const torch::Tensor& masked_m,
    int direct_blocks_per_expert,
    int mode,
    int persistent_multiplier,
    bool enable_overlap,
    int32_t* signal) {
  const int experts = input.size(0);
  const int size_m = input.size(1);
  const int size_k = input.size(2);
  const int size_n = output.size(2);

  GemmParams_w4a8<char, scalar_t> params(
      reinterpret_cast<const char*>(input.data_ptr<int8_t>()),
      reinterpret_cast<const char*>(b_qweight.data_ptr<int8_t>()),
      reinterpret_cast<scalar_t*>(output.data_ptr()),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      masked_m.data_ptr<int32_t>(),
      experts,
      size_m,
      size_n,
      size_k,
      1,
      1,
      b_scale.stride(0),
      b_scale.stride(1),
      b_scale.stride(2),
      direct_blocks_per_expert,
      0,
      enable_overlap,
      signal);

  // 如果block_m小于64
  // 返回{64, (64/block_m) * (size_n / (block_n * n_loop_num))}
  // 每次完成一个`(block_m, block_n×nloop)` tile 的写回显存时，
  // 对signal[e*ceil_div(M/64)+m64]原子加一
  // valid_m 不满 64 行时（末带实际 tile 数少于 64/block_m），内核在末带
  // 数据写完时补齐差额
  // 如果block_m大于等于64
  // 返回{block_m, size_n / (block_n * n_loop_num)}
  // 每次完成一个`(block_m, block_n×nloop)` tile 的写回显存时，
  // 对signal[e*ceil_div(M/block_m)+m]原子加一
  if (mode == 1002 || mode == 1644) {
    launch_w4a8_hipc_masked_prefill_kernel<
        64, 128, 64, 64, 32, 64, 2, char, scalar_t, 4, true, false>(params);
    return {64, DIVIDE(size_n, 128 * 4)};
  } else if (mode == 2282) {
    // A 128-row tile reuses each packed B tile across twice as many rows.
    params.blocks_per_expert = DIVIDE(size_m, 128);
    launch_w4a8_hipc_masked_prefill_kernel<
        128, 128, 64, 128, 32, 64, 2, char, scalar_t, 2, true, false>(params);
    return {64, DIVIDE(size_n, 128 * 2)};
  } else if (mode == 1003 || mode == 1161) {
    launch_w4a8_hipc_masked_persistent<16, 1, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, 4 * DIVIDE(size_n, 128 * 1)};
  } else if (mode == 1016 || mode == 1162) {
    launch_w4a8_hipc_masked_persistent<16, 2, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, 4 * DIVIDE(size_n, 128 * 2)};
  } else if (mode == 1017 || mode == 1164) {
    launch_w4a8_hipc_masked_persistent<16, 4, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, 4 * DIVIDE(size_n, 128 * 4)};
  } else if (mode == 1031 || mode == 1321) {
    launch_w4a8_hipc_masked_persistent<32, 1, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, 2 * DIVIDE(size_n, 128 * 1)};
  } else if (mode == 1032 || mode == 1322) {
    launch_w4a8_hipc_masked_persistent<32, 2, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, 2 * DIVIDE(size_n, 128 * 2)};
  } else if (mode == 1034 || mode == 1324) {
    launch_w4a8_hipc_masked_persistent<32, 4, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, 2 * DIVIDE(size_n, 128 * 4)};
  } else if (mode == 1061 || mode == 1641) {
    launch_w4a8_hipc_masked_persistent<64, 1, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, DIVIDE(size_n, 128 * 1)};
  } else if (mode == 1062 || mode == 1642) {
    launch_w4a8_hipc_masked_persistent<64, 2, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k, persistent_multiplier,
        enable_overlap, signal);
    return {64, DIVIDE(size_n, 128 * 2)};
  } else if (mode == 1070){
    m_grouped_w4a8_gemm_nt_masked_pair32_bm128_bn256_bk128::launch(
            input, b_qweight, a_scale, b_scale, masked_m, output,
            enable_overlap, signal);
    return {128, DIVIDE(size_n, 256)};
  } else {
    TORCH_CHECK(false, mode, " mode error.");
  }
}

std::pair<int, int> launch_m_grouped_w4a8_hipc_masked_fastpath(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    int mode,
    int persistent_multiplier,
    bool enable_overlap,
    int32_t* signal) {
  const int size_m = input.size(1);
  const int blocks_per_expert = DIVIDE(size_m, kW4A8HipcBlockM);

  std::pair<int, int> batch_m_and_done_threshold;
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    batch_m_and_done_threshold = launch_w4a8_hipc_masked_prefill<bhalf_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        blocks_per_expert, mode, persistent_multiplier, enable_overlap, signal);
  } else if (output.scalar_type() == at::ScalarType::Half) {
    batch_m_and_done_threshold = launch_w4a8_hipc_masked_prefill<half_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        blocks_per_expert, mode, persistent_multiplier, enable_overlap, signal);
  } else {
    TORCH_CHECK(false, "HIPC W4A8 fastpath supports fp16/bf16 output only");
  }
  DG_HIP_CHECK(hipGetLastError());
  return batch_m_and_done_threshold;
}

std::tuple<int64_t, int64_t> m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m,
  int expected_m_per_group,
  int mode,
  int persistent_multiplier,
  bool enable_overlap,
  const c10::optional<torch::Tensor>& signal) {

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  TORCH_CHECK(
      mode == 1002 || mode == 1003 || mode == 1016 || mode == 1017 ||
          mode == 1031 || mode == 1032 || mode == 1034 ||
          mode == 1061 || mode == 1062 || mode == 1070 ||
          mode == 1161 || mode == 1162 || mode == 1164 ||
          mode == 1321 || mode == 1322 || mode == 1324 ||
          mode == 1641 || mode == 1642 || mode == 1644 || mode == 2282 ,
      mode, " mode error.");

  int32_t* signal_ptr = nullptr;
  if (enable_overlap) {
    // signal 槽位按 64 行粒度组织：signal[e * ceil(M/64) + m64]。
    // block_m<64 的内核将 64/block_m 个 m tile 聚合进同一槽位，
    // 槽位终值统一为 (64/block_m) * ceil_div(N, block_n)。
    TORCH_CHECK(signal.has_value() && signal->defined() && signal->is_cuda() &&
                    signal->scalar_type() == at::kInt && signal->is_contiguous(),
                "signal must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(signal->numel() >=
                    static_cast<int64_t>(input.size(0)) *
                        ((input.size(1) + 63) / 64),
                "signal must have at least E * ceil(M/64) elements");
    signal_ptr = signal->data_ptr<int32_t>();
  }
  TORCH_CHECK(input.is_cuda() && b_qweight.is_cuda() && output.is_cuda() &&
                  a_scale.is_cuda() && b_scale.is_cuda() && masked_m.is_cuda(),
              "m_grouped_marlin_w4a8_gemm_nt_masked_hipc expects CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Char, "input must be int8");
  TORCH_CHECK(b_qweight.scalar_type() == at::ScalarType::Char,
              "HIPC W4A8 path expects int8 shuffled packed weight");
  TORCH_CHECK(a_scale.scalar_type() == at::ScalarType::Float, "a_scale must be float32");
  TORCH_CHECK(b_scale.scalar_type() == at::ScalarType::Float, "b_scale must be float32");
  TORCH_CHECK(masked_m.scalar_type() == at::ScalarType::Int, "masked_m must be int32");
  TORCH_CHECK(input.dim() == 3, "input must be [E, M, K]");
  TORCH_CHECK(output.dim() == 3, "output must be [E, M, N]");
  TORCH_CHECK(a_scale.dim() == 3 && a_scale.size(2) == 1, "a_scale must be [E, M, 1]");
  TORCH_CHECK(b_scale.dim() == 3 && b_scale.size(2) == 1, "b_scale must be [E, N, 1]");
  TORCH_CHECK(masked_m.dim() == 1, "masked_m must be [E]");
  TORCH_CHECK(input.size(0) == output.size(0) && input.size(0) == b_qweight.size(0) &&
                  input.size(0) == a_scale.size(0) && input.size(0) == b_scale.size(0) &&
                  input.size(0) == masked_m.size(0),
              "expert dimension mismatch");
  TORCH_CHECK(input.size(1) == output.size(1) && input.size(1) == a_scale.size(1),
              "M dimension mismatch");
  TORCH_CHECK(output.size(2) == b_scale.size(1), "N dimension mismatch");
  TORCH_CHECK(input.size(2) % 64 == 0, "K must be divisible by 64 for current W4A8 packed layout");
  TORCH_CHECK(output.size(2) % 16 == 0, "N must be divisible by 16 for current W4A8 packed layout");
  TORCH_CHECK(
      b_qweight.numel() == input.size(0) * output.size(2) * input.size(2) / 2,
      "unexpected HIPC W4A8 b_qweight size");
  TORCH_CHECK(input.is_contiguous() && b_qweight.is_contiguous() && output.is_contiguous() &&
                  a_scale.is_contiguous() && b_scale.is_contiguous() && masked_m.is_contiguous(),
              "all tensors must be contiguous");

  const auto batch_m_and_done_threshold = launch_m_grouped_w4a8_hipc_masked_fastpath(
      input, b_qweight, output, a_scale, b_scale, masked_m,
      mode, persistent_multiplier, enable_overlap, signal_ptr);
  DG_HIP_CHECK(hipGetLastError());
  return std::make_tuple(
      batch_m_and_done_threshold.first,
      batch_m_and_done_threshold.second);
}



}

}  // namespace gfx92a
