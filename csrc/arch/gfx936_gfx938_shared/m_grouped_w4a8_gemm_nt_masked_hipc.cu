#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "m_grouped_gemm_nt_masked.h"
#include "hipc_w4a8/w4a8_intrinsics.h"

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
               uint32_t blocks_per_expert
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
          blocks_per_expert(blocks_per_expert)
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
    bool ABC_ORDER = false>
__forceinline__ __device__ void  gemm_nt_marlin_prefill_w4a8(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset, 
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
    const int lds_stage_offset = WARP_M * WARP_K;

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
            int row = std::min(s_offset + col_id, max_n_len_offset - 1);
            g_row_A[m_tile] = (expert_id * max_n_len_offset + row) * size_k + row_id * 4;
            

    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K ) + kloop * (128);
    
    int i =0;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 

                inline_buffer_load_dword_lds( A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K ) / 4 +  (0) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4 , (g_row_A[m_tile] )/4);
            

        }
    }
    

    i = 1;

    
    k_start += stage_offset;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 

                inline_buffer_load_dword_lds(  A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 +  (1) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4, (g_row_A[m_tile] )/4);

        }
    }

for(kloop  ;kloop < k_loop_count ;kloop++){

    __builtin_hcu_sched_barrier(0);
    vmcnt_wait(0);

    int k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (128 / 2) * size_n;
    

    
    

    
    

        

        

    

    

   

    

    
    union {
        vec<int, 4> v;
        uint32_t i[4];
    } weight_reg_tmp[ WARP_N / 32][2][STAGE];
    

if constexpr (STAGE == 2){

const Element* cur_weight_ptr = weight_ptr ;

int i =0;        
int stage_b_flag =0;
int n_loop =0;

__builtin_hcu_sched_barrier(0);
#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ] );
        
        }

}

__builtin_hcu_sched_barrier(0);

i = 1;

k_start_b += stage_offset_b;

      
__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile ]   );
            
        }

}
__builtin_hcu_sched_barrier(0);

vmcnt_wait(2 *( WARP_N / 32 ) );

	    #pragma unroll
	    for(int i =0;i<2;i++){
	        #pragma unroll
	        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
	            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                int s_index = m_tile * (16) * 64 + + (i) * WARP_M / 16 * (16) * WARP_K;
                A_reg[m_tile][i].int4_array[k_tile] = *( vec<Element, 16> *) (&A_lds[ s_index + row_id * 64 + col_id * 16      ]);

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

__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ]);
        

        }

}
__builtin_hcu_sched_barrier(0);

i = 1;

k_start_b += stage_offset_b;
        
__builtin_hcu_sched_barrier(0);
#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
            

        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ] );
        
        }

}
__builtin_hcu_sched_barrier(0);

   

__builtin_hcu_sched_barrier(0);
vmcnt_only_wait( 3* ( WARP_N / 32 ));

    
n_loop--;
i = 0;
stage_b_flag ^= 1;

__builtin_hcu_sched_barrier(0);

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

__builtin_hcu_sched_barrier(0);

vmcnt_only_wait( 2* ( WARP_N / 32 ));

i = 1;

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

    
__builtin_hcu_sched_barrier(0);
stage_b_flag ^= 1;
n_loop ++;

k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (WARP_K * 2 / 2) * size_n;
   
}

    n_loop--;
    wave_barrier();
    if(kloop <  k_loop_count - 1 ) {

        kloop++;
        k_start = warp_k_id * (WARP_K ) + kloop * (128);

        i =0;

        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 

                    inline_buffer_load_dword_lds( A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K ) / 4 +  (0) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4 , (g_row_A[m_tile] )/4);
                    

            }
        }
        

        i = 1;

        
        k_start += stage_offset;

        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 

                    inline_buffer_load_dword_lds(  A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 +  (1) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4, (g_row_A[m_tile] )/4);

            }
        }
        kloop --;
    }
    
    

    vmcnt_only_wait(  ( WARP_N / 32 ) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)) );

if(kloop == k_loop_count - 1 ){
    vmcnt_wait(  ( WARP_N / 32 ) );
}

i = 0;

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

__builtin_hcu_sched_barrier(0);

    vmcnt_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));

    if(kloop == k_loop_count - 1 ){
        vmcnt_wait(  0 );
    }
    

i = 1;

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
__builtin_hcu_sched_barrier(0);

__builtin_hcu_sched_barrier(0);

    }
else if constexpr (STAGE == 1){
;
}

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
  bool PERSISTENT = false>
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
  uint32_t blocks_per_expert
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
  if constexpr (DIRECT_DWORDX4) {
    #pragma unroll
    for(int n_loop=0;n_loop < n_loop_num;n_loop++){
      const int scale_base = weight_scale_offset + BLOCK_SIZE_N * n_loop +
                             warp_id * WARP_N + col_id * 8;
      #pragma unroll
      for(int i = 0; i < 8; i++){
        b_scale_dwordx4[n_loop][i] = weight_scale[scale_base + i];
      }
    }
  }

   

  

    
    intx4 C_reg[n_loop_num][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};
    __builtin_hcu_sched_barrier(0);
      

      constexpr int SIZE_K = 0;
      gemm_nt_marlin_prefill_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, N_LOOP_NUM, DIRECT_DWORDX4>
      (g_input, g_qweight, input_lds, qweight_lds, input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, expert_id, block_in_expert);
    __builtin_hcu_sched_barrier(0);
    

    

  

    const bool full_direct_block = block_m_base + BLOCK_SIZE_M <= valid_m;

    if constexpr (DIRECT_DWORDX4) {
      auto store_direct_row = [&](int n_loop, int min_tile_m, int row) {
        const float a_scale_value =
            input_scale[(expert_id * size_m + row) * stride_asm];
        vec_element_8<scalar_t> packed_store_value;
        const int tile_idx = min_tile_m * (WARP_N / mfma_n);
        #pragma unroll
        for(int reg_id =0; reg_id < 4 ; reg_id ++){
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
  }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden, int N_LOOP_NUM, bool DIRECT_DWORDX4, bool PERSISTENT = false>
void launch_w4a8_hipc_masked_prefill_kernel(
    const GemmParams_w4a8<T,T_hidden>& params) {
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    constexpr int n_loop_num = N_LOOP_NUM;
    dim3 blockDim(WARP_NUM * 64, 1, 1);
    dim3 gridDim;
    if constexpr (PERSISTENT) {
      const int n_blocks = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
      int device = 0;
      hipDeviceProp_t device_properties;
      DG_HIP_CHECK(hipGetDevice(&device));
      DG_HIP_CHECK(hipGetDeviceProperties(&device_properties, device));
      int hcu_cus = device_properties.multiProcessorCount;
      if (const char* env = std::getenv("DEEPGEMM_GPU_CUS")) {
        const int env_cus = std::strtol(env, nullptr, 10);
        if (env_cus > 0) hcu_cus = env_cus;
      }
      int persistent_multiplier = 2;
      if (const char* env = std::getenv("DEEPGEMM_W4A8_MASKED_PERSISTENT_MULT")) {
        const int env_multiplier = std::strtol(env, nullptr, 10);
        if (env_multiplier > 0) persistent_multiplier = env_multiplier;
      }
      const int persistent_thread_blocks =
          std::lcm(n_blocks, hcu_cus) * persistent_multiplier;
      gridDim = dim3(persistent_thread_blocks, 1, 1);
    } else {
      gridDim = dim3(1, DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num),
                     params.experts * params.blocks_per_expert);
    }
    const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    DEEPGEMM_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_MASKED<T_hidden,char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
    GROUP_N, GROUP_K, STAGES, N_LOOP_NUM, DIRECT_DWORDX4, PERSISTENT><<<gridDim, blockDim, shared_mem_size, stream>>>(
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
    params.blocks_per_expert);
}

namespace deepgemm {

constexpr int kW4A8HipcBlockM = 64;

template <int BLOCK_SIZE_M, int N_LOOP_NUM, typename scalar_t>
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
    int size_k) {
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
      persistent_blocks_per_expert);
  launch_w4a8_hipc_masked_prefill_kernel<
      BLOCK_SIZE_M, 128, 64, BLOCK_SIZE_M, 32, 64, 2,
      char, scalar_t, N_LOOP_NUM, true, true>(
          persistent_params);
}

template <typename scalar_t>
void launch_w4a8_hipc_masked_prefill(
    const torch::Tensor& input,
    const torch::Tensor& b_qweight,
    torch::Tensor& output,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    const torch::Tensor& masked_m,
    int direct_blocks_per_expert,
    int mode) {
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
      direct_blocks_per_expert);

  if (mode == 1002 || mode == 1644) {
    launch_w4a8_hipc_masked_prefill_kernel<
        64, 128, 64, 64, 32, 64, 2, char, scalar_t, 4, true, false>(params);
  } else if (mode == 1003 || mode == 1161) {
    launch_w4a8_hipc_masked_persistent<16, 1, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else if (mode == 1016 || mode == 1162) {
    launch_w4a8_hipc_masked_persistent<16, 2, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else if (mode == 1017 || mode == 1164) {
    launch_w4a8_hipc_masked_persistent<16, 4, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else if (mode == 1031 || mode == 1321) {
    launch_w4a8_hipc_masked_persistent<32, 1, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else if (mode == 1032 || mode == 1322) {
    launch_w4a8_hipc_masked_persistent<32, 2, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else if (mode == 1034 || mode == 1324) {
    launch_w4a8_hipc_masked_persistent<32, 4, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else if (mode == 1061 || mode == 1641) {
    launch_w4a8_hipc_masked_persistent<64, 1, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else if (mode == 1062 || mode == 1642) {
    launch_w4a8_hipc_masked_persistent<64, 2, scalar_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        experts, size_m, size_n, size_k);
  } else {
    TORCH_CHECK(false, mode, " mode error.");
  }
}

void launch_m_grouped_w4a8_hipc_masked_fastpath(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor masked_m,
    int mode) {
  const int size_m = input.size(1);
  const int blocks_per_expert = DIVIDE(size_m, kW4A8HipcBlockM);

  if (output.scalar_type() == at::ScalarType::BFloat16) {
    launch_w4a8_hipc_masked_prefill<bhalf_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        blocks_per_expert, mode);
  } else if (output.scalar_type() == at::ScalarType::Half) {
    launch_w4a8_hipc_masked_prefill<half_t>(
        input, b_qweight, output, a_scale, b_scale, masked_m,
        blocks_per_expert, mode);
  } else {
    TORCH_CHECK(false, "HIPC W4A8 fastpath supports fp16/bf16 output only");
  }
  DG_HIP_CHECK(hipGetLastError());
}

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_masked_hipc(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m,
  int expected_m_per_group,
  int mode) {

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  TORCH_CHECK(
      mode == 1002 || mode == 1003 || mode == 1016 || mode == 1017 ||
          mode == 1031 || mode == 1032 || mode == 1034 ||
          mode == 1061 || mode == 1062 ||
          mode == 1161 || mode == 1162 || mode == 1164 ||
          mode == 1321 || mode == 1322 || mode == 1324 ||
          mode == 1641 || mode == 1642 || mode == 1644,
      mode, " mode error.");
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

  launch_m_grouped_w4a8_hipc_masked_fastpath(
      input, b_qweight, output, a_scale, b_scale, masked_m,
      mode);
  DG_HIP_CHECK(hipGetLastError());
  return output;
}



}
