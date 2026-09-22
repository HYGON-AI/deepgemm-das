#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <vector>

#include "hip_common.h"
#include "m_grouped_gemm_nt_masked.h"

#include "hipc_w16a16/w16a16_utils.h"

#define DEBUG 0

using half_t = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))




/* module function args */
template <typename T>
struct HipFunctionArgs {
  uint32_t numWorkGroups0;
  uint32_t numWorkGroups1;
  T* ptr_C;                          // output
  const T* ptr_A;              // input
  const T* ptr_B;               // weight
  float* ptr_A_scale;                // input scale
  float* ptr_B_scale;                // weight scale
  // const float* topk_weights;         // topk weights
  // const int32_t* expert_offsets;     // null
  const int32_t* masked_m;       // tokens per expert
  uint32_t experts_num;              // num of experts
  uint32_t size_m;                   // input size m
  uint32_t size_n;                   // output size n
  uint32_t size_k;                   // input size k
  uint32_t stride_ase;               // weight scale stride expert
  uint32_t stride_asm;               // input scale stride m
  uint32_t stride_ask;               // input scale stride k
  uint32_t stride_bse;               // weight scale stride expert
  uint32_t stride_bsn;               // weight scale stride n
  uint32_t stride_bsk;               // weight scale stride k
  uint32_t numFullBlocks;
  uint32_t wgmRemainder1;
  uint32_t magicNumberWgmRemainder1;
  void *debugBuffer;
};

namespace deepgemm {

#define buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, N_row_len, BLOCK_M, WARP_K, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id,\
                                                vec_size, per_phase, max_phase)\
{\
    int bytes_per_Element = 2;\
    const int bytes_num_per_dword = 4;\
    const int bytes_num_per_dwordx2 = 8;\
    int thread_num_n = WARP_K*bytes_per_Element/bytes_num_per_dwordx2;\
    int thread_num_m = 64/thread_num_n;\
    int lane_M_idx = lane_id / thread_num_n;\
    int lane_N_idx = lane_id % thread_num_n;\
    const int lds_load_num = (BLOCK_M*WARP_K*bytes_per_Element) / (bytes_num_per_dwordx2*64);\
    for(int load = 0,warp_loop = warp_id; load < std::max(1,lds_load_num/WARP_NUM); warp_loop += WARP_NUM, ++load) {\
        int block_row = warp_loop * thread_num_m + lane_M_idx;\
        int sorted_token_idx = 0;\
        int g_row = 0;\
        if(sorted_token_ids_offset == nullptr) {\
            sorted_token_idx = bidx * BLOCK_M + block_row;\
            g_row = std::min(sorted_token_idx, max_n_len_offset - 1);\
        }\
        else{\
            sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + block_row, int(sorted_token_lens - 1))];\
            g_row = w16a16_ck_token_clamped(sorted_token_idx, max_n_len_offset);\
        }\
        int phase = (block_row/per_phase) % max_phase;\
        int g_col = lane_N_idx;\
        int vec_size_dword = (vec_size * bytes_per_Element) / bytes_num_per_dword;\
        int g_col_swizzle = (g_col ^ phase) * vec_size_dword;\
        int gsOffset = global_offset;\
        int gvOffset = g_row * N_row_len/bytes_num_per_dword + g_col_swizzle;\
        int lds_offset = lds_stage_offset/bytes_per_Element + warp_loop * 64 * 4;\
        buffer_load_lds_dwordx2(global_ptr, lds_ptr, lds_offset, (gvOffset*4), gsOffset);\
    }\
}


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
[&] {                                         \
  if (COND) {                                 \
    constexpr static bool CONST_NAME = true;  \
    return __VA_ARGS__();                     \
  } else {                                    \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }                                           \
}()

#define DG_HIP_CHECK(call)                                                   \
  do {                                                                       \
    hipError_t err = (call);                                                  \
    TORCH_CHECK(                                                              \
        err == hipSuccess, #call " failed: ", hipGetErrorString(err));       \
  } while (0)


template <typename T>
struct GemmParams3 {
    GemmParams3(const T* ptr_A,
                const T* ptr_B0,
                T* ptr_C,
                const float* topk_weights,
                const int32_t* sorted_token_ids,
                const int32_t* expert_ids,
                const int32_t num_tokens_post_pad,
                const int32_t* num_tokens_post_pad_ptr,
                uint32_t size_m,
                uint32_t size_n,
                uint32_t size_k,
                uint32_t size_kofb,
                uint32_t sorted_token_lens,
                uint32_t top_k,
                uint32_t delta,
                uint32_t expert_num,
                uint32_t expected_m_per_group)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          size_kofb(size_kofb),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          delta(delta),
          expert_num(expert_num),
          expected_m_per_group(expected_m_per_group) {}

    const T* ptr_A;
    const T* ptr_B0;
    T* ptr_C;
    const float* topk_weights;
    const int32_t* sorted_token_ids;
    const int32_t* expert_ids;
    const int32_t num_tokens_post_pad;
    const int32_t* num_tokens_post_pad_ptr;
    uint32_t size_m;
    uint32_t size_n;
    uint32_t size_k;
    uint32_t size_kofb;
    uint32_t sorted_token_lens;
    uint32_t top_k;
    uint32_t delta;
    uint32_t expert_num;
    uint32_t expected_m_per_group;
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
    int STAGES,
    typename Element,
    typename ElementAccum = int32_t>
    __forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel(
    const Element* input_ptr,
    const Element* weight_ptr,
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset,
    w16_union_vec<Element,4> A_reg[][WARP_K/16],
    w16_union_vec<Element,8> B_reg[][WARP_K/16],
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id,
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k,
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;

    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_stride = marlin_warp_k * marlin_warp_n * bytes_per_element/*bytes*/;

    const int warp_n_num = BLOCK_N / WARP_N; //1
    int warp_n_id = warp_id % warp_n_num;//0
    int warp_m_id = warp_id / warp_n_num;//0
    int weight_vec_size = 8;

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)];
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)];

    typename w16_vec<Element,4>::type *B_lds_v4bf16 = (typename w16_vec<Element,4>::type *)(B_lds);
    typename w16_vec<Element,4>::type *A_lds_v4bf16 = (typename w16_vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //4
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //4
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element;
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element;
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = (input_ptr + 0);
        auto B_ptr = (weight_ptr + 0);

        #pragma unroll
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element;
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset;
                buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * (WARP_K/marlin_warp_k) * seqlen_B_stride;
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset;
                buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop,
                                                            marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
            }
        }
    }

    // 构建A矩阵读取索引
    #pragma unroll
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_row = (lane_id & 15);
            int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16;

            int vec_size_dword = vec_size/2/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/2 + (min_tile_m * MFMA_M + lds_row)*WARP_K/2/*dword*/ + col_swizzle/*dword*/;
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引,完成一个warp读取WARP_NxWARP_K个元素
    #pragma unroll
    for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/marlin_warp_k); min_tile_k++) {
            int warp_offset = lane_id * weight_vec_size * bytes_per_element/*bytes*/;
            int lds_offset = warp_n_id * (WARP_N/marlin_warp_n) * (WARP_K/marlin_warp_k) * marlin_warp_stride +
                                min_tile_n * (WARP_K/marlin_warp_k) * marlin_warp_stride + min_tile_k * marlin_warp_stride + warp_offset;
            precompute_B_lds_offset[min_tile_n*(WARP_K/marlin_warp_k) + min_tile_k] = lds_offset;
        }
    }

    //main loop 2048/16 = 128次迭代
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        int kb_offset = k * (WARP_K/marlin_warp_k) * seqlen_B_k;
        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息

        {
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element;
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset;
            buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * (WARP_K/marlin_warp_k) * seqlen_B_stride;
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset;
            buffer_load_lds_tile_pad_weight_marlin(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset,
                                                        B_lds_stage_offset_loop, marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k);
        }

        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }

            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            }
        }

        // lgkmcnt_wait_barrier(0);
        lgkmcnt_wait(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename w16_vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]),
                            *(typename w16_vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]),
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
                // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            }
        }

        // lgkmcnt_wait_barrier(0);
        lgkmcnt_wait(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename w16_vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]),
                            *(typename w16_vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]),
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    // 为了最后一次等到vmcnt_wait=0
    vmcnt_wait(0);
    {
        stage_id_reg_A = ((size_k/WARP_K-1))%STAGES;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
            // 加载A矩阵
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            // 加载B矩阵
            ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
        }
    }

    // lgkmcnt_wait_barrier(0);
    lgkmcnt_wait(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {
                #pragma unroll
                for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                    C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                        *(typename w16_vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]),
                        *(typename w16_vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]),
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);
                }
            }
        }
    }
    lgkmcnt_wait_barrier(0);
}


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
    int STAGES,
    typename Element,
    typename ElementAccum = int32_t>
__forceinline__ __device__ void  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_remainder(
    const Element* input_ptr,
    const Element* weight_ptr,
    Element* A_lds,
    Element* B_lds,
    int max_n_len_offset,
    w16_union_vec<Element,4> A_reg[][WARP_K/16],
    w16_union_vec<Element,8> B_reg[][WARP_K/16],
    floatx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id,
    int seqlen_A_k,  //2048
    int seqlen_B_k,  //2048
    int top_k,
    const int* sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int bidy) {

    const int size_k = seqlen_A_k;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    constexpr int bytes_per_element = 2;
    constexpr int bytes_per_dword = 4;
    const int seqlen_A_stride = seqlen_A_k * bytes_per_element;
    const int seqlen_B_stride = seqlen_B_k * bytes_per_element;

    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_stride = marlin_warp_k * marlin_warp_n * bytes_per_element/*bytes*/;

    const int warp_n_num = BLOCK_N / WARP_N; //1
    int warp_n_id = warp_id % warp_n_num;//0
    int warp_m_id = warp_id / warp_n_num;//0
    int weight_vec_size = 8;
    int warp_n_id_1 = warp_n_id;
    if((bidy * (BLOCK_N* marlin_warp_k)) + warp_n_id *marlin_warp_k * marlin_warp_n >= seqlen_B_k){
        warp_n_id_1 =0;
    }
    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)];
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)];

    typename w16_vec<Element,4>::type *B_lds_v4bf16 = (typename w16_vec<Element,4>::type *)(B_lds);
    typename w16_vec<Element,4>::type *A_lds_v4bf16 = (typename w16_vec<Element,4>::type *)(A_lds);

    const int type_width_in_bit = 16;
    const int num_banks = 32;
    const int bank_bit_width = 32;
    const int elems_per_one_banks_row = (num_banks * bank_bit_width) / type_width_in_bit;
    const int simd_width = 16;
    const int inner_dim_length = WARP_K;
    const int vec_size = 4;
    const int per_phase = std::max(1, elems_per_one_banks_row / inner_dim_length); //4
    const int max_phase = std::min(simd_width / per_phase, inner_dim_length / vec_size); //4
    // const int max_phase = 32 / (vec_size * type_width_in_bit / bank_bit_width); // 32
    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K * bytes_per_element;
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K * bytes_per_element;
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = (input_ptr + 0);
        auto B_ptr = (weight_ptr + 0);

        #pragma unroll
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(A_prefetch_level == 0) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K * bytes_per_element;
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset;
                buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                            vec_size, per_phase, max_phase);
            }

            if(B_prefetch_level == 0) { // 预加载B矩阵
                int B_block_buffer_load_global_offset = stage_id_num * (WARP_K/marlin_warp_k) * seqlen_B_stride;
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset;
                buffer_load_lds_tile_pad_weight_marlin_remiander(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop,
                                                            marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k,warp_n_id_1);
            }
        }
    }

    // 构建A矩阵读取索引
    #pragma unroll
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            int lds_row = (lane_id & 15);
            int lds_col = (min_tile_k * MFMA_K) / 4 + lane_id/16;

            int vec_size_dword = vec_size/2/*dword*/;
            int phase = (lds_row/per_phase) % max_phase;
            int col_swizzle = (lds_col ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/2 + (min_tile_m * MFMA_M + lds_row)*WARP_K/2/*dword*/ + col_swizzle/*dword*/;
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引,完成一个warp读取WARP_NxWARP_K个元素
    #pragma unroll
    for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/marlin_warp_k); min_tile_k++) {
            int warp_offset = lane_id * weight_vec_size * bytes_per_element/*bytes*/;
            int lds_offset = warp_n_id * (WARP_N/marlin_warp_n) * (WARP_K/marlin_warp_k) * marlin_warp_stride +
                                min_tile_n * (WARP_K/marlin_warp_k) * marlin_warp_stride + min_tile_k * marlin_warp_stride + warp_offset;
            precompute_B_lds_offset[min_tile_n*(WARP_K/marlin_warp_k) + min_tile_k] = lds_offset;
        }
    }

    //main loop 2048/16 = 128次迭代
    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        int kb_offset = k * (WARP_K/marlin_warp_k) * seqlen_B_k;
        auto A_ptr = (input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = (weight_ptr + kb_offset); // 配置全局显存信息
        {
            // 预加载A矩阵
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K * bytes_per_element;
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset;
            buffer_load_lds_tile_pad_sorted_token_dwordx2(WARP_NUM, seqlen_A_stride, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size, per_phase, max_phase);

            // 预加载B矩阵
            int B_block_buffer_load_global_offset = (STAGES - 1) * (WARP_K/marlin_warp_k) * seqlen_B_stride;
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset;
            buffer_load_lds_tile_pad_weight_marlin_remiander(WARP_NUM, seqlen_B_stride, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset,
                                                        B_lds_stage_offset_loop, marlin_warp_stride, warp_id, lane_id, marlin_warp_n, marlin_warp_k,warp_n_id_1);

        }

        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }

            { // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            }
        }

        // lgkmcnt_wait_barrier(0);
        lgkmcnt_wait(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N / marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename w16_vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]),
                            *(typename w16_vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]),
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait((std::max(1, (BLOCK_N * WARP_K) / (8*64)/WARP_NUM) + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
                // 加载B矩阵
                ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
            }
        }

        // lgkmcnt_wait_barrier(0);
        lgkmcnt_wait(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {
                    #pragma unroll
                    for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                            *(typename w16_vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]),
                            *(typename w16_vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]),
                            C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }
    vmcnt_wait(0);
    {
        stage_id_reg_A = ((size_k/WARP_K-1))%STAGES;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){
            // 加载A矩阵
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v4bf16, precompute_A_lds_offset, A_lds_stage_offset_loop/bytes_per_dword, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            // 加载B矩阵
            ds_read_b128_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v4bf16, precompute_B_lds_offset, B_lds_stage_offset_loop, B_reg);
        }
    }

    // lgkmcnt_wait_barrier(0);
    lgkmcnt_wait(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/marlin_warp_n); min_tile_n++) {
                #pragma unroll
                for(int n_tile=0; n_tile<marlin_warp_n/MFMA_N; n_tile++) {
                    C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile] = mmac<Element>(
                        *(typename w16_vec<Element, 4>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]),
                        *(typename w16_vec<Element, 4>::type*)(&B_reg[min_tile_n][min_tile_k].int2_array[n_tile]),
                        C_reg[0][min_tile_m*(WARP_N/marlin_warp_n) * (marlin_warp_n/MFMA_N) + min_tile_n * (marlin_warp_n/MFMA_N) + n_tile]);
                }
            }
        }
    }

    __syncthreads();
}

// <32, 128, 128, 16, 32, 64, scalar_t>
template<
  typename Element,
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M,
  uint16_t BLOCK_SIZE_N,
  uint16_t BLOCK_SIZE_K,
  uint16_t WARP_M,
  uint16_t WARP_N,
  uint16_t WARP_K,
  bool mul_topk_weight,
  bool tail_N_process>
__global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP(
  const Element* __restrict__ input, // deepgemm: EMK ; 原意：MK
  const Element* __restrict__ qweight,
  Element* __restrict__ output,  // 写回 flat top-k 顺序 [M * top_k, N]
  const float* __restrict__ topk_weights, // nullptr-->没用到
  const int32_t* __restrict__ sorted_token_ids, // nullptr
  const int32_t* __restrict__ expert_ids, // masked_m 原意：每个 M tile 对应哪个 expert
  const int32_t* __restrict__ num_tokens_post_pad, // nullptr-->没用到，判断每个专家接受的token数量最大值
  uint32_t size_m, // sorted_token_lens=M*top_k
  uint32_t size_n,
  uint32_t size_k,
  uint32_t size_kb, // 原意：每个专家的weight的K方向大小，单位个数
  uint32_t sorted_token_lens, // 重要 被用作了direct_blocks_per_expert，重要得改buffer_load_lds_tile_pad_sorted_token_dwordx2中有使用，有用到 原意：M*top_k (+ padding)
  uint32_t top_k, // 1 重要-带topk的地方得改 原意：每个token对应的专家数量
  uint32_t delta, // 1 一个expert对应的block数量
  uint32_t expert_num,
  uint32_t expected_m_per_group
) {
    // 原aiter-w16a16代码理解:
    // input: [m, size_k]，每个元素是一个 token 的 embedding。
    // sorted_token_ids: MoE token 按照专家重排后的 token_id, 维度为 [m*topk]，每个元素是包含 token_id 信息，route_id信息。
    // sorted_token_id: 
    // 低24位表示tokeninput中的位置， 范围0~m-1
    // 高8位表示route_id， 范围0~topk-1
    // weight的权重原本ENK，在经过shuffle之后，形状变成了 EKbN


    // ===== 1. CTA 坐标解析：定位当前 block 负责的 M/N/K 分块 =====
    // grid 维度映射：blockIdx.z 对应 M 方向（专家内 token block），blockIdx.y 对应 N 方向，blockIdx.x 预留给 K 方向。
    const int bidx = blockIdx.z; // m方向，分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    const int bidy = blockIdx.y; // n方向
    const int bidz = blockIdx.x;

    // ===== 2. Masked/direct 调度信息解析：确定专家、专家内 block 以及有效 token 数 =====
    // direct_masked 模式下 sorted_token_ids 为空，expert_ids 实际传入 masked_m，用 bidx 直接映射到 expert/block_m。
    // 非 direct 模式下沿用 sorted_token_ids/expert_ids 的 MoE token 重排路径。
    // 我的修改
    const bool direct_masked = sorted_token_ids == nullptr;
    const int32_t* masked_m = expert_ids;
    // 方案一
    // const int direct_blocks_per_expert = DIVIDE(size_m, BLOCK_SIZE_M);
    // 方案二
    const int direct_blocks_per_expert = DIVIDE(expected_m_per_group, BLOCK_SIZE_M); 
    const int blocks_per_expert = direct_masked ? direct_blocks_per_expert : 0;
    const int expert_id = direct_masked ? (bidx / blocks_per_expert) : expert_ids[bidx];
    const int block_in_expert = direct_masked ? (bidx - expert_id * blocks_per_expert) : bidx;
    // 方案三
    // const int expert_id = direct_masked ? expert_ids_of_blk[bidx] : expert_ids[bidx];
    // const int block_in_expert = direct_masked ? blk_ids_in_expert[bidx] : bidx;

    const int block_m_base = direct_masked ? block_in_expert * BLOCK_SIZE_M : 0; //当前block在专家内的M方向起始位置
    const int valid_m = direct_masked ? masked_m[expert_id] : 0;


    // ===== 3. Marlin packed weight 形状常量：描述 qweight 中一个 16x32 子 tile 的布局 =====
    const int bytes_per_element = 2;
    const int marlin_warp_k = 16;// marlin weight tile: 16x32
    const int marlin_warp_n = 32;
    const int marlin_warp_size = marlin_warp_k * marlin_warp_n;
    const int marlin_stride_b = size_n * bytes_per_element;
    const int real_size_n = (size_n * size_kb) / size_k;

    // ===== 4. CTA 有效性过滤：跳过超出 masked_m 或 token 重排范围的无效 M block =====
    // 我的修改
    if (direct_masked) {
        if (block_m_base >= valid_m) return; // 该blk被掩码全掩住，就不计算，直接返回
    } else {
        if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回
        if (!w16a16_ck_valid_up(sorted_token_ids[bidx * BLOCK_SIZE_M], size_m, top_k)) return; // 对于无效的block,直接返回
    }

    if(expert_id >=expert_num || expert_id <0) return;

    // ===== 5. 专家权重定位：按 expert_id 与 N 方向 block 定位到当前 CTA 需要的 qweight tile =====
    constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 

    const uint64_t expert_offset = ((uint64_t)size_n) * size_kb * expert_id; // weight的专家偏移
    const uint64_t qweight_offset = expert_offset + bidy * (BLOCK_SIZE_N/marlin_warp_n) * marlin_warp_size ; // bidy:0~real_n/BLOCK_SIZE_N-1

    const uint32_t output_offset = bidy *  BLOCK_SIZE_N; // 输出n方向偏移
    auto g_qweight = qweight + qweight_offset;
    auto g_input = direct_masked ? input + ((uint64_t)expert_id) * size_m * size_k : input;

    // ===== 6. LDS 切分：计算阶段放 A/B tile，写回阶段复用同一片 LDS 暂存 C tile =====
    Element* g_output;
    extern __shared__ uint8_t smem[]; // 声明lds信息
    Element* input_lds = (Element*)&(smem); // 输入lds首地址,首地址开始分配给input
    Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // input + blk_m * blk_k
    uint16_t* output_lds = (uint16_t*)&(smem); // 重复使用lds,留给output_lds

    g_output = direct_masked ? output + ((uint64_t)expert_id) * size_m * real_size_n + output_offset
                              : output + output_offset; // 专家偏移+n方向偏移 （每个blk负责blk_m条token）

    // ===== 7. MFMA/线程拓扑：将 1024-thread CTA 拆成 wave64，并计算 lane 在 M/N 写回中的职责 =====
    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 16;

    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_hcu_readfirstlane(warp_id_vec); // 让整个wave中所有线程读取lane0的值
    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;  // 一个warp中16行4列lane时的行索引 行主序
    int col_id = lane_id / 16; // 一个warp中16行4列lane时的行索引 行主序

    const int warp_n_num = BLOCK_SIZE_N / WARP_N; // 一个blk在n方向warp数量

    int warp_n_id = warp_id % warp_n_num; // blk中warp在n方向的索引
    int warp_m_id = warp_id / warp_n_num; // blk中warp在m方向的索引
    constexpr int dwordx4_num_elements = 8; // 128bit向量load，dwordx4 8个bf16元素
    constexpr int N_thread = BLOCK_SIZE_N / dwordx4_num_elements; // 原码 默认N_thread<=64，即BLOCK_SIZE_N<=512
    // const int m_num_threads = std::max(64 / N_thread, 1);
    const int m_num_threads = 64 / N_thread; // 一个warp中m方向的线程数
    const int n_num_threads = 64 / m_num_threads; // 一个warp中n方向的线程数
    const int m_thread_idx = lane_id / n_num_threads;   // lane在m方向的线程索引
    const int n_thread_idx = lane_id & (n_num_threads-1); // lane在n方向的线程索引 lane n主序

    // ===== 8. 寄存器与 token 索引准备：为 A/B/C 寄存器 tile 分配空间，并预先确定每组 M 行对应的输出 token =====
    const int *sorted_token_ids_offset = sorted_token_ids; // 原aiter用于计算token_id和route_id的偏移量


    // 以下寄存器是每个warp在计算一个blk_k中用到的寄存器，blk_k中有k方向pipeline，因此申请*STAGES个寄存器
    w16_union_vec<Element,4> A_reg[WARP_M/mfma_m * STAGES][WARP_K/mfma_k]; //  STAGES:blk_k内pipeline stages，一个blk算完blk_k, 还要计算整个k，证据： for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++)
    w16_union_vec<Element,8> B_reg[WARP_N/marlin_warp_n][WARP_K/marlin_warp_k];

    floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};

    // warp按照m方向排列，n方向不切分warp
    // int token_index_arr[BLOCK_SIZE_M / (m_num_threads*WARP_NUM)]; // 原码
    constexpr int TOKEN_INDEX_RAW = BLOCK_SIZE_M / (m_num_threads * WARP_NUM);
    constexpr int TOKEN_INDEX_ITERS = TOKEN_INDEX_RAW > 0 ? TOKEN_INDEX_RAW : 1;
    int token_index_arr[TOKEN_INDEX_ITERS];  // 用于输出

    // direct_masked 目前按专家内连续 token 写回；非 direct 模式则从 sorted_token_ids 反查原始 token/topk 位置。
    //理解：token_index_arr[m_idx] = token_index_arr[token_id*topk+route_id]
    //输出：token_index * real_size_n + n_idx + n_thread_idx*dwordx4_num_elements
    #pragma unroll
    for(int m_idx = 0; m_idx < TOKEN_INDEX_ITERS; m_idx++) {
        const int local_m = m_idx * m_num_threads * WARP_NUM + warp_id * m_num_threads + m_thread_idx;
        const int row = block_m_base + local_m;
        int token_index = direct_masked ? row : w16a16_ck_flat_id(sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + local_m, int(sorted_token_lens - 1))], top_k);
        token_index_arr[m_idx] = token_index;
    }

    // ===== 9. 主 GEMM 计算：direct_masked 下 helper 的 bidx 必须是专家内 block id =====
    const int helper_bidx = direct_masked ? block_in_expert : bidx;
    if(tail_N_process){
        MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_remainder<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
        (g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, helper_bidx,bidy);
    }
    else{
        MOE_W16A16_MARLIN_HIP_NT_prefill_kernel<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
        (g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, helper_bidx);
    }


    // ===== 10. C 寄存器落 LDS：把每个 lane 的 MFMA 累加结果转成 bf16/f16，并按全 CTA 的 MxN 布局写入 output_lds =====
    #pragma unroll
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / marlin_warp_n; min_tile_n++){
        #pragma unroll
        for(int n_tile = 0; n_tile < marlin_warp_n / mfma_n; n_tile++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
            float value = C_reg[0][min_tile_m * ((WARP_N/marlin_warp_n) * (marlin_warp_n / mfma_n)) + min_tile_n * (marlin_warp_n / mfma_n) + n_tile][reg_id];
            int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * marlin_warp_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N +
                    reg_id * (marlin_warp_n / mfma_n) * 4 + (lane_id / 16) * (marlin_warp_n / mfma_n) + n_tile + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16))/2 * 2/*padding*/;// + (lane_id & 15 )/2 * 2/*padding*/

            output_lds[index] = w16_f32_to_f16<Element>(value);
        }
        }
    }
    }

    __syncthreads();

    // ===== 11. LDS 写回全局内存：按 token_index 映射回输出 token，并按 N 向量化写出 8 个 bf16 =====
    {

    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    
        #pragma unroll
        for(int m_idx = 0; m_idx < BLOCK_SIZE_M; m_idx += m_num_threads*WARP_NUM) {
            const int32_t token_index = token_index_arr[m_idx/(m_num_threads*WARP_NUM)];
            #pragma unroll
            for(int n_idx = 0; n_idx < BLOCK_SIZE_N; n_idx += n_num_threads*dwordx4_num_elements) {
            // token_index 保护 M 边界，N 条件保护 tail_N_process 下最后一个 N block 的有效范围。
            const int n_global = bidy * BLOCK_SIZE_N + n_idx + n_thread_idx*dwordx4_num_elements;
            const bool valid_m_store = direct_masked ? (token_index < valid_m) : (token_index < int(size_m * top_k));
            if(valid_m_store && n_global < real_size_n) 
            {
                //printf("token_index:%d  n_idx:%d n_thread_idx:%d addr writing to %d\n", token_index, n_idx, n_thread_idx, token_index * real_size_n + n_idx + n_thread_idx*dwordx4_num_elements);
                *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * real_size_n + n_idx + n_thread_idx*dwordx4_num_elements]) =
                *reinterpret_cast<vec_bf16_8*>(&output_lds[(m_idx + m_thread_idx + warp_id * m_num_threads) * BLOCK_SIZE_N
                + n_idx + n_thread_idx*dwordx4_num_elements + (m_idx + m_thread_idx + warp_id * m_num_threads)/2 * 2/*padding*/]);
            }
            }
        }
    }
}


template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP(const GemmParams3<T>& params) {

    // constexpr int STAGES = BLOCK_SIZE_K/WARP_K; //1
    // constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    // const bool mul_topk_weight = false;

    // dim3 blockDim, gridDim;
    // blockDim.x = WARP_NUM*64;
    // blockDim.y = 1;
    // blockDim.z = 1;
    // constexpr int bytes_per_element = 2;
    // const int real_size_n = (params.size_n * params.size_kofb) / params.size_k;
    // //规定BLOCKSIZE_N固定位256
    // gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向 原意：申请m*topk/blk_m个blk。
    // gridDim.y = DIV_ceil(real_size_n , BLOCK_SIZE_N); // n方向 16大小分块 320/128数量为2余64
    // gridDim.x = 1; // k方向

    // const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K) * bytes_per_element, (BLOCK_SIZE_M * BLOCK_SIZE_N * bytes_per_element + BLOCK_SIZE_M*(BLOCK_SIZE_M/2)*2*bytes_per_element)/*padding*/);
    // const hipStream_t stream = at::cuda::getCurrentHIPStream();

    constexpr int STAGES = BLOCK_SIZE_K/WARP_K; //1
    constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool mul_topk_weight = false;


    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    constexpr int bytes_per_element = 2;
    const int real_size_n = (params.size_n * params.size_kofb) / params.size_k; // size_n和size_kof来自weight的shape，真实n，k可能在shuffle中打乱了，所以这样计算

    const uint32_t blocks_per_expert = DIVIDE(params.expected_m_per_group, BLOCK_SIZE_M);
    gridDim.z = params.expert_num * blocks_per_expert; // m方向 

    gridDim.y = DIV_ceil(real_size_n , BLOCK_SIZE_N); // n方向 16大小分块 320/128数量为2余64
    gridDim.x = 1; // k方向

    // LDS大小为结果矩阵大小
    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K) * bytes_per_element, (BLOCK_SIZE_M * BLOCK_SIZE_N * bytes_per_element + BLOCK_SIZE_M*(BLOCK_SIZE_M/2)*2*bytes_per_element)/*padding*/);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    // printf("launch_moe_w16a16_first_stage_prefill_masked_direct: gridDim = (%d, %d, %d), blockDim = (%d, %d, %d), lds_size = %d\n",
    //     gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, lds_size);

    BOOL_SWITCH(((real_size_n % BLOCK_SIZE_N)!=0),tail_N_process, [&] {

    MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight,tail_N_process><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A,
      params.ptr_B0,
      params.ptr_C,
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids,
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta,
      params.expert_num,
      params.expected_m_per_group
      );
    });
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}


// launch_aiter_prefill =对应=> dispatch_w16a16_marlin_gemm
template <typename scalar_t>
void dispatch_w16a16_marlin_gemm(
const torch::Tensor& input,
const torch::Tensor& b_qweight,
torch::Tensor& output,
const torch::Tensor& masked_m,
int expected_m_per_group) {
    const int experts = input.size(0);
    const int size_m = input.size(1);
    const int size_k = input.size(2);
    // dispatch_w16a16_marlin_gemm 这里为什么这么写？
    // const int size_n = b_qweight.size(2);
    // const int size_kb = b_qweight.size(1);
    const int size_n = b_qweight.size(2);
    const int size_kb = b_qweight.size(1);


  GemmParams3<scalar_t> params(
      reinterpret_cast<const scalar_t*>(input.data_ptr()),
      reinterpret_cast<const scalar_t*>(b_qweight.data_ptr()),
      reinterpret_cast<scalar_t*>(output.data_ptr()),
      nullptr,
      nullptr,
      masked_m.data_ptr<int32_t>(), //expert_ids
      experts,  // num_tokens_post_pad
      nullptr,
      size_m,
      size_n,
      size_k,
      size_kb, // kofb
      0, //sorted_token_lens
      1, //topk
      1, //delta
      experts,
      expected_m_per_group
    );

    //   模板参数正确性确认 TODO

        // // 默认dispatch数据
        // launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<
        // 16, 64, 128, 16, 16, 64, scalar_t>(params);

        // // test-case:
        // m_list=390
        // n_list=[384]
        // k_list=[4096]
        // E=256
        // expected_m_per_group_list=[32]
        // BLOCK_SIZE_M: 32, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 128
        // WARP_M: 16, WARP_N: 32, WARP_K: 64
        launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<
        32, 128, 128, 16, 32, 64, scalar_t>(params);
}

// launch_m_grouped_w16a16_aiter_fastpath =对应=> moe_c_moe_gemm_marlin_w16a16
void launch_m_grouped_w16a16_aiter_fastpath(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor masked_m,
    int expected_m_per_group) {
  const int size_m = input.size(1);

  if (output.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_w16a16_marlin_gemm<bhalf_t>(
        input, b_qweight, output, masked_m, expected_m_per_group);
  } else {
    TORCH_CHECK(false, "HIPC W4A8 fastpath supports bf16 output only");
  }
  DG_HIP_CHECK(hipGetLastError());
}

torch::Tensor m_grouped_bf16_gemm_nt_masked_hipc(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor masked_m, 
    int expected_m_per_group,
    int mode) {
  
	const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
    launch_m_grouped_w16a16_aiter_fastpath(
        input, b_qweight, output, masked_m,
        expected_m_per_group);
    return output;
}

} // namespace deepgemm
