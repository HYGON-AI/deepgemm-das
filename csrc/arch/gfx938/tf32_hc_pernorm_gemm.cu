#include<hip/hip_runtime.h>
#include<hip/hip_hcc.h>
#include<torch/all.h>
#include<ATen/hip/HIPContext.h>
#include<c10/cuda/CUDAGuard.h>
#include<vector>
#include<optional>

#include"hip/hip_bf16.h"
#include"tf32_hc_pernorm_gemm.h"
#include"intrinsic.h"
#include"utils.h"
#include"exception.h"

using bhalf_t = __hip_bfloat16;

namespace deepgemm {

template<uint32_t BLOCK_M,  uint32_t BLOCK_K, uint32_t BLOCK_N,
         uint32_t WARP_M, uint32_t WARP_N, uint32_t WARP_K, uint32_t WARP_NUM,
         uint32_t kNumStages>
__global__ void __launch_bounds__(1024) tf32_hc_pernorm_gemm_kernel(
                                            const uint32_t shape_m,
                                            const uint32_t shape_n,
                                            const uint32_t shape_k,
                                            const uint32_t num_splits,    
                                            const bhalf_t* a,
                                            const float* b,
                                            float* d,
                                            float* sqr_sum) {
    //确定使用的MMAC
    constexpr int mmac_m = 16;
    constexpr int mmac_n = 16;
    constexpr int mmac_k = 8;

    //确定WARP_K_NUM, WARP_M_NUM, WARP_N_NUM
    static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be divisible by WARP_K");
    static_assert(BLOCK_K == WARP_K, "BLOCK_K must be equal to WARP_K");

    constexpr int WARP_K_NUM = BLOCK_K / WARP_K;   // 1
    constexpr int WARP_M_NUM = BLOCK_M / WARP_M;   // 4
    constexpr int WARP_N_NUM = BLOCK_N / WARP_N;   // 1

    //确定split block_K数量
    const uint32_t kNumBlocks = (shape_k + BLOCK_K - 1) / BLOCK_K;   // 1
    const uint32_t kNumKBlocksPerSplit = kNumBlocks / num_splits;   // 1
    const uint32_t kRemainKBlocks = kNumBlocks % num_splits;        // 0

    //确定block_idx, split_idx
    const uint32_t block_idx = __builtin_hcu_readfirstlane(blockIdx.x);    // 0
    const uint32_t block_idy = __builtin_hcu_readfirstlane(blockIdx.y);    // 0
    const uint32_t warp_idx = __builtin_hcu_readfirstlane(threadIdx.x >> 6);   // 0 - 3

    const uint32_t m_block_idx = block_idx / num_splits;                    // 0
    const uint32_t k_split_idx = block_idx % num_splits;                    // 0

    //确定k_start_offset k_end_offset
    const uint32_t k_num_block_split = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);
    const uint32_t k_start_offset = (k_split_idx * (kNumKBlocksPerSplit + min<uint32_t>(k_split_idx, kRemainKBlocks))) * BLOCK_K;
    const uint32_t k_end_offset = min<uint32_t>(k_start_offset + k_num_block_split * BLOCK_K, shape_k);
    const uint32_t k_actual_length = k_end_offset - k_start_offset;
        
    //确定warp_idx_k,warp_idx_m,warp_idx_n
    const uint32_t warp_idx_k = warp_idx % WARP_K_NUM;                  // 0
    const uint32_t warp_idx_n = (warp_idx / WARP_K_NUM) % WARP_N_NUM;   // 0
    const uint32_t warp_idx_m = (warp_idx / WARP_K_NUM) / WARP_N_NUM;   // 0 - 3
    //确定a_offset, b_offset, d_offset, sqr_sum_offset
    const uint32_t split_m_offset = shape_m * k_split_idx;
    auto a_ptr = a + (m_block_idx * BLOCK_M) * shape_k + k_start_offset;
    auto b_ptr = b + block_idy * BLOCK_N * shape_k + k_start_offset;
    auto d_ptr = d + (split_m_offset + m_block_idx * BLOCK_M) * shape_n + block_idy * BLOCK_N;
    auto sqr_sum_ptr = sqr_sum + split_m_offset + m_block_idx * BLOCK_M;

    //确定 a_lds, b_lds
    extern __shared__ uint8_t ab_smem[];
    bhalf_t* a_lds = reinterpret_cast<bhalf_t*>(ab_smem);
    int a_lds_base = reinterpret_cast<size_t>(a_lds);

    union_vec<float, WARP_K / 4> A_reg[WARP_M / mmac_m][kNumStages];
    union_vec<float, WARP_K / 4> B_reg[WARP_N / mmac_n][kNumStages];
    union_vec<bhalf_t, WARP_K / 4> A_reg_b16[WARP_M / mmac_m][kNumStages];
    floatx4 C_reg[1][(WARP_M / mmac_m) * (WARP_N / mmac_n)] = {0,0,0,0};

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;

    //线程读B的offset
    int g_row_B[WARP_N / mmac_n];
    #pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
        int n_offset = warp_idx_n * WARP_N + row_id + n_tile * mmac_n;
        //每个线程读取4个float
        g_row_B[n_tile] = n_offset * shape_k + col_id * 4; 
    }
    // for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
    //     printf("warp_idx: %d, lane_id: %d, n_tile: %d, g_row_B: %d\n", warp_idx, lane_id, n_tile, g_row_B[n_tile]);
    // }
    //A使用matrix_load， B使用buffer_load
    uintx4 a_rsrc = make_rscr_matrix_load(a_ptr, shape_k);
    uintx4 a_read_rsrc;
    a_read_rsrc[2] = shape_k;
    int actual_m = shape_m - m_block_idx * BLOCK_M;
    int actual_n = shape_n - block_idy * BLOCK_N;
    int k_start_stage_offset = 0;
    
    //加载STAGE - 1次
    int i = 0;
    int lds_offset = 0;
    int lds_offset2 = 0;
    #pragma unroll
    for (; i < kNumStages - 1; ++i) {
        // load A
        *(uint64_t*)&a_read_rsrc = (*(uint64_t*)&a_rsrc + (k_start_stage_offset + (warp_idx * 16) * shape_k) * sizeof(bhalf_t)) & 0xffffffffffff;
        int nm_filter = deepgemm::inline_min_max<0, 16>(16 * warp_idx + 16 - actual_m);
        a_read_rsrc[3] = actual_m % BLOCK_M == 0 ? 0 : nm_filter << 8;
        lds_offset = (i * BLOCK_M * BLOCK_K + warp_idx * 16 * 64) * sizeof(bhalf_t);
        matrix_load_b16_lds_trans<64, 16, 1, 0, bhalf_t>(a_lds + lds_offset, a_read_rsrc, lds_offset, 0);

        //load B
        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
            #pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / 16; ++k_tile) {
                buffer_load_reg_dwordx4(b_ptr, B_reg[n_tile][i].float4_array[k_tile], 0, (g_row_B[n_tile] + k_tile * 16 + k_start_stage_offset) * sizeof(float), shape_k * actual_n * sizeof(float));
            }
        }
        k_start_stage_offset += WARP_K;
    }

    float sqr_sum_acc_0 = 0;
    for (; k_start_stage_offset < k_actual_length; k_start_stage_offset += WARP_K, ++i) {
        // load A
        *(uint64_t*)&a_read_rsrc = (*(uint64_t*)&a_rsrc + (k_start_stage_offset + (warp_idx * 16) * shape_k) * sizeof(bhalf_t)) & 0xffffffffffff;
        int nm_filter = deepgemm::inline_min_max<0, 16>(16 * warp_idx + 16 - actual_m);
        a_read_rsrc[3] = actual_m % BLOCK_M == 0 ? 0 : nm_filter << 8;
        lds_offset = ((i % kNumStages) * BLOCK_M * BLOCK_K + warp_idx * 16 * 64) * sizeof(bhalf_t);
        matrix_load_b16_lds_trans<64, 16, 1, 0, bhalf_t>(a_lds, a_read_rsrc, lds_offset, 0);

        //load B
        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
            #pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / 16; ++k_tile) {   // 64 / 16 = 4
                buffer_load_reg_dwordx4(b_ptr, B_reg[n_tile][i % kNumStages].float4_array[k_tile], 0, 
                    (g_row_B[n_tile] + k_tile * 16 + k_start_stage_offset) * sizeof(float), shape_k * actual_n * sizeof(float));
            }
        }
        //vmcnt_wait((kNumStages - 1));
        vmcnt_wait((1 + WARP_N / mmac_n * WARP_K / 16) * (kNumStages - 1));

        //读取DS 使用ds_read_matrix_trans

        lds_offset = a_lds_base + (((i + 1) % kNumStages) * BLOCK_M * BLOCK_K + warp_idx * 16 * 64) * sizeof(bhalf_t);
        lds_offset2 = lds_offset + 16 * 32 * sizeof(bhalf_t);
        #if defined(__gfx938__)
        DS_READ_MATRIX_32X16_B16(lds_offset, A_reg_b16[0][(i + 1) % kNumStages].b8t_array[0], true);
        DS_READ_MATRIX_32X16_B16(lds_offset2, A_reg_b16[0][(i + 1) % kNumStages].b8t_array[1], true);
        #endif
        
        lgkmcnt_wait(1);
        
        //将 bf16转换为float
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            A_reg[0][(i+1) % kNumStages].scalar_array[j] = static_cast<float>(A_reg_b16[0][(i+1) % kNumStages].scalar_array[j]);
            sqr_sum_acc_0 += A_reg[0][(i+1) % kNumStages].scalar_array[j] * A_reg[0][(i+1) % kNumStages].scalar_array[j];
        }
        
        
        // for (int j = 0; j < WARP_N / mmac_n; ++j) {
        //     for (int k = 0; k < WARP_K / 4; ++k) {
        //         float res = B_reg[j][(i+1) % kNumStages].scalar_array[k];
        //         printf("warp_idx: %d, lane_id: %d, j: %d, k: %d, res: %f\n", warp_idx, lane_id, j, k, res);
        //     }
        // }
        

        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
            #if defined(__gfx938__)
            // floatx2 A_reg_float2 = *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[0]);
            // floatx2 B_reg_float2 = *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[0]);
            // printf("warp_idx: %d, lane_id: %d, n_tile: %d, A_reg_float2: %f, %f, B_reg_float2: %f, %f\n", warp_idx, lane_id, n_tile, A_reg_float2[0], A_reg_float2[1], B_reg_float2[0], B_reg_float2[1]);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[0]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[0]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[1]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[1]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[2]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[2]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[3]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[3]), 
                C_reg[0][n_tile], true, false);
            #endif
        }
    
        

        lgkmcnt_wait(0);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            A_reg[0][(i+1) % kNumStages].scalar_array[j+8] = static_cast<float>(A_reg_b16[0][(i+1) % kNumStages].scalar_array[j + 8]);
            sqr_sum_acc_0 += A_reg[0][(i+1) % kNumStages].scalar_array[j+8] * A_reg[0][(i+1) % kNumStages].scalar_array[j+8];
        }

        
        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
            #if defined(__gfx938__)
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[4]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[4]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[5]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[5]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[6]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[6]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[7]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[7]), 
                C_reg[0][n_tile], true, false);
            #endif
        }
        
    }

    // for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
    //     for (int k = 0; k < 4; ++k) {
    //         float res = C_reg[0][n_tile][k];
    //         printf("warp_idx: %d, lane_id: %d, n_tile: %d, k: %d, res: %f\n", warp_idx, lane_id, n_tile, k, res);
    //     }
    // }

    //还剩stage - 1 次
    #pragma unroll
    for (int last_stage = 0; last_stage < kNumStages - 1; ++last_stage, ++i) {
        vmcnt_wait((1 + WARP_N / mmac_n * WARP_K / 16) * (kNumStages - 1 - (last_stage + 1)));

        lds_offset = ((i + 1) % kNumStages * BLOCK_M * BLOCK_K + warp_idx_m * 16 * 64) * sizeof(bhalf_t);
        lds_offset2 = lds_offset + 16 * 32 * sizeof(bhalf_t);
        #if defined(__gfx938__)
        DS_READ_MATRIX_32X16_B16(lds_offset, A_reg_b16[0][(i + 1) % kNumStages].b8t_array[0], true);
        DS_READ_MATRIX_32X16_B16(lds_offset2, A_reg_b16[0][(i + 1) % kNumStages].b8t_array[1], true);
        #endif
        lgkmcnt_wait(1);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            A_reg[0][(i+1) % kNumStages].scalar_array[j] = static_cast<float>(A_reg_b16[0][(i+1) % kNumStages].scalar_array[j]);
            sqr_sum_acc_0 += A_reg[0][(i+1) % kNumStages].scalar_array[j] * A_reg[0][(i+1) % kNumStages].scalar_array[j];
        }
        
        
        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
            #if defined(__gfx938__)
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[0]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[0]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[1]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[1]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[2]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[2]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[3]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[3]), 
                C_reg[0][n_tile], true, false);
            #endif
        }   
        
        lgkmcnt_wait(0);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            A_reg[0][(i+1) % kNumStages].scalar_array[j+8] = static_cast<float>(A_reg_b16[0][(i+1) % kNumStages].scalar_array[j + 8]);
            sqr_sum_acc_0 += A_reg[0][(i+1) % kNumStages].scalar_array[j+8] * A_reg[0][(i+1) % kNumStages].scalar_array[j+8];
        }
        
        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
            #if defined(__gfx938__)
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[4]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[4]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[5]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[5]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[6]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[6]), 
                C_reg[0][n_tile], true, false);
            C_reg[0][n_tile] = __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(
                *(floatx2*)(&A_reg[0][(i+1) % kNumStages].scalar2_array[7]), 
                *(floatx2*)(&B_reg[n_tile][(i+1) % kNumStages].scalar2_array[7]), 
                C_reg[0][n_tile], true, false);
            #endif
        }
        
    }
    // printf("warp_idx: %d, lane_id: %d, sqr_sum_acc_0: %f\n", warp_idx, lane_id, sqr_sum_acc_0);
    // LDS 规约 sqr_acc
    float* sqr_lds = reinterpret_cast<float*>(ab_smem);
    sqr_lds[warp_idx * 64 + lane_id] = sqr_sum_acc_0;
    lgkmcnt_wait_barrier(0);

    //0 + 16 + 32 + 48
    if (lane_id < 16 && warp_idx * WARP_M + lane_id < actual_m) {
        float sqr_sum_acc_1 = 0;
        sqr_sum_acc_1 += sqr_lds[warp_idx * 64 + lane_id];
        sqr_sum_acc_1 += sqr_lds[warp_idx * 64 + lane_id + 16];
        sqr_sum_acc_1 += sqr_lds[warp_idx * 64 + lane_id + 32];
        sqr_sum_acc_1 += sqr_lds[warp_idx * 64 + lane_id + 48];
        //存出
        sqr_sum_ptr[warp_idx * WARP_M + lane_id] = sqr_sum_acc_1;
    }

    // 写出（先用标量路径保证正确性）
    const int out_row = warp_idx * WARP_M + row_id;
    if (out_row < actual_m) {
        #pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / mmac_n; ++n_tile) {
            const int out_col_base = n_tile * mmac_n + col_id * 4;
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int out_col = out_col_base + v;
                if (out_col < actual_n) {
                    d_ptr[out_row * shape_n + out_col] = C_reg[0][n_tile][v];
                }
            }
        }
    }
    
}


void tf32_hc_pernorm_gemm_impl(const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& sqr_sum,
    const int& m, const int& n, const int& k,
    const int& num_splits) {
    
    constexpr int block_m = 64;
    constexpr int block_k = 64;
    const int block_n = align(n, 16);

    constexpr int WARP_NUM = 4;
    constexpr int WARP_M = constexpr_ceil_div(block_m, WARP_NUM);
    constexpr int WARP_K = block_k;
    LIGHTOP_HOST_ASSERT(n <= block_n);
    LIGHTOP_HOST_ASSERT(n <= 32 && n % 8 == 0);
    LIGHTOP_HOST_ASSERT(k % block_k == 0);

    int num_stage = 4, smem_size = 0;
    int kblock_num = k / block_k;
    num_stage = min<int>(num_stage, kblock_num);
    smem_size = block_m * block_k * static_cast<int>(sizeof(bhalf_t));
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    dim3 gridDim(num_splits * ceil_div(m, block_m), 1, 1);
    dim3 blockDim(WARP_NUM * 64, 1, 1);

    if (n <= 16)
      tf32_hc_pernorm_gemm_kernel<block_m, block_k, 16, WARP_M, 16, WARP_K, WARP_NUM, 1>
      <<<gridDim, blockDim, smem_size, stream>>>(
        m, n, k, num_splits,
        reinterpret_cast<const bhalf_t*>(a.data_ptr<c10::BFloat16>()),
        b.data_ptr<float>(),
        d.data_ptr<float>(),
        sqr_sum.data_ptr<float>()
      );
    else 
      tf32_hc_pernorm_gemm_kernel<block_m, block_k, 32, WARP_M, 32, WARP_K, WARP_NUM, 1>
      <<<gridDim, blockDim, smem_size, stream>>>(
        m, n, k, num_splits,
        reinterpret_cast<const bhalf_t*>(a.data_ptr<c10::BFloat16>()),
        b.data_ptr<float>(),
        d.data_ptr<float>(),
        sqr_sum.data_ptr<float>()
      );
}


void tf32_hc_pernorm_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& D,
    const torch::Tensor& sqr_sum,
    const std::optional<int>& num_splits
) {
    check_major_type_cd(D);

    LIGHTOP_HOST_ASSERT(sqr_sum.is_contiguous());

    int m = A.size(0);
    int n = B.size(0);
    int k = A.size(1);

    LIGHTOP_HOST_ASSERT(A.scalar_type() == torch::kBFloat16);
    LIGHTOP_HOST_ASSERT(B.scalar_type() == torch::kFloat);
    LIGHTOP_HOST_ASSERT(D.scalar_type() == torch::kFloat);
    LIGHTOP_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat);


    if (m == 0)
        return;

    tf32_hc_pernorm_gemm_impl(A, B, D, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
}
}
