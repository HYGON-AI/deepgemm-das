// SPDX-License-Identifier: Apache-2.0
#ifndef MOE_MARLIN_PREFILL_DOWN_FP8_DEVICE_INC_H
#define MOE_MARLIN_PREFILL_DOWN_FP8_DEVICE_INC_H

// MoE Marlin GEMM2 prefill-down FP8 — slice of moe/csrc_for_aiter/moe_w8a8_opt.h.
// Tile: <32,128,64,32,32,64,2> (mode 86).

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
  bool mul_topk_weight> // true
__global__ void __launch_bounds__(512,1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_FP8(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  scalar_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const float* __restrict__ topk_weights,
  const int32_t*  sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn, 
  uint32_t stride_bsk,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t real_topk 
) {
  // const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  // const int bidy = blockIdx.y; // pid_n
  // const int bidz = blockIdx.x; // pid_k 

  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k 

  // int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  // int bidy = blockIdx.y; // pid_n
  // int bidz = blockIdx.x; // pid_k 
  // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);
  
  //constexpr int STAGES = 2; 
  uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
  if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  

  const uint32_t input_offset = bidz * BLOCK_SIZE_K/* bidz * BLOCK_SIZE_K */; //输入k方向分块的位置
  const int32_t delta_bidx = bidx ;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

   


  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移

  // auto g_input = input;
  auto g_input = input ;
  auto g_input_scale = input_scale; // 配置全局显存信息




  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // decode这里没用
  Element* qweight_lds = input_lds; // decode这里没用
  scalar_t* output_lds = reinterpret_cast<scalar_t*>(&(smem)); // 重复使用lds,留给output_lds
  // float* output_lds_float = (float*)&(smem); // 重复使用lds,留给output_lds

  float* b_scale_lds = (float*)&(smem);


  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];



  float weight_dot_a_scale[WARP_M / mfma_m][4];


  


  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
    for(int i = 0 ;i< 4;i++){
      int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
      int token_ids = sorted_token_ids_element & 0x00FFFFFF;
      int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
      int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1)  ;
      float input_scale_value = *(input_scale + token_index_safe * stride_asm); //计算M方向的偏移
      weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
  
    }
  }



  

  constexpr  int  n_loop_num = 4;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */ ; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
  scalar_t* g_output;
  g_output = output + output_offset;


  if(expert_id == -1){ //EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    vec8_Element<scalar_t> zero_element_8{};

    int m_idx = threadIdx.x / N_thread; 
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))] ;
      const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
      const int32_t topk_ids   = sorted_token_ids_element & 0xFF000000;
      int token_index = token_ids * real_topk /* top_k */ + topk_ids;
      
      if(topk_ids < real_topk ) {
        *reinterpret_cast<vec8_Element<scalar_t>*>(&g_output[(token_index) * size_n + n_idx * 8]) = zero_element_8;
      }
    } 
    return;
  }  



  constexpr int store_size = WARP_M  / 4;
  int token_index_store[store_size];
  int tok_ids_store[store_size];
  int32_t sorted_token_ids_element_store[store_size];
  auto g_sorted_token_ids_offset =  tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

  {


  // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){ // 会导致循环无法展开 最终导致core dump
  for(int min_tile_m = 0 ; min_tile_m < WARP_M / mfma_m; min_tile_m ++){
  for(int i = 0; i < 4 ; i++){
    int m_idx = min_tile_m * 16 + i * 4 + col_id ;
    int it =  m_idx / 4;
    // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
    inline_buffer_load_dword(sorted_token_ids_element_store[it],  m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
    
    // int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
    // tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24; 
    // token_index_store[it] = token_ids_store * 8 + tok_ids_store[it];
  }
  }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id*WARP_N;

  float b_scale[n_loop_num][(WARP_N / mfma_n) ];

  {
    #pragma unroll
    for(int n_loop=0;n_loop < n_loop_num;n_loop++){

      #pragma unroll
      for(int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++){
        for(int i = 0 ;i < 32 / mfma_n ;i++){
        
            vec<uint,4> b_scale_ptr_prepared =  tcp_cache_swizzle_func_no<128,float>(b_scale_ptr + BLOCK_SIZE_N* n_loop);

        
            // b_scale[min_tile_n*4 + reg_id] = (b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4)[0];
            inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i ], row_id *  2 ,b_scale_ptr_prepared,min_tile_n * 32 + i );
            
        
        }
      }
    }

  }
    



  

    
    vec4_fp32 C_reg[n_loop_num][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
    __builtin_hcu_sched_barrier(0);
      
      // scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
      float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
        gemm_nt_marlin_prefill_2_fp8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element , scalar_t>
        (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx,sorted_token_ids_element_store,tok_ids_store,token_index_store, weight_dot_a_scale,b_scale, tmp ,real_topk);
      
    __builtin_hcu_sched_barrier(0);
    



    

  

    scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    #pragma unroll
    for(int n_loop =0 ;n_loop < n_loop_num ; n_loop++){

      #pragma unroll
      for(int min_tile_m =0 ; min_tile_m < (WARP_M / mfma_m) ; min_tile_m ++){
          #pragma unroll
          for(int min_tile_n = 0; min_tile_n < (WARP_N / 32) ; min_tile_n ++ ){
            #pragma unroll
            for(int i =0; i < 2 ;i++){
            
              const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
              #pragma unroll
              for(int reg_id =0; reg_id < 4 ; reg_id ++){
                // float  tmp = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + i];
                value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] =  b32_to_b16<scalar_t>/* f32_to_bf16  */(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
              }
                
            }
            
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
                #pragma unroll
                for(int reg_id =0; reg_id < 4 ; reg_id ++){
                      int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                      if(tok_ids_store[it] < real_topk){
                      
                        *( vec2_Element<scalar_t>*)(&g_output[ token_index_store[it] * size_n + index  ] )= *( vec2_Element<scalar_t>*)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2 ]);
                      }
                }
          }
          
              
            


      }

        // __syncthreads();

      // }

    }





}


#endif /* MOE_MARLIN_PREFILL_DOWN_FP8_DEVICE_INC_H */

