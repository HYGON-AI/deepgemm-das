// SPDX-License-Identifier: Apache-2.0
#ifndef M_GROUPED_FP8_GEMM_NT_CONTIG_DEVICE_INC_H
#define M_GROUPED_FP8_GEMM_NT_CONTIG_DEVICE_INC_H

// DeepGEMM contiguous FP8 GEMM (marlin-tile, mode 86 = <32,128,64,32,32,64,2>)
// Direct consumer of m_indices — derived from
// MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_FP8 with the routing
// preprocessing (sorted_token_ids / expert_ids / num_tokens_post_pad /
// topk_weights) inlined into the kernel.
//
// Invariants (enforced by the wrapper):
//   * m_indices is int32 of shape (size_m,)
//   * size_m % BLOCK_SIZE_M == 0
//   * within each BLOCK_SIZE_M-row block, all rows share the same expert
//     (i.e. groups are aligned to a multiple of BLOCK_SIZE_M; tests use 256
//     which is a multiple of 32)
//   * m_indices[row] < 0 marks a pad row; m_indices[row] >= 0 is the expert id

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
  int N_LOOP_NUM,
  bool B_TENSOR_WISE = false>
__global__ void __launch_bounds__(512,1) M_GROUPED_FP8_GEMM_NT_CONTIG_MARLIN(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  scalar_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ m_indices,
  uint32_t size_m,
  uint32_t size_n,
  uint32_t size_k,
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn,
  uint32_t stride_bsk
) {
  const int bidx = blockIdx.z;     // m direction tile (per block of BLOCK_SIZE_M rows)
  const int bidy = blockIdx.y;     // n direction tile

  // Block-level early-out: skip if block is past M, or block is fully pad
  // (the first row of the block carries the expert id for the whole block).
  if (bidx * BLOCK_SIZE_M >= size_m) return;
  const int32_t expert_id = m_indices[bidx * BLOCK_SIZE_M];
  if (expert_id < 0) return;

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;

  auto g_input = input;
  auto g_input_scale = input_scale;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;

  extern __shared__ Element smem[];
  Element* input_lds = (Element*)&(smem);
  Element* qweight_lds = input_lds;
  scalar_t* output_lds = reinterpret_cast<scalar_t*>(&(smem));
  float* b_scale_lds = (float*)&(smem);

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

  float weight_dot_a_scale[WARP_M / mfma_m][4];

  // For contiguous: token_index == row, topk_weights ≡ 1.0.
  // a_scale is indexed by the raw row, with the BLOCK_SIZE_M-row block guaranteed
  // to stay within size_m (the block-level early-out above clamps bidx).
  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
    for(int i = 0 ;i< 4;i++){
      int row = bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4;
      int row_safe = std::min(uint32_t(row), size_m - 1);
      float input_scale_value = *(input_scale + row_safe * stride_asm);
      weight_dot_a_scale[idx][i] = input_scale_value; // topk_weights ≡ 1.0
    }
  }

  constexpr int n_loop_num = N_LOOP_NUM;

  const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num;
  const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = B_TENSOR_WISE ? expert_id : stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  scalar_t* g_output;
  g_output = output + output_offset;

  // Pre-load per-row m_indices into the same register array the inner GEMM
  // expects (sorted_token_ids_element_store). Use normal scalar loads here to
  // isolate whether the swizzled buffer-load path causes intermittent faults.
  constexpr int store_size = WARP_M / 4;
  int token_index_store[store_size];
  int tok_ids_store[store_size];
  int32_t sorted_token_ids_element_store[store_size];

  {
    for(int min_tile_m = 0 ; min_tile_m < WARP_M / mfma_m; min_tile_m ++){
      for(int i = 0; i < 4 ; i++){
        int m_idx = min_tile_m * 16 + i * 4 + col_id;
        int it = m_idx / 4;
        sorted_token_ids_element_store[it] = m_indices[bidx * BLOCK_SIZE_M + m_idx];
      }
    }
  }

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  float tensor_weight_scale = 1.0f;
  if constexpr (B_TENSOR_WISE)
  {
    tensor_weight_scale = weight_scale[expert_id];
  }
  float *b_scale_ptr = weight_scale + weight_scale_offset;
  if constexpr (!B_TENSOR_WISE)
  {
    b_scale_ptr += warp_n_id * WARP_N;
  }

  float b_scale[n_loop_num][(WARP_N / mfma_n)];
  {
    #pragma unroll
    for(int n_loop=0;n_loop < n_loop_num;n_loop++){
      #pragma unroll
      for(int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++){
        for(int i = 0 ;i < 32 / mfma_n ;i++){
          if constexpr (B_TENSOR_WISE)
          {
            b_scale[n_loop][min_tile_n * 2 + i] = tensor_weight_scale;
          }
          else
          {
            vec<uint,4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128,float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);
            inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i], row_id * 2, b_scale_ptr_prepared, min_tile_n * 32 + i);
          }
        }
      }
    }
  }

  vec4_fp32 C_reg[n_loop_num][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};
  __builtin_hcu_sched_barrier(0);

  // float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];

  // real_topk parameter is irrelevant in kFromMIndices=true mode but kept for
  // call-site compatibility; pass 1 so the (dead) `topk_ids < real_topk` ==
  // `pad_bit < 1` check inside the inner gemm degenerates correctly.
  gemm_nt_marlin_prefill_2_fp8<
      false, 0, 0, WARP_NUM,
      BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
      WARP_M, WARP_N, WARP_K, STAGES,
      GROUP_N, GROUP_K, Element, scalar_t,
      /*kFromMIndices=*/true,
      N_LOOP_NUM>
    (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale,
     size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k,
     stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk,
     /*top_k=*/1, m_indices, /*sorted_token_lens=*/size_m,
     expert_id, bidx,
     sorted_token_ids_element_store, tok_ids_store, token_index_store,
     weight_dot_a_scale, b_scale, nullptr, /*real_topk=*/1);

  __builtin_hcu_sched_barrier(0);

 scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
  #pragma unroll
  for(int n_loop = 0; n_loop < n_loop_num; n_loop++){
    #pragma unroll
    for(int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m ++){
      #pragma unroll
      for(int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n ++){
        #pragma unroll
        for(int i = 0; i < 2; i++){
          const int n_idx = min_tile_n * 2 + i;
          const int tile_idx = min_tile_m * (WARP_N / mfma_n) + n_idx;
          #pragma unroll
          for(int reg_id = 0; reg_id < 4; reg_id ++){
            float tmp = C_reg[n_loop][tile_idx][reg_id] *
                        weight_dot_a_scale[min_tile_m][reg_id] *
                        b_scale[n_loop][n_idx];
            value[n_loop][min_tile_m][reg_id][n_idx] =
                b32_to_b16<scalar_t>(tmp);
          }
        }
      }
    }
  }

  // Write-back: token_index_store[it] is now the raw row index (set by the
  // inner gemm in kFromMIndices mode). tok_ids_store[it] is 1 for pad rows
  // and 0 for real rows, so the gate `tok_ids_store[it] < 1` keeps pad rows
  // from being written out.
  #pragma unroll
  for(int n_loop = 0; n_loop < n_loop_num; n_loop++){
    #pragma unroll
    for(int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m ++){
      #pragma unroll
      for(int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n ++){
        int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id ++){
          int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
          if(tok_ids_store[it] < 1){
            *(vec2_Element<scalar_t>*)(&g_output[token_index_store[it] * size_n + index])
              = *(vec2_Element<scalar_t>*)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
          }
        }
      }
    }
  }
}

#endif /* M_GROUPED_FP8_GEMM_NT_CONTIG_DEVICE_INC_H */
