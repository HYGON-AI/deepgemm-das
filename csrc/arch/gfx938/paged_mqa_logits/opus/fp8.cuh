// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once
#include "../../../gfx936_gfx938_shared/paged_mqa_logits/opus/common.cuh"
namespace deepgemm::opus {
template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kBatchSplit, int kNumWarps = 4,
          int kTokensPerWarp = BLOCK_KV, bool WaveLocal = false, int QGROUP = 1, bool Repacked = true>

__device__ __forceinline__ void grouped_device(const Element* q,
                                     const Element* kv_block,
                                     const float* kv_block_scales,
                                     const float* weights,
                                     const int batch_size,
                                     const int num_kv_blocks,
                                     const uint64_t kv_cache_stride_bytes,
                                     const uint64_t logits_stride,
                                     const uint64_t block_table_stride,
                                     const int* context_lens,
                                     float* logits,
                                     const int* block_table,
                                     int q_index, bool clean_logits) {
    const int& warp_idx = threadIdx.x / 64;
    const int& lane_idx = threadIdx.x % 64;
    const int& t_id = threadIdx.x;

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    static constexpr int K_TILE = 64;
    static constexpr int Q_TILE = 16;
    static constexpr int KV_TILE = 16;
    static constexpr int Q_ITER = constexpr_ceil_div(kNumHeads, Q_TILE);
    static constexpr int KV_ITER = kTokensPerWarp / KV_TILE;
    static constexpr int PAGES_PER_WARP = BLOCK_KV == 1 ? 1 : kTokensPerWarp / BLOCK_KV;
    static constexpr int Stages = kHeadDim / K_TILE;
    static_assert(BLOCK_KV == 1 || BLOCK_KV == 16 || BLOCK_KV == 32 || BLOCK_KV == 64, "Invalid BLOCK_KV");
    static_assert(kTokensPerWarp % BLOCK_KV == 0 &&
                          kTokensPerWarp <= kWaveSize,
                          "Invalid tokens per warp");

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    constexpr bool is_e4m3 = std::is_same<Element, uint8_t>::value;

    auto* smem_kv_block = reinterpret_cast<uint8_t*>(smem_buffer);

    const auto next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;

    int q_idx = q_index;
    int start_kv_idx = blockIdx.x / kNextN * kNumWarps * kTokensPerWarp / BLOCK_KV;
    int kv_idx = start_kv_idx + warp_idx * kTokensPerWarp / BLOCK_KV;
    int context_len = context_lens[q_idx];
    int num_kv = ceil_div(context_len, BLOCK_KV);

    // Calculate logits KV block offset in advance
    auto base_logits_offset = q_idx * kNextN * logits_stride * QGROUP + next_n * logits_stride;
    auto base_seq_kv_offset = kv_idx * BLOCK_KV;

    if (start_kv_idx >= num_kv) {
        if (!clean_logits) return;
        uint32_t fill_offset = base_logits_offset + base_seq_kv_offset + lane_idx;
        if (lane_idx < kTokensPerWarp && base_seq_kv_offset + lane_idx < logits_stride) {
            for(int g=0;g<QGROUP;++g)logits[fill_offset+g*logits_stride] = -INFINITY;
        }
        return;
    }

    // fetch Q && Q weights
    auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(uint8_t);
    constexpr int fetch_q_cnt = constexpr_ceil_div(kNumHeads * kHeadDim,
                                                   kNumWarps * kWaveSize * kAlignmentQ);
    constexpr int fetch_q_stride = kAlignmentQ * kNumWarps * kWaveSize;

    #pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        auto pos = t_id * kAlignmentQ + i * fetch_q_stride;
        if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
            *reinterpret_cast<f8x16_t *>(&smem_kv_block[pos]) = load_q16(&gQ[pos]);
        }
    }

    auto gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto smem_weight = reinterpret_cast<float *>(smem_kv_block + kNumHeads * kHeadDim);
    for(int h=threadIdx.x;h<kNumHeads;h+=blockDim.x)smem_weight[h]=gW[h];

    // fetch kv block idx
    int kv_block_idx[PAGES_PER_WARP];
    uint64_t block_table_offset = q_idx * block_table_stride + kv_idx;
    #pragma unroll
    for (int i = 0; i < PAGES_PER_WARP; ++i) {
        kv_block_idx[i] = kv_idx + i < num_kv ? block_table[block_table_offset + i] : -1;
    }

    bool contiguous_page=false;
    if constexpr(BLOCK_KV==1 && Repacked){
        const int token=base_seq_kv_offset+lane_idx;
        const int p0=kv_block_idx[0];
        const bool match=token>=context_len || block_table[int64_t(q_idx)*block_table_stride+token]==p0+lane_idx;
        contiguous_page=(p0>=0 && p0%64==0 && __ballot(match)==0xffffffffffffffffULL);
    }
    constexpr int heads_cnt = Q_ITER;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    f8x16_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();
    // pre-fetch Q from smem
    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
    #pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
        #pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            if constexpr (kNumHeads < Q_TILE) {
                reg_q_operand[i][j] = load_padded_q_operand<kNumHeads, Q_TILE, f8x16_t>(
                    smem_kv_block, offset, i * Q_TILE + lane_idx % Q_TILE);
            } else {
                reg_q_operand[i][j] = *reinterpret_cast<f8x16_t*>(&smem_kv_block[offset]);
            }
        }
    }

    // pre-fetch weights
    float reg_weights[heads_cnt * 4];
    #pragma unroll
    for (int j = 0; j < Q_ITER; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if constexpr (kNumHeads < Q_TILE) {
                reg_weights[j * 4 + k] = load_padded_weight<kNumHeads, Q_TILE>(
                    smem_weight, j * Q_TILE + k * 4 + lane_idx / Q_TILE);
            } else {
                reg_weights[j * 4 + k] = smem_weight[j * Q_TILE + k * 4 + lane_idx / Q_TILE];
            }
        }
    }

    __syncthreads();

    // fetch kv block
    constexpr int kAlignmentKV = 16 / sizeof(uint8_t);
    constexpr long stride = 0x0;

    constexpr int prefetch_kv_stage = KV_ITER >= 2 ? 2 : 1;
    constexpr int smem_kv_block_per_warp = prefetch_kv_stage * KV_TILE * kHeadDim;

    auto smem_warp_kv_start = smem_kv_block + warp_idx * smem_kv_block_per_warp;

    #pragma unroll
    for (int i = 0; i < prefetch_kv_stage; ++i) {
        const int page_delta = i * KV_TILE / BLOCK_KV;
        const int row_in_page = BLOCK_KV==1?i*KV_TILE:i * KV_TILE % BLOCK_KV;
        const int current_block_idx = kv_block_idx[BLOCK_KV == 1 ? 0 : page_delta];
        auto gKv = current_block_idx != -1 ?
            kv_block + (BLOCK_KV==1?int64_t(current_block_idx/64)*8448+(current_block_idx%64)*128:
                        (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4)) : kv_block;
        long glob_kv_desc[2];
        glob_kv_desc[0] = *reinterpret_cast<const long *>(&gKv);
        glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
        glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
        auto *src = reinterpret_cast<v4i *>(glob_kv_desc);
        auto glob_stage_offset = row_in_page * kHeadDim;
        auto smem_per_stage = smem_warp_kv_start + i * KV_TILE * kHeadDim;
        #pragma unroll
        for (int j = 0; j < Stages; ++j) {
            auto smem_ptr = smem_per_stage + j * KV_TILE * K_TILE;
            auto inner_warp_offset = current_block_idx != -1 ? glob_stage_offset +
                ((lane_idx % 4) ^ ((lane_idx / 8) % 4)) * kAlignmentKV + lane_idx / 4 * kHeadDim + j * K_TILE : -1;
            if constexpr(BLOCK_KV == 1) {
                if(contiguous_page)buffer_load_lds_x4(src,smem_ptr,inner_warp_offset);
                else load_s1<Repacked>(reinterpret_cast<const uint8_t*>(kv_block),block_table+q_idx*block_table_stride,
                        base_seq_kv_offset+i*KV_TILE,j,lane_idx,context_len,smem_ptr);
            } else { buffer_load_lds_x4(src, smem_ptr, inner_warp_offset); }
        }
    }

    // pre-fetch KV from smem
    f8x16_t reg_kv_operand;

    auto base_fetch_lds_KV = lane_idx % 16 * K_TILE + ((lane_idx / 16) ^ ((lane_idx % 16) / 2 % 4)) * 16;

    fp32x4 acc[Q_ITER][KV_ITER];
    OPUS_MQA_CLEAR_ACC_2D(acc, Q_ITER, KV_ITER);
    float sum_result[QGROUP][KV_ITER] = {};
    constexpr int num_issue_kv_prefetch = Stages * prefetch_kv_stage;

    #pragma unroll
    for (uint32_t kv_iter = 0; kv_iter < KV_ITER - prefetch_kv_stage; ++kv_iter) {
        const uint32_t load_kv_iter = kv_iter + prefetch_kv_stage;
        const int page_delta = load_kv_iter * KV_TILE / BLOCK_KV;
        const int row_in_page = BLOCK_KV==1?load_kv_iter*KV_TILE:load_kv_iter * KV_TILE % BLOCK_KV;
        const int current_block_idx = kv_block_idx[BLOCK_KV == 1 ? 0 : page_delta];
        auto gKv = current_block_idx != -1 ?
            kv_block + (BLOCK_KV==1?int64_t(current_block_idx/64)*8448+(current_block_idx%64)*128:
                        (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4)) : kv_block;
        long glob_kv_desc[2];
        glob_kv_desc[0] = *reinterpret_cast<const long *>(&gKv);
        glob_kv_desc[0] = glob_kv_desc[0] | stride << 48;
        glob_kv_desc[1] = (long) 0x20000 << 32 | 0xFFFFFFFE;
        auto *src = reinterpret_cast<v4i *>(glob_kv_desc);
        auto glob_stage_offset = row_in_page * kHeadDim;
        auto smem_per_stage = smem_warp_kv_start + kv_iter % prefetch_kv_stage * KV_TILE * kHeadDim;
        #pragma unroll
        for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            auto fetch_lds_kv =
                base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
            OPUS_MQA_WAIT_STAGE(num_issue_kv_prefetch - 1);

            int lds_addr = reinterpret_cast<size_t>(smem_per_stage + fetch_lds_kv);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile(
                "\n ds_read_b128 %0 ,%1\n\t"
                "s_waitcnt lgkmcnt(0) \n\t"
                ""
                : "=v"(reg_kv_operand)
                : "v"(lds_addr)
                :);
            __builtin_amdgcn_sched_barrier(0);

            if constexpr(!WaveLocal) __syncthreads();
            auto smem_ptr = smem_per_stage + k_block * KV_TILE * K_TILE;
            int inner_warp_offset = current_block_idx != -1 ? glob_stage_offset +
                ((lane_idx % 4) ^ ((lane_idx / 8) % 4)) * kAlignmentKV + lane_idx / 4 * kHeadDim + k_block * K_TILE : -1;
            if constexpr(BLOCK_KV == 1) {
                if(contiguous_page)buffer_load_lds_x4(src,smem_ptr,inner_warp_offset);
                else load_s1<Repacked>(reinterpret_cast<const uint8_t*>(kv_block),block_table+q_idx*block_table_stride,
                        base_seq_kv_offset+load_kv_iter*KV_TILE,k_block,lane_idx,context_len,smem_ptr);
            } else { buffer_load_lds_x4(src, smem_ptr, inner_warp_offset); }

            #pragma unroll
            for (int i = 0; i < Q_ITER; ++i) {
                builtin_fp8_mmac<is_e4m3>(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_fp8_mmac<is_e4m3>(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                #pragma unroll
                for(int g=0;g<QGROUP;++g){
                    float sum=0.f;
                    #pragma unroll
                    for(int j=0;j<Q_ITER/QGROUP;++j){
                        #pragma unroll
                        for(int c=0;c<4;++c)sum+=fmaxf(acc[g*(Q_ITER/QGROUP)+j][kv_iter][c],0.f)*reg_weights[(g*(Q_ITER/QGROUP)+j)*4+c];
                    }
                    sum+=__shfl_down_sync(0xffffffffffffffffULL,sum,16);
                    sum+=__shfl_down_sync(0xffffffffffffffffULL,sum,32);
                    sum_result[g][kv_iter]=sum;
                }
            }
        }
    }

    #pragma unroll
    for (uint32_t kv_iter = KV_ITER - prefetch_kv_stage; kv_iter < KV_ITER; ++kv_iter) {
        auto smem_per_stage = smem_warp_kv_start + kv_iter % prefetch_kv_stage * KV_TILE * kHeadDim;
        #pragma unroll
        for (uint32_t k_block = 0; k_block < Stages; ++k_block) {
            auto fetch_lds_kv =
               base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
            OPUS_MQA_WAIT_STAGE(num_issue_kv_prefetch - (kv_iter - (KV_ITER - prefetch_kv_stage)) * Stages - k_block - 1);

            int lds_addr = reinterpret_cast<size_t>(smem_per_stage + fetch_lds_kv);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile(
                "\n ds_read_b128 %0 ,%1\n\t"
                "s_waitcnt lgkmcnt(0) \n\t"
                // ""
                : "=v"(reg_kv_operand)
                : "v"(lds_addr)
                :);
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for (int i = 0; i < Q_ITER; ++i) {
                builtin_fp8_mmac<is_e4m3>(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_fp8_mmac<is_e4m3>(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                #pragma unroll
                for(int g=0;g<QGROUP;++g){
                    float sum=0.f;
                    #pragma unroll
                    for(int j=0;j<Q_ITER/QGROUP;++j){
                        #pragma unroll
                        for(int c=0;c<4;++c)sum+=fmaxf(acc[g*(Q_ITER/QGROUP)+j][kv_iter][c],0.f)*reg_weights[(g*(Q_ITER/QGROUP)+j)*4+c];
                    }
                    sum+=__shfl_down_sync(0xffffffffffffffffULL,sum,16);
                    sum+=__shfl_down_sync(0xffffffffffffffffULL,sum,32);
                    sum_result[g][kv_iter]=sum;
                }
            }
        }
    }

    __syncthreads();
    auto* smem_write_back = reinterpret_cast<float*>(smem_kv_block);
    if (lane_idx < 16) {
        #pragma unroll
        for (int i = 0; i < KV_ITER; ++i) {
            #pragma unroll
            for(int g=0;g<QGROUP;++g)
                smem_write_back[g*kNumWarps*kTokensPerWarp+warp_idx*kTokensPerWarp+i*KV_TILE+lane_idx%16]=sum_result[g][i];
        }
    }
    __syncthreads();

    const int token=base_seq_kv_offset+lane_idx;
    if(lane_idx<kTokensPerWarp && token<logits_stride && (clean_logits || token<context_len)){
        const bool valid=token<context_len-kNextN+next_n+1;
        const int page=valid?block_table[int64_t(q_idx)*block_table_stride+token/BLOCK_KV]:0;
        const int64_t offset=BLOCK_KV==1?(Repacked?int64_t(page/64)*8448+8192+(page%64)*4:int64_t(page)*132+128):
            int64_t(page)*BLOCK_KV*132+BLOCK_KV*128+(token%BLOCK_KV)*4;
        const float sf=valid?*reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(kv_block)+offset):0.f;
        #pragma unroll
        for(int g=0;g<QGROUP;++g){
            const float value=smem_write_back[g*kNumWarps*kTokensPerWarp+warp_idx*kTokensPerWarp+lane_idx];
            logits[base_logits_offset+g*logits_stride+token]=valid?value*sf:-INFINITY;
        }
    }

}




} // namespace deepgemm::opus
