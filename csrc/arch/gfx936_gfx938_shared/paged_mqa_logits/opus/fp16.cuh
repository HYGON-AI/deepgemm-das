// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once
#include "common.cuh"
namespace deepgemm::opus {
template <typename Element, int kNextN = 1, int kNumHeads = 64,
          int kHeadDim = 128, int BLOCK_KV = 64,
          int kBatchSplit, int kNumWarps = 4,
          int kTokensPerWarp = BLOCK_KV, int QGROUP = 1>

__device__ __forceinline__ void grouped_half_device(const Element* q,
                                 const Element* kv_block,
                                 const float* weights,
                                 const int batch_size,
                                 const int num_kv_blocks,
                                 const uint64_t kv_cache_stride_bytes,
                                 const uint64_t logits_stride,
                                 const uint64_t block_table_stride,
                                 const int* context_lens,
                                 float* logits,
                                 const int* block_table,
                                 const float* kv_scales, int q_index, bool clean_logits) {
    const int warp_idx = threadIdx.x / kWaveSize;
    const int lane_idx = threadIdx.x % kWaveSize;
    const int t_id = threadIdx.x;

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr int K_TILE = 32;
    static constexpr int Q_TILE = 16;
    static constexpr int KV_TILE = 16;
    static constexpr int Q_ITER = constexpr_ceil_div(kNumHeads, Q_TILE);
    static constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    static constexpr int KV_ITER = kTokensPerWarp / KV_TILE;
    static constexpr int kTokensPerCta = kTokensPerWarp * kNumWarps;
    static_assert(BLOCK_KV == 1 || BLOCK_KV == 16 || BLOCK_KV == 32 || BLOCK_KV == 64, "Invalid page-major BLOCK_KV");
    static_assert(kTokensPerWarp == 64, "Invalid page-major tiling");

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    auto* smem_kv_block = reinterpret_cast<Element*>(smem_buffer);
    constexpr bool is_half = true;

    const int next_n = kNextN != 1 ? blockIdx.x % kNextN : 0;
    const int q_idx = q_index;
    const int cta_idx = blockIdx.x / kNextN;
    const int context_len = context_lens[q_idx];
    const int num_kv = ceil_div(context_len, BLOCK_KV);
    const int cta_seq_kv_offset = cta_idx * kTokensPerCta;
    const uint64_t base_logits_offset = q_idx * kNextN * logits_stride * QGROUP + next_n * logits_stride;

    if (cta_seq_kv_offset >= num_kv * BLOCK_KV) {
        if (!clean_logits) return;
        #pragma unroll
        for (int i = t_id; i < kTokensPerCta; i += blockDim.x) {
            const int seq_kv_offset = cta_seq_kv_offset + i;
            if (seq_kv_offset < logits_stride) {
                for(int g=0;g<QGROUP;++g)logits[base_logits_offset+g*logits_stride+seq_kv_offset]=-INFINITY;
            }
        }
        return;
    }

    auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(Element);
    constexpr int fetch_q_cnt = constexpr_ceil_div(kNumHeads * kHeadDim,
                                                    kNumWarps * kWaveSize * kAlignmentQ);
    constexpr int fetch_q_stride = kAlignmentQ * kNumWarps * kWaveSize;

    #pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        const int pos = t_id * kAlignmentQ + i * fetch_q_stride;
        if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
            *reinterpret_cast<f16x8_t*>(&smem_kv_block[pos]) = *reinterpret_cast<const f16x8_t*>(&gQ[pos]);
        }
    }

    f16x8_t reg_q_operand[Q_ITER][K_BLOCK_MAX];
    const int base_fetch_lds_Q = lane_idx % Q_TILE * kHeadDim + lane_idx / Q_TILE * kAlignmentQ;

    auto gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto* smem_weight = reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim);
    float reg_weights[Q_ITER * 4];

    if constexpr (kNumHeads >= KV_TILE * kNumWarps) {
        smem_weight = reinterpret_cast<float*>(smem_kv_block);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < Q_ITER; ++i) {
            #pragma unroll
            for (int j = 0; j < K_BLOCK_MAX; ++j) {
                const int offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
                if constexpr (kNumHeads < Q_TILE) {
                    reg_q_operand[i][j] = load_padded_q_operand<kNumHeads, Q_TILE, f16x8_t>(
                        smem_kv_block, offset, i * Q_TILE + lane_idx % Q_TILE);
                } else {
                    reg_q_operand[i][j] = *reinterpret_cast<f16x8_t*>(&smem_kv_block[offset]);
                }
            }
        }

        __syncthreads();
        for(int h=threadIdx.x;h<kNumHeads;h+=blockDim.x)smem_weight[h]=gW[h];
        __syncthreads();

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
    } else {
        for(int h=threadIdx.x;h<kNumHeads;h+=blockDim.x)smem_weight[h]=gW[h];
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < Q_ITER; ++i) {
            #pragma unroll
            for (int j = 0; j < K_BLOCK_MAX; ++j) {
                const int offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
                if constexpr (kNumHeads < Q_TILE) {
                    reg_q_operand[i][j] = load_padded_q_operand<kNumHeads, Q_TILE, f16x8_t>(
                        smem_kv_block, offset, i * Q_TILE + lane_idx % Q_TILE);
                } else {
                    reg_q_operand[i][j] = *reinterpret_cast<f16x8_t*>(&smem_kv_block[offset]);
                }
            }
        }

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
    }

    __syncthreads();

    constexpr int smem_kv_block_per_warp = KV_TILE * kHeadDim;
    auto* smem_warp_kv_start = smem_kv_block + warp_idx * smem_kv_block_per_warp;
    const uint64_t block_table_offset = q_idx * block_table_stride;
    float sum_result[QGROUP][KV_ITER] = {};
    const auto prefetch_tile = [&](const int kv_iter) {
        const int cta_tile_idx = kv_iter * kNumWarps + warp_idx;
        const int seq_kv_tile_offset = cta_seq_kv_offset + cta_tile_idx * KV_TILE;
        const int page_idx = seq_kv_tile_offset / BLOCK_KV;
        if (page_idx < num_kv) {
            if constexpr(BLOCK_KV==1){
                long desc[2];desc[0]=reinterpret_cast<long>(kv_block);desc[1]=(long(0x20000)<<32)|0xFFFFFFFE;
                #pragma unroll
                for(int j=0;j<4;++j){
                    const int linear=(j*64+lane_idx)*8;
                    const int token=seq_kv_tile_offset+linear/128;
                    const int p=token<context_len?block_table[block_table_offset+token]:0;
                    const int offset=token<context_len?int(uint32_t(p)*256+(linear%128)*2):-1;
                    buffer_load_lds_x4(reinterpret_cast<v4i*>(desc),reinterpret_cast<uint8_t*>(smem_warp_kv_start)+j*1024,offset);
                }
            }else{
                const int block_idx=block_table[block_table_offset+page_idx];
                const auto* page=kv_block+int64_t(block_idx)*BLOCK_KV*kHeadDim;
                buffer_load_page_tile_lds<Element,kHeadDim,KV_TILE>(page,seq_kv_tile_offset%BLOCK_KV,lane_idx,smem_warp_kv_start);
            }
        }
    };

    prefetch_tile(0);

    #pragma unroll
    for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
        const int cta_tile_idx = kv_iter * kNumWarps + warp_idx;
        const int seq_kv_tile_offset = cta_seq_kv_offset + cta_tile_idx * KV_TILE;
        const bool valid = seq_kv_tile_offset / BLOCK_KV < num_kv;

        #if defined(__gfx936__) || defined(__gfx938__)
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_waitcnt vmcnt(0)\n\t"
            );
        __builtin_amdgcn_sched_barrier(0);
        #else
        __syncthreads();
        #endif

        fp32x4 acc[Q_ITER];
        if (valid) {
            OPUS_MQA_CLEAR_ACC(acc, Q_ITER);
        }

        const int base_fetch = lane_idx % KV_TILE * kHeadDim + lane_idx / KV_TILE * kAlignmentQ;
        #pragma unroll
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            f16x8_t reg_kv_operand;
            if (valid) {
                const int fetch_lds = base_fetch + k_block * K_TILE;
                const int lds_addr = reinterpret_cast<size_t>(smem_warp_kv_start + fetch_lds);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile(
                    "\n ds_read_b128 %0, %1\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t"
                    : "=v"(reg_kv_operand)
                    : "v"(lds_addr)
                    :);
                __builtin_amdgcn_sched_barrier(0);
            }

            if (k_block == K_BLOCK_MAX - 1 && kv_iter + 1 < KV_ITER) {
                prefetch_tile(kv_iter + 1);
            }

            if (valid) {
                #pragma unroll
                for (int i = 0; i < Q_ITER; ++i) {
                    builtin_b16_mmac<is_half>(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i]);
                    builtin_b16_mmac<is_half>(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i]);
                }
            }
        }

        if(valid){
            #pragma unroll
            for(int g=0;g<QGROUP;++g){
                float sum=0.f;
                #pragma unroll
                for(int j=0;j<Q_ITER/QGROUP;++j){
                    #pragma unroll
                    for(int c=0;c<4;++c)sum+=fmaxf(acc[g*(Q_ITER/QGROUP)+j][c],0.f)*reg_weights[(g*(Q_ITER/QGROUP)+j)*4+c];
                }
                sum+=__shfl_down_sync(0xffffffffffffffffULL,sum,16);
                sum+=__shfl_down_sync(0xffffffffffffffffULL,sum,32);
                sum_result[g][kv_iter]=sum;
            }
        }

    }

    #pragma unroll
    for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
        const int cta_tile_idx = kv_iter * kNumWarps + warp_idx;
        const int seq_kv_tile_offset = cta_seq_kv_offset + cta_tile_idx * KV_TILE;
        if (lane_idx < KV_TILE) {
            const int seq_kv_offset = seq_kv_tile_offset + lane_idx;
            if (seq_kv_offset < context_len - (kNextN - next_n) + 1) {
                const int p=block_table[block_table_offset+seq_kv_offset/BLOCK_KV];
                const float sf=kv_scales[p*BLOCK_KV+seq_kv_offset%BLOCK_KV];
                #pragma unroll
                for(int g=0;g<QGROUP;++g)logits[base_logits_offset+g*logits_stride+seq_kv_offset]=sum_result[g][kv_iter]*sf;
            } else if (seq_kv_offset < logits_stride && (clean_logits || seq_kv_offset < context_len)) {
                for(int g=0;g<QGROUP;++g)logits[base_logits_offset+g*logits_stride+seq_kv_offset]=-INFINITY;
            }
        }
    }
}



} // namespace deepgemm::opus
