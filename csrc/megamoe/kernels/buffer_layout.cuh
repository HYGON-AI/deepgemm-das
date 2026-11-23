// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// Buffer layout and slicing for HCU MegaMoE.
// Ported from SM90 sm90_fp8_mega_moe.cuh:227-254 and layout/mega_moe.cuh:320-403.
//
// Key difference from SM90: SF layouts use per-token (1 float/token)
// instead of per-128-K (kHidden/32) or per-64-K (kIntermediateHidden/16).

#include <cstdint>
#include <hip/hip_runtime.h>
#include "params.h"
#include "workspace.cuh"

namespace deep_gemm::mega::hcu {

// Data layout descriptor (simplified from SM90 layout::Data)
// require_alignment removed 鈥?never used in device code, saves 4 bytes/Buffer in registers
struct DataLayout {
    uint32_t num_bytes = 0;       // bytes per token

    __host__ __device__
    constexpr DataLayout() = default;
    __host__ __device__
    constexpr explicit DataLayout(const uint32_t bytes, const bool /*align*/ = true)
        : num_bytes(bytes) {}
};

// Buffer: a contiguous region of tokens with a per-token data layout
struct Buffer {
    DataLayout data_layout{};
    uint32_t num_ranks = 0;              // for input: 1; for combine: num_topk
    uint32_t num_max_tokens = 0;
    uint8_t* base = nullptr;             // device pointer

    __host__ __device__
    Buffer() = default;
    __host__ __device__
    Buffer(const DataLayout& layout, const uint32_t ranks,
           const uint32_t tokens, void* ptr = nullptr)
        : data_layout(layout), num_ranks(ranks),
          num_max_tokens(tokens), base(static_cast<uint8_t*>(ptr)) {}

    __host__ __device__
    uint64_t get_num_bytes_per_rank() const {
        return static_cast<uint64_t>(num_max_tokens) * data_layout.num_bytes;
    }

    __host__ __device__
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    __host__ __device__
    void* get_end_ptr() const {
        return base + get_num_bytes();
    }

    // Get pointer to a specific token's data within a rank
    __host__ __device__
    void* get_data_buffer(const uint32_t token_idx, const uint32_t rank_idx = 0) const {
        return base + (static_cast<uint64_t>(rank_idx) * num_max_tokens + token_idx) * data_layout.num_bytes;
    }

    template <typename T>
    __host__ __device__
    T* get_data_buffer(const uint32_t token_idx, const uint32_t rank_idx = 0) const {
        return static_cast<T*>(get_data_buffer(token_idx, rank_idx));
    }
};

// Compute the total sym_buffer size and create a slicing function.
// Mirrors SM90 get_symm_buffer_size_for_sm90_mega_moe (sm90_mega.hpp:33-138)
// but with per-token SF layouts.
//
// Buffer allocation is split into two regions:
//   - IPC buffer (fine-grained, no L2 cache): workspace + P2P areas
//     (input_token, input_sf, input_topk_idx, input_topk_weights, combine_token)
//   - Local buffer (regular hipMalloc, with L2 cache): non-P2P GEMM areas
//     (l1_token, l1_sf, l2_token, l2_sf, l2_output)
// The local buffers are only accessed on the local GPU (no sym.map() calls),
// so they benefit from L2 cache for GEMM performance.
struct BufferSlices {
    Buffer input_token_buffer;
    Buffer input_sf_buffer;
    Buffer input_topk_idx_buffer;
    Buffer input_topk_weights_buffer;
    Buffer l1_token_buffer;
    Buffer l1_sf_buffer;
    Buffer l2_token_buffer;
    Buffer l2_sf_buffer;
    Buffer l2_output_buffer;       // Per-block BF16 scratch for HT L2 GEMM output: [kNumSMs, kHTL2ScratchBlockM, hidden]
    Buffer combine_token_buffer;
};

// Size of the IPC (P2P) portion: workspace + input areas + combine area
__host__ __device__ static inline
uint64_t get_ipc_buffer_total_bytes(
    const uint32_t num_ranks, const uint32_t num_experts,
    const uint32_t num_max_tokens_per_rank, const uint32_t num_topk,
    const uint32_t hidden, const uint32_t intermediate_hidden) {

    HCUMegaMoeWorkspace<> workspace(nullptr, nullptr, num_ranks, num_experts,
                                   num_max_tokens_per_rank, num_topk);
    const uint64_t workspace_bytes = workspace.get_num_bytes();

    const DataLayout fp8_token_layout(hidden);
    const DataLayout bf16_token_layout(hidden * 2);  // sizeof(bf16) = 2
    const DataLayout fp8_sf_layout(sizeof(float), false);              // V1 per-token
    const DataLayout input_topk_idx_layout(num_topk * sizeof(int64_t), false);
    const DataLayout input_topk_weights_layout(num_topk * sizeof(float), false);

    uint64_t offset = workspace_bytes;

    // Registered input area (P2P — written by remote ranks via sym.map)
    Buffer input_token_buffer(fp8_token_layout, 1, num_max_tokens_per_rank, nullptr);
    offset += input_token_buffer.get_num_bytes();
    Buffer input_sf_buffer(fp8_sf_layout, 1, num_max_tokens_per_rank, nullptr);
    offset += input_sf_buffer.get_num_bytes();
    Buffer input_topk_idx_buffer(input_topk_idx_layout, 1, num_max_tokens_per_rank, nullptr);
    offset += input_topk_idx_buffer.get_num_bytes();
    Buffer input_topk_weights_buffer(input_topk_weights_layout, 1, num_max_tokens_per_rank, nullptr);
    offset += input_topk_weights_buffer.get_num_bytes();

    // Combine area (P2P — read by remote ranks via sym.map)
    Buffer combine_token_buffer(bf16_token_layout, num_topk, num_max_tokens_per_rank, nullptr);
    offset += combine_token_buffer.get_num_bytes();

    return align_u64(offset, 16);
}

// Size of the local (non-P2P) portion: L1/L2 GEMM staging areas
// num_sms: runtime CU count (64 or 72) — L2 output scratch has one slot per block.
__host__ __device__ static inline
uint64_t get_local_buffer_total_bytes(
    const uint32_t num_ranks, const uint32_t num_experts,
    const uint32_t num_max_tokens_per_rank, const uint32_t num_topk,
    const uint32_t hidden, const uint32_t intermediate_hidden,
    const uint32_t num_sms = kNumSMs,
    const bool l1_ring_enabled = true) {

    HCUMegaMoeWorkspace<> workspace(nullptr, nullptr, num_ranks, num_experts,
                                   num_max_tokens_per_rank, num_topk, num_sms,
                                   l1_ring_enabled);

    const DataLayout fp8_token_layout(hidden);
    const DataLayout fp8_intermediate_token_layout(intermediate_hidden);
    const DataLayout fp8_sf_layout(sizeof(float), false);
    // Group-wise SF: [num_groups per token] floats = [intermediate_hidden / kGroupQuantSize]
    const DataLayout fp8_intermediate_sf_layout(
        (intermediate_hidden / kGroupQuantSize) * sizeof(float), false);
    // Per-block scratch: each slot holds one [BLOCK_M x hidden] region so the
    // existing MatrixC[row * L2_SHAPE_N + n_blk * BLOCK_N + col] addressing is
    // unchanged. One slot per block (indexed by blockIdx.x).
    const DataLayout l2_output_layout(
        static_cast<uint32_t>(kHTL2ScratchBlockM) * hidden * 2, false);

    const uint32_t num_max_pool_tokens = workspace.num_max_pool_blocks * kMinCandidateBlockM;

    // Local sync workspace (grid sync, send count, L1/L2 arrival, quant_done,
    // token_src_metadata) occupies the head of the local buffer.
    uint64_t offset = workspace.get_local_num_bytes();

    // L1 pool area (local — GEMM A matrix, no P2P access).
    // Ring-capped at the 16K-token-equivalent pool when the full pool exceeds it.
    const uint32_t num_l1_pool_tokens = workspace.num_l1_pool_tokens;
    Buffer l1_token_buffer(fp8_token_layout, 1, num_l1_pool_tokens, nullptr);
    offset += l1_token_buffer.get_num_bytes();
    Buffer l1_sf_buffer(fp8_sf_layout, 1, num_l1_pool_tokens, nullptr);
    offset += l1_sf_buffer.get_num_bytes();

    // L2 pool area (local — GEMM A matrix, no P2P access)
    Buffer l2_token_buffer(fp8_intermediate_token_layout, 1, num_max_pool_tokens, nullptr);
    offset += l2_token_buffer.get_num_bytes();
    Buffer l2_sf_buffer(fp8_intermediate_sf_layout, 1, num_max_pool_tokens, nullptr);
    offset += l2_sf_buffer.get_num_bytes();

    // L2 output staging (HT kernel: per-block scratch — GEMM writes one tile
    // then immediately scatters it to the remote combine buffer)
    Buffer l2_output_buffer(l2_output_layout, 1, num_sms, nullptr);
    offset += l2_output_buffer.get_num_bytes();

    return align_u64(offset, 16);
}

// Backward-compat: total = IPC + local (used by old code paths)
__host__ __device__ static inline
uint64_t get_sym_buffer_total_bytes(
    const uint32_t num_ranks, const uint32_t num_experts,
    const uint32_t num_max_tokens_per_rank, const uint32_t num_topk,
    const uint32_t hidden, const uint32_t intermediate_hidden,
    const uint32_t num_sms = kNumSMs,
    const bool l1_ring_enabled = true) {
    return get_ipc_buffer_total_bytes(num_ranks, num_experts,
                                       num_max_tokens_per_rank, num_topk,
                                       hidden, intermediate_hidden)
         + get_local_buffer_total_bytes(num_ranks, num_experts,
                                         num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate_hidden, num_sms,
                                         l1_ring_enabled);
}

// Slice the buffers into sub-buffers.
// ipc_workspace_base: device pointer to the start of the workspace region
//                     (after IPC header: peer ptrs + signal ptrs + runtime_num_tokens)
// local_base: device pointer to the start of the local (non-P2P) buffer
__host__ __device__ static inline
BufferSlices slice_sym_buffers(uint8_t* ipc_workspace_base,
                               uint8_t* local_base,
                               const uint32_t num_ranks,
                               const uint32_t num_experts,
                               const uint32_t num_max_tokens_per_rank,
                               const uint32_t num_topk,
                               const uint32_t hidden,
                               const uint32_t intermediate_hidden,
                               const uint32_t num_sms = kNumSMs,
                               const bool l1_ring_enabled = true) {
    HCUMegaMoeWorkspace<> workspace(ipc_workspace_base, local_base, num_ranks, num_experts,
                                   num_max_tokens_per_rank, num_topk, num_sms,
                                   l1_ring_enabled);

    const DataLayout fp8_token_layout(hidden);
    const DataLayout bf16_token_layout(hidden * 2);
    const DataLayout fp8_intermediate_token_layout(intermediate_hidden);
    const DataLayout fp8_sf_layout(sizeof(float), false);
    // Group-wise SF: [num_groups per token] floats = [intermediate_hidden / kGroupQuantSize]
    const DataLayout fp8_intermediate_sf_layout(
        (intermediate_hidden / kGroupQuantSize) * sizeof(float), false);
    const DataLayout input_topk_idx_layout(num_topk * sizeof(int64_t), false);
    const DataLayout input_topk_weights_layout(num_topk * sizeof(float), false);
    // Per-block scratch: [kHTL2ScratchBlockM x hidden] per block (indexed by blockIdx.x)
    const DataLayout l2_output_layout(
        static_cast<uint32_t>(kHTL2ScratchBlockM) * hidden * 2, false);

    const uint32_t num_max_pool_tokens = workspace.num_max_pool_blocks * kMinCandidateBlockM;

    // --- IPC (P2P) region: slice from ipc_workspace_base ---
    uint8_t* ipc_ptr = static_cast<uint8_t*>(workspace.get_end_ptr());

    Buffer input_token_buffer(fp8_token_layout, 1, num_max_tokens_per_rank, ipc_ptr);
    ipc_ptr = static_cast<uint8_t*>(input_token_buffer.get_end_ptr());
    Buffer input_sf_buffer(fp8_sf_layout, 1, num_max_tokens_per_rank, ipc_ptr);
    ipc_ptr = static_cast<uint8_t*>(input_sf_buffer.get_end_ptr());
    Buffer input_topk_idx_buffer(input_topk_idx_layout, 1, num_max_tokens_per_rank, ipc_ptr);
    ipc_ptr = static_cast<uint8_t*>(input_topk_idx_buffer.get_end_ptr());
    Buffer input_topk_weights_buffer(input_topk_weights_layout, 1, num_max_tokens_per_rank, ipc_ptr);
    ipc_ptr = static_cast<uint8_t*>(input_topk_weights_buffer.get_end_ptr());

    // Combine area (P2P — placed right after input area in IPC buffer)
    Buffer combine_token_buffer(bf16_token_layout, num_topk, num_max_tokens_per_rank, ipc_ptr);

    // --- Local (non-P2P) region: slice from local_base ---
    // Head of local_base is reserved for the local sync workspace.
    uint8_t* lcl_ptr = local_base + workspace.get_local_num_bytes();

    // L1 staging buffers are ring-capped: at large token counts they hold only
    // the 16K-token-equivalent pool and are recycled by the pull<->L1 handshake.
    const uint32_t num_l1_pool_tokens = workspace.num_l1_pool_tokens;
    Buffer l1_token_buffer(fp8_token_layout, 1, num_l1_pool_tokens, lcl_ptr);
    lcl_ptr = static_cast<uint8_t*>(l1_token_buffer.get_end_ptr());
    Buffer l1_sf_buffer(fp8_sf_layout, 1, num_l1_pool_tokens, lcl_ptr);
    lcl_ptr = static_cast<uint8_t*>(l1_sf_buffer.get_end_ptr());

    Buffer l2_token_buffer(fp8_intermediate_token_layout, 1, num_max_pool_tokens, lcl_ptr);
    lcl_ptr = static_cast<uint8_t*>(l2_token_buffer.get_end_ptr());
    Buffer l2_sf_buffer(fp8_intermediate_sf_layout, 1, num_max_pool_tokens, lcl_ptr);
    lcl_ptr = static_cast<uint8_t*>(l2_sf_buffer.get_end_ptr());

    Buffer l2_output_buffer(l2_output_layout, 1, num_sms, lcl_ptr);

    return {
        input_token_buffer, input_sf_buffer,
        input_topk_idx_buffer, input_topk_weights_buffer,
        l1_token_buffer, l1_sf_buffer,
        l2_token_buffer, l2_sf_buffer,
        l2_output_buffer,
        combine_token_buffer
    };
}

// IPC header layout (before workspace):
//   [0] peer buffer ptrs:    num_ranks * sizeof(uint8_t*)
//   [align16] peer signal ptrs: num_ranks * sizeof(int*)
//   [align16] runtime_num_tokens: int32_t
//   [align16] workspace starts here
__host__ __device__ static inline
uint64_t ipc_header_bytes(const uint32_t num_ranks) {
    uint64_t offset = 0;
    offset += static_cast<uint64_t>(num_ranks) * sizeof(uint8_t*);  // peer buffer ptrs
    offset = align_u64(offset, 16);
    offset += static_cast<uint64_t>(num_ranks) * sizeof(int*);      // peer signal ptrs
    offset = align_u64(offset, 16);
    offset += sizeof(int32_t);                                       // runtime_num_tokens
    return align_u64(offset, 16);
}

// Total IPC buffer size = header + workspace + P2P data areas only
// (non-P2P areas are allocated separately via regular hipMalloc)
__host__ __device__ static inline
uint64_t get_total_ipc_buffer_bytes(
    const uint32_t num_ranks, const uint32_t num_experts,
    const uint32_t num_max_tokens_per_rank, const uint32_t num_topk,
    const uint32_t hidden, const uint32_t intermediate_hidden) {
    return ipc_header_bytes(num_ranks) +
           get_ipc_buffer_total_bytes(num_ranks, num_experts,
                                       num_max_tokens_per_rank, num_topk,
                                       hidden, intermediate_hidden);
}

// Get workspace base pointer from sym_buffer base (skip IPC header)
__host__ __device__ static inline
uint8_t* get_workspace_base(uint8_t* sym_buffer, const uint32_t num_ranks) {
    return sym_buffer + ipc_header_bytes(num_ranks);
}

// Get peer buffer pointer array from sym_buffer header
__host__ __device__ static inline
uint8_t** get_peer_buffer_ptrs(uint8_t* sym_buffer) {
    return reinterpret_cast<uint8_t**>(sym_buffer);
}

// Get peer signal pointer array from sym_buffer header
__host__ __device__ static inline
int** get_peer_signal_ptrs(uint8_t* sym_buffer, const uint32_t num_ranks) {
    uint64_t offset = static_cast<uint64_t>(num_ranks) * sizeof(uint8_t*);
    offset = align_u64(offset, 16);
    return reinterpret_cast<int**>(sym_buffer + offset);
}

// Get runtime_num_tokens pointer from sym_buffer header
__host__ __device__ static inline
int32_t* get_runtime_num_tokens_ptr(uint8_t* sym_buffer, const uint32_t num_ranks) {
    uint64_t offset = static_cast<uint64_t>(num_ranks) * sizeof(uint8_t*);
    offset = align_u64(offset, 16);
    offset += static_cast<uint64_t>(num_ranks) * sizeof(int*);
    offset = align_u64(offset, 16);
    return reinterpret_cast<int32_t*>(sym_buffer + offset);
}

// Symmetric buffer helper: map a local pointer to a remote rank's pointer
// via the IPC peer buffer pointer array.
struct SymBuffer {
    uint8_t** peer_buffer_ptrs;  // [num_ranks] from IPC header
    int rank_idx;
    int num_ranks;

    __device__ void* map(void* local_ptr, int dst_rank) const {
        const auto offset = static_cast<uint8_t*>(local_ptr) - peer_buffer_ptrs[rank_idx];
        return peer_buffer_ptrs[dst_rank] + offset;
    }
};

} // namespace deep_gemm::mega::hcu
