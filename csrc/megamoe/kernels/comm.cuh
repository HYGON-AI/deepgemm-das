// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// Communication primitives: rank barrier, local blocks barrier, and subset barriers.
//
// Subset barriers replace SM90's named barriers (bar.sync N, count):
//   sync_aligned(count, idx)   → subset_barrier (contiguous thread range)
//   sync_unaligned(count, idx) → subset_barrier (non-contiguous thread ranges)
// HCU has no named barriers, so we use shared-memory sense-reversal counters.

#include <cstdint>
#include <hip/hip_runtime.h>
#include "warp_reduce.cuh"
#include "params.h"  // for DG_DBUG_PRINTF macro

namespace deep_gemm::mega::hcu {

static constexpr int kBarrierFinishedTag = 1024;
static constexpr uint64_t kFullWaveMask = ~0ull;

// Atomic OR with system scope for uint64_t
// (HIP's atomicOr_system supports unsigned long long natively)
__device__ static inline uint64_t atomicOr_system_u64(uint64_t* ptr, uint64_t value) {
    return atomicOr_system(reinterpret_cast<unsigned long long*>(ptr),
                           static_cast<unsigned long long>(value));
}

// Atomic CAS with system scope for int32_t
// HIP lacks atomicCAS_system for int — use __atomic_compare_exchange (GCC builtin)
__device__ static inline int32_t atomicCAS_system(int32_t* ptr, int32_t expected, int32_t desired) {
    int32_t old = expected;
    __atomic_compare_exchange_n(ptr, &old, desired, false,
                                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    return old;
}

// Atomic exchange with system scope for int32_t
__device__ static inline int32_t atomicExch_system(int32_t* ptr, int32_t value) {
    return __atomic_exchange_n(ptr, value, __ATOMIC_ACQ_REL);
}

// Atomic CAS with agent (device) scope for int32_t — for local (per-GPU)
// coordination state in the L2-cached local workspace. NOT for IPC fields.
__device__ static inline int32_t atomicCAS_agent(int32_t* ptr, int32_t expected, int32_t desired) {
    int32_t old = expected;
    __hip_atomic_compare_exchange_strong(ptr, &old, desired,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                         __HIP_MEMORY_SCOPE_AGENT);
    return old;
}

__device__ static inline int load_signal_system(const volatile int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ static inline int load_signal_agent(const volatile int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ============================================================================
// Subset barrier: synchronizes a subset of threads within a block.
// Replaces SM90's ptx::sync_aligned / ptx::sync_unaligned (named barriers).
//
// Uses sense-reversal in shared memory. Each barrier instance has its own
// count + sense pair. Threads call subset_barrier() and block until all
// `expected_count` threads have arrived.
// ============================================================================
struct SubsetBarrier {
    uint32_t count;
    uint32_t sense;
};

__device__ static inline void subset_barrier_init(SubsetBarrier* bar) {
    if (threadIdx.x == 0) {
        bar->count = 0;
        bar->sense = 0;
    }
}

__device__ static inline void subset_barrier(volatile SubsetBarrier* bar,
                                              uint32_t expected_count,
                                              const char* tag = nullptr) {
    // Wave-level barrier: only lane 0 of each wave does the atomic.
    // expected_count = number of waves (NOT threads).
    // HCU: all 64 lanes in a wave execute in lockstep (mask-based divergence),
    // so the entire wave enters and exits this function together.
    __syncwarp();
    __threadfence_block();

    const uint32_t lane = threadIdx.x & (kWarpSize - 1);

    // Only lane 0 does atomic + spin. Other lanes are masked off (HCU lockstep)
    // and stay at the same PC until lane 0 finishes the spin loop.
    if (lane == 0) {
        if (blockIdx.x == 0 && threadIdx.x < kWarpSize) {
            DG_DBUG_PRINTF("[DBUG] BAR [%s] enter blk=0 count=%u sense=%u exp=%u\n",
                   tag ? tag : "?", bar->count, bar->sense, expected_count);
        }
        // Read sense BEFORE atomic_add to avoid race:
        // if we read sense after atomic_add, the last wave may have already
        // flipped sense, and we'd read the new sense and spin forever.
        uint32_t s = __hip_atomic_load(const_cast<uint32_t*>(&bar->sense),
                                       __ATOMIC_ACQUIRE, __memory_scope_work_group);
        uint32_t old = __hip_atomic_fetch_add(const_cast<uint32_t*>(&bar->count), 1u,
                                              __ATOMIC_ACQUIRE, __memory_scope_work_group);
        if (old == expected_count - 1) {
            // Last wave — release the barrier
            if (blockIdx.x == 0 && threadIdx.x < kWarpSize) {
                DG_DBUG_PRINTF("[DBUG] BAR [%s] RELEASE blk=0 old=%u\n", tag ? tag : "?", old);
            }
            __threadfence_block();
            __hip_atomic_fetch_sub(const_cast<uint32_t*>(&bar->count), expected_count,
                                   __ATOMIC_RELEASE, __memory_scope_work_group);
            __hip_atomic_fetch_add(const_cast<uint32_t*>(&bar->sense), 1u,
                                   __ATOMIC_RELEASE, __memory_scope_work_group);
            __threadfence_block();
        } else {
            // Not the last wave — spin on sense
#if DG_HANG_DIAG
            const auto start_time = clock64();
            bool printed = false;
#endif
            while (__hip_atomic_load(const_cast<uint32_t*>(&bar->sense),
                                     __ATOMIC_ACQUIRE, __memory_scope_work_group) == s) {
#if DG_HANG_DIAG
                if (!printed && clock64() - start_time > DG_HANG_TIMEOUT_CYCLES) {
                    DG_HANG_PRINTF("MegaMoE barrier timeout [%s]: old=%u expected=%u sense=%u tid=%d blk=%d\n",
                           tag ? tag : "?", old, expected_count, s,
                           static_cast<int>(threadIdx.x), static_cast<int>(blockIdx.x));
                    DG_HANG_MARK(1, old, expected_count, s);
                    printed = true;
                }
#endif
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    // All waves arrive here together (lane 0 finished, other lanes were masked).
    // Flush LDS one more time.
    __syncwarp();
    __threadfence_block();
}

// ============================================================================
// Cross-rank barrier using HIP IPC signal buffers.
// Replaces SM90's comm::nvlink_barrier.
//
// Called by a subset of threads (dispatch or epilogue). All participating
// threads in all blocks across all ranks must call this.
// Uses barrier_id to separate independent rank barriers (dispatch pre-pull,
// dispatch post-cleanup, epilogue pre-combine) that may overlap in time.
// Each barrier_id occupies [num_ranks] int slots in the signal buffer.
// ============================================================================
__device__ static inline void rank_barrier(int** signal_buffers,
                                                    const int rank_idx,
                                                    const int num_ranks,
                                                    const int barrier_id = 0) {
    const int thread_id = static_cast<int>(threadIdx.x);
    const int slot = barrier_id * num_ranks;
    auto* my_signals = signal_buffers[rank_idx];
    __threadfence_system();
    // NOTE: no __syncthreads here — caller must ensure participating threads
    // are already synchronized (e.g., via subset_barrier) before calling.

    if (thread_id < num_ranks) {
        auto* peer_signals = signal_buffers[thread_id];
        atomicAdd_system(my_signals + slot + thread_id, kBarrierFinishedTag);
        atomicSub_system(peer_signals + slot + rank_idx, kBarrierFinishedTag);
    }

#if DG_HANG_DIAG
    const auto start_time = clock64();
    bool printed = false;
#endif
    while (true) {
        int value = 0;
        if (thread_id < num_ranks) {
            volatile int* signal = reinterpret_cast<volatile int*>(my_signals + slot + thread_id);
            value = load_signal_system(signal);
        }
        // Check if all lanes see value <= 0
        value = warp_reduce_max(value);
        if (value <= 0)
            break;
#if DG_HANG_DIAG
        if (clock64() - start_time > DG_HANG_TIMEOUT_CYCLES && thread_id < num_ranks) {
            if (!printed) {
                DG_HANG_PRINTF("MegaMoE HIP rank barrier timeout: rank=%d thread=%d value=%d barrier=%d\n",
                       rank_idx, thread_id, value, barrier_id);
                DG_HANG_MARK(2, value, barrier_id, rank_idx);
                printed = true;
            }
        }
#endif
        __builtin_amdgcn_s_sleep(1);
    }
}

// ============================================================================
// Local grid barrier using sense-reversal counter in workspace.
// Replaces SM90's comm::grid_sync.
//
// Leader version: caller must ensure only ONE thread per block calls this.
// NO __syncthreads — caller must ensure calling threads are already
// synchronized (e.g., via subset_barrier) before calling, and must sync
// after if needed.
// ============================================================================
__device__ static inline void grid_sync_leader(uint32_t* grid_sync_counter,
                                                            const int num_blocks) {
    // Packed sense-reversing counter: low 16 bits hold the arrival count,
    // high 16 bits the epoch. The last arriver's add of (0x10000 -
    // num_blocks) wraps the count back to 0 and bumps the epoch in ONE
    // atomic, closing the lap window where a count reset and a sense
    // flip as two separate atomics can be observed out of order (a fast
    // block re-arriving on the next barrier could otherwise pile onto an
    // un-reset count and corrupt or hang the barrier).
    // Waiters take the epoch from their own arrival's return value, so a
    // separate sense read cannot race the flip either.
    // Assumes num_blocks <= 65535.
    volatile uint32_t* sync_word = grid_sync_counter;
    const uint32_t v = static_cast<uint32_t>(__hip_atomic_fetch_add(
        sync_word, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT));
    if ((v & 0xffffu) == static_cast<uint32_t>(num_blocks) - 1u) {
        __hip_atomic_fetch_add(sync_word,
                               0x10000u - static_cast<uint32_t>(num_blocks),
                               __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    } else {
        const uint32_t epoch = v & ~0xffffu;
#if DG_HANG_DIAG
        const auto start_time = clock64();
        bool printed = false;
#endif
        while ((static_cast<uint32_t>(__hip_atomic_load(
                    sync_word, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT))
                & ~0xffffu) == epoch) {
#if DG_HANG_DIAG
            if (!printed && clock64() - start_time > DG_HANG_TIMEOUT_CYCLES) {
                DG_HANG_PRINTF("MegaMoE HIP local barrier timeout: block=%d epoch=%u\n",
                       static_cast<int>(blockIdx.x), epoch);
                DG_HANG_MARK(3, epoch, num_blocks, 0);
                printed = true;
            }
#endif
            __builtin_amdgcn_s_sleep(1);
        }
    }
}

// Default version: only thread 0 of each block participates.
__device__ static inline void grid_sync(uint32_t* grid_sync_counter,
                                                            const int num_blocks) {
    if (threadIdx.x == 0) {
        grid_sync_leader(grid_sync_counter, num_blocks);
    }
}

} // namespace deep_gemm::mega::hcu
