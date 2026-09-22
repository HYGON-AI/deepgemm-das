#pragma once

#include <algorithm>
#include <stdint.h>

#ifndef lgkmcnt_wait
#define lgkmcnt_wait(X)                    \
  __builtin_hcu_sched_barrier(0);       \
  asm volatile("s_waitcnt lgkmcnt(%0)"     \
               :                           \
               : "I"(X));                  \
  __builtin_hcu_sched_barrier(0);
#endif

namespace w4a8_nwave_bm128_bn512 {

constexpr int kComputeWaves = 8;
constexpr int kLoaderWaves = 4;
constexpr int kThreads = (kComputeWaves + kLoaderWaves) * 64;
constexpr int kSharedBytes = 64 * 1024;

__forceinline__ __device__ uint32_t input_srd_stride() { return 0x40400000; }

namespace detail {

__forceinline__ __device__ void ds_read_b128(
    char* lds, const int lds_byte_offset, vec<int, 4>& dst) {
  const int lds_addr =
      static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_byte_offset;
  asm volatile("ds_read_b128 %0, %1\n\t" : "=v"(dst) : "v"(lds_addr));
}

template<int M_TILES>
constexpr int a_stage_bytes() {
  return M_TILES * 4096;
}

template<int STAGE_M_TILES>
__forceinline__ __device__ void load_a_direct4(
    const intx4& global_ptr,
    const int lds_base,
    const int (&a_row_offsets)[4],
    const int stage,
    const int k_start,
    const int loader_wave_id) {
  static_assert(STAGE_M_TILES == 4 || STAGE_M_TILES == 8,
                "load_a_direct4 supports BM64/BM128 stages");
  constexpr int A_STAGE_BYTES = a_stage_bytes<STAGE_M_TILES>();
  constexpr int A_M_TILE_BYTES = 4096;
  constexpr int A_K_TILE_BYTES = 1024;
  const int lds_stage_base =
      lds_base + stage * A_STAGE_BYTES + loader_wave_id * A_K_TILE_BYTES;
  const int lds0 = lds_stage_base + 0 * A_M_TILE_BYTES;
  const int lds1 = lds_stage_base + 1 * A_M_TILE_BYTES;
  const int lds2 = lds_stage_base + 2 * A_M_TILE_BYTES;
  const int lds3 = lds_stage_base + 3 * A_M_TILE_BYTES;
  const int s_offset = k_start + loader_wave_id * 64;
  asm volatile(
      "s_mov_b32 m0, %4\n\t"
      "buffer_load_dwordx4 %0, %8, %9, offen offset:0, lds\n\t"
      "s_mov_b32 m0, %5\n\t"
      "buffer_load_dwordx4 %1, %8, %9, offen offset:0, lds\n\t"
      "s_mov_b32 m0, %6\n\t"
      "buffer_load_dwordx4 %2, %8, %9, offen offset:0, lds\n\t"
      "s_mov_b32 m0, %7\n\t"
      "buffer_load_dwordx4 %3, %8, %9, offen offset:0, lds\n\t"
      :
      : "v"(a_row_offsets[0]),
        "v"(a_row_offsets[1]),
        "v"(a_row_offsets[2]),
        "v"(a_row_offsets[3]),
        "s"(lds0),
        "s"(lds1),
        "s"(lds2),
        "s"(lds3),
        "s"(global_ptr),
        "s"(s_offset)
      : "memory");
}

template<int M_TILES>
__forceinline__ __device__ void load_a_to_lds_direct(
    const char* __restrict__ input,
    char* A_lds,
    const intx4& global_ptr,
    const int lds_base,
    const int (&a_row_offsets)[M_TILES],
    const int stage,
    const int k_start,
    const int loader_wave_id) {
  (void)input;
  constexpr int A_M_TILE_BYTES = 4096;
  static_assert(M_TILES == 4 || M_TILES == 8,
                "load_a_to_lds_direct supports BM64/BM128");
  int offsets0[4] = {
      a_row_offsets[0], a_row_offsets[1], a_row_offsets[2], a_row_offsets[3]};
  load_a_direct4<M_TILES>(
      global_ptr, lds_base, offsets0, stage, k_start, loader_wave_id);
  if constexpr (M_TILES == 8) {
    int offsets1[4] = {
        a_row_offsets[4], a_row_offsets[5], a_row_offsets[6], a_row_offsets[7]};
    load_a_direct4<M_TILES>(
        global_ptr, lds_base + 4 * A_M_TILE_BYTES, offsets1, stage, k_start,
        loader_wave_id);
  }
}

template<int M_TILES>
__forceinline__ __device__ void load_a_tiles_direct(
    const char* __restrict__ input,
    char* A_lds,
    const int physical_warp_id,
    const int expert_id,
    const int block_m_base,
    const int valid_m,
    const uint32_t size_m) {
  constexpr int SIZE_K = 7168;
  constexpr int K_LOOP_STEP = 256;
  constexpr int KLOOPS = SIZE_K / K_LOOP_STEP;
  static_assert(M_TILES == 4 || M_TILES == 8,
                "load_a_tiles_direct supports BM64/BM128");
  const int lane_id = threadIdx.x & 63;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  const int loader_wave_id = physical_warp_id - kComputeWaves;

  int a_row_offsets[M_TILES];
  const int safe_valid_m = valid_m > 0 ? valid_m : 1;
#pragma unroll
  for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
    const int m_idx = m_tile * 16 + row_id;
    const int row = std::min(block_m_base + m_idx, safe_valid_m - 1);
    a_row_offsets[m_tile] = (expert_id * int(size_m) + row) * SIZE_K + col_id * 16;
  }

  intx4 a_global_ptr;
  *reinterpret_cast<uint64_t*>(&a_global_ptr) = reinterpret_cast<uint64_t>(input);
  a_global_ptr[1] += input_srd_stride();
  a_global_ptr[2] = 0x80000000;
  a_global_ptr[3] = 0x00020000;

  const int a_lds_base = __builtin_hcu_readfirstlane(
      static_cast<int>(reinterpret_cast<size_t>(A_lds)));
  load_a_to_lds_direct<M_TILES>(
      input, A_lds, a_global_ptr, a_lds_base, a_row_offsets, 0, 0,
      loader_wave_id);

  for (int kloop = 0; kloop < KLOOPS; ++kloop) {
    const int next_slot = (kloop & 1) ^ 1;
    vmcnt_only_wait(0);
    lgkmcnt_wait(0);
    asm volatile("s_barrier\n" ::: "memory");
    if (kloop + 1 < KLOOPS) {
      load_a_to_lds_direct<M_TILES>(
          input, A_lds, a_global_ptr, a_lds_base, a_row_offsets, next_slot,
          (kloop + 1) * K_LOOP_STEP, loader_wave_id);
    }
    __builtin_hcu_sched_barrier(0);
    vmcnt_only_wait(0);
  }
}

template<int M_TILES>
__forceinline__ __device__ void read_a_mtile_kgroup(
    char* A_lds,
    union_vec_opt<char, 16>& A_frag,
    const int current_slot,
    const int m_tile,
    const int k_read,
    const int row_id,
    const int col_id) {
  constexpr int A_STAGE_BYTES = a_stage_bytes<M_TILES>();
  constexpr int A_M_TILE_BYTES = 4096;
  constexpr int A_K_TILE_BYTES = 1024;
  const int a_lds_offset =
      current_slot * A_STAGE_BYTES + m_tile * A_M_TILE_BYTES +
      k_read * A_K_TILE_BYTES + col_id * 256 + row_id * 16;
  ds_read_b128(A_lds, a_lds_offset, A_frag.int4_array[0]);
}

__forceinline__ __device__ void set_mmac_prio() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_setprio 1\n" ::: "memory");
}

__forceinline__ __device__ void clear_mmac_prio() {
  asm volatile("s_setprio 0\n" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

template<int M_TILES>
__forceinline__ __device__ void mmac_rolling_mtile2_bn512(
    char* A_lds,
    union_vec_opt<char, 64> B_reg[4],
    intx4 C_reg[1][M_TILES * 4],
    const int current_slot,
    const int row_id,
    const int col_id) {
  static_assert(M_TILES == 4 || M_TILES == 8,
                "BM64/BM128 BN512 expects 4 or 8 M tiles");
#pragma unroll
  for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
    union_vec_opt<char, 16> A_frag[2];
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[0], current_slot, m_tile, 0, row_id, col_id);
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[1], current_slot, m_tile, 1, row_id, col_id);
    lgkmcnt_wait(1);
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int c_idx = m_tile * 4 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[kk]),
            *(vec<char, 8>*)(&A_frag[0].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[0], current_slot, m_tile, 2, row_id, col_id);
    lgkmcnt_wait(1);
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int c_idx = m_tile * 4 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k_tile = 2 + kk;
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[k_tile]),
            *(vec<char, 8>*)(&A_frag[1].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[1], current_slot, m_tile, 3, row_id, col_id);
    lgkmcnt_wait(1);
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int c_idx = m_tile * 4 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k_tile = 4 + kk;
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[k_tile]),
            *(vec<char, 8>*)(&A_frag[0].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
    lgkmcnt_wait(0);
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int c_idx = m_tile * 4 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k_tile = 6 + kk;
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[k_tile]),
            *(vec<char, 8>*)(&A_frag[1].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
  }
}

__forceinline__ __device__ void load_unpack_b_w4_bn512(
    const char* __restrict__ qweight,
    union_vec_opt<char, 64> B_reg[4],
    const int warp_n_id,
    const int lane_id,
    const int k_base,
    const uint32_t size_n) {
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  union {
    vec<int, 4> v;
    uint32_t i[4];
  } tmp[2][4];

#pragma unroll
  for (int n_tile = 0; n_tile < 2; ++n_tile) {
    const int n32 = warp_n_id * 2 + n_tile;
    const int v_offset = n32 * 1024 + row_id * 16 + col_id * 256;
#pragma unroll
    for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
      buffer_load_reg_dwordx4_w4a8(
          qweight, tmp[n_tile][k_chunk].v,
          (k_base / 2) * int(size_n) + k_chunk * 32 * int(size_n),
          v_offset);
    }
  }
  vmcnt_only_wait(0);

#pragma unroll
  for (int n_tile = 0; n_tile < 2; ++n_tile) {
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int b_idx = n_tile * 2 + it;
#pragma unroll
      for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const uint32_t packed = tmp[n_tile][k_chunk].i[it * 2 + pair];
          const uint32_t high = packed & 0xf0f0f0f0u;
          const uint32_t low = (packed << 4) & 0xf0f0f0f0u;
          B_reg[b_idx].int2_array[k_chunk * 2 + pair][0] = *reinterpret_cast<const int*>(&high);
          B_reg[b_idx].int2_array[k_chunk * 2 + pair][1] = *reinterpret_cast<const int*>(&low);
        }
      }
    }
  }
}


template<int M_TILES>
__forceinline__ __device__ void mmac_rolling_mtile2_bn256(
    char* A_lds,
    union_vec_opt<char, 64> B_reg[2],
    intx4 C_reg[1][M_TILES * 2],
    const int current_slot,
    const int row_id,
    const int col_id) {
  static_assert(M_TILES == 8, "BM128/BN256 expects 8 M tiles");
#pragma unroll
  for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
    union_vec_opt<char, 16> A_frag[2];
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[0], current_slot, m_tile, 0, row_id, col_id);
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[1], current_slot, m_tile, 1, row_id, col_id);
    lgkmcnt_wait(1);
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int c_idx = m_tile * 2 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[kk]),
            *(vec<char, 8>*)(&A_frag[0].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[0], current_slot, m_tile, 2, row_id, col_id);
    lgkmcnt_wait(1);
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int c_idx = m_tile * 2 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k_tile = 2 + kk;
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[k_tile]),
            *(vec<char, 8>*)(&A_frag[1].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
    read_a_mtile_kgroup<M_TILES>(A_lds, A_frag[1], current_slot, m_tile, 3, row_id, col_id);
    lgkmcnt_wait(1);
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int c_idx = m_tile * 2 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k_tile = 4 + kk;
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[k_tile]),
            *(vec<char, 8>*)(&A_frag[0].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
    lgkmcnt_wait(0);
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int c_idx = m_tile * 2 + it;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k_tile = 6 + kk;
        C_reg[0][c_idx] = mmac<char>(
            *(vec<char, 8>*)(&B_reg[it].int8t_array[k_tile]),
            *(vec<char, 8>*)(&A_frag[1].int8t_array[kk]),
            C_reg[0][c_idx]);
      }
    }
  }
}

__forceinline__ __device__ void load_unpack_b_w4_bn256(
    const char* __restrict__ qweight,
    union_vec_opt<char, 64> B_reg[2],
    const int warp_n_id,
    const int lane_id,
    const int k_base,
    const uint32_t size_n) {
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  union {
    vec<int, 4> v;
    uint32_t i[4];
  } tmp[4];

  const int v_offset = warp_n_id * 1024 + row_id * 16 + col_id * 256;
#pragma unroll
  for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
    buffer_load_reg_dwordx4_w4a8(
        qweight, tmp[k_chunk].v,
        (k_base / 2) * int(size_n) + k_chunk * 32 * int(size_n),
        v_offset);
  }
  vmcnt_only_wait(0);

#pragma unroll
  for (int it = 0; it < 2; ++it) {
#pragma unroll
    for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const uint32_t packed = tmp[k_chunk].i[it * 2 + pair];
        const uint32_t high = packed & 0xf0f0f0f0u;
        const uint32_t low = (packed << 4) & 0xf0f0f0f0u;
        B_reg[it].int2_array[k_chunk * 2 + pair][0] = *reinterpret_cast<const int*>(&high);
        B_reg[it].int2_array[k_chunk * 2 + pair][1] = *reinterpret_cast<const int*>(&low);
      }
    }
  }
}

template<typename scalar_t, int M_TILES>
__forceinline__ __device__ void compute_direct_7168(
    const char* __restrict__ qweight,
    scalar_t* __restrict__ g_output,
    float* __restrict__ input_scale,
    float* __restrict__ g_weight_scale,
    char* A_lds,
    const int physical_warp_id,
    const int expert_id,
    const int block_m_base,
    const int valid_m,
    const uint32_t size_m,
    const uint32_t size_n,
    const uint32_t stride_asm) {
  constexpr int SIZE_K = 7168;
  static_assert(M_TILES == 4 || M_TILES == 8,
                "compute_direct_7168 supports BM64/BM128");
  constexpr int WARP_M = M_TILES * 16;
  constexpr int WARP_N = 64;
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int K_LOOP_STEP = 256;
  constexpr int KLOOPS = SIZE_K / K_LOOP_STEP;

  const int lane_id = threadIdx.x & 63;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  const int warp_n_id = physical_warp_id;

  union_vec_opt<char, 64> B_reg[4];
  intx4 C_reg[1][M_TILES * (WARP_N / mfma_n)] = {0};

  load_unpack_b_w4_bn512(qweight, B_reg, warp_n_id, lane_id, 0, size_n);
  for (int kloop = 0; kloop < KLOOPS; ++kloop) {
    const int a_current_slot = kloop & 1;
    lgkmcnt_wait(0);
    asm volatile("s_barrier" ::: "memory");
    vmcnt_only_wait(0);
    set_mmac_prio();
    mmac_rolling_mtile2_bn512<M_TILES>(A_lds, B_reg, C_reg, a_current_slot, row_id, col_id);
    clear_mmac_prio();
    __builtin_hcu_sched_barrier(0);
    if (kloop + 1 < KLOOPS) {
      load_unpack_b_w4_bn512(
          qweight, B_reg, warp_n_id, lane_id, (kloop + 1) * K_LOOP_STEP,
          size_n);
    }
  }

  __builtin_hcu_sched_barrier(0);
  vmcnt_only_wait(0);

#pragma unroll
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; ++min_tile_m) {
#pragma unroll
    for (int n_pair = 0; n_pair < 2; ++n_pair) {
      const int even_tile_idx = min_tile_m * (WARP_N / mfma_n) + n_pair * 2;
      const int odd_tile_idx = even_tile_idx + 1;
      const int n_idx = warp_n_id * WARP_N + n_pair * 32 + row_id * 2;
      const float b0 = g_weight_scale[n_idx + 0];
      const float b1 = g_weight_scale[n_idx + 1];
#pragma unroll
      for (int reg_id = 0; reg_id < 4; ++reg_id) {
        const int row = block_m_base + min_tile_m * mfma_m + reg_id * 4 + col_id;
        if (row < valid_m) {
          const int row_safe = std::min(row, int(size_m) - 1);
          const float scale = input_scale[(expert_id * int(size_m) + row_safe) * stride_asm];
          vec_element_2<scalar_t> packed_store_value;
          packed_store_value.data[0] = b32_to_b16<scalar_t>(
              float(C_reg[0][even_tile_idx][reg_id]) * scale * b0);
          packed_store_value.data[1] = b32_to_b16<scalar_t>(
              float(C_reg[0][odd_tile_idx][reg_id]) * scale * b1);
          *reinterpret_cast<vec_element_2<scalar_t>*>(
              &g_output[uint64_t(row) * size_n + uint64_t(n_idx)]) = packed_store_value;
        }
      }
    }
  }

}



template<typename scalar_t>
__forceinline__ __device__ void compute_direct_bn256_7168(
    const char* __restrict__ qweight,
    scalar_t* __restrict__ g_output,
    float* __restrict__ input_scale,
    float* __restrict__ g_weight_scale,
    char* A_lds,
    const int physical_warp_id,
    const int expert_id,
    const int block_m_base,
    const int valid_m,
    const uint32_t size_m,
    const uint32_t size_n,
    const uint32_t stride_asm) {
  constexpr int SIZE_K = 7168;
  constexpr int M_TILES = 8;
  constexpr int WARP_M = 128;
  constexpr int WARP_N = 32;
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int K_LOOP_STEP = 256;
  constexpr int KLOOPS = SIZE_K / K_LOOP_STEP;

  const int lane_id = threadIdx.x & 63;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  const int warp_n_id = physical_warp_id;

  union_vec_opt<char, 64> B_reg[2];
  intx4 C_reg[1][M_TILES * (WARP_N / mfma_n)] = {0};

  load_unpack_b_w4_bn256(qweight, B_reg, warp_n_id, lane_id, 0, size_n);
  for (int kloop = 0; kloop < KLOOPS; ++kloop) {
    const int a_current_slot = kloop & 1;
    lgkmcnt_wait(0);
    asm volatile("s_barrier" ::: "memory");
    vmcnt_only_wait(0);
    set_mmac_prio();
    mmac_rolling_mtile2_bn256<M_TILES>(A_lds, B_reg, C_reg, a_current_slot, row_id, col_id);
    clear_mmac_prio();
    __builtin_hcu_sched_barrier(0);
    if (kloop + 1 < KLOOPS) {
      load_unpack_b_w4_bn256(
          qweight, B_reg, warp_n_id, lane_id, (kloop + 1) * K_LOOP_STEP,
          size_n);
    }
  }

  __builtin_hcu_sched_barrier(0);
  vmcnt_only_wait(0);

#pragma unroll
  for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; ++min_tile_m) {
    const int even_tile_idx = min_tile_m * (WARP_N / mfma_n);
    const int odd_tile_idx = even_tile_idx + 1;
    const int n_idx = warp_n_id * WARP_N + row_id * 2;
    const float b0 = g_weight_scale[n_idx + 0];
    const float b1 = g_weight_scale[n_idx + 1];
#pragma unroll
    for (int reg_id = 0; reg_id < 4; ++reg_id) {
      const int row = block_m_base + min_tile_m * mfma_m + reg_id * 4 + col_id;
      if (row < valid_m) {
        const int row_safe = std::min(row, int(size_m) - 1);
        const float scale = input_scale[(expert_id * int(size_m) + row_safe) * stride_asm];
        vec_element_2<scalar_t> packed_store_value;
        packed_store_value.data[0] = b32_to_b16<scalar_t>(
            float(C_reg[0][even_tile_idx][reg_id]) * scale * b0);
        packed_store_value.data[1] = b32_to_b16<scalar_t>(
            float(C_reg[0][odd_tile_idx][reg_id]) * scale * b1);
        *reinterpret_cast<vec_element_2<scalar_t>*>(
            &g_output[uint64_t(row) * size_n + uint64_t(n_idx)]) = packed_store_value;
      }
    }
  }
}

template<typename scalar_t, int BLOCK_M>
__forceinline__ __device__ void run_direct_7168(
    const char* __restrict__ input,
    const char* __restrict__ qweight,
    scalar_t* __restrict__ output,
    float* __restrict__ input_scale,
    float* __restrict__ weight_scale,
    const int32_t* __restrict__ masked_m,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t stride_asm,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t blocks_per_expert,
    char* A_lds) {
  constexpr int SIZE_K = 7168;
  constexpr int BLOCK_N = 512;
  constexpr int M_TILES = BLOCK_M / 16;
  static_assert(BLOCK_M == 64 || BLOCK_M == 128,
                "run_direct_7168 supports BM64/BM128");
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int expert_id = bidx / int(blocks_per_expert);
  const int block_in_expert = bidx - expert_id * int(blocks_per_expert);
  const int block_m_base = block_in_expert * BLOCK_M;
  const int valid_m = masked_m[expert_id];
  if (block_m_base >= valid_m) return;

  const int physical_warp_id = __builtin_hcu_readfirstlane(int(threadIdx.x / 64));
  const uint64_t expert_offset = uint64_t(size_n) * SIZE_K / 2 * expert_id;
  const uint64_t qweight_offset = expert_offset + uint64_t(bidy) * BLOCK_N * 32;
  const char* g_qweight = qweight + qweight_offset;
  scalar_t* g_output = output + uint64_t(expert_id) * size_m * size_n + uint64_t(bidy) * BLOCK_N;
  const uint64_t weight_scale_offset =
      uint64_t(stride_bse) * stride_bsn * expert_id + uint64_t(bidy) * BLOCK_N;
  float* g_weight_scale = weight_scale + weight_scale_offset;

  if (physical_warp_id < kComputeWaves) {
    compute_direct_7168<scalar_t, M_TILES>(
        g_qweight, g_output, input_scale, g_weight_scale, A_lds,
        physical_warp_id, expert_id, block_m_base, valid_m, size_m, size_n,
        stride_asm);
  } else {
    load_a_tiles_direct<M_TILES>(
        input, A_lds, physical_warp_id, expert_id, block_m_base, valid_m,
        size_m);
  }
}



template<typename scalar_t>
__forceinline__ __device__ void run_direct_bn256_7168(
    const char* __restrict__ input,
    const char* __restrict__ qweight,
    scalar_t* __restrict__ output,
    float* __restrict__ input_scale,
    float* __restrict__ weight_scale,
    const int32_t* __restrict__ masked_m,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t stride_asm,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t blocks_per_expert,
    char* A_lds) {
  constexpr int SIZE_K = 7168;
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 256;
  constexpr int M_TILES = 8;
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int expert_id = bidx / int(blocks_per_expert);
  const int block_in_expert = bidx - expert_id * int(blocks_per_expert);
  const int block_m_base = block_in_expert * BLOCK_M;
  const int valid_m = masked_m[expert_id];
  if (block_m_base >= valid_m) return;

  const int physical_warp_id = __builtin_hcu_readfirstlane(int(threadIdx.x / 64));
  const uint64_t expert_offset = uint64_t(size_n) * SIZE_K / 2 * expert_id;
  const uint64_t qweight_offset = expert_offset + uint64_t(bidy) * BLOCK_N * 32;
  const char* g_qweight = qweight + qweight_offset;
  scalar_t* g_output = output + uint64_t(expert_id) * size_m * size_n + uint64_t(bidy) * BLOCK_N;
  const uint64_t weight_scale_offset =
      uint64_t(stride_bse) * stride_bsn * expert_id + uint64_t(bidy) * BLOCK_N;
  float* g_weight_scale = weight_scale + weight_scale_offset;

  if (physical_warp_id < kComputeWaves) {
    compute_direct_bn256_7168<scalar_t>(
        g_qweight, g_output, input_scale, g_weight_scale, A_lds,
        physical_warp_id, expert_id, block_m_base, valid_m, size_m, size_n,
        stride_asm);
  } else {
    load_a_tiles_direct<M_TILES>(
        input, A_lds, physical_warp_id, expert_id, block_m_base, valid_m,
        size_m);
  }
}

}  // namespace detail

template<typename scalar_t, int BLOCK_M>
__global__ __launch_bounds__(kThreads, 1) void kernel_direct_7168(
    const char* __restrict__ input,
    const char* __restrict__ qweight,
    scalar_t* __restrict__ output,
    float* __restrict__ input_scale,
    float* __restrict__ weight_scale,
    const int32_t* __restrict__ masked_m,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t stride_asm,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t blocks_per_expert) {
  extern __shared__ char smem[];
  detail::run_direct_7168<scalar_t, BLOCK_M>(
      input, qweight, output, input_scale, weight_scale, masked_m, size_m,
      size_n, stride_asm, stride_bse, stride_bsn, blocks_per_expert, smem);
}



template<typename scalar_t>
__global__ __launch_bounds__(kThreads, 1) void kernel_direct_bn256_7168(
    const char* __restrict__ input,
    const char* __restrict__ qweight,
    scalar_t* __restrict__ output,
    float* __restrict__ input_scale,
    float* __restrict__ weight_scale,
    const int32_t* __restrict__ masked_m,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t stride_asm,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t blocks_per_expert) {
  extern __shared__ char smem[];
  detail::run_direct_bn256_7168<scalar_t>(
      input, qweight, output, input_scale, weight_scale, masked_m, size_m,
      size_n, stride_asm, stride_bse, stride_bsn, blocks_per_expert, smem);
}

}  // namespace w4a8_nwave_bm128_bn512

template <typename T, typename T_hidden>
void launch_moe_w4a8_nwave_bm128_bn512_masked_direct(
    const GemmParams_w4a8<T, T_hidden>& params) {
  if (params.size_k != 7168 || params.size_n % 512 != 0) {
    launch_moe_w4a8_first_stage_prefill_masked_direct<
        128, 128, 64, 128, 32, 64, 2>(params);
    return;
  }
  const bool use_bm64 = params.size_m <= 64;
  const int block_m = use_bm64 ? 64 : 128;
  const int blocks_per_expert = DIVIDE(params.size_m, block_m);
  const dim3 block(w4a8_nwave_bm128_bn512::kThreads, 1, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  if (use_bm64) {
    const dim3 grid(1, DIVIDE(params.size_n, 512), params.num_tokens_post_pad * blocks_per_expert);
    w4a8_nwave_bm128_bn512::kernel_direct_7168<T_hidden, 64><<<
        grid, block, w4a8_nwave_bm128_bn512::kSharedBytes, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.expert_ids,
        params.size_m,
        params.size_n,
        params.stride_asm,
        params.stride_bse,
        params.stride_bsn,
        blocks_per_expert);
  } else {
    const dim3 grid(1, DIVIDE(params.size_n, 256), params.num_tokens_post_pad * blocks_per_expert);
    w4a8_nwave_bm128_bn512::kernel_direct_bn256_7168<T_hidden><<<
        grid, block, w4a8_nwave_bm128_bn512::kSharedBytes, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.expert_ids,
        params.size_m,
        params.size_n,
        params.stride_asm,
        params.stride_bse,
        params.stride_bsn,
        blocks_per_expert);
  }
}
