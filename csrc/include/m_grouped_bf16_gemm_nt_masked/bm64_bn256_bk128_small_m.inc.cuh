#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace deepgemm::grouped_bf16_gemm_nt_masked_bm64_bn256_bk128_small_m {

using bf16 = __hip_bfloat16;
using fp32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using bf16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bf16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using bf16x2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
using uintx4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;

constexpr int kBlockM = 64;
constexpr int kBlockN = 256;
constexpr int kBlockK = 128;
constexpr int kMmacM = 16;
constexpr int kMmacN = 16;
constexpr int kComputeWaves = 4;
constexpr int kALoaderWaves = 4;
constexpr int kPipelineStages = 2;
constexpr int kAStageElements = kBlockM * kBlockK;

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float value) {
  uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ fp32x4 zero_fp32x4() {
  return fp32x4{0.0f, 0.0f, 0.0f, 0.0f};
}

struct AccFragment {
  fp32x4 value;
};

struct BFragment {
  bf16x4 k0;
  bf16x4 k1;
};

struct BPairFragment {
  BFragment even;
  BFragment odd;
};

struct AccPairFragment {
  AccFragment even;
  AccFragment odd;
};

__device__ __forceinline__ AccFragment zero_acc_fragment() {
  return AccFragment{zero_fp32x4()};
}

__device__ __forceinline__ AccPairFragment zero_acc_pair_fragment() {
  return AccPairFragment{zero_acc_fragment(), zero_acc_fragment()};
}

__device__ __forceinline__ fp32x4 bf16_mmac_16x16x16(const bf16x4& a,
                                                     const bf16x4& b,
                                                     fp32x4 acc) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__)
  return __builtin_hcu_mmac_f32_16x16x16_bf16(a, b, acc);
#elif defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, acc, false, false);
#else
  return acc;
#endif
}

__device__ __forceinline__ void mmac_priority_on() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_setprio 1\n" ::: "memory");
}

__device__ __forceinline__ void mmac_priority_off() {
  asm volatile("s_setprio 0\n" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt_barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void buffer_load_dwordx4_to_lds(
    char* __restrict__ lds_base,
    int lds_byte_offset,
    const uintx4& global_ptr,
    int global_byte_offset) {
  char* lds_ptr = lds_base + lds_byte_offset;
  __builtin_hcu_sched_barrier(0);
  __builtin_hcu_raw_buffer_load_lds(
      global_ptr,
      (__attribute__((address_space(3))) int*)lds_ptr,
      16,
      global_byte_offset,
      0,
      0,
      0);
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ bf16x8 buffer_load_dwordx4_to_vgpr(
    const uintx4& global_ptr,
    int global_byte_offset) {
  uintx4 data_words;
  asm volatile("buffer_load_dwordx4 %0,%1,%2,0 offen offset:0\n"
               : "=v"(data_words)
               : "v"(global_byte_offset), "s"(global_ptr)
               : "memory");
  return *reinterpret_cast<bf16x8*>(&data_words);
}

__device__ __forceinline__ bf16x8 ds_read_b128_bf16(
    const bf16* __restrict__ lds_base,
    int byte_offset) {
  uintx4 data_words;
  const int lds_addr =
      static_cast<int>(reinterpret_cast<size_t>(lds_base)) + byte_offset;
  asm volatile("ds_read_b128 %0, %1\n\t"
               : "=v"(data_words)
               : "v"(lds_addr)
               : "memory");
  return *reinterpret_cast<bf16x8*>(&data_words);
}

__device__ __forceinline__ void wait_lgkmcnt_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_barrier\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

template <int kBlockM, int kBlockK>
__device__ __forceinline__ void load_a_tile_to_lds_4w(
    bf16* __restrict__ a_lds,
    const bf16* __restrict__ input,
    int64_t input_base,
    int m_block_start,
    int k_block_start,
    int valid_m,
    int size_m,
    int size_k,
    int wave_id,
    int lane_id,
    bool tile_has_valid_rows) {
  constexpr int kRowsPerMSubtile = 16;
  constexpr int kKPerLoad = 32;
  constexpr int kElementsPerLane = 8;
  constexpr int kMSubtiles = kBlockM / kRowsPerMSubtile;
  constexpr int kK32Tiles = kBlockK / kKPerLoad;
  constexpr int kMSubtileBytes =
      kRowsPerMSubtile * kBlockK * static_cast<int>(sizeof(bf16));
  constexpr int kK32TileBytes =
      kRowsPerMSubtile * kKPerLoad * static_cast<int>(sizeof(bf16));
  constexpr int kColGroupBytes =
      kRowsPerMSubtile * kElementsPerLane * static_cast<int>(sizeof(bf16));
  constexpr int kLaneBytes = kElementsPerLane * static_cast<int>(sizeof(bf16));

  static_assert(kMSubtiles == 4, "BM64 BN256 BK128 kernel expects BM=64");
  static_assert(kK32Tiles == 4, "BM64 BN256 BK128 kernel expects BK=128");

  if (wave_id >= kALoaderWaves) {
    return;
  }

  const int loader_k32 = wave_id;
  const int row_id = lane_id & 15;
  const int col_group = lane_id >> 4;
  const int local_k = loader_k32 * kKPerLoad +
                      col_group * kElementsPerLane;
  const bool full_m_tile =
      tile_has_valid_rows && (m_block_start + kBlockM <= valid_m) &&
      (m_block_start + kBlockM <= size_m);

  if (full_m_tile) {
    const bf16* tile_input =
        input + input_base + static_cast<int64_t>(m_block_start) * size_k;
    uintx4 global_ptr;
    *reinterpret_cast<uint64_t*>(&global_ptr) =
        reinterpret_cast<uint64_t>(tile_input);
    global_ptr[1] += 0x41000000;
    global_ptr[2] = static_cast<uint32_t>(kBlockM * size_k *
                                          static_cast<int>(sizeof(bf16)) + 16);
    global_ptr[3] = 0x00020000;

#pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
      const int local_row = m_loop * kRowsPerMSubtile + row_id;
      const int global_byte_offset =
          static_cast<int>((static_cast<int64_t>(local_row) * size_k +
                            k_block_start + local_k) *
                           static_cast<int>(sizeof(bf16)));
      const int lds_byte_offset =
          m_loop * kMSubtileBytes + loader_k32 * kK32TileBytes;

      buffer_load_dwordx4_to_lds(reinterpret_cast<char*>(a_lds),
                                  lds_byte_offset,
                                  global_ptr,
                                  global_byte_offset);
    }
    return;
  }

#pragma unroll
  for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
    const int local_row = m_loop * kRowsPerMSubtile + row_id;
    const int global_row = m_block_start + local_row;
    const int global_k = k_block_start + local_k;
    bf16x8 data = {0, 0, 0, 0, 0, 0, 0, 0};
    if (tile_has_valid_rows && global_row < valid_m && global_row < size_m &&
        global_k + 7 < size_k) {
      data = *reinterpret_cast<const bf16x8*>(
          input + input_base + static_cast<int64_t>(global_row) * size_k +
          global_k);
    }
    const int lds_byte_offset =
        m_loop * kMSubtileBytes + loader_k32 * kK32TileBytes +
        col_group * kColGroupBytes + row_id * kLaneBytes;
    *reinterpret_cast<bf16x8*>(
        reinterpret_cast<char*>(a_lds) + lds_byte_offset) = data;
  }
}

template <int kNPairs>
__device__ __forceinline__ void load_b_pairs(
    BPairFragment (&b_pair)[kNPairs],
    const bf16* __restrict__ packed_weight,
    int64_t weight_base,
    int n_block_base,
    int k_pack_iter,
    int size_n,
    int wave_id,
    int row_id,
    int col_group) {
  const int n16_tiles = size_n >> 4;

  const bf16* expert_weight = packed_weight + weight_base;
  uintx4 global_ptr;
  *reinterpret_cast<uint64_t*>(&global_ptr) =
      reinterpret_cast<uint64_t>(expert_weight);
  global_ptr[1] += 0x40800000;
  global_ptr[2] = 0x80000000;
  global_ptr[3] = 0x00020000;

#pragma unroll
  for (int n_pair = 0; n_pair < kNPairs; ++n_pair) {
    const int global_n32 = n_pair * kComputeWaves + wave_id;
    const int n16_even = (n_block_base >> 4) + global_n32 * 2;
    const int n16_odd = n16_even + 1;
    const int lane_offset =
        (col_group * kMmacN + row_id) * 8;

    const int even_byte_offset =
        ((k_pack_iter * n16_tiles + n16_even) * 512 + lane_offset) *
        static_cast<int>(sizeof(bf16));
    const int odd_byte_offset =
        ((k_pack_iter * n16_tiles + n16_odd) * 512 + lane_offset) *
        static_cast<int>(sizeof(bf16));

    const bf16x8 even_b8 =
        buffer_load_dwordx4_to_vgpr(global_ptr, even_byte_offset);
    const bf16x8 odd_b8 =
        buffer_load_dwordx4_to_vgpr(global_ptr, odd_byte_offset);

    b_pair[n_pair].even.k0 = *reinterpret_cast<const bf16x4*>(
        reinterpret_cast<const uint16_t*>(&even_b8) + 0);
    b_pair[n_pair].even.k1 = *reinterpret_cast<const bf16x4*>(
        reinterpret_cast<const uint16_t*>(&even_b8) + 4);
    b_pair[n_pair].odd.k0 = *reinterpret_cast<const bf16x4*>(
        reinterpret_cast<const uint16_t*>(&odd_b8) + 0);
    b_pair[n_pair].odd.k1 = *reinterpret_cast<const bf16x4*>(
        reinterpret_cast<const uint16_t*>(&odd_b8) + 4);
  }
}

__device__ __forceinline__ void compute_m16n16_k32(const bf16x4& a_k0,
                                                   const bf16x4& a_k1,
                                                   const BFragment& b_frag,
                                                   AccFragment& acc) {
  acc.value = bf16_mmac_16x16x16(a_k0, b_frag.k0, acc.value);
  acc.value = bf16_mmac_16x16x16(a_k1, b_frag.k1, acc.value);
}

template <int kMSubtiles, int kNPairsPerWave>
__device__ __forceinline__ void compute_one_k32(
    bf16* __restrict__ a_lds,
    int current_stage,
    int k32,
    int row_id,
    int col_group,
    const BPairFragment (&b_pair)[kNPairsPerWave],
    AccPairFragment (&acc)[kMSubtiles][kNPairsPerWave]) {
  constexpr int kAStageBytes =
      kAStageElements * static_cast<int>(sizeof(bf16));
  const int a_stage_byte_offset = current_stage * kAStageBytes;
  const int k32_id = k32 >> 5;

#pragma unroll

  for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
    const int a_lds_byte_offset =
        a_stage_byte_offset +
        m_loop * kMmacM * kBlockK * static_cast<int>(sizeof(bf16)) +
        k32_id * kMmacM * 32 * static_cast<int>(sizeof(bf16)) +
        col_group * kMmacM * 8 * static_cast<int>(sizeof(bf16)) +
        row_id * 8 * static_cast<int>(sizeof(bf16));
    const bf16x8 a8 = ds_read_b128_bf16(a_lds, a_lds_byte_offset);
    wait_lgkmcnt_zero();

    const bf16x4 a_k0 = *reinterpret_cast<const bf16x4*>(
        reinterpret_cast<const uint16_t*>(&a8) + 0);
    const bf16x4 a_k1 = *reinterpret_cast<const bf16x4*>(
        reinterpret_cast<const uint16_t*>(&a8) + 4);

#pragma unroll
    for (int n_pair = 0; n_pair < kNPairsPerWave; ++n_pair) {
      compute_m16n16_k32(a_k0, a_k1, b_pair[n_pair].even,
                         acc[m_loop][n_pair].even);
      compute_m16n16_k32(a_k0, a_k1, b_pair[n_pair].odd,
                         acc[m_loop][n_pair].odd);
    }
  }
}

__global__ __launch_bounds__(256, 1) void grouped_bf16_gemm_nt_masked_bm64_bn256_bk128_small_m_kernel(
    const bf16* __restrict__ input,
    const bf16* __restrict__ packed_weight,
    bf16* __restrict__ output,
    const int32_t* __restrict__ masked_m,
    int experts,
    int size_m,
    int size_n,
    int size_k,
    int n_tiles,
    int,
    int) {

  extern __shared__ uint16_t a_lds_raw[];
  bf16* a_lds = reinterpret_cast<bf16*>(a_lds_raw);

  const int wave_id = threadIdx.x >> 6;
  const int lane_id = threadIdx.x & 63;
  const int row_id = lane_id & 15;
  const int col_group = lane_id >> 4;

  constexpr int kMSubtiles = kBlockM / kMmacM;
  constexpr int kNPairsPerWave =
      kBlockN / (kComputeWaves * 32);
  static_assert(kMSubtiles == 4, "BM64 BN256 BK128 kernel expects four 16-row M subtiles");
  static_assert(kNPairsPerWave == 2,
                "BM64 BN256 BK128 kernel expects two 32-column N pairs per consumer wave");

  int tile_id = blockIdx.x;
  const int tile_stride = gridDim.x;
  int expert_tile_begin = 0;

  for (int expert_id = 0; expert_id < experts; ++expert_id) {
    const int valid_m = max(0, min(masked_m[expert_id], size_m));
    const int m_tiles_for_expert = ceil_div(valid_m, kBlockM);
    const int expert_active_tiles = m_tiles_for_expert * n_tiles;
    const int expert_tile_end = expert_tile_begin + expert_active_tiles;

    while (tile_id >= expert_tile_begin && tile_id < expert_tile_end) {
      const int local_tile = tile_id - expert_tile_begin;
      const int m_tile = local_tile / n_tiles;
      const int n_tile = local_tile - m_tile * n_tiles;
      const int m_block_start = m_tile * kBlockM;
      const int n_block_base = n_tile * kBlockN;

      const int64_t input_base =
          static_cast<int64_t>(expert_id) * size_m * size_k;
      const int64_t weight_base =
          static_cast<int64_t>(expert_id) * size_n * size_k;

      AccPairFragment acc[kMSubtiles][kNPairsPerWave];

#pragma unroll
      for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
#pragma unroll
        for (int n_pair = 0; n_pair < kNPairsPerWave; ++n_pair) {
          acc[m_loop][n_pair] = zero_acc_pair_fragment();
        }
      }

      load_a_tile_to_lds_4w<kBlockM, kBlockK>(
          a_lds,
          input,
          input_base,
          m_block_start,
          0,
          valid_m,
          size_m,
          size_k,
          wave_id,
          lane_id,
          true);
      wait_vmcnt_barrier();

      int current_stage = 0;
      int next_stage = 1;

      for (int k_base_outer = 0; k_base_outer < size_k;
           k_base_outer += kBlockK) {
        const int next_k_base = k_base_outer + kBlockK;
        const bool has_next_k = next_k_base < size_k;

        if (has_next_k) {
          load_a_tile_to_lds_4w<kBlockM, kBlockK>(
              a_lds + next_stage * kAStageElements,
              input,
              input_base,
              m_block_start,
              next_k_base,
              valid_m,
              size_m,
              size_k,
              wave_id,
              lane_id,
              true);
        }

        mmac_priority_on();

        BPairFragment b_cur[kNPairsPerWave];
        BPairFragment b_next[kNPairsPerWave];

        load_b_pairs<kNPairsPerWave>(
            b_cur,
            packed_weight,
            weight_base,
            n_block_base,
            k_base_outer / 32,
            size_n,
            wave_id,
            row_id,
            col_group);
        wait_vmcnt_zero();

#pragma unroll
        for (int k32 = 0; k32 < kBlockK; k32 += 32) {
          const int next_k32 = k32 + 32;
          const bool has_next_k32 = next_k32 < kBlockK;

          if (has_next_k32) {
            load_b_pairs<kNPairsPerWave>(
                b_next,
                packed_weight,
                weight_base,
                n_block_base,
                (k_base_outer + next_k32) / 32,
                size_n,
                wave_id,
                row_id,
                col_group);
          }

          compute_one_k32<kMSubtiles, kNPairsPerWave>(
              a_lds, current_stage, k32, row_id, col_group, b_cur, acc);

          if (has_next_k32) {
            wait_vmcnt_zero();
#pragma unroll
            for (int n_pair = 0; n_pair < kNPairsPerWave; ++n_pair) {
              b_cur[n_pair] = b_next[n_pair];
            }
          }
        }

        mmac_priority_off();

        if (has_next_k) {
          wait_vmcnt_barrier();
          current_stage = 1 - current_stage;
          next_stage = 1 - next_stage;
        }
      }

#pragma unroll
      for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int row = m_block_start + m_loop * kMmacM + row_id;
        const bool row_valid = row < valid_m && row < size_m;

#pragma unroll
        for (int n_pair = 0; n_pair < kNPairsPerWave; ++n_pair) {
          const int global_n32 = n_pair * kComputeWaves + wave_id;
          const int col = n_block_base + global_n32 * 32 + col_group * 8;

          if (row < size_m && col + 7 < size_n) {
            bf16x8 out8 = {0, 0, 0, 0, 0, 0, 0, 0};
            uint16_t* out_bits = reinterpret_cast<uint16_t*>(&out8);

#pragma unroll
            for (int elem = 0; elem < 4; ++elem) {
              out_bits[elem * 2 + 0] =
                  row_valid
                      ? f32_to_bf16_bits(acc[m_loop][n_pair].even.value[elem])
                      : 0;
              out_bits[elem * 2 + 1] =
                  row_valid
                      ? f32_to_bf16_bits(acc[m_loop][n_pair].odd.value[elem])
                      : 0;
            }

            *reinterpret_cast<bf16x8*>(
                output +
                (static_cast<int64_t>(expert_id) * size_m + row) * size_n +
                col) = out8;
          }
        }
      }
      barrier();
      tile_id += tile_stride;
    }

    expert_tile_begin = expert_tile_end;
  }
}

void launch(torch::Tensor input,
            torch::Tensor packed_weight,
            torch::Tensor output,
            torch::Tensor masked_m) {
  const int experts = static_cast<int>(input.size(0));
  const int size_m = static_cast<int>(input.size(1));
  const int size_k = static_cast<int>(input.size(2));
  const int size_n = static_cast<int>(packed_weight.size(1));

  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bf16");
  TORCH_CHECK(packed_weight.scalar_type() == at::kBFloat16,
              "packed_weight must be bf16");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(masked_m.scalar_type() == at::kInt, "masked_m must be int32");
  TORCH_CHECK(input.dim() == 3, "input must be [E,M,K]");
  TORCH_CHECK(packed_weight.dim() == 3,
              "packed_weight must be [E,N,K] in BM64 BN256 BK128 packed layout");
  TORCH_CHECK(output.dim() == 3, "output must be [E,M,N]");
  TORCH_CHECK(packed_weight.size(0) == experts, "packed_weight E mismatch");
  TORCH_CHECK(packed_weight.size(2) == size_k, "packed_weight K mismatch");
  TORCH_CHECK(output.size(0) == experts, "output E mismatch");
  TORCH_CHECK(output.size(1) == size_m, "output M mismatch");
  TORCH_CHECK(output.size(2) == size_n, "output N mismatch");
  TORCH_CHECK(masked_m.numel() == experts, "masked_m length mismatch");
  TORCH_CHECK((size_n % 256) == 0, "N must be divisible by 256");
  TORCH_CHECK((size_k % 128) == 0, "K must be divisible by 128");

  constexpr int kSharedMemBytes =
      kPipelineStages * kAStageElements * static_cast<int>(sizeof(bf16));
  const int n_tiles = ceil_div(size_n, kBlockN);

  const c10::OptionalDeviceGuard guard(device_of(input));
  int device = 0;
  hipGetDevice(&device);
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device);

  int blocks_per_cu = 2;
  if (const char* env = std::getenv("DEEPGEMM_BF16_MASKED_BLOCKS_PER_CU")) {
    const int parsed = std::atoi(env);
    if (parsed > 0) {
      blocks_per_cu = parsed;
    }
  }
  const int grid_x = prop.multiProcessorCount * blocks_per_cu;

  const dim3 block(256, 1, 1);
  const dim3 grid(grid_x, 1, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  grouped_bf16_gemm_nt_masked_bm64_bn256_bk128_small_m_kernel<<<grid, block, kSharedMemBytes, stream>>>(
      reinterpret_cast<const bf16*>(input.data_ptr()),
      reinterpret_cast<const bf16*>(packed_weight.data_ptr()),
      reinterpret_cast<bf16*>(output.data_ptr()),
      masked_m.data_ptr<int32_t>(),
      experts,
      size_m,
      size_n,
      size_k,
      n_tiles,
      0,
      0);
}

torch::Tensor run(torch::Tensor input,
                                   torch::Tensor packed_weight,
                                   torch::Tensor output,
                                   torch::Tensor masked_m) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA/HIP tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA/HIP tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA/HIP tensor");
  TORCH_CHECK(masked_m.is_cuda(), "masked_m must be CUDA/HIP tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
  launch(input, packed_weight, output, masked_m);
  return output;
}

}
