// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <c10/core/DeviceGuard.h>

#include <cstdint>
#include <cstdlib>

namespace gfx92a::m_grouped_int8_gemm_nt_masked_pair32_bm128_bn256_bk128 {

using int8x8 = int8_t __attribute__((ext_vector_type(8)));
using int32x4 = int32_t __attribute__((ext_vector_type(4)));
using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));
using bf16x8 = uint16_t __attribute__((ext_vector_type(8)));

constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 128;
constexpr int kProducerWaves = 0;
constexpr int kALoaderWaves = 4;
constexpr int kConsumerWaves = 8;
constexpr int kWaveSize = 64;
constexpr int kThreads = (kProducerWaves + kConsumerWaves) * kWaveSize;
constexpr int kProducerRows = kBlockM / kALoaderWaves;
constexpr int kMSubtiles = kBlockM / 16;
constexpr int kAStageBytes = kBlockM * kBlockK;
constexpr int kASlabBytes = kProducerRows * kBlockK;
constexpr int kSharedBytes = 2 * kAStageBytes;
static_assert(kBlockM == 128);

static_assert(kALoaderWaves <= kConsumerWaves);
static_assert(kSharedBytes <= 65536);
static_assert((kBlockM + kBlockN) * sizeof(float) <= kAStageBytes);

union Int8Fragment {
  int8x8 i8[2];
  int32x4 words;
};

struct BPacket {
  Int8Fragment fragment[4];
};

struct Params {
  const int8_t* inputs;
  const int8_t* weights;
  const float* a_scale;
  const float* b_scale;
  const int32_t* masked_m;
  uint16_t* output;
  int experts;
  int size_m;
  int size_n;
  int size_k;
};

__device__ __forceinline__ void block_barrier() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_barrier\n" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_global() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_lds() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

template <int Pending>
__device__ __forceinline__ void wait_lds_pending() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(Pending) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ uint32x4 buffer_resource(const void* pointer,
                                                  uint32_t size_bytes) {
  const uint64_t address = reinterpret_cast<uint64_t>(pointer);
  return uint32x4{static_cast<uint32_t>(address),
                  static_cast<uint32_t>(address >> 32),
                  size_bytes, 0x00020000u};
}

__device__ __forceinline__ void load_a_stage(
    int8_t* shared, const uint32x4& resource, int producer_id,
    int lane_id, int size_k, int stage_id) {
  const int row_group = lane_id >> 3;
  const int k_panel = ((lane_id & 7) ^ row_group) * 16;
  const int stage_offset = (stage_id & 1) * kAStageBytes;
  const int producer_row = producer_id * kProducerRows;
#pragma unroll
  for (int panel = 0; panel < kProducerRows / 8; ++panel) {
    const int global_offset =
        (producer_row + row_group + panel * 8) * size_k +
        stage_id * kBlockK + k_panel;
    int8_t* destination = shared + stage_offset + producer_id * kASlabBytes +
                          panel * 1024;
    __builtin_amdgcn_raw_buffer_load_lds(
        resource,
        (__attribute__((address_space(3))) int*)destination,
        16, global_offset, 0, 0, 0);
  }
}

template <int StageOffset>
__device__ __forceinline__ void read_a_half(
    uint32_t base, Int8Fragment (&fragments)[kMSubtiles]) {
  asm volatile(
      "ds_read_b128 %0, %8 offset:%c9\n\t"
      "ds_read_b128 %1, %8 offset:%c10\n\t"
      "ds_read_b128 %2, %8 offset:%c11\n\t"
      "ds_read_b128 %3, %8 offset:%c12\n\t"
      "ds_read_b128 %4, %8 offset:%c13\n\t"
      "ds_read_b128 %5, %8 offset:%c14\n\t"
      "ds_read_b128 %6, %8 offset:%c15\n\t"
      "ds_read_b128 %7, %8 offset:%c16\n"
      : "=&v"(fragments[0].words), "=&v"(fragments[1].words),
        "=&v"(fragments[2].words), "=&v"(fragments[3].words),
        "=&v"(fragments[4].words), "=&v"(fragments[5].words),
        "=&v"(fragments[6].words), "=&v"(fragments[7].words)
      : "v"(base), "n"(StageOffset), "n"(StageOffset + 2048),
        "n"(StageOffset + 4096), "n"(StageOffset + 6144),
        "n"(StageOffset + 8192), "n"(StageOffset + 10240),
        "n"(StageOffset + 12288), "n"(StageOffset + 14336)
      : "memory");
}

__device__ __forceinline__ void load_b_packet(
    BPacket& packet, const uint32x4& resource, int stage_id,
    int size_n, int consumer_id, int col_base, int lane_id) {
  const int n32_count = size_n / 32;
  const int n32_band = col_base / 32 + consumer_id;
  const int row = lane_id & 15;
  const int col_group = lane_id >> 4;
#pragma unroll
  for (int fragment = 0; fragment < 4; ++fragment) {
    const int k64_tile = stage_id * 2 + fragment / 2;
    const int pair = fragment & 1;
    const int offset =
        (((((k64_tile * n32_count + n32_band) * 2 + pair) * 4 +
            col_group) * 16 + row) * 16);
    asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:0\n"
                 : "=v"(packet.fragment[fragment].words)
                 : "v"(offset), "s"(resource)
                 : "memory");
  }
}

template <int Begin = 0, int End = kMSubtiles>
__device__ __forceinline__ void accumulate_half(
    int32x4 (&accumulators)[kMSubtiles][2],
    const Int8Fragment (&a_fragments)[kMSubtiles],
    const BPacket& packet, int half, int valid_rows) {
  asm volatile("" : "+s"(valid_rows) : : "memory");
#pragma unroll
  for (int subtile = Begin; subtile < End; ++subtile) {
    if (subtile != 0 && subtile * 16 >= valid_rows) {
      return;
    }
    {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const Int8Fragment& weight = packet.fragment[half * 2 + pair];
        accumulators[subtile][pair] = __builtin_hcu_mmac_i32_16x16x32_i8(
            a_fragments[subtile].i8[0], weight.i8[0],
            accumulators[subtile][pair]);
        accumulators[subtile][pair] = __builtin_hcu_mmac_i32_16x16x32_i8(
            a_fragments[subtile].i8[1], weight.i8[1],
            accumulators[subtile][pair]);
      }
    }
  }
}

__device__ __forceinline__ uint16_t bf16_bits(float value) {
  union FloatBits {
    float floating;
    uint32_t bits;
  } converted;
  converted.floating = value;
  converted.bits += 0x7fffu + ((converted.bits >> 16) & 1u);
  return static_cast<uint16_t>(converted.bits >> 16);
}

__device__ __forceinline__ void consumer_body(
    const Params& params, int8_t* shared, int expert, int row_base,
    int col_base, int valid_m, int consumer_id, int lane_id) {
  const int row_in_tile = lane_id & 15;
  const int col_group = lane_id >> 4;
  const int stage_count = params.size_k / kBlockK;
  const int valid_rows = min(kBlockM, valid_m - row_base);
  int32x4 accumulators[kMSubtiles][2] = {};
  Int8Fragment a_fragments[kMSubtiles];
  BPacket b_packets[2];
  const uint32x4 weight_resource = buffer_resource(
      params.weights + static_cast<int64_t>(expert) * params.size_n * params.size_k,
      static_cast<uint32_t>(params.size_n * params.size_k));
  const uint32_t shared_base = static_cast<uint32_t>(
      reinterpret_cast<uintptr_t>(shared));
  const uint32_t a_base_first = shared_base + row_in_tile * kBlockK +
      ((col_group ^ (row_in_tile & 7)) * 16);
  const uint32_t a_base_second = shared_base + row_in_tile * kBlockK +
      (((col_group + 4) ^ (row_in_tile & 7)) * 16);

  const int64_t a_offset =
      (static_cast<int64_t>(expert) * params.size_m + row_base) * params.size_k;
  const uint32x4 a_resource = buffer_resource(
      params.inputs + a_offset, static_cast<uint32_t>(valid_rows * params.size_k));
  auto load_next_a = [&](int stage) {
    if (consumer_id * kProducerRows < valid_rows) {
      load_a_stage(shared, a_resource, consumer_id, lane_id, params.size_k, stage);
    }
  };
  load_next_a(0);
  load_b_packet(b_packets[0], weight_resource, 0, params.size_n,
                consumer_id, col_base, lane_id);
  wait_global();
  block_barrier();
  auto stage_cooperative = [&]<int Current>(int stage_id) {
    constexpr int Next = Current ^ 1;
    if (stage_id + 1 < stage_count) {
      load_next_a(stage_id + 1);
    }
    read_a_half<Current * kAStageBytes>(a_base_first, a_fragments);
    wait_lds_pending<4>();
    accumulate_half<0, 4>(accumulators, a_fragments, b_packets[Current], 0, valid_rows);
    wait_lds();
    accumulate_half<4, 8>(accumulators, a_fragments, b_packets[Current], 0, valid_rows);
    read_a_half<Current * kAStageBytes>(a_base_second, a_fragments);
    if (stage_id + 1 < stage_count) {
      load_b_packet(b_packets[Next], weight_resource, stage_id + 1,
                    params.size_n, consumer_id, col_base, lane_id);
    }
    wait_lds_pending<4>();
    accumulate_half<0, 4>(accumulators, a_fragments, b_packets[Current], 1, valid_rows);
    wait_lds();
    accumulate_half<4, 8>(accumulators, a_fragments, b_packets[Current], 1, valid_rows);
    wait_global();
    block_barrier();
  };
#pragma unroll 1
  for (int stage = 0; stage < stage_count; stage += 2) {
    stage_cooperative.template operator()<0>(stage);
    if (stage + 1 < stage_count) {
      stage_cooperative.template operator()<1>(stage + 1);
    }
  }
  float* scale_destination = reinterpret_cast<float*>(
      shared + (stage_count & 1) * kAStageBytes);
  const int consumer_thread = consumer_id * kWaveSize + lane_id;
  if (consumer_thread < kBlockM) {
    const int row = row_base + consumer_thread;
    scale_destination[consumer_thread] = row < valid_m
        ? params.a_scale[static_cast<int64_t>(expert) * params.size_m + row]
        : 0.0f;
  }
  if (consumer_thread < kBlockN) {
    scale_destination[kBlockM + consumer_thread] = params.b_scale[
        static_cast<int64_t>(expert) * params.size_n + col_base + consumer_thread];
  }
  wait_global();
  wait_lds();
  block_barrier();
  const float* scales = reinterpret_cast<const float*>(
      shared + (stage_count & 1) * kAStageBytes);
  const int local_col = consumer_id * 32 + col_group * 8;
  float weight_scales[8];
#pragma unroll
  for (int column = 0; column < 8; ++column) {
    weight_scales[column] = scales[kBlockM + local_col + column];
  }
#pragma unroll
  for (int subtile = 0; subtile < kMSubtiles; ++subtile) {
    const int local_row = subtile * 16 + row_in_tile;
    if (local_row < valid_rows) {
      const float row_scale = scales[local_row];
      bf16x8 result;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        result[column * 2] = bf16_bits(
            static_cast<float>(accumulators[subtile][0][column]) *
            row_scale * weight_scales[column * 2]);
        result[column * 2 + 1] = bf16_bits(
            static_cast<float>(accumulators[subtile][1][column]) *
            row_scale * weight_scales[column * 2 + 1]);
      }
      const int64_t output_offset =
          (static_cast<int64_t>(expert) * params.size_m + row_base + local_row) *
          params.size_n + col_base + local_col;
      *reinterpret_cast<bf16x8*>(params.output + output_offset) = result;
    }
  }
}

__global__ __launch_bounds__(kThreads, 1)
void w8a8_int8_masked_cooperative_bm128_persistent(Params params) {
  extern __shared__ int8_t shared[];
  const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
  const int lane_id = threadIdx.x & (kWaveSize - 1);
  int task_id = blockIdx.x;
  int expert_task_begin = 0;
  const int n_tiles = params.size_n / kBlockN;
  for (int expert = 0; expert < params.experts; ++expert) {
    int valid_m = __builtin_amdgcn_readfirstlane(params.masked_m[expert]);
    valid_m = max(0, min(valid_m, params.size_m));
    const int m_tiles = valid_m / kBlockM + (valid_m % kBlockM != 0);
    const int expert_task_end = expert_task_begin + m_tiles * n_tiles;
    while (task_id >= expert_task_begin && task_id < expert_task_end) {
      const int local_task = task_id - expert_task_begin;
      const int row_base = (local_task / n_tiles) * kBlockM;
      const int col_base = (local_task % n_tiles) * kBlockN;
      if (params.size_k == 0) {
        const int tile_rows = min(kBlockM, valid_m - row_base);
        for (int element = threadIdx.x; element < tile_rows * kBlockN;
             element += kThreads) {
          const int64_t offset =
              (static_cast<int64_t>(expert) * params.size_m + row_base +
               element / kBlockN) * params.size_n + col_base + element % kBlockN;
          params.output[offset] = 0;
        }
      } else {
        consumer_body(params, shared, expert, row_base, col_base,
                      valid_m, wave_id, lane_id);
      }
      block_barrier();
      task_id += gridDim.x;
    }
    expert_task_begin = expert_task_end;
  }
}

pybind11::dict resources() {
  int device = 0;
  C10_HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  C10_HIP_CHECK(hipGetDeviceProperties(&properties, device));
  hipFuncAttributes attributes{};
  C10_HIP_CHECK(hipFuncGetAttributes(
      &attributes,
      reinterpret_cast<const void*>(w8a8_int8_masked_cooperative_bm128_persistent)));
  int resident_blocks = 0;
  C10_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident_blocks,
      w8a8_int8_masked_cooperative_bm128_persistent,
      kThreads, kSharedBytes));
  pybind11::dict result;
  result["arch"] = properties.gcnArchName;
  result["cu_count"] = properties.multiProcessorCount;
  result["block_m"] = kBlockM;
  result["block_n"] = kBlockN;
  result["block_k"] = kBlockK;
  result["cooperative"] = true;
  result["a_loader_waves"] = kALoaderWaves;

  result["producer_waves"] = kProducerWaves;
  result["c_waves"] = kConsumerWaves / 2;
  result["cp_waves"] = kConsumerWaves / 2;
  result["threads_per_block"] = kThreads;
  result["vgprs_per_thread"] = attributes.numRegs;
  result["local_bytes_per_thread"] = attributes.localSizeBytes;
  result["static_lds_bytes"] = attributes.sharedSizeBytes;
  result["dynamic_lds_bytes"] = kSharedBytes;
  result["active_blocks_per_cu"] = resident_blocks;
  result["max_threads_per_block"] = attributes.maxThreadsPerBlock;
  return result;
}

void launch(torch::Tensor inputs, torch::Tensor packed_b,
            torch::Tensor a_scale, torch::Tensor b_scale,
            torch::Tensor masked_m, torch::Tensor output,
            int64_t blocks_per_cu = 2, int64_t grid_blocks = 0) {
  TORCH_CHECK(inputs.is_cuda(), "inputs must be inputs HIP tensor");
  const c10::hip::HIPGuardMasqueradingAsCUDA guard(inputs.device());
  for (const auto& tensor : {inputs, packed_b, a_scale, b_scale, masked_m, output}) {
    TORCH_CHECK(tensor.device() == inputs.device(), "all tensors must share inputs device");
    TORCH_CHECK(tensor.is_contiguous(), "all tensors must be contiguous");
  }
  TORCH_CHECK(inputs.scalar_type() == at::kChar && packed_b.scalar_type() == at::kChar,
              "inputs and packed_b must be int8");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat &&
                  b_scale.scalar_type() == at::kFloat, "scales must be float32");
  TORCH_CHECK(masked_m.scalar_type() == at::kInt, "masked_m must be int32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(inputs.dim() == 3, "inputs must be [E,M,K]");
  TORCH_CHECK(packed_b.dim() == 7, "packed_b must be [E,K/64,N/32,2,4,16,16]");
  TORCH_CHECK(a_scale.dim() == 2 || (a_scale.dim() == 3 && a_scale.size(-1) == 1), "a_scale must be [E,M] or [E,M,1]");
  TORCH_CHECK(b_scale.dim() == 2 || (b_scale.dim() == 3 && b_scale.size(-1) == 1), "b_scale must be [E,N] or [E,N,1]");
  TORCH_CHECK(output.dim() == 3, "output must be [E,M,N]");
  TORCH_CHECK(masked_m.dim() == 1, "masked_m must be [E]");
  const int64_t experts = inputs.size(0);
  const int64_t size_m = inputs.size(1);
  const int64_t size_k = inputs.size(2);
  const int64_t size_n = output.size(2);
  TORCH_CHECK(experts <= INT_MAX && size_m <= INT_MAX &&
                  size_n <= INT_MAX && size_k <= INT_MAX / kBlockM,
              "dimensions exceed kernel indexing limits");
  TORCH_CHECK(size_k % kBlockK == 0 && size_n % kBlockN == 0,
              "K must be divisible by 128 and N by 256");
  TORCH_CHECK(size_n * size_k <= INT_MAX, "one packed expert exceeds buffer range");
  TORCH_CHECK(output.size(0) == experts && output.size(1) == size_m, "output shape mismatch");
  TORCH_CHECK(packed_b.size(0) == experts && packed_b.size(1) == size_k / 64 &&
                  packed_b.size(2) == size_n / 32 && packed_b.size(3) == 2 &&
                  packed_b.size(4) == 4 && packed_b.size(5) == 16 &&
                  packed_b.size(6) == 16, "packed_b shape mismatch");
  TORCH_CHECK(a_scale.size(0) == experts && a_scale.size(1) == size_m &&
                  b_scale.size(0) == experts && b_scale.size(1) == size_n &&
                  masked_m.numel() == experts, "scale or mask shape mismatch");
  if (experts == 0 || size_m == 0 || size_n == 0) {
    return;
  }
  int device = 0;
  C10_HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  C10_HIP_CHECK(hipGetDeviceProperties(&properties, device));
  if (blocks_per_cu == 0) {
    blocks_per_cu = 2;
  }
  TORCH_CHECK(blocks_per_cu > 0 &&
                  blocks_per_cu <= INT_MAX / properties.multiProcessorCount,
              "invalid blocks_per_cu");
  const int64_t max_m_tiles = size_m / kBlockM + (size_m % kBlockM != 0);
  if (grid_blocks == 0) {
    grid_blocks = properties.multiProcessorCount * blocks_per_cu;
  }
  TORCH_CHECK(grid_blocks > 0 && grid_blocks <= INT_MAX, "invalid grid_blocks");
  TORCH_CHECK(experts <= (INT_MAX - grid_blocks) / max_m_tiles / (size_n / kBlockN),
              "persistent task count exceeds int32 indexing range");
  Params params{
      inputs.data_ptr<int8_t>(), packed_b.data_ptr<int8_t>(),
      a_scale.data_ptr<float>(), b_scale.data_ptr<float>(),
      masked_m.data_ptr<int32_t>(), reinterpret_cast<uint16_t*>(output.data_ptr()),
      static_cast<int>(experts), static_cast<int>(size_m),
      static_cast<int>(size_n), static_cast<int>(size_k)};
  const dim3 grid(grid_blocks);
  const dim3 block(kThreads);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  hipLaunchKernelGGL(w8a8_int8_masked_cooperative_bm128_persistent,
                     grid, block, kSharedBytes, stream, params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}  // namespace m_grouped_int8_gemm_nt_masked_pair32_bm128_bn256_bk128
