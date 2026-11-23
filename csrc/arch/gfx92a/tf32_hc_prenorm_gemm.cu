// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <string>

#include "tf32_hc_prenorm_gemm.h"

namespace gfx92a {
namespace deepgemm {

using Float2 = float __attribute__((ext_vector_type(2)));
using Float4 = float __attribute__((ext_vector_type(4)));

__global__ void tf32_hc_prenorm_gemm_gfx92a_small_fp32_kernel(
    const __hip_bfloat16* activation,
    const float* weight,
    float* output,
    float* square_sum,
    int rows,
    int columns,
    int depth,
    int splits) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    const int column = threadIdx.x / 8;
    const int worker = threadIdx.x % 8;
    const int blocks = depth / 64;
    const int blocks_per_split = blocks / splits;
    const int remaining_blocks = blocks % splits;
    const int first_block = split * blocks_per_split + min(split, remaining_blocks);
    const int begin = first_block * 64;
    const int end = begin + (blocks_per_split + (split < remaining_blocks)) * 64;
    float accumulator = 0;
    float square_accumulator = 0;
    for (int inner = begin + worker; inner < end; inner += 8) {
        const float left = static_cast<float>(activation[static_cast<int64_t>(row) * depth + inner]);
        const float right = weight[static_cast<int64_t>(column) * depth + inner];
        accumulator = fmaf(left, right, accumulator);
        square_accumulator = fmaf(left, left, square_accumulator);
    }
    for (int offset = 4; offset > 0; offset /= 2) {
        accumulator += __shfl_down(accumulator, offset, 8);
        square_accumulator += __shfl_down(square_accumulator, offset, 8);
    }
    if (worker == 0) {
        const int64_t split_row = static_cast<int64_t>(split) * rows + row;
        output[split_row * columns + column] = accumulator;
        if (column == 0) {
            square_sum[split_row] = square_accumulator;
        }
    }
}

template <int ColumnTiles>
__global__ void tf32_hc_prenorm_gemm_gfx92a_kernel(
    const __hip_bfloat16* activation,
    const float* weight,
    float* output,
    float* square_sum,
    int rows,
    int columns,
    int depth,
    int splits) {
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int tile_row = lane % 16;
    const int group = lane / 16;
    const int row = blockIdx.x * 64 + wave * 16 + tile_row;
    const int split = blockIdx.y;
    const int blocks = depth / 64;
    const int blocks_per_split = blocks / splits;
    const int remaining_blocks = blocks % splits;
    const int first_block = split * blocks_per_split + min(split, remaining_blocks);
    const int split_blocks = blocks_per_split + (split < remaining_blocks);
    const int begin = first_block * 64;
    const int end = begin + split_blocks * 64;
    Float4 accumulators[ColumnTiles];
#pragma unroll
    for (int tile = 0; tile < ColumnTiles; ++tile) {
        accumulators[tile] = {0, 0, 0, 0};
    }
    float square_accumulator = 0;
    for (int inner = begin; inner < end; inner += 8) {
        const int inner_offset = inner + group * 2;
        Float2 left = {0, 0};
        if (row < rows) {
            const int64_t offset = static_cast<int64_t>(row) * depth + inner_offset;
            left[0] = static_cast<float>(activation[offset]);
            left[1] = static_cast<float>(activation[offset + 1]);
        }
        square_accumulator += left[0] * left[0] + left[1] * left[1];
#pragma unroll
        for (int tile = 0; tile < ColumnTiles; ++tile) {
            const int weight_row = tile * 16 + tile_row;
            Float2 right = {0, 0};
            if (weight_row < columns) {
                const int64_t offset = static_cast<int64_t>(weight_row) * depth + inner_offset;
                right = *reinterpret_cast<const Float2*>(weight + offset);
            }
            accumulators[tile] = __builtin_hcu_mmac_f32_16x16x8_tf32(left, right, accumulators[tile]);
        }
    }
    square_accumulator += __shfl_xor(square_accumulator, 16, 64);
    square_accumulator += __shfl_xor(square_accumulator, 32, 64);
    if (row < rows) {
        const int64_t split_row = static_cast<int64_t>(split) * rows + row;
        if (group == 0) {
            square_sum[split_row] = square_accumulator;
        }
#pragma unroll
        for (int tile = 0; tile < ColumnTiles; ++tile) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int column = tile * 16 + item * 4 + group;
                if (column < columns) {
                    output[split_row * columns + column] = accumulators[tile][item];
                }
            }
        }
    }
}

template <int ColumnTiles, int BlockRows = 64>
__global__ void tf32_hc_prenorm_gemm_gfx92a_tiled_kernel(
    const __hip_bfloat16* __restrict__ activation,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ square_sum,
    int rows,
    int columns,
    int depth,
    int splits) {
    constexpr int BlockThreads = BlockRows * 4;
    using UInt4 = unsigned int __attribute__((ext_vector_type(4)));
    __shared__ __align__(16) unsigned int activation_tile[BlockRows][36];
    __shared__ __align__(16) float weight_tile[ColumnTiles * 16][68];
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int tile_row = lane % 16;
    const int group = lane / 16;
    const int block_row = blockIdx.x * BlockRows;
    const int row = block_row + wave * 16 + tile_row;
    const int split = blockIdx.y;
    const int blocks = depth / 64;
    const int blocks_per_split = blocks / splits;
    const int remaining_blocks = blocks % splits;
    const int first_block = split * blocks_per_split + min(split, remaining_blocks);
    const int split_blocks = blocks_per_split + (split < remaining_blocks);
    const int begin = first_block * 64;
    const int end = begin + split_blocks * 64;
    Float4 accumulators[ColumnTiles];
#pragma unroll
    for (int tile = 0; tile < ColumnTiles; ++tile) {
        accumulators[tile] = {0, 0, 0, 0};
    }
    float square_accumulator = 0;
    for (int block_inner = begin; block_inner < end; block_inner += 64) {
#pragma unroll
        for (int offset = threadIdx.x * 8; offset < BlockRows * 64; offset += BlockThreads * 8) {
            const int local_row = offset / 64;
            const int local_inner = offset % 64;
            UInt4 values = {0, 0, 0, 0};
            if (block_row + local_row < rows) {
                const int64_t address = static_cast<int64_t>(block_row + local_row) * depth + block_inner + local_inner;
                values = *reinterpret_cast<const UInt4*>(activation + address);
            }
            *reinterpret_cast<UInt4*>(&activation_tile[local_row][local_inner / 2]) = values;
        }
#pragma unroll
        for (int offset = threadIdx.x * 4; offset < ColumnTiles * 16 * 64; offset += BlockThreads * 4) {
            const int local_row = offset / 64;
            const int local_inner = offset % 64;
            Float4 values = {0, 0, 0, 0};
            if (local_row < columns) {
                const int64_t address = static_cast<int64_t>(local_row) * depth + block_inner + local_inner;
                values = *reinterpret_cast<const Float4*>(weight + address);
            }
            *reinterpret_cast<Float4*>(&weight_tile[local_row][local_inner]) = values;
        }
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < 64; inner += 8) {
            const unsigned int packed = activation_tile[wave * 16 + tile_row][inner / 2 + group];
            const Float2 left = {__uint_as_float(packed << 16), __uint_as_float(packed & 0xffff0000u)};
            square_accumulator += left[0] * left[0] + left[1] * left[1];
#pragma unroll
            for (int tile = 0; tile < ColumnTiles; ++tile) {
                const Float2 right = *reinterpret_cast<const Float2*>(&weight_tile[tile * 16 + tile_row][inner + group * 2]);
                accumulators[tile] = __builtin_hcu_mmac_f32_16x16x8_tf32(left, right, accumulators[tile]);
            }
        }
        __syncthreads();
    }
    square_accumulator += __shfl_xor(square_accumulator, 16, 64);
    square_accumulator += __shfl_xor(square_accumulator, 32, 64);
    if (row < rows) {
        const int64_t split_row = static_cast<int64_t>(split) * rows + row;
        if (group == 0) {
            square_sum[split_row] = square_accumulator;
        }
#pragma unroll
        for (int tile = 0; tile < ColumnTiles; ++tile) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int column = tile * 16 + item * 4 + group;
                if (column < columns) {
                    output[split_row * columns + column] = accumulators[tile][item];
                }
            }
        }
    }
}

using UInt4 = unsigned int __attribute__((ext_vector_type(4)));

template <int BlockRows, int BlockDepth, int WeightRows, int Threads>
__device__ __forceinline__ void tf32_hc_prefetch_tile(
    const __hip_bfloat16* activation,
    const float* weight,
    int rows,
    int columns,
    int depth,
    int block_row,
    int block_inner,
    int end,
    UInt4 (&left)[(BlockRows * BlockDepth + Threads * 8 - 1) / (Threads * 8)],
    Float4 (&right)[(WeightRows * BlockDepth + Threads * 4 - 1) / (Threads * 4)]) {
#pragma unroll
    for (int item = 0; item < (BlockRows * BlockDepth + Threads * 8 - 1) / (Threads * 8); ++item) {
        const int offset = (threadIdx.x + item * Threads) * 8;
        const int local_row = offset / BlockDepth;
        const int local_inner = offset % BlockDepth;
        left[item] = {0, 0, 0, 0};
        if (offset < BlockRows * BlockDepth && block_row + local_row < rows && block_inner + local_inner < end) {
            const int64_t address = static_cast<int64_t>(block_row + local_row) * depth + block_inner + local_inner;
            left[item] = *reinterpret_cast<const UInt4*>(activation + address);
        }
    }
#pragma unroll
    for (int item = 0; item < (WeightRows * BlockDepth + Threads * 4 - 1) / (Threads * 4); ++item) {
        const int offset = (threadIdx.x + item * Threads) * 4;
        const int local_row = offset / BlockDepth;
        const int local_inner = offset % BlockDepth;
        right[item] = {0, 0, 0, 0};
        if (offset < WeightRows * BlockDepth && local_row < columns && block_inner + local_inner < end) {
            const int64_t address = static_cast<int64_t>(local_row) * depth + block_inner + local_inner;
            right[item] = *reinterpret_cast<const Float4*>(weight + address);
        }
    }
}

template <int BlockRows, int BlockDepth, int WeightRows, int Threads, int SharedPadding,
          bool SwizzleA, bool SwizzleB, bool PackedWeights>
__device__ __forceinline__ void tf32_hc_store_tile(
    const UInt4 (&left)[(BlockRows * BlockDepth + Threads * 8 - 1) / (Threads * 8)],
    const Float4 (&right)[(WeightRows * BlockDepth + Threads * 4 - 1) / (Threads * 4)],
    unsigned int (&activation_tile)[BlockRows][BlockDepth / 2 + SharedPadding],
    unsigned int (&weight_tile)[WeightRows][(PackedWeights ? BlockDepth / 4 * 3 : BlockDepth) + SharedPadding]) {
#pragma unroll
    for (int item = 0; item < (BlockRows * BlockDepth + Threads * 8 - 1) / (Threads * 8); ++item) {
        const int offset = (threadIdx.x + item * Threads) * 8;
        if (offset < BlockRows * BlockDepth) {
            const int tile_row = offset / BlockDepth;
            const int tile_inner = (offset % BlockDepth) / 2;
            const int swizzle = SwizzleA ? (tile_row * 4) & (BlockDepth / 2 - 1) : 0;
            *reinterpret_cast<UInt4*>(&activation_tile[tile_row][tile_inner ^ swizzle]) = left[item];
        }
    }
#pragma unroll
    for (int item = 0; item < (WeightRows * BlockDepth + Threads * 4 - 1) / (Threads * 4); ++item) {
        const int offset = (threadIdx.x + item * Threads) * 4;
        if (offset < WeightRows * BlockDepth) {
            const int tile_row = offset / BlockDepth;
            const int tile_inner = offset % BlockDepth;
            if constexpr (PackedWeights) {
                const int swizzle = SwizzleB ? (tile_row * 4) & (BlockDepth / 4 - 1) : 0;
                const int packed_inner = ((tile_inner / 4) ^ swizzle) * 3;
                const unsigned int first = __float_as_uint(right[item][0]) >> 8;
                const unsigned int second = __float_as_uint(right[item][1]) >> 8;
                const unsigned int third = __float_as_uint(right[item][2]) >> 8;
                const unsigned int fourth = __float_as_uint(right[item][3]) >> 8;
                weight_tile[tile_row][packed_inner] = first | (second << 24);
                weight_tile[tile_row][packed_inner + 1] = (second >> 8) | (third << 16);
                weight_tile[tile_row][packed_inner + 2] = (third >> 16) | (fourth << 8);
            } else {
                const int swizzle = SwizzleB ? (tile_row * 8) & (BlockDepth - 1) : 0;
                *reinterpret_cast<Float4*>(&weight_tile[tile_row][tile_inner ^ swizzle]) = right[item];
            }
        }
    }
}

template <int ColumnTiles, int BlockRows, int BlockDepth, int InnerSplits, bool Prefetch,
          int AccumulatorGroups = 1, int SharedPadding = 4, bool SwizzleA = false, bool SwizzleB = false,
          bool ParallelColumns = false, int WeightRows = ColumnTiles * 16, bool DoubleBuffer = false,
          bool PackedWeights = false, bool IndependentSquares = false, int StepUnroll = 0,
          bool GridColumns = false>
__global__ __launch_bounds__((BlockRows + 15) / 16 * InnerSplits * (ParallelColumns ? ColumnTiles : 1) * 64) void tf32_hc_prenorm_gemm_gfx92a_staged_kernel(
    const __hip_bfloat16* __restrict__ activation,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ square_sum,
    int rows,
    int columns,
    int depth,
    int splits) {
    constexpr int RowWaves = (BlockRows + 15) / 16;
    constexpr int ColumnWaves = ParallelColumns ? ColumnTiles : 1;
    constexpr int ComputedTiles = ParallelColumns ? 1 : ColumnTiles;
    constexpr int Waves = RowWaves * InnerSplits * ColumnWaves;
    constexpr int Threads = Waves * 64;
    constexpr int LeftLoads = (BlockRows * BlockDepth + Threads * 8 - 1) / (Threads * 8);
    constexpr int RightLoads = (WeightRows * BlockDepth + Threads * 4 - 1) / (Threads * 4);
    union SharedStorage {
        struct {
            unsigned int activation_tile[BlockRows][BlockDepth / 2 + SharedPadding];
            unsigned int weight_tile[WeightRows][(PackedWeights ? BlockDepth / 4 * 3 : BlockDepth) + SharedPadding];
        } inputs[DoubleBuffer ? 2 : 1];
        struct {
            Float4 values[Waves][ComputedTiles][64];
            float squares[Waves][16];
        } reduction;
    };
    __shared__ SharedStorage storage;
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int tile_row = lane % 16;
    const int group = lane / 16;
    const int row_wave = wave % RowWaves;
    const int column_wave = wave / RowWaves % ColumnWaves;
    const int inner_split = wave / (RowWaves * ColumnWaves);
    const int block_row = blockIdx.x * BlockRows;
    const int block_column = GridColumns ? blockIdx.z * 16 : 0;
    const int tile_columns = GridColumns ? min(16, columns - block_column) : columns;
    const float* tile_weight = weight + static_cast<int64_t>(block_column) * depth;
    const int local_row = row_wave * 16 + tile_row;
    const int row = block_row + local_row;
    const int split = blockIdx.y;
    const int blocks = depth / 64;
    const int blocks_per_split = blocks / splits;
    const int remaining_blocks = blocks % splits;
    const int first_block = split * blocks_per_split + min(split, remaining_blocks);
    const int split_blocks = blocks_per_split + (split < remaining_blocks);
    const int begin = first_block * 64;
    const int end = begin + split_blocks * 64;
    Float4 accumulators[ComputedTiles][AccumulatorGroups];
#pragma unroll
    for (int tile = 0; tile < ComputedTiles; ++tile) {
#pragma unroll
        for (int accumulator = 0; accumulator < AccumulatorGroups; ++accumulator) {
            accumulators[tile][accumulator] = {0, 0, 0, 0};
        }
    }
    constexpr int SquareGroups = IndependentSquares ? AccumulatorGroups : 1;
    float square_accumulators[SquareGroups] = {};
    UInt4 left_values[LeftLoads];
    Float4 right_values[RightLoads];
    if constexpr (Prefetch || DoubleBuffer) {
        tf32_hc_prefetch_tile<BlockRows, BlockDepth, WeightRows, Threads>(
            activation, tile_weight, rows, tile_columns, depth, block_row, begin, end, left_values, right_values);
    }
    if constexpr (DoubleBuffer) {
        tf32_hc_store_tile<BlockRows, BlockDepth, WeightRows, Threads, SharedPadding, SwizzleA, SwizzleB, PackedWeights>(
            left_values, right_values, storage.inputs[0].activation_tile, storage.inputs[0].weight_tile);
        __syncthreads();
    }
    int current_buffer = 0;
    for (int block_inner = begin; block_inner < end; block_inner += BlockDepth) {
        if constexpr (!DoubleBuffer) {
            if constexpr (!Prefetch) {
                tf32_hc_prefetch_tile<BlockRows, BlockDepth, WeightRows, Threads>(
                    activation, tile_weight, rows, tile_columns, depth, block_row, block_inner, end, left_values, right_values);
            }
            tf32_hc_store_tile<BlockRows, BlockDepth, WeightRows, Threads, SharedPadding, SwizzleA, SwizzleB, PackedWeights>(
                left_values, right_values, storage.inputs[0].activation_tile, storage.inputs[0].weight_tile);
            __syncthreads();
        }
        if constexpr (Prefetch || DoubleBuffer) {
            tf32_hc_prefetch_tile<BlockRows, BlockDepth, WeightRows, Threads>(
                activation, tile_weight, rows, tile_columns, depth, block_row,
                block_inner + BlockDepth, end, left_values, right_values);
        }
        constexpr int InnerSteps = (BlockDepth + InnerSplits * 8 - 1) / (InnerSplits * 8);
        constexpr int MicroSteps = StepUnroll == 0 ? InnerSteps : StepUnroll;
#pragma unroll 1
        for (int outer = 0; outer < (InnerSteps + MicroSteps - 1) / MicroSteps; ++outer) {
#pragma unroll
            for (int micro = 0; micro < MicroSteps; ++micro) {
                const int step = outer * MicroSteps + micro;
                const int inner = inner_split * 8 + step * InnerSplits * 8;
                if (step < InnerSteps && inner < BlockDepth) {
                    const int accumulator = micro % AccumulatorGroups;
                    const int activation_swizzle = SwizzleA ? (local_row * 4) & (BlockDepth / 2 - 1) : 0;
                    unsigned int packed = 0;
                    if (local_row < BlockRows) {
                        packed = storage.inputs[current_buffer].activation_tile[local_row][(inner / 2 + group) ^ activation_swizzle];
                    }
                    const Float2 left = {__uint_as_float(packed << 16), __uint_as_float(packed & 0xffff0000u)};
                    if (column_wave == 0) {
                        const int square_group = IndependentSquares ? accumulator : 0;
                        square_accumulators[square_group] += left[0] * left[0] + left[1] * left[1];
                    }
#pragma unroll
                    for (int tile = 0; tile < ComputedTiles; ++tile) {
                        const int weight_row = (tile + column_wave) * 16 + tile_row;
                        Float2 right = {0, 0};
                        if (weight_row < WeightRows) {
                            if constexpr (PackedWeights) {
                                const int weight_inner = inner + group * 2;
                                const int weight_swizzle = SwizzleB ? (weight_row * 4) & (BlockDepth / 4 - 1) : 0;
                                const int pair = weight_inner % 4 / 2;
                                const int packed_inner = ((weight_inner / 4) ^ weight_swizzle) * 3 + pair;
                                const unsigned int lower = storage.inputs[current_buffer].weight_tile[weight_row][packed_inner];
                                const unsigned int upper = storage.inputs[current_buffer].weight_tile[weight_row][packed_inner + 1];
                                const unsigned int first = pair == 0 ? lower << 8 : ((lower >> 16) | (upper << 16)) << 8;
                                const unsigned int second = pair == 0 ? ((lower >> 24) | (upper << 8)) << 8 : upper & 0xffffff00u;
                                right = {__uint_as_float(first), __uint_as_float(second)};
                            } else {
                                const int weight_swizzle = SwizzleB ? (weight_row * 8) & (BlockDepth - 1) : 0;
                                right = *reinterpret_cast<const Float2*>(&storage.inputs[current_buffer].weight_tile[weight_row][(inner + group * 2) ^ weight_swizzle]);
                            }
                        }
                        accumulators[tile][accumulator] = __builtin_hcu_mmac_f32_16x16x8_tf32(
                            left, right, accumulators[tile][accumulator]);
                    }
                }
            }
        }
        if constexpr (DoubleBuffer) {
            const int next_buffer = current_buffer ^ 1;
            if (block_inner + BlockDepth < end) {
                tf32_hc_store_tile<BlockRows, BlockDepth, WeightRows, Threads, SharedPadding, SwizzleA, SwizzleB, PackedWeights>(
                    left_values, right_values, storage.inputs[next_buffer].activation_tile,
                    storage.inputs[next_buffer].weight_tile);
            }
            __syncthreads();
            current_buffer = next_buffer;
        } else {
            __syncthreads();
        }
    }
    float square_accumulator = square_accumulators[0];
#pragma unroll
    for (int square_group = 1; square_group < SquareGroups; ++square_group) {
        square_accumulator += square_accumulators[square_group];
    }
    square_accumulator += __shfl_xor(square_accumulator, 16, 64);
    square_accumulator += __shfl_xor(square_accumulator, 32, 64);
#pragma unroll
    for (int tile = 0; tile < ComputedTiles; ++tile) {
#pragma unroll
        for (int accumulator = 1; accumulator < AccumulatorGroups; ++accumulator) {
            accumulators[tile][0] += accumulators[tile][accumulator];
        }
    }
    if constexpr (InnerSplits > 1) {
#pragma unroll
        for (int tile = 0; tile < ComputedTiles; ++tile) {
            storage.reduction.values[wave][tile][lane] = accumulators[tile][0];
        }
        if (group == 0 && column_wave == 0) {
            storage.reduction.squares[wave][tile_row] = square_accumulator;
        }
        __syncthreads();
        if (inner_split == 0) {
#pragma unroll 1
            for (int partial = 1; partial < InnerSplits; ++partial) {
#pragma unroll
                for (int tile = 0; tile < ComputedTiles; ++tile) {
                    accumulators[tile][0] += storage.reduction.values[row_wave + (column_wave + partial * ColumnWaves) * RowWaves][tile][lane];
                }
                if (group == 0 && column_wave == 0) {
                    square_accumulator += storage.reduction.squares[row_wave + partial * ColumnWaves * RowWaves][tile_row];
                }
            }
        }
    }
    if (inner_split == 0 && local_row < BlockRows && row < rows) {
        const int64_t split_row = static_cast<int64_t>(split) * rows + row;
        if (group == 0 && column_wave == 0 && block_column == 0) {
            square_sum[split_row] = square_accumulator;
        }
#pragma unroll
        for (int tile = 0; tile < ComputedTiles; ++tile) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int column = block_column + (tile + column_wave) * 16 + item * 4 + group;
                if (column < columns) {
                    output[split_row * columns + column] = accumulators[tile][0][item];
                }
            }
        }
    }
}

template <int ColumnTiles, int BlockRows, int BlockDepth, int InnerSplits, int RowTiles,
          bool Prefetch, int AccumulatorGroups, int StepUnroll, int WeightRows,
          int SharedPadding = 4, bool CompactReduction = false, int ReductionColumns = WeightRows>
__global__ __launch_bounds__((BlockRows + RowTiles * 16 - 1) / (RowTiles * 16) * InnerSplits * 64)
void tf32_hc_prenorm_gemm_gfx92a_row_kernel(
    const __hip_bfloat16* __restrict__ activation,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ square_sum,
    int rows, int columns, int depth, int splits) {
    constexpr int RowWaves = (BlockRows + RowTiles * 16 - 1) / (RowTiles * 16);
    constexpr int Waves = RowWaves * InnerSplits;
    constexpr int Threads = Waves * 64;
    constexpr int ReductionWaves = InnerSplits > 1 ? Waves : 1;
    constexpr int LeftLoads = (BlockRows * BlockDepth + Threads * 8 - 1) / (Threads * 8);
    constexpr int RightLoads = (WeightRows * BlockDepth + Threads * 4 - 1) / (Threads * 4);
    union SharedStorage {
        struct {
            unsigned int activation_tile[BlockRows][BlockDepth / 2 + SharedPadding];
            unsigned int weight_tile[WeightRows][BlockDepth + SharedPadding];
        } inputs;
        struct {
            union {
                Float4 values[CompactReduction ? 1 : ReductionWaves][CompactReduction ? 1 : RowTiles][ColumnTiles][64];
                float compact_values[CompactReduction ? ReductionWaves : 1][CompactReduction ? RowTiles : 1][64][ReductionColumns / 4];
            } matrices;
            float squares[ReductionWaves][RowTiles][16];
        } reduction;
    };
    __shared__ SharedStorage storage;
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int tile_row = lane % 16;
    const int group = lane / 16;
    const int row_wave = wave % RowWaves;
    const int inner_split = wave / RowWaves;
    const int block_row = blockIdx.x * BlockRows;
    const int split = blockIdx.y;
    const int blocks = depth / 64;
    const int blocks_per_split = blocks / splits;
    const int remaining_blocks = blocks % splits;
    const int first_block = split * blocks_per_split + min(split, remaining_blocks);
    const int split_blocks = blocks_per_split + (split < remaining_blocks);
    const int begin = first_block * 64;
    const int end = begin + split_blocks * 64;
    Float4 accumulators[RowTiles][ColumnTiles][AccumulatorGroups];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int tile = 0; tile < ColumnTiles; ++tile) {
#pragma unroll
            for (int accumulator = 0; accumulator < AccumulatorGroups; ++accumulator) {
                accumulators[row_tile][tile][accumulator] = {0, 0, 0, 0};
            }
        }
    }
    float square_accumulators[RowTiles] = {};
    UInt4 left_values[LeftLoads];
    Float4 right_values[RightLoads];
    if constexpr (Prefetch) {
        tf32_hc_prefetch_tile<BlockRows, BlockDepth, WeightRows, Threads>(
            activation, weight, rows, columns, depth, block_row, begin, end, left_values, right_values);
    }
    for (int block_inner = begin; block_inner < end; block_inner += BlockDepth) {
        if constexpr (!Prefetch) {
            tf32_hc_prefetch_tile<BlockRows, BlockDepth, WeightRows, Threads>(
                activation, weight, rows, columns, depth, block_row, block_inner, end, left_values, right_values);
        }
        tf32_hc_store_tile<BlockRows, BlockDepth, WeightRows, Threads, SharedPadding, false, false, false>(
            left_values, right_values, storage.inputs.activation_tile, storage.inputs.weight_tile);
        __syncthreads();
        if constexpr (Prefetch) {
            tf32_hc_prefetch_tile<BlockRows, BlockDepth, WeightRows, Threads>(
                activation, weight, rows, columns, depth, block_row,
                block_inner + BlockDepth, end, left_values, right_values);
        }
        constexpr int InnerSteps = (BlockDepth + InnerSplits * 8 - 1) / (InnerSplits * 8);
        constexpr int MicroSteps = StepUnroll == 0 ? InnerSteps : StepUnroll;
#pragma unroll 1
        for (int outer = 0; outer < (InnerSteps + MicroSteps - 1) / MicroSteps; ++outer) {
#pragma unroll
            for (int micro = 0; micro < MicroSteps; ++micro) {
                const int step = outer * MicroSteps + micro;
                const int inner = inner_split * 8 + step * InnerSplits * 8;
                if (step < InnerSteps && inner < BlockDepth) {
                    Float2 weights[ColumnTiles];
#pragma unroll
                    for (int tile = 0; tile < ColumnTiles; ++tile) {
                        const int weight_row = tile * 16 + tile_row;
                        weights[tile] = {0, 0};
                        if (weight_row < WeightRows) {
                            weights[tile] = *reinterpret_cast<const Float2*>(
                                &storage.inputs.weight_tile[weight_row][inner + group * 2]);
                        }
                    }
#pragma unroll
                    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                        const int local_row = (row_wave * RowTiles + row_tile) * 16 + tile_row;
                        unsigned int packed = 0;
                        if (local_row < BlockRows) {
                            packed = storage.inputs.activation_tile[local_row][inner / 2 + group];
                        }
                        const Float2 left = {__uint_as_float(packed << 16), __uint_as_float(packed & 0xffff0000u)};
                        square_accumulators[row_tile] += left[0] * left[0] + left[1] * left[1];
#pragma unroll
                        for (int tile = 0; tile < ColumnTiles; ++tile) {
                            accumulators[row_tile][tile][micro % AccumulatorGroups] =
                                __builtin_hcu_mmac_f32_16x16x8_tf32(
                                    left, weights[tile], accumulators[row_tile][tile][micro % AccumulatorGroups]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        square_accumulators[row_tile] += __shfl_xor(square_accumulators[row_tile], 16, 64);
        square_accumulators[row_tile] += __shfl_xor(square_accumulators[row_tile], 32, 64);
#pragma unroll
        for (int tile = 0; tile < ColumnTiles; ++tile) {
#pragma unroll
            for (int accumulator = 1; accumulator < AccumulatorGroups; ++accumulator) {
                accumulators[row_tile][tile][0] += accumulators[row_tile][tile][accumulator];
            }
        }
    }
    if constexpr (InnerSplits > 1) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int tile = 0; tile < ColumnTiles; ++tile) {
                if constexpr (CompactReduction) {
#pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        if (tile * 4 + item < ReductionColumns / 4) {
                            storage.reduction.matrices.compact_values[wave][row_tile][lane][tile * 4 + item] =
                                accumulators[row_tile][tile][0][item];
                        }
                    }
                } else {
                    storage.reduction.matrices.values[wave][row_tile][tile][lane] = accumulators[row_tile][tile][0];
                }
            }
            if (group == 0) {
                storage.reduction.squares[wave][row_tile][tile_row] = square_accumulators[row_tile];
            }
        }
        __syncthreads();
        if (inner_split == 0) {
#pragma unroll 1
            for (int partial = 1; partial < InnerSplits; ++partial) {
#pragma unroll
                for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
                    for (int tile = 0; tile < ColumnTiles; ++tile) {
                        if constexpr (CompactReduction) {
#pragma unroll
                            for (int item = 0; item < 4; ++item) {
                                if (tile * 4 + item < ReductionColumns / 4) {
                                    accumulators[row_tile][tile][0][item] +=
                                        storage.reduction.matrices.compact_values[row_wave + partial * RowWaves][row_tile][lane][tile * 4 + item];
                                }
                            }
                        } else {
                            accumulators[row_tile][tile][0] +=
                                storage.reduction.matrices.values[row_wave + partial * RowWaves][row_tile][tile][lane];
                        }
                    }
                    if (group == 0) {
                        square_accumulators[row_tile] +=
                            storage.reduction.squares[row_wave + partial * RowWaves][row_tile][tile_row];
                    }
                }
            }
        }
    }
    if (inner_split == 0) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            const int local_row = (row_wave * RowTiles + row_tile) * 16 + tile_row;
            const int row = block_row + local_row;
            if (local_row < BlockRows && row < rows) {
                const int64_t split_row = static_cast<int64_t>(split) * rows + row;
                if (group == 0) {
                    square_sum[split_row] = square_accumulators[row_tile];
                }
#pragma unroll
                for (int tile = 0; tile < ColumnTiles; ++tile) {
#pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        const int column = tile * 16 + item * 4 + group;
                        if (column < columns) {
                            output[split_row * columns + column] = accumulators[row_tile][tile][0][item];
                        }
                    }
                }
            }
        }
    }
}

void tf32_hc_prenorm_gemm(
    const torch::Tensor& activation,
    const torch::Tensor& weight,
    const torch::Tensor& output,
    const torch::Tensor& square_sum,
    const std::optional<int>& num_splits) {
    TORCH_CHECK(activation.is_cuda() && weight.is_cuda() && output.is_cuda() && square_sum.is_cuda(),
                "all tensors must be GPU tensors");
    TORCH_CHECK(activation.device() == weight.device() && activation.device() == output.device() &&
                activation.device() == square_sum.device(), "all tensors must be on the same device");
    TORCH_CHECK(activation.dim() == 2 && weight.dim() == 2, "activation and weight must be matrices");
    TORCH_CHECK(activation.scalar_type() == torch::kBFloat16, "activation must be BF16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32 &&
                square_sum.scalar_type() == torch::kFloat32, "weight and outputs must be FP32");
    TORCH_CHECK(activation.is_contiguous() && weight.is_contiguous() && output.is_contiguous() &&
                square_sum.is_contiguous(), "all tensors must be contiguous");
    const int rows = activation.size(0);
    const int columns = weight.size(0);
    const int depth = activation.size(1);
    const int splits = num_splits.value_or(1);
    TORCH_CHECK(weight.size(1) == depth, "activation and weight K dimensions must agree");
    TORCH_CHECK(columns > 0 && columns <= 32 && columns % 8 == 0, "N must be 8, 16, 24 or 32");
    TORCH_CHECK(depth > 0 && depth % 64 == 0, "K must be a positive multiple of 64");
    TORCH_CHECK(splits > 0 && splits <= depth / 64, "num_splits must lie in [1, K / 64]");
    TORCH_CHECK((output.dim() == 2 && output.size(0) == static_cast<int64_t>(splits) * rows &&
                 output.size(1) == columns) ||
                (output.dim() == 3 && output.size(0) == splits && output.size(1) == rows &&
                 output.size(2) == columns), "output must have shape [S*M,N] or [S,M,N]");
    TORCH_CHECK((square_sum.dim() == 1 && square_sum.size(0) == static_cast<int64_t>(splits) * rows) ||
                (square_sum.dim() == 2 && square_sum.size(0) == splits && square_sum.size(1) == rows),
                "square_sum must have shape [S*M] or [S,M]");
    c10::cuda::CUDAGuard device_guard(activation.device());
    hipDeviceProp_t properties;
    TORCH_CHECK(hipGetDeviceProperties(&properties, activation.get_device()) == hipSuccess,
                "unable to query GPU architecture");
    TORCH_CHECK(std::string(properties.gcnArchName).rfind("gfx92a", 0) == 0,
                "this extension was built for gfx92a; detected ", properties.gcnArchName);
    if (rows == 0) {
        return;
    }
    const hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto* activation_pointer = reinterpret_cast<const __hip_bfloat16*>(activation.data_ptr<c10::BFloat16>());
    if (rows < 16 && depth <= 128) {
        const dim3 grid(rows, splits);
        tf32_hc_prenorm_gemm_gfx92a_small_fp32_kernel<<<grid, dim3(columns * 8), 0, stream>>>(
            activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
            square_sum.data_ptr<float>(), rows, columns, depth, splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    const bool aligned_inputs = (reinterpret_cast<uintptr_t>(activation_pointer) % 16 == 0) &&
                                (reinterpret_cast<uintptr_t>(weight.data_ptr<float>()) % 16 == 0);
    if (aligned_inputs) {
        if ((depth / 64 / splits <= 1) && (static_cast<int64_t>(rows) <= static_cast<int64_t>(properties.multiProcessorCount) * 4)) {
            const dim3 grid((rows + 31) / 32, splits, 1);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 32, 64, 1, true, 1, 4, false, false, true, 8, false, false, false, 0, false><<<grid, dim3(128), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 32, 64, 1, true, 1, 4, false, false, true, 16, false, false, false, 0, false><<<grid, dim3(128), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 32, 64, 1, true, 1, 4, false, false, true, 24, false, false, false, 0, false><<<grid, dim3(256), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 32, 64, 1, true, 1, 4, false, false, true, 32, false, false, false, 0, false><<<grid, dim3(256), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits <= 1)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_tiled_kernel<1, 64><<<grid, dim3(256), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_tiled_kernel<1, 64><<<grid, dim3(256), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_tiled_kernel<2, 64><<<grid, dim3(256), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_tiled_kernel<2, 64><<<grid, dim3(256), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 16) && (static_cast<int64_t>(rows) <= static_cast<int64_t>(properties.multiProcessorCount) * 1 / splits / ((columns + 15) / 16))) {
            const dim3 grid((rows + 0) / 1, splits, (columns + 15) / 16);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 1, 512, 16, true, 1, 0, true, true, false, 8, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 1, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 1, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 1, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 16) && (static_cast<int64_t>(rows) <= static_cast<int64_t>(properties.multiProcessorCount) * 2 / splits / ((columns + 15) / 16))) {
            const dim3 grid((rows + 1) / 2, splits, (columns + 15) / 16);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 2, 512, 16, true, 1, 0, true, true, false, 8, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 2, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 2, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 2, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 16) && (static_cast<int64_t>(rows) <= static_cast<int64_t>(properties.multiProcessorCount) * 4 / splits / ((columns + 15) / 16))) {
            const dim3 grid((rows + 3) / 4, splits, (columns + 15) / 16);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 4, 512, 16, true, 1, 0, true, true, false, 8, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 4, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 4, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 4, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 16) && (static_cast<int64_t>(rows) <= static_cast<int64_t>(properties.multiProcessorCount) * 8 / splits / ((columns + 15) / 16))) {
            const dim3 grid((rows + 7) / 8, splits, (columns + 15) / 16);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 8, 512, 16, true, 1, 0, true, true, false, 8, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 8, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 8, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 8, 512, 16, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) <= static_cast<int64_t>(properties.multiProcessorCount) * 8 / splits / ((columns + 15) / 16)) && (columns >= 24)) {
            const dim3 grid((rows + 7) / 8, splits, (columns + 15) / 16);
            if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 8, 256, 8, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 8, 256, 8, true, 1, 0, true, true, false, 16, false, false, false, 0, true><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) <= static_cast<int64_t>(properties.multiProcessorCount) * 1)) {
            const dim3 grid((rows + 15) / 16, splits, 1);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 16, 128, 8, true, 1, 4, false, false, false, 16, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 16, 128, 8, true, 1, 4, false, false, false, 16, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 16, 128, 8, true, 1, 4, false, false, false, 32, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 16, 128, 8, true, 1, 4, false, false, false, 32, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 256) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 96) && (columns >= 24) && (columns <= 24)) {
            const dim3 grid((rows + 127) / 128, splits, 1);
            tf32_hc_prenorm_gemm_gfx92a_row_kernel<2, 128, 128, 4, 2, true, 1, 0, 24, 4, true, 24><<<grid, dim3(1024), 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
        else if ((depth / 64 / splits >= 64) && (static_cast<int64_t>(rows) * splits > static_cast<int64_t>(properties.multiProcessorCount) * 32) && (static_cast<int64_t>(rows) * splits <= static_cast<int64_t>(properties.multiProcessorCount) * 64) && (columns >= 24) && (columns <= 24)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            tf32_hc_prenorm_gemm_gfx92a_row_kernel<2, 64, 256, 4, 2, true, 1, 0, 24, 4, true, 24><<<grid, dim3(512), 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) * splits > static_cast<int64_t>(properties.multiProcessorCount) * 64) && (columns >= 24) && (columns <= 24) && (depth / 64 % splits == 0 && depth / splits % 128 != 0)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            tf32_hc_prenorm_gemm_gfx92a_row_kernel<2, 64, 64, 2, 2, true, 1, 0, 24, 4, true, 24><<<grid, dim3(256), 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 63) && (static_cast<int64_t>(rows) * splits > static_cast<int64_t>(properties.multiProcessorCount) * 64) && (columns >= 24) && (columns <= 24)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            tf32_hc_prenorm_gemm_gfx92a_row_kernel<2, 64, 128, 4, 2, true, 1, 0, 24, 4, true, 24><<<grid, dim3(512), 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
        else if ((depth / 64 / splits >= 16) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 64) && (columns >= 24)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 64, 128, 2, true, 1, 4, false, false, false, 24, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 64, 128, 2, true, 1, 4, false, false, false, 32, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 16) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 64) && (columns <= 16)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 64, 128, 2, true, 4, 4, false, false, false, 8, false, false, true, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 64, 128, 2, true, 4, 4, false, false, false, 16, false, false, true, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 64) && (static_cast<int64_t>(rows) * splits > static_cast<int64_t>(properties.multiProcessorCount) * 32) && (columns >= 24)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 64, 128, 4, true, 2, 4, false, false, false, 24, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 64, 128, 4, true, 2, 4, false, false, false, 32, false, false, false, 0, false><<<grid, dim3(1024), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 48) && (columns >= 32) && (columns <= 32)) {
            const dim3 grid((rows + 79) / 80, splits, 1);
            tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 80, 64, 1, true, 4, 4, false, false, false, 32, false, false, false, 0, false><<<grid, dim3(320), 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 4) && (columns >= 24)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 64, 64, 1, true, 4, 4, false, false, true, 24, false, false, true, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 64, 64, 1, true, 4, 4, false, false, true, 32, false, false, true, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 4) && (columns <= 8)) {
            const dim3 grid((rows + 31) / 32, splits, 1);
            tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 32, 128, 2, true, 4, 4, false, false, false, 8, false, false, true, 0, false><<<grid, dim3(256), 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 16) && (columns <= 16) && (columns >= 16) && (depth / 64 % splits == 0 && depth / splits % 112 == 0)) {
            const dim3 grid((rows + 31) / 32, splits, 1);
            tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 32, 112, 4, true, 1, 4, false, false, true, 16, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
        else if ((depth / 64 / splits >= 4) && (depth / 64 / splits <= 15) && (static_cast<int64_t>(rows) >= static_cast<int64_t>(properties.multiProcessorCount) * 16) && (columns <= 16)) {
            const dim3 grid((rows + 63) / 64, splits, 1);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 64, 128, 2, true, 2, 4, false, false, false, 16, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 64, 128, 2, true, 2, 4, false, false, false, 16, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
        else {
            const dim3 grid((rows + 31) / 32, splits, 1);
            if (columns == 8) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 32, 128, 4, true, 2, 4, false, false, false, 8, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 16) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<1, 32, 128, 4, true, 2, 4, false, false, false, 16, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 24) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 32, 128, 4, true, 2, 4, false, false, false, 24, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
            else if (columns == 32) {
                tf32_hc_prenorm_gemm_gfx92a_staged_kernel<2, 32, 128, 4, true, 2, 4, false, false, false, 32, false, false, false, 0, false><<<grid, dim3(512), 0, stream>>>(
                    activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                    square_sum.data_ptr<float>(), rows, columns, depth, splits);
            }
        }
    } else {
        const dim3 grid((rows + 63) / 64, splits);
        const dim3 block(256);
        if (columns <= 16) {
            tf32_hc_prenorm_gemm_gfx92a_kernel<1><<<grid, block, 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        } else {
            tf32_hc_prenorm_gemm_gfx92a_kernel<2><<<grid, block, 0, stream>>>(
                activation_pointer, weight.data_ptr<float>(), output.data_ptr<float>(),
                square_sum.data_ptr<float>(), rows, columns, depth, splits);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}

}
