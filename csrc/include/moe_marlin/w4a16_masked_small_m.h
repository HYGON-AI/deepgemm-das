#pragma once

// Experimental small-M multi-expert-per-CTA prototype.
//
// This file is intentionally not included by the default W4A16 path. It was
// tested on gemm2 (N=7168, K=1024, size_m=16) and passed correctness, but was
// slower than the current default:
//   default 12 nonzero experts: 0.0977 ms
//   v1 2 experts/CTA, N_REUSE=1: 0.1331 ms
//   v1 4 experts/CTA, N_REUSE=1: 0.1300 ms
//   v2 2 experts/CTA, N_REUSE=2: 0.1216 ms
//   default 5 nonzero experts: 0.0529 ms
//   v1 2 experts/CTA, N_REUSE=1: 0.0674 ms
//   v1 4 experts/CTA, N_REUSE=1: 0.0728 ms
//   v2 2 experts/CTA, N_REUSE=2: 0.0757 ms
//
// The experiment reduces expert scheduling overhead, but loses too much N-side
// parallelism for these large-N gemm2 shapes.
//
// If included for future experiments, include it from masked_w4a16_gemm.h
// inside namespace deepgemm_w4a16_masked.

__global__ void w4a16_small_m_build_nonzero_experts_kernel(
    const int32_t* __restrict__ actual_m,
    int32_t* __restrict__ workspace,
    int experts) {
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    if (actual_m[expert] > 0) {
      const int idx = atomicAdd(workspace, 1);
      workspace[idx + 1] = expert;
    }
  }
}

__global__ void w4a16_small_m_build_tile_workspace_kernel(
    const int32_t* __restrict__ actual_m,
    int32_t* __restrict__ workspace,
    int experts,
    int n_groups) {
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    if (actual_m[expert] > 0) {
      for (int n_group_idx = 0; n_group_idx < n_groups; ++n_group_idx) {
        const int idx = atomicAdd(workspace, 1);
        workspace[idx + 1] = (n_group_idx << 16) | expert;
      }
    }
  }
}

template <typename T, int NUM_WARPS, int N_REUSE>
__global__ void __launch_bounds__(NUM_WARPS * 64, 1)
w4a16_small_m_tile_list_kernel(
    const T* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const void* __restrict__ matrix_b_scale,
    const int32_t* __restrict__ actual_m,
    const int32_t* __restrict__ tile_workspace,
    T* __restrict__ matrix_c,
    int tile_count,
    int size_m,
    int size_n,
    int size_k) {
  using Dtype = ScalarType<T>;
  using scalar_t2 = typename ScalarType<T>::scalar_t2;

  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  constexpr int GROUP_WISE_K = 32;
  constexpr int WARP_K = 128;
  constexpr int MFMA_K = 32;

  const int tid = threadIdx.x;
  const int warp_id = tid / 64;
  const int lane_id = tid & 63;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  const int n_tiles = size_n / TILE_N;
  const T* __restrict__ matrix_b_scale_t =
      reinterpret_cast<const T*>(matrix_b_scale);

  const bool tile_count_in_workspace = tile_count < 0;
  const int actual_tile_count =
      tile_count_in_workspace ? tile_workspace[0] : tile_count;
  const int tile_workspace_base = tile_count_in_workspace ? 1 : 0;

  for (int tile_idx = blockIdx.x; tile_idx < actual_tile_count; tile_idx += gridDim.x) {
    const int packed_tile = tile_workspace[tile_workspace_base + tile_idx];
    const int expert_id = packed_tile & 0xffff;
    const int n_group_idx = static_cast<uint32_t>(packed_tile) >> 16;
    const int expert_m = actual_m[expert_id];
    const int block_actual_m = expert_m > TILE_M ? TILE_M : expert_m;
    if (block_actual_m <= 0) {
      continue;
    }

    const int n_tile_idx_base = (n_group_idx * NUM_WARPS + warp_id) * N_REUSE;
    int n_tile_idx_arr[N_REUSE];
    int n_base_arr[N_REUSE];
    bool valid_n_tile_arr[N_REUSE];
    #pragma unroll
    for (int nr = 0; nr < N_REUSE; ++nr) {
      n_tile_idx_arr[nr] = n_tile_idx_base + nr;
      n_base_arr[nr] = n_tile_idx_arr[nr] * TILE_N;
      valid_n_tile_arr[nr] = n_tile_idx_arr[nr] < n_tiles;
    }

    union_vec_opt_w4a16_A<T, WARP_K / 4> A_reg;
    union_vec_opt_w4a16<uint32_t, WARP_K / 32> B_int_reg;
    reg_bf16_fp16<scalar_t2> B_reg[WARP_K / 32];
    floatx4 C_reg[N_REUSE];
    const floatx4 Zero_reg = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint64_t a_base = static_cast<uint64_t>(expert_id) * size_m * size_k;
    const uint64_t b_base =
        static_cast<uint64_t>(expert_id) * (size_k / WARP_K) * (size_n / TILE_N) *
        (WARP_K / GROUP_WISE_K) * TILE_N * 4;
    const uint64_t s_base =
        static_cast<uint64_t>(expert_id) * size_n * (size_k / GROUP_WISE_K);

    uint64_t b_tile_base[N_REUSE];
    uint64_t s_tile_base[N_REUSE];
    #pragma unroll
    for (int nr = 0; nr < N_REUSE; ++nr) {
      b_tile_base[nr] =
          b_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) * 4 * 16 * 4 +
          static_cast<uint64_t>(col_id) * 16 * 4 + static_cast<uint64_t>(row_id) * 4;
      s_tile_base[nr] =
          s_base + static_cast<uint64_t>(n_base_arr[nr] + row_id) *
                       (size_k / GROUP_WISE_K) +
          col_id;
    }

    for (int k128 = 0; k128 < size_k / WARP_K; ++k128) {
      #pragma unroll
      for (int k_tile = 0; k_tile < WARP_K / GROUP_WISE_K; ++k_tile) {
        const int k_base = col_id * 32 + k_tile * 8;
        if (row_id < block_actual_m) {
          A_reg.input_half8[k_tile] =
              *reinterpret_cast<const vec8_Element<T>*>(
                  &matrix_a[a_base + static_cast<uint64_t>(row_id) * size_k +
                            k128 * WARP_K + k_base]);
        } else {
          A_reg.input_half8[k_tile] = {};
        }
      }

      #pragma unroll
      for (int nr = 0; nr < N_REUSE; ++nr) {
        scalar_t2 scale_f2_vec = Dtype::num2num2(from_float<T>(0.0f));
        if (valid_n_tile_arr[nr]) {
          const uint64_t b_offset =
              b_tile_base[nr] + static_cast<uint64_t>(k128) * (size_n / TILE_N) * 4 * 16 * 4;
          B_int_reg.int4_array[0] =
              *reinterpret_cast<const vec<uint32_t, 4>*>(&matrix_b[b_offset]);
          const T scale_num = matrix_b_scale_t[s_tile_base[nr] + k128 * 4];
          scale_f2_vec = Dtype::num2num2(scale_num);
        } else {
          B_int_reg.int4_array[0] = {};
        }

        #pragma unroll
        for (int j = 0; j < WARP_K / GROUP_WISE_K; ++j) {
          dequant_w4_scaled_pk_f32<scalar_t2>(
              B_int_reg.uint_array[j], scale_f2_vec, B_reg[j].bf162);
        }

        if (valid_n_tile_arr[nr]) {
          #pragma unroll
          for (int k_tile = 0; k_tile < WARP_K / MFMA_K; ++k_tile) {
            if (k128 == 0 && k_tile == 0) {
              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 0],
                      A_reg.input_half[k_tile * 8 + 4],
                      A_reg.input_half[k_tile * 8 + 1],
                      A_reg.input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[0][0],
                      B_reg[k_tile].vec2_i16[0][1],
                      B_reg[k_tile].vec2_i16[1][0],
                      B_reg[k_tile].vec2_i16[1][1]},
                  Zero_reg);
            } else {
              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 0],
                      A_reg.input_half[k_tile * 8 + 4],
                      A_reg.input_half[k_tile * 8 + 1],
                      A_reg.input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[0][0],
                      B_reg[k_tile].vec2_i16[0][1],
                      B_reg[k_tile].vec2_i16[1][0],
                      B_reg[k_tile].vec2_i16[1][1]},
                  C_reg[nr]);
            }

            C_reg[nr] = mmac<T, float>(
                vec4_Element<T>{
                    A_reg.input_half[k_tile * 8 + 2],
                    A_reg.input_half[k_tile * 8 + 6],
                    A_reg.input_half[k_tile * 8 + 3],
                    A_reg.input_half[k_tile * 8 + 7]},
                vec4_Element<T>{
                    B_reg[k_tile].vec2_i16[2][0],
                    B_reg[k_tile].vec2_i16[2][1],
                    B_reg[k_tile].vec2_i16[3][0],
                    B_reg[k_tile].vec2_i16[3][1]},
                C_reg[nr]);
          }
        }
      }
    }

    if (row_id < block_actual_m) {
      #pragma unroll
      for (int nr = 0; nr < N_REUSE; ++nr) {
        if (valid_n_tile_arr[nr]) {
          #pragma unroll
          for (int reg_id = 0; reg_id < 4; ++reg_id) {
            const int n_idx = n_base_arr[nr] + reg_id * 4 + col_id;
            matrix_c[(static_cast<uint64_t>(expert_id) * size_m + row_id) * size_n + n_idx] =
                Dtype::float2num(C_reg[nr][reg_id]);
          }
        }
      }
    }
  }
}

template <
    typename T,
    int EXPERTS_PER_CTA,
    int WARPS_PER_EXPERT,
    int N_REUSE>
__global__ void __launch_bounds__(EXPERTS_PER_CTA * WARPS_PER_EXPERT * 64, 1)
w4a16_small_m_multi_expert_kernel(
    const T* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const void* __restrict__ matrix_b_scale,
    const int32_t* __restrict__ actual_m,
    const int32_t* __restrict__ nonzero_experts,
    T* __restrict__ matrix_c,
    int size_m,
    int size_n,
    int size_k) {
  using Dtype = ScalarType<T>;
  using scalar_t2 = typename ScalarType<T>::scalar_t2;

  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  constexpr int GROUP_WISE_K = 32;
  constexpr int WARP_K = 128;
  constexpr int MFMA_K = 32;
  constexpr int NUM_WARPS = EXPERTS_PER_CTA * WARPS_PER_EXPERT;
  static_assert(EXPERTS_PER_CTA == 2 || EXPERTS_PER_CTA == 4);
  static_assert(WARPS_PER_EXPERT == 4 || WARPS_PER_EXPERT == 8);
  static_assert(NUM_WARPS <= 16);

  const int tid = threadIdx.x;
  const int warp_id = tid / 64;
  const int lane_id = tid & 63;
  const int expert_slot = warp_id / WARPS_PER_EXPERT;
  const int local_warp = warp_id - expert_slot * WARPS_PER_EXPERT;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;

  const int nonzero_count = nonzero_experts[0];
  const int n_tiles = size_n / TILE_N;
  const int n_groups =
      (n_tiles + WARPS_PER_EXPERT * N_REUSE - 1) / (WARPS_PER_EXPERT * N_REUSE);
  const int expert_groups = (nonzero_count + EXPERTS_PER_CTA - 1) / EXPERTS_PER_CTA;
  const int total_tiles = expert_groups * n_groups;
  const T* __restrict__ matrix_b_scale_t =
      reinterpret_cast<const T*>(matrix_b_scale);

  for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
    const int expert_group_idx = tile_idx / n_groups;
    const int n_group_idx = tile_idx - expert_group_idx * n_groups;
    const int list_idx = expert_group_idx * EXPERTS_PER_CTA + expert_slot;
    const bool valid_expert = list_idx < nonzero_count;
    const int expert_id = valid_expert ? nonzero_experts[list_idx + 1] : 0;
    const int expert_m = valid_expert ? actual_m[expert_id] : 0;
    const int block_actual_m = expert_m > TILE_M ? TILE_M : expert_m;

    const int n_tile_idx_base = (n_group_idx * WARPS_PER_EXPERT + local_warp) * N_REUSE;
    int n_tile_idx_arr[N_REUSE];
    int n_base_arr[N_REUSE];
    bool valid_n_tile_arr[N_REUSE];
    #pragma unroll
    for (int nr = 0; nr < N_REUSE; ++nr) {
      n_tile_idx_arr[nr] = n_tile_idx_base + nr;
      n_base_arr[nr] = n_tile_idx_arr[nr] * TILE_N;
      valid_n_tile_arr[nr] =
          valid_expert && block_actual_m > 0 && n_tile_idx_arr[nr] < n_tiles;
    }

    union_vec_opt_w4a16_A<T, WARP_K / 4> A_reg;
    union_vec_opt_w4a16<uint32_t, WARP_K / 32> B_int_reg;
    reg_bf16_fp16<scalar_t2> B_reg[WARP_K / 32];
    floatx4 C_reg[N_REUSE];
    const floatx4 Zero_reg = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint64_t a_base = static_cast<uint64_t>(expert_id) * size_m * size_k;
    const uint64_t b_base =
        static_cast<uint64_t>(expert_id) * (size_k / WARP_K) * (size_n / TILE_N) *
        (WARP_K / GROUP_WISE_K) * TILE_N * 4;
    const uint64_t s_base =
        static_cast<uint64_t>(expert_id) * size_n * (size_k / GROUP_WISE_K);

    uint64_t b_tile_base[N_REUSE];
    uint64_t s_tile_base[N_REUSE];
    #pragma unroll
    for (int nr = 0; nr < N_REUSE; ++nr) {
      b_tile_base[nr] =
          b_base + static_cast<uint64_t>(n_tile_idx_arr[nr]) * 4 * 16 * 4 +
          static_cast<uint64_t>(col_id) * 16 * 4 + static_cast<uint64_t>(row_id) * 4;
      s_tile_base[nr] =
          s_base + static_cast<uint64_t>(n_base_arr[nr] + row_id) *
                       (size_k / GROUP_WISE_K) +
          col_id;
    }

    for (int k128 = 0; k128 < size_k / WARP_K; ++k128) {
      #pragma unroll
      for (int k_tile = 0; k_tile < WARP_K / GROUP_WISE_K; ++k_tile) {
        const int k_base = col_id * 32 + k_tile * 8;
        if (row_id < block_actual_m) {
          A_reg.input_half8[k_tile] =
              *reinterpret_cast<const vec8_Element<T>*>(
                  &matrix_a[a_base + static_cast<uint64_t>(row_id) * size_k +
                            k128 * WARP_K + k_base]);
        } else {
          A_reg.input_half8[k_tile] = {};
        }
      }

      #pragma unroll
      for (int nr = 0; nr < N_REUSE; ++nr) {
        scalar_t2 scale_f2_vec = Dtype::num2num2(from_float<T>(0.0f));
        if (valid_n_tile_arr[nr]) {
          const uint64_t b_offset =
              b_tile_base[nr] + static_cast<uint64_t>(k128) * (size_n / TILE_N) * 4 * 16 * 4;
          B_int_reg.int4_array[0] =
              *reinterpret_cast<const vec<uint32_t, 4>*>(&matrix_b[b_offset]);
          const T scale_num = matrix_b_scale_t[s_tile_base[nr] + k128 * 4];
          scale_f2_vec = Dtype::num2num2(scale_num);
        } else {
          B_int_reg.int4_array[0] = {};
        }

        #pragma unroll
        for (int j = 0; j < WARP_K / GROUP_WISE_K; ++j) {
          dequant_w4_scaled_pk_f32<scalar_t2>(
              B_int_reg.uint_array[j], scale_f2_vec, B_reg[j].bf162);
        }

        if (valid_n_tile_arr[nr]) {
          #pragma unroll
          for (int k_tile = 0; k_tile < WARP_K / MFMA_K; ++k_tile) {
            if (k128 == 0 && k_tile == 0) {
              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 0],
                      A_reg.input_half[k_tile * 8 + 4],
                      A_reg.input_half[k_tile * 8 + 1],
                      A_reg.input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[0][0],
                      B_reg[k_tile].vec2_i16[0][1],
                      B_reg[k_tile].vec2_i16[1][0],
                      B_reg[k_tile].vec2_i16[1][1]},
                  Zero_reg);
            } else {
              C_reg[nr] = mmac<T, float>(
                  vec4_Element<T>{
                      A_reg.input_half[k_tile * 8 + 0],
                      A_reg.input_half[k_tile * 8 + 4],
                      A_reg.input_half[k_tile * 8 + 1],
                      A_reg.input_half[k_tile * 8 + 5]},
                  vec4_Element<T>{
                      B_reg[k_tile].vec2_i16[0][0],
                      B_reg[k_tile].vec2_i16[0][1],
                      B_reg[k_tile].vec2_i16[1][0],
                      B_reg[k_tile].vec2_i16[1][1]},
                  C_reg[nr]);
            }

            C_reg[nr] = mmac<T, float>(
                vec4_Element<T>{
                    A_reg.input_half[k_tile * 8 + 2],
                    A_reg.input_half[k_tile * 8 + 6],
                    A_reg.input_half[k_tile * 8 + 3],
                    A_reg.input_half[k_tile * 8 + 7]},
                vec4_Element<T>{
                    B_reg[k_tile].vec2_i16[2][0],
                    B_reg[k_tile].vec2_i16[2][1],
                    B_reg[k_tile].vec2_i16[3][0],
                    B_reg[k_tile].vec2_i16[3][1]},
                C_reg[nr]);
          }
        }
      }
    }

    if (row_id < block_actual_m) {
      #pragma unroll
      for (int nr = 0; nr < N_REUSE; ++nr) {
        if (valid_n_tile_arr[nr]) {
          #pragma unroll
          for (int reg_id = 0; reg_id < 4; ++reg_id) {
            const int n_idx = n_base_arr[nr] + reg_id * 4 + col_id;
            matrix_c[(static_cast<uint64_t>(expert_id) * size_m + row_id) * size_n + n_idx] =
                Dtype::float2num(C_reg[nr][reg_id]);
          }
        }
      }
    }
  }
}

template <
    typename T,
    int NUM_SM,
    int EXPERTS_PER_CTA,
    int WARPS_PER_EXPERT,
    int N_REUSE>
void launch_w4a16_small_m_multi_expert(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    const torch::Tensor& compact_workspace,
    int size_m,
    int size_n,
    int size_k,
    int experts,
    int grid_x,
    hipStream_t stream) {
  static_assert(EXPERTS_PER_CTA * WARPS_PER_EXPERT == 16);
  hipMemsetAsync(compact_workspace.data_ptr(), 0, sizeof(int32_t), stream);
  w4a16_small_m_build_nonzero_experts_kernel<<<dim3(1), dim3(256), 0, stream>>>(
      reinterpret_cast<const int32_t*>(actual_tokens.data_ptr()),
      reinterpret_cast<int32_t*>(compact_workspace.data_ptr()),
      experts);
  w4a16_small_m_multi_expert_kernel<T, EXPERTS_PER_CTA, WARPS_PER_EXPERT, N_REUSE>
      <<<dim3(grid_x), dim3(EXPERTS_PER_CTA * WARPS_PER_EXPERT * 64), 0, stream>>>(
          reinterpret_cast<const T*>(matrix_a.data_ptr()),
          reinterpret_cast<const uint32_t*>(matrix_b.data_ptr()),
          matrix_b_scale.data_ptr(),
          reinterpret_cast<const int32_t*>(actual_tokens.data_ptr()),
          reinterpret_cast<const int32_t*>(compact_workspace.data_ptr()),
          reinterpret_cast<T*>(matrix_c.data_ptr()),
          size_m,
          size_n,
          size_k);
}

template <
    typename T,
    int NUM_SM,
    int EXPERTS_PER_CTA,
    int WARPS_PER_EXPERT,
    int N_REUSE>
void launch_w4a16_small_m_multi_expert_from_workspace(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    const torch::Tensor& compact_workspace,
    int size_m,
    int size_n,
    int size_k,
    int grid_x,
    hipStream_t stream) {
  static_assert(EXPERTS_PER_CTA * WARPS_PER_EXPERT == 16);
  w4a16_small_m_multi_expert_kernel<T, EXPERTS_PER_CTA, WARPS_PER_EXPERT, N_REUSE>
      <<<dim3(grid_x), dim3(EXPERTS_PER_CTA * WARPS_PER_EXPERT * 64), 0, stream>>>(
          reinterpret_cast<const T*>(matrix_a.data_ptr()),
          reinterpret_cast<const uint32_t*>(matrix_b.data_ptr()),
          matrix_b_scale.data_ptr(),
          reinterpret_cast<const int32_t*>(actual_tokens.data_ptr()),
          reinterpret_cast<const int32_t*>(compact_workspace.data_ptr()),
          reinterpret_cast<T*>(matrix_c.data_ptr()),
          size_m,
          size_n,
          size_k);
}

template <
    typename T,
    int NUM_SM,
    int NUM_WARPS,
    int N_REUSE>
void launch_w4a16_small_m_tile_list(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    const torch::Tensor& tile_workspace,
    int tile_count,
    int size_m,
    int size_n,
    int size_k,
    int grid_x,
    hipStream_t stream) {
  w4a16_small_m_tile_list_kernel<T, NUM_WARPS, N_REUSE>
      <<<dim3(grid_x), dim3(NUM_WARPS * 64), 0, stream>>>(
          reinterpret_cast<const T*>(matrix_a.data_ptr()),
          reinterpret_cast<const uint32_t*>(matrix_b.data_ptr()),
          matrix_b_scale.data_ptr(),
          reinterpret_cast<const int32_t*>(actual_tokens.data_ptr()),
          reinterpret_cast<const int32_t*>(tile_workspace.data_ptr()),
          reinterpret_cast<T*>(matrix_c.data_ptr()),
          tile_count,
          size_m,
          size_n,
          size_k);
}

template <
    typename T,
    int NUM_SM,
    int NUM_WARPS,
    int N_REUSE>
void launch_w4a16_small_m_tile_list_auto(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b,
    const torch::Tensor& matrix_b_scale,
    const torch::Tensor& actual_tokens,
    torch::Tensor& matrix_c,
    const torch::Tensor& tile_workspace,
    int experts,
    int size_m,
    int size_n,
    int size_k,
    int grid_x,
    hipStream_t stream) {
  constexpr int TILE_N = 16;
  const int n_tiles = size_n / TILE_N;
  const int n_groups = (n_tiles + NUM_WARPS * N_REUSE - 1) / (NUM_WARPS * N_REUSE);
  hipMemsetAsync(tile_workspace.data_ptr(), 0, sizeof(int32_t), stream);
  w4a16_small_m_build_tile_workspace_kernel<<<dim3(1), dim3(256), 0, stream>>>(
      reinterpret_cast<const int32_t*>(actual_tokens.data_ptr()),
      reinterpret_cast<int32_t*>(tile_workspace.data_ptr()),
      experts,
      n_groups);
  w4a16_small_m_tile_list_kernel<T, NUM_WARPS, N_REUSE>
      <<<dim3(grid_x), dim3(NUM_WARPS * 64), 0, stream>>>(
          reinterpret_cast<const T*>(matrix_a.data_ptr()),
          reinterpret_cast<const uint32_t*>(matrix_b.data_ptr()),
          matrix_b_scale.data_ptr(),
          reinterpret_cast<const int32_t*>(actual_tokens.data_ptr()),
          reinterpret_cast<const int32_t*>(tile_workspace.data_ptr()),
          reinterpret_cast<T*>(matrix_c.data_ptr()),
          -1,
          size_m,
          size_n,
          size_k);
}
