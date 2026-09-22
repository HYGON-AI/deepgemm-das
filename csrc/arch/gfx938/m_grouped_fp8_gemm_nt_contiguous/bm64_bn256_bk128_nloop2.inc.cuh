#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <string>

namespace m_grouped_fp8_gemm_nt_contiguous_bm64_bn256_bk128_nloop2 {

template <typename scalar_t, size_t len>
using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

using bfloat16_t = __hip_bfloat16;
using vec4_u32 = vec<uint32_t, 4>;
using vec4_i32 = vec<int32_t, 4>;
using vec2_i32 = vec<int32_t, 2>;
using vec8_i8 = vec<int8_t, 8>;
using vec4_f32 = vec<float, 4>;

union AOperand {
  vec8_i8 fp8x8[4];
  vec4_i32 i32x4[2];
};

union BOperand {
  vec8_i8 fp8x8[2];
  vec2_i32 i32x2[2];
  vec4_i32 i32x4;
};

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
#if defined(__gfx938__)
  using bf16x2_t = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;
  const bf16x2_t packed = __builtin_hcu_cvt_pk_bf16_f32(lo, hi, false);
  return *reinterpret_cast<const uint32_t*>(&packed);
#else
  const bfloat16_t lo_bf16(lo);
  const bfloat16_t hi_bf16(hi);
  const __hip_bfloat16_raw lo_raw = lo_bf16;
  const __hip_bfloat16_raw hi_raw = hi_bf16;
  return static_cast<uint32_t>(lo_raw.x) |
         (static_cast<uint32_t>(hi_raw.x) << 16);
#endif
}
__device__ __forceinline__ uint64_t pack_bf16x4(float v0,
                                                float v1,
                                                float v2,
                                                float v3) {
  const uint32_t lo = pack_bf16x2(v0, v1);
  const uint32_t hi = pack_bf16x2(v2, v3);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ __forceinline__ vec4_u32 make_matrix_resource(const T* ptr,
                                                         int stride) {
  vec4_u32 rsrc;
  *reinterpret_cast<uint64_t*>(&rsrc) =
      reinterpret_cast<uint64_t>(ptr) & 0xffffffffffffULL;
  rsrc[2] = stride;
  rsrc[3] = 0;
  return rsrc;
}

__device__ __forceinline__ vec2_i32 load_i32x2(const char* ptr) {
  return *reinterpret_cast<const vec2_i32*>(ptr);
}

__forceinline__ __device__ void buffer_load_i32x2(const char* base,
                                                  vec2_i32& value,
                                                  int byte_offset) {
  __builtin_hcu_sched_barrier(0);
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  intx4 resource;
  *reinterpret_cast<uint64_t*>(&resource) = reinterpret_cast<uint64_t>(base);
  resource[1] += 0x40400000;
  resource[2] = 0x80000000;
  resource[3] = 0x00020000;

  asm volatile(
      "s_nop 2 \n\t"
      "buffer_load_dwordx2 %0,%1,%2,0, offen offset:0\n"
	      : "=v"(value), "+v"(byte_offset), "+s"(resource));
}

__forceinline__ __device__ void buffer_load_i32x4(const char* base,
                                                  vec4_i32& value,
                                                  int byte_offset) {
  __builtin_hcu_sched_barrier(0);
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  intx4 resource;
  *reinterpret_cast<uint64_t*>(&resource) = reinterpret_cast<uint64_t>(base);
  resource[1] += 0x40400000;
  resource[2] = 0x80000000;
  resource[3] = 0x00020000;

  asm volatile(
      "s_nop 2 \n\t"
      "buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n"
      : "=v"(value), "+v"(byte_offset), "+s"(resource));
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

__device__ __forceinline__ void wait_vmcnt4_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(4)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt7_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(7)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt5_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(5)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt3_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(3)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt1_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(1)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt2_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(2)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt2_barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(2)\n\ts_barrier\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt4_barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(4)\n\ts_barrier\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt1_barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(1)\n\ts_barrier\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_lgkmcnt_barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_lgkmcnt_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

#define MATRIX_LOAD_128X16_B8_T_LDS(LDSADDR, SRSRC)                           \
  do {                                                                         \
    int soffset = (LDSADDR) + 0x80000000;                                      \
    asm volatile("s_nop 4\n\t"                                                \
                 "matrix_load_128x16_b8 %0, %1, moffset:0 t lds\n\t"          \
                 :: "s"(SRSRC), "s"(soffset)                                 \
                 : );                                                 \
  } while (0)

#define DS_READ_MATRIX_64X16_B8(OFFSET, REG)                                  \
  do {                                                                         \
    asm volatile( \
        "s_add_u32 m0, %1, 0x80000000\n\t" \
        "s_nop 4\n\t" \
        "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x3 col:0x1 alt:0x0\n" \
        : "=v"(REG) \
        : "s"(OFFSET) \
        :); \
	  } while (0)

__device__ __forceinline__ void ds_read_b128_lds(const char* lds_base,
                                                 int byte_offset,
                                                 vec4_i32& value) {
  const int lds_addr =
      static_cast<int>(reinterpret_cast<size_t>(lds_base)) + byte_offset;
  asm volatile("ds_read_b128 %0, %1\n\t"
               : "=v"(value)
               : "v"(lds_addr)
               : "memory");
}

__device__ __forceinline__ vec4_f32 fp8_mmac_16x16x32_direct(
    const vec8_i8& a,
    const vec8_i8& b,
    vec4_f32 acc) {
#if defined(__gfx938__)
  acc = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, acc, 1, 0);
#endif
  return acc;
}

__device__ __forceinline__ vec4_f32 fp8_mmac_16x16x64_direct(
    const vec8_i8& a0,
    const vec8_i8& a1,
    const BOperand& b,
    vec4_f32 acc) {
#if defined(__gfx938__)
  acc =
      __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a0, b.fp8x8[0], acc, 1, 0);
  acc =
      __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a1, b.fp8x8[1], acc, 1, 0);
#endif
  return acc;
}

template <int BlockK,
          int K64Loop,
          int MSubtiles,
          int NFragments,
          int BWaitMode,
          int ALdsTileBytes,
          int K64Tile,
          int MmacRows>
// 从固定的 A LDS stage 和一个 B 寄存器计算一段 K64。
// K64Loop 使用模板参数，保证 A LDS 偏移在编译期固定；BWaitMode 控制 B 的延迟等待策略。
__device__ __forceinline__ void compute_k64_from_lds(
    const char* a_lds_stage,
    const BOperand (&b_reg)[NFragments],
    int lane_id,
    vec4_f32 (&acc)[MSubtiles][NFragments]) {
  AOperand a_regs[MSubtiles];

#if defined(__gfx938__)
  const int a_lane_row_pair = (lane_id & 15) >> 1;
  const int a_lane_col_group = lane_id >> 4;
  const int a_col_swizzle = (a_lane_row_pair + a_lane_col_group) & 3; // 0 1 具有相同sw, 2 3具有相同sw
  const int a_lane_odd_k = (lane_id & 1) * K64Tile;
#endif

  // const int a_lds_stage_addr =
  //   static_cast<int>(reinterpret_cast<size_t>(a_lds_stage));
#pragma unroll
  for (int m_loop = 0; m_loop < MSubtiles; ++m_loop) {

#if defined(__gfx938__)
    const int a_lds_load_base =
        m_loop * ALdsTileBytes + K64Loop * K64Tile * MmacRows; // m_loop * 2048 + K64Loop * 1024
    const int a_lds_load_offset =
        a_lds_load_base + a_lane_row_pair * BlockK + a_col_swizzle * 16 +
        a_lane_odd_k;
    ds_read_b128_lds(a_lds_stage, a_lds_load_offset,
      a_regs[m_loop].i32x4[0]);
#endif

    // const int a_lds_tile_offset =
    //     a_lds_stage_addr + m_loop * ALdsTileBytes + K64Loop * 4096;
    // DS_READ_MATRIX_64X16_B8(a_lds_tile_offset, a_regs[m_loop].i32x4[0]);
  }

  wait_lgkmcnt_zero();
  if constexpr (BWaitMode == 1) {
    if constexpr (K64Loop == 0) {
      wait_vmcnt7_zero();
    } else {
      wait_vmcnt5_zero();
    }
  } else if constexpr (BWaitMode == 2) {
    if constexpr (K64Loop == 0) {
      wait_vmcnt2_zero();
    } else {
      wait_vmcnt_zero();
    }
  }
  
#pragma unroll
  for (int m_loop = 0; m_loop < MSubtiles; ++m_loop) {
#pragma unroll
    for (int n_frag = 0; n_frag < NFragments; ++n_frag) {
      //
      acc[m_loop][n_frag] = fp8_mmac_16x16x32_direct(
          a_regs[m_loop].fp8x8[0], b_reg[n_frag].fp8x8[0],
          acc[m_loop][n_frag]);
      acc[m_loop][n_frag] = fp8_mmac_16x16x32_direct(
          a_regs[m_loop].fp8x8[1], b_reg[n_frag].fp8x8[1],
          acc[m_loop][n_frag]);
    }
  }
}

__device__ __forceinline__ int w6_offset(int size_n, int n, int k) {
  const int n_band = n / 16;
  const int n_inner = n & 15;
  const int k_band = k / 64;
  const int k_mid = (k & 63) / 16;
  const int k_inner = k & 15;
  return (((k_band * (size_n / 16) + n_band) * 4 + k_mid) * 16 + n_inner) *
             16 +
         k_inner;
}

template <int NFragments, int K64Tile>
__device__ __forceinline__ void load_b_tile_bn128(
    const char* packed_weight_w6,
    int expert_b_base,
    int size_n,
    int logical_n_base,
    int k_base,
    int lane_group,
    BOperand (&b0)[NFragments],
    BOperand (&b1)[NFragments]) {
#pragma unroll
  for (int n_frag = 0; n_frag < NFragments; ++n_frag) {
    const int logical_n = logical_n_base + n_frag * 16;
    int b_offset0 = expert_b_base +
                    w6_offset(size_n, logical_n, k_base + lane_group * 16);
    buffer_load_i32x4(packed_weight_w6, b0[n_frag].i32x4, b_offset0);
  }
#pragma unroll
  for (int n_frag = 0; n_frag < NFragments; ++n_frag) {
    const int logical_n = logical_n_base + n_frag * 16;
    int b_offset1 = expert_b_base +
                    w6_offset(size_n, logical_n,
                              k_base + K64Tile + lane_group * 16);
    buffer_load_i32x4(packed_weight_w6, b1[n_frag].i32x4, b_offset1);
  }
}

template <int BlockM, int BlockN, int BlockK, int NumWarps, int NLoopNum>
__global__ __launch_bounds__(256, 1) void m_grouped_fp8_gemm_nt_contiguous_kernel(
    const char* __restrict__ input,
    const char* __restrict__ packed_weight_w6,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    const int32_t* __restrict__ m_indices,
    bfloat16_t* __restrict__ output,
    int size_m,
    int size_n,
    int size_k,
    int experts) {
  static_assert(BlockM == 64, "contiguous FP8 compiles BM=64 only");
  static_assert(BlockN == 256, "contiguous FP8 compiles BN=256 only");
  static_assert(BlockK == 128, "contiguous FP8 uses BK=128");
  static_assert(NumWarps == 4, "contiguous FP8 uses four waves per CTA");
  static_assert(NLoopNum == 2, "contiguous FP8 uses four N fragments per warp");

#if !(defined(__gfx938__))
  return;
#else

  constexpr int kWarpSize = 64;
  constexpr int kMmacRows = 16;
  constexpr int kMmacCols = 16;
  constexpr int kMmacVgprs = 4;
  constexpr int kMSubtiles = BlockM / kMmacRows;
  constexpr int kAccumTile = 32;
  constexpr int kK64Tile = 64;
  constexpr int kKLoops = BlockK / kAccumTile;
  constexpr int kK64Loops = BlockK / kK64Tile;
  constexpr int kNFragmentsPerWarp = 2;
  constexpr int kNFragments = NLoopNum * kNFragmentsPerWarp;
  constexpr int kAStageBytes = BlockM * BlockK;
  constexpr int kAStages = 2;
  static_assert(kMSubtiles == 4,
                "contiguous FP8 expects four 16-row M subtiles");
  static_assert(kKLoops == 4,
                "contiguous FP8 expects four 32-wide K MMAC loops");
  static_assert(kK64Loops == 2,
                "contiguous FP8 expects two K64 wrappers per BK128");
  static_assert(kNFragmentsPerWarp == 2,
                "BN256 expects each warp to own four 16-column fragments");

  __align__(16) __shared__ char a_lds[kAStages * kAStageBytes];

  const int n_block_id = blockIdx.x;
  const int m_block_id = blockIdx.y;
  const int m_block_start = m_block_id * BlockM;
  if (m_block_start >= size_m) {
    return;
  }

  const int32_t expert_id = m_indices[m_block_start];
  if (expert_id < 0 || expert_id >= experts) {
    return;
  }
  const int32_t expert_id_s = __builtin_hcu_readfirstlane(expert_id);

  const int thread_id = threadIdx.x;
  const int warp_id_v = thread_id >> 6;
  const int warp_id = __builtin_hcu_readfirstlane(warp_id_v);
  const int lane_id = thread_id & (kWarpSize - 1);
  const int lane_row = lane_id & 15;
  const int lane_group = lane_id >> 4;

  const int n_block_start = n_block_id * BlockN;
  const int k_tiles = size_k / BlockK;
  const int expert_b_base =
      static_cast<int>(static_cast<int64_t>(expert_id_s) * size_n * size_k);

  vec4_f32 acc[kMSubtiles][kNFragments];
#pragma unroll
  for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
#pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
      acc[m_loop][n_frag] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
  }

  const int local_m = warp_id * kMmacRows;
  // a_lds_write_bytes warp之间一次MLS写入LDS的偏移

#if defined(__gfx938__)
  const int a_lds_write_bytes = local_m * BlockK; // nmz => a_lds_write_bytes = warp_id * 2048
#endif

  const int a_lds_base_addr =
      static_cast<int>(reinterpret_cast<size_t>(a_lds));
  const int logical_n_base =
      n_block_start + warp_id * kNFragments * kMmacCols +
      (lane_id & 15);
  // kALdsTileBytes m方向偏移
#if defined(__gfx938__)
  constexpr int kALdsTileBytes = kMmacRows * BlockK; // nmz => kALdsTileBytes = 2048
#endif
 

  // 显式 B 寄存器双缓冲。不要用 BOperand 数组，动态 stage 下标会让
  // 当前工具链把它放进 private memory，并触发 VMFault。
  BOperand b_even0[kNFragments];
  BOperand b_even1[kNFragments];
  BOperand b_odd0[kNFragments];
  BOperand b_odd1[kNFragments];

  auto compute_even_tile = [&](const char* a_stage, int b_wait_mode) {
    if (b_wait_mode == 1) {
      compute_k64_from_lds<BlockK, 0, kMSubtiles, kNFragments, 1,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_even0, lane_id, acc);
      compute_k64_from_lds<BlockK, 1, kMSubtiles, kNFragments, 1,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_even1, lane_id, acc);
    } else {
      compute_k64_from_lds<BlockK, 0, kMSubtiles, kNFragments, 2,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_even0, lane_id, acc);
      compute_k64_from_lds<BlockK, 1, kMSubtiles, kNFragments, 2,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_even1, lane_id, acc);
    }
  };

  auto compute_odd_tile = [&](const char* a_stage, int b_wait_mode) {
    if (b_wait_mode == 1) {
      compute_k64_from_lds<BlockK, 0, kMSubtiles, kNFragments, 1,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_odd0, lane_id, acc);
      compute_k64_from_lds<BlockK, 1, kMSubtiles, kNFragments, 1,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_odd1, lane_id, acc);
    } else {
      compute_k64_from_lds<BlockK, 0, kMSubtiles, kNFragments, 2,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_odd0, lane_id, acc);
      compute_k64_from_lds<BlockK, 1, kMSubtiles, kNFragments, 2,
                           kALdsTileBytes, kK64Tile, kMmacRows>(
          a_stage, b_odd1, lane_id, acc);
    }
  };

  // 序章：把 tile0 加载到 even stage，并预取它的两段 B K64。
  const char* a_ptr0 =
      input + static_cast<int64_t>(m_block_start + local_m) * size_k;
  vec4_u32 a_srsrc0 = make_matrix_resource(a_ptr0, size_k);
  const int lds_addr0 = a_lds_base_addr + a_lds_write_bytes;
  MATRIX_LOAD_128X16_B8_T_LDS(lds_addr0, a_srsrc0);

  load_b_tile_bn128<kNFragments, kK64Tile>(
      packed_weight_w6, expert_b_base, size_n, logical_n_base, 0, lane_group,
      b_even0, b_even1);

  // 序章只等待 A tile0 写入 LDS，保留 B tile0 到使用前再等待。
  wait_vmcnt4_barrier();

  int k_tile = 0;
  for (; k_tile + 2 < k_tiles; k_tile += 2) {
    // 当前计算 even tile 时，把 odd tile 加载到 stage 1。
    const char* a_odd_ptr =
        input + static_cast<int64_t>(m_block_start + local_m) * size_k +
        static_cast<int64_t>(k_tile + 1) * BlockK;
    vec4_u32 a_odd_srsrc = make_matrix_resource(a_odd_ptr, size_k);
    const int odd_lds_addr =
        a_lds_base_addr + kAStageBytes + a_lds_write_bytes;
    MATRIX_LOAD_128X16_B8_T_LDS(odd_lds_addr, a_odd_srsrc);

    load_b_tile_bn128<kNFragments, kK64Tile>(
        packed_weight_w6, expert_b_base, size_n, logical_n_base,
        (k_tile + 1) * BlockK, lane_group, b_odd0, b_odd1);

    const char* a_even_stage = a_lds;
    compute_even_tile(a_even_stage, 1);

    wait_vmcnt4_barrier();

    // 当前计算 odd tile 时，把后续 even tile 加载回 stage 0。
    const char* a_even_ptr =
        input + static_cast<int64_t>(m_block_start + local_m) * size_k +
        static_cast<int64_t>(k_tile + 2) * BlockK;
    vec4_u32 a_even_srsrc = make_matrix_resource(a_even_ptr, size_k);
    const int even_lds_addr = a_lds_base_addr + a_lds_write_bytes;
    MATRIX_LOAD_128X16_B8_T_LDS(even_lds_addr, a_even_srsrc);

    load_b_tile_bn128<kNFragments, kK64Tile>(
        packed_weight_w6, expert_b_base, size_n, logical_n_base,
        (k_tile + 2) * BlockK, lane_group, b_even0, b_even1);

    const char* a_odd_stage = a_lds + kAStageBytes;
    compute_odd_tile(a_odd_stage, 1);

    wait_vmcnt4_barrier();
  }

  const int remaining_tiles = k_tiles - k_tile;
  if (remaining_tiles == 1) {
    // K 只有一个剩余 BK128 tile：当前 even stage 已经就绪，不能再加载 odd。
    const char* a_even_stage = a_lds;
    compute_even_tile(a_even_stage, 2);
  } else {
    // 最后一组 even/odd tile。
    const char* a_tail_odd_ptr =
        input + static_cast<int64_t>(m_block_start + local_m) * size_k +
        static_cast<int64_t>(k_tile + 1) * BlockK;
    vec4_u32 a_tail_odd_srsrc = make_matrix_resource(a_tail_odd_ptr, size_k);
    const int tail_odd_lds_addr =
        a_lds_base_addr + kAStageBytes + a_lds_write_bytes;
    MATRIX_LOAD_128X16_B8_T_LDS(tail_odd_lds_addr, a_tail_odd_srsrc);

    load_b_tile_bn128<kNFragments, kK64Tile>(
        packed_weight_w6, expert_b_base, size_n, logical_n_base,
        (k_tile + 1) * BlockK, lane_group, b_odd0, b_odd1);

    const char* a_even_stage = a_lds;
    compute_even_tile(a_even_stage, 1);

    wait_vmcnt4_barrier();

    const char* a_odd_stage = a_lds + kAStageBytes;
    compute_odd_tile(a_odd_stage, 2);
  }

  // N 维在 launch 侧保证按 BN=256 对齐；先缓存本 wave 负责列的 scale，
  // 避免每个 M subtile 重复加载同一组 weight_scale。
  float col_scale[kNFragments][kMmacVgprs];
#pragma unroll
  for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
    const int col_base = n_block_start +
                         warp_id * kNFragments * kMmacCols +
                         n_frag * kMmacCols + lane_group * kMmacVgprs;
#pragma unroll
    for (int elem = 0; elem < kMmacVgprs; ++elem) {
      col_scale[n_frag][elem] =
          weight_scale[static_cast<int64_t>(expert_id_s) * size_n +
                       col_base + elem];
    }
  }

  const bool full_block_valid =
      (m_block_start + BlockM <= size_m) &&
      (m_indices[m_block_start + BlockM - 1] >= 0);

  if (full_block_valid) {
#pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
      const int row = m_block_start + m_loop * kMmacRows + lane_row;
      const float row_scale = input_scale[row];
#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int col_base = n_block_start +
                             warp_id * kNFragments * kMmacCols +
                             n_frag * kMmacCols + lane_group * kMmacVgprs;
        const int col = col_base;
        const float value0 = acc[m_loop][n_frag][0] * row_scale *
                             col_scale[n_frag][0];
        const float value1 = acc[m_loop][n_frag][1] * row_scale *
                             col_scale[n_frag][1];
        const float value2 = acc[m_loop][n_frag][2] * row_scale *
                             col_scale[n_frag][2];
        const float value3 = acc[m_loop][n_frag][3] * row_scale *
                             col_scale[n_frag][3];
        *reinterpret_cast<uint64_t*>(output + static_cast<int64_t>(row) *
                                             size_n + col) =
            pack_bf16x4(value0, value1, value2, value3);
      }
    }
  } else {
#pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
      const int row = m_block_start + m_loop * kMmacRows + lane_row;
      if (row >= size_m || m_indices[row] < 0) {
        continue;
      }
      const float row_scale = input_scale[row];
#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {

        const int col_base = n_block_start +
                             warp_id * kNFragments * kMmacCols +
                             n_frag * kMmacCols + lane_group * kMmacVgprs;
        const int col = col_base;
        const float value0 = acc[m_loop][n_frag][0] * row_scale *
                             col_scale[n_frag][0];
        const float value1 = acc[m_loop][n_frag][1] * row_scale *
                             col_scale[n_frag][1];
        const float value2 = acc[m_loop][n_frag][2] * row_scale *
                             col_scale[n_frag][2];
        const float value3 = acc[m_loop][n_frag][3] * row_scale *
                             col_scale[n_frag][3];
        *reinterpret_cast<uint64_t*>(output + static_cast<int64_t>(row) *
                                             size_n + col) =
            pack_bf16x4(value0, value1, value2, value3);
      }
    }
  }
#endif
}

int resolve_n_loop_num(int size_n, int requested) {
  constexpr int kBlockN = 256;
  TORCH_CHECK(requested == 0 || requested == 2,
              "contiguous FP8 BN256 only supports N_LOOP_NUM=2; pass 0 or 2");
  TORCH_CHECK(size_n % kBlockN == 0,
              "N must be divisible by BLOCK_N=256 for contiguous FP8 BN256");
  return 2;
}

template <int NLoopNum>
void launch_m_grouped_fp8_gemm_nt_contiguous_nloop(torch::Tensor input,
                                torch::Tensor b_qweight,
                                torch::Tensor output,
                                torch::Tensor a_scale,
                                torch::Tensor b_scale,
                                torch::Tensor m_indices) {
  static_assert(NLoopNum == 2,
                "contiguous FP8 BN256 supports N_LOOP_NUM=2 only");
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 256;
  constexpr int kBlockK = 128;
  constexpr int kNumWarps = 4;

  const int size_m = static_cast<int>(input.size(0));
  const int size_k = static_cast<int>(input.size(1));
  const int experts = static_cast<int>(b_qweight.size(0));
  const int b_k = static_cast<int>(b_qweight.size(1) * b_qweight.size(3) *
                                   b_qweight.size(5));
  const int size_n = static_cast<int>(b_qweight.size(2) * b_qweight.size(4));

  TORCH_CHECK(size_k == b_k, "input K=", size_k, " but packed weight K=", b_k);
  TORCH_CHECK(output.size(0) == size_m && output.size(1) == size_n,
              "output must be [M,N] matching input and packed weight");
  TORCH_CHECK(size_k % kBlockK == 0, "K must be divisible by 128 for contiguous FP8");
  TORCH_CHECK(size_k >= kBlockK, "K must be at least 128 for contiguous FP8");
  TORCH_CHECK(size_n % kBlockN == 0,
              "N must be divisible by BLOCK_N=256 for contiguous FP8 BN256");
  TORCH_CHECK(input.scalar_type() == at::kFloat8_e4m3fn,
              "contiguous FP8 only supports the fp8_e4m3fn input path");

  const dim3 block(kNumWarps * 64, 1, 1);
  const dim3 grid(ceil_div(size_n, kBlockN),
                  ceil_div(size_m, kBlockM), 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  m_grouped_fp8_gemm_nt_contiguous_kernel<kBlockM, kBlockN, kBlockK, kNumWarps, NLoopNum>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const char*>(input.data_ptr()),
          reinterpret_cast<const char*>(b_qweight.data_ptr()),
          a_scale.data_ptr<float>(),
          b_scale.data_ptr<float>(),
          m_indices.data_ptr<int32_t>(),
          reinterpret_cast<bfloat16_t*>(output.data_ptr()),
          size_m,
          size_n,
          size_k,
          experts);
}

bool current_device_is_valid() {
  int device_id = 0;
  hipError_t err = hipGetDevice(&device_id);
  TORCH_CHECK(err == hipSuccess, "failed to get current HIP device");

  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, device_id);
  TORCH_CHECK(err == hipSuccess, "failed to get HIP device properties");

  std::string arch(prop.gcnArchName);
  return arch.rfind("gfx938", 0) == 0;
}


torch::Tensor run(torch::Tensor input,
                            torch::Tensor b_qweight,
                            torch::Tensor output,
                            torch::Tensor a_scale,
                            torch::Tensor b_scale,
                            torch::Tensor m_indices,
                            int64_t block_m,
                            int64_t block_n,
                            int64_t n_loop_num) {
  const c10::OptionalDeviceGuard device_guard(device_of(input));


  TORCH_CHECK(input.is_cuda() && b_qweight.is_cuda() && output.is_cuda(),
              "input, packed weight, and output must be CUDA tensors");
  TORCH_CHECK(current_device_is_valid(),
              "m_grouped_fp8_gemm_nt_contiguous requires gfx938");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(b_qweight.is_contiguous(), "packed weight must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(a_scale.is_contiguous(), "a_scale must be contiguous");
  TORCH_CHECK(b_scale.is_contiguous(), "b_scale must be contiguous");
  TORCH_CHECK(m_indices.is_cuda() && m_indices.is_contiguous(),
              "m_indices must be a contiguous CUDA tensor");
  TORCH_CHECK(m_indices.scalar_type() == at::kInt, "m_indices must be int32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat, "a_scale must be float32");
  TORCH_CHECK(b_scale.scalar_type() == at::kFloat, "b_scale must be float32");
  TORCH_CHECK(input.dim() == 2, "input must be [M,K]");
  TORCH_CHECK(b_qweight.dim() == 6,
              "packed weight must be [E,K/64,N/16,4,16,16]");
  TORCH_CHECK(output.dim() == 2, "output must be [M,N]");
  TORCH_CHECK(a_scale.dim() == 1 || (a_scale.dim() == 2 && a_scale.size(1) == 1),
              "contiguous FP8 expects a_scale as [M] or [M,1]");
  TORCH_CHECK(a_scale.size(0) == input.size(0),
              "a_scale M dimension must match input");
  TORCH_CHECK(b_scale.dim() == 2 ||
                  (b_scale.dim() == 3 && b_scale.size(2) == 1),
              "contiguous FP8 expects b_scale as [E,N] or [E,N,1]");
  TORCH_CHECK(m_indices.numel() == input.size(0),
              "m_indices length must match M");
  TORCH_CHECK(b_qweight.size(3) == 4 && b_qweight.size(4) == 16 &&
                  b_qweight.size(5) == 16,
              "packed weight trailing dims must be [4,16,16]");
  TORCH_CHECK(b_scale.size(0) == b_qweight.size(0),
              "b_scale expert dimension must match packed weight");
  TORCH_CHECK(b_scale.size(1) == b_qweight.size(2) * b_qweight.size(4),
              "b_scale N dimension must match packed weight");
  TORCH_CHECK(block_m == 64 && block_n == 256,
              "contiguous FP8 is compiled only for BM=64, BN=256");

  auto a_scale_kernel =
      a_scale.dim() == 2 ? a_scale.view({a_scale.size(0)}) : a_scale;
  auto b_scale_kernel =
      b_scale.dim() == 3 ? b_scale.view({b_scale.size(0), b_scale.size(1)})
                         : b_scale;

  const int resolved_n_loop =
      resolve_n_loop_num(static_cast<int>(output.size(1)),
                         static_cast<int>(n_loop_num));
  (void)resolved_n_loop;
  launch_m_grouped_fp8_gemm_nt_contiguous_nloop<2>(input, b_qweight, output, a_scale_kernel, b_scale_kernel,
                                m_indices);
  return output;
}

}  // namespace m_grouped_fp8_gemm_nt_contiguous_bm64_bn256_bk128_nloop2
