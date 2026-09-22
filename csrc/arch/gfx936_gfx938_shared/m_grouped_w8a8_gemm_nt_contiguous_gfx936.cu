#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <string>

namespace i8_contig_ll_v92 {

template <typename scalar_t, size_t len>
using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

using bfloat16_t = __hip_bfloat16;
using vec4_u32 = vec<uint32_t, 4>;
using vec4_i32 = vec<int32_t, 4>;
using vec2_i32 = vec<int32_t, 2>;
using vec8_i8 = vec<int8_t, 8>;
using vec4_f32 = vec<float, 4>;

union AOperand {
  vec8_i8 int8x8[4];
  vec4_i32 i32x4[2];
};

union BOperand {
  vec8_i8 int8x8[2];
  vec2_i32 i32x2[2];
  vec4_i32 i32x4;
};

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const bfloat16_t lo_bf16(lo);
  const bfloat16_t hi_bf16(hi);
  const __hip_bfloat16_raw lo_raw = lo_bf16;
  const __hip_bfloat16_raw hi_raw = hi_bf16;
  return static_cast<uint32_t>(lo_raw.x) |
         (static_cast<uint32_t>(hi_raw.x) << 16);
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

__forceinline__ __device__ void buffer_load_dword_lds(const char* base,
                                                      int global_dword_offset,
                                                      int lds_dword_offset) {
  __builtin_hcu_sched_barrier(0);
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  intx4 resource;
  *reinterpret_cast<uint64_t*>(&resource) = reinterpret_cast<uint64_t>(base);
  resource[1] += 0x40400000;
  resource[2] = 0x80000000;
  resource[3] = 0x00020000;

  int lds_addr = lds_dword_offset << 2;
  int gv_offset = global_dword_offset << 2;
  int scalar_offset = 0;
  asm volatile("s_mov_b32 m0, %1\n\t"
               "buffer_load_dword %0, %2, %3, offen offset:0, lds\n\t"
               :
               : "v"(gv_offset), "s"(lds_addr), "s"(resource),
                 "s"(scalar_offset)
               : "memory");
}

__device__ __forceinline__ void load_a_m128_k64_buffer_to_rowmajor_lds(
    const char* __restrict__ a_ptr,
    int size_k,
    int lds_base_addr,
    int thread_id) {
  constexpr int kWarpNum = 4;
  constexpr int kMmacRows = 16;
  constexpr int kWarpK = 64;
  const int warp_id_v = thread_id >> 6;
  const int warp_id = __builtin_hcu_readfirstlane(warp_id_v);
  const int lane_id = thread_id & 63;
  const int row_id = lane_id & 15;
  const int col_id = lane_id >> 4;
  const int lds_base_dword = lds_base_addr >> 2;

#pragma unroll
  for (int m_tile = 0; m_tile < 8; ++m_tile) {
    for (int index = warp_id; index < kMmacRows / 4; index += kWarpNum) {
      const int row = m_tile * kMmacRows + index * 4 + col_id;
      const int k = row_id * 4;
      const int lds_dword_offset =
          lds_base_dword + (m_tile * kMmacRows * kWarpK +
                            index * 4 * kWarpK) / 4;
      const int global_dword_offset = (row * size_k + k) >> 2;
      buffer_load_dword_lds(a_ptr, global_dword_offset, lds_dword_offset);
    }
  }
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

__device__ __forceinline__ void wait_vmcnt12_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(12)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt16_zero() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(16)\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt4_barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(4)\n\ts_barrier\n\t" ::: "memory");
  __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt8_barrier() {
  __builtin_hcu_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(8)\n\ts_barrier\n\t" ::: "memory");
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

#if defined(__gfx938__)
#define MMAC_I8_K32(ACC, A_REG, B_REG)                                          do {                                                                             (ACC) = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(                          (A_REG), (B_REG), (ACC), 1, 0, 0);                                       } while (0)
#elif defined(__gfx936__) || defined(__gfx92a__) || defined(__gfx928__)
#define MMAC_I8_K32(ACC, A_REG, B_REG)                                          do {                                                                             (ACC) = __builtin_hcu_mmac_i32_16x16x32_i8((A_REG), (B_REG), (ACC));         } while (0)
#else
#define MMAC_I8_K32(ACC, A_REG, B_REG) do { } while (0)
#endif

#define LOAD_A_M16_FROM_LDS(STAGE_BASE, M_LOOP, A_REG)                        \
  do {                                                                         \
    const int a_lds_load_offset =                                              \
        (M_LOOP) * kALdsTileBytes + lane_row * BlockK + lane_group * 16;       \
    (A_REG).i32x4[0] =                                                         \
        *reinterpret_cast<vec4_i32*>(&a_lds[(STAGE_BASE) - a_lds_base_addr +   \
                                            a_lds_load_offset]);               \
  } while (0)

#define B_EVEN_AT(IDX)                                                         \
  ((IDX) == 0 ? b_even0 :                                                       \
   (IDX) == 1 ? b_even1 :                                                       \
   (IDX) == 2 ? b_even2 : b_even3)

#define B_ODD_AT(IDX)                                                          \
  ((IDX) == 0 ? b_odd0 :                                                        \
   (IDX) == 1 ? b_odd1 :                                                        \
   (IDX) == 2 ? b_odd2 : b_odd3)

#define B_PREF_AT(IDX)                                                         \
  ((IDX) == 0 ? b_pref0 :                                                       \
   (IDX) == 1 ? b_pref1 :                                                       \
   (IDX) == 2 ? b_pref2 : b_pref3)

// v91 数字总览：
// 1. CTA 负责 C[BM, BN] = C[128, 256]。
// 2. K 方向每次推进 BK=64，所以一次 K tile 做 A[128,64] x B[64,256]。
// 3. 一个 CTA 有 4 个 wave/warp，每个 wave=64 lanes，总线程数 4*64=256。
// 4. 4 个 warp 按 1M x 4N 排布，每个 warp 覆盖四个连续 N16 fragment：
//      warp0 -> C[ 0:128,   0: 64]
//      warp1 -> C[ 0:128,  64:128]
//      warp2 -> C[ 0:128, 128:192]
//      warp3 -> C[ 0:128, 192:256]
// 5. 每个 warp 负责 C[128,64]，内部拆成 8*4 个 M16xN16 MMAC subtile。
// 6. MMAC 指令是 16x16x32，BK64 需要连续做 2 次 K32 MMAC。
// 7. A 进 LDS：每个 wave 发 8 条 buffer_load_dword ... lds，
//    4 个 wave 合起来覆盖 A[M128,K64]，LDS 内为 row-major。
// 8. A LDS 使用双 stage：
//      stage0: byte [0, 8192)
//      stage1: byte [8192, 16384)
//    total LDS = 128*64*2 = 16384B。
// 9. B 不进 LDS：每个 warp 直接 buffer_load B[K64,N64] 到寄存器。
// 10. 当前版本严格按序章 / 主循环 / 收尾组织 A 双 stage pipeline，
//     B 继续沿用 v78 的 even/odd/pref 三组寄存器：
//     prologue 预加载 even tile0；
//     steady state 先 load odd A/B 和 next-even B，再 compute even；
//     随后 wait odd A、load next-even A、compute odd，最后收住 next-even A/B；
//     tail 先发最后一个 odd tile，再 compute final even，最后 compute final odd。
// 11. v91 基于 v78 验证 A 的 K32 split 寄存器流水：
//     AOperand 仍是 2 槽滚动，但每个 M16 先计算 K32#0；
//     然后发射下一 M16 的普通 LDS vector load；
//     再计算当前 M16 的 K32#1；
//     最后用 lgkmcnt(0) 收住下一组 A。
//     这样试图用当前 M16 的第二半 K32 MMAC 覆盖下一组 A LDS->VGPR 延迟。
template <int BlockM, int BlockN, int BlockK, int NumWarps, int NLoopNum>
__global__ __launch_bounds__(256, 1) void i8_contig_ll_kernel(
    const char* __restrict__ input,
    const char* __restrict__ packed_weight_w6,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    const int32_t* __restrict__ m_indices,
    bfloat16_t* __restrict__ output,
    int size_m,
    int size_n,
    int size_k) {
  static_assert(BlockM == 128, "contiguous int8 compiles BM=128 only");
  static_assert(BlockN == 256, "contiguous int8 compiles BN=256 only");
  static_assert(BlockK == 64, "contiguous int8 uses BK=64");
  static_assert(NumWarps == 4, "contiguous int8 uses four waves per CTA");
  static_assert(NLoopNum == 4,
                "contiguous int8 uses four N fragments per warp");

#if !(defined(__gfx938__) || defined(__gfx936__) || defined(__gfx92a__) || defined(__gfx928__))
  return;
#else

  constexpr int kWarpSize = 64;          // AMD 一个 wave 有 64 lanes。
  constexpr int kMmacRows = 16;          // 单条 MMAC 输出 M 方向 16 行。
  constexpr int kMmacCols = 16;          // 单条 MMAC 输出 N 方向 16 列。
  constexpr int kMmacVgprs = 4;          // 每个 lane 输出 4 个 fp32 accumulator。
  constexpr int kWarpM = 128;            // 每个 warp 覆盖 M128。
  constexpr int kMSubtiles = kWarpM / kMmacRows;  // 128/16=8 个 M subtile。
  constexpr int kAccumTile = 32;         // int8 MMAC 的 K 粒度是 32。
  constexpr int kK64Tile = 64;           // 本版 BK=64，也是 A LDS 一行的宽度。
  constexpr int kKLoops = BlockK / kAccumTile;   // 64/32=2 次 MMAC。
  constexpr int kK64Loops = BlockK / kK64Tile;   // 64/64=1 个 K64 包。
  constexpr int kNFragments = 4;                 // 本版每 warp 计算 4 个 N16。
  constexpr int kAStageBytes = BlockM * BlockK;  // A LDS stage: 128*64=8192B。
  constexpr int kAStages = 2;                    // A LDS 使用 even/odd 双 stage。
  constexpr int kALdsTotalBytes = kAStages * kAStageBytes;  // 2*8192=16384B。
  constexpr int kALdsTileBytes = kMmacRows * BlockK;  // 一个 M16xK64 小块 1024B。
  static_assert(kMSubtiles == 8,
                "each warp computes eight 16-row M subtiles");
  static_assert(kKLoops == 2,
                "BK64 expects two 32-wide K MMAC loops");
  static_assert(kK64Loops == 1,
                "BK64 expects one K64 wrapper per K tile");
  static_assert(kNFragments == 4,
                "BN256 expects each warp to own four 16-column fragments");

  __align__(16) __shared__ char a_lds[kALdsTotalBytes];

  const int n_block_id = blockIdx.x;       // grid.x 沿 N，步长 BN=256。
  const int m_block_id = blockIdx.y;       // grid.y 沿 M，步长 BM=128。
  const int m_block_start = m_block_id * BlockM;

  // v91 scene assumption: launch M is 256-aligned, so grid.y is exact.
  // Padding rows are still represented by m_indices < 0 and must be skipped.
  const int32_t expert_id = m_indices[m_block_start];
  if (expert_id < 0) {
    return;
  }
  const int32_t expert_id_s = __builtin_hcu_readfirstlane(expert_id);

  const int thread_id = threadIdx.x;       // 0..255。
  const int warp_id_v = thread_id >> 6;    // 0..3，每 64 lanes 一个 warp。
  const int warp_id = __builtin_hcu_readfirstlane(warp_id_v);
  const int warp_m_group = 0;              // 4 个 warp 都复用同一个 M128 区域。
  const int warp_n_group = warp_id;        // 0..3，选择 4 个 N64 分片。
  const int lane_id = thread_id & (kWarpSize - 1);  // warp 内 0..63。
  const int lane_row = lane_id & 15;                // 0..15，对应 MMAC M16。
  const int lane_group = lane_id >> 4;              // 0..3，对应 K64 中 4 个 K16。

  const int n_block_start = n_block_id * BlockN;  // 当前 CTA 的 N256 起点。
  const int k_tiles = size_k / BlockK;            // K 总循环次数，例如 4096/64=64。
  const int expert_b_base =
      static_cast<int>(static_cast<int64_t>(expert_id_s) * size_n * size_k);
  const int b_n_tiles = size_n / kMmacCols;  // packed W6 的 N/16 tile 数。

  vec4_i32 acc[kMSubtiles][kNFragments];
#pragma unroll
  for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
#pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
      acc[m_loop][n_frag] = {0, 0, 0, 0};
    }
  }

  // A 搬运：4 个 warp 合起来覆盖 A[M128,K64]，写入 row-major LDS。
  const int a_lds_base_addr =
      static_cast<int>(reinterpret_cast<size_t>(a_lds));
  // N 计算：warp_n_group 选择四个连续 N16 fragment，lane_id&15 选择 fragment 内具体列。
  const int logical_n_base =
      n_block_start + warp_n_group * kNFragments * kMmacCols +
      (lane_id & 15);
  const int warp_m_start = warp_m_group * kWarpM;

  // --- 1. 初始化 ---
  // 一个 CTA 做 BM128 x BN256 x BK64。
  // 4 个 warp 做 1M x 4N 排布；每个 warp 负责 M128 x N64。
  // A 在 LDS 中共享；B 不进 LDS，每个 warp 直接从 global 取自己负责的 N64。
  // 本版 A LDS 使用双 stage：
  //   stage0/even 保存偶数 k_tile；
  //   stage1/odd  保存奇数 k_tile。
  BOperand b_even0;
  BOperand b_even1;
  BOperand b_even2;
  BOperand b_even3;
  BOperand b_odd0;
  BOperand b_odd1;
  BOperand b_odd2;
  BOperand b_odd3;
  BOperand b_pref0;
  BOperand b_pref1;
  BOperand b_pref2;
  BOperand b_pref3;
  AOperand a_regs[2];
  const int a_stage0_base_addr = a_lds_base_addr;
  const int a_stage1_base_addr = a_lds_base_addr + kAStageBytes;

  // --- 2. 序章 (Prolog) ---
  // prologue 只准备第一个 even tile：
  //   A tile0 -> LDS stage0；
  //   B tile0 -> b_even0..3；
  //   wait 后 stage0/even 成为主循环第一段 compute 的输入。
  {
    const int k_base = 0;
    const char* a_ptr =
        input + static_cast<int64_t>(m_block_start) * size_k +
        static_cast<int64_t>(k_base);
    const int lds_addr = a_stage0_base_addr;
    load_a_m128_k64_buffer_to_rowmajor_lds(a_ptr, size_k, lds_addr, thread_id);

#pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
      const int logical_n = logical_n_base + n_frag * kMmacCols;
      const int n_band = logical_n / kMmacCols;
      const int n_inner = logical_n & (kMmacCols - 1);
      const int b_k = k_base + lane_group * 16;
      const int b_k_band = b_k / kK64Tile;
      const int b_k_mid = (b_k & (kK64Tile - 1)) / 16;
      const int b_k_inner = b_k & 15;
      const int b_tile = (b_k_band * b_n_tiles + n_band) * 4 + b_k_mid;
      const int b_offset =
          expert_b_base + (b_tile * kMmacCols + n_inner) * 16 + b_k_inner;
      buffer_load_i32x4(packed_weight_w6, B_EVEN_AT(n_frag).i32x4, b_offset);
    }
  }
  wait_vmcnt4_barrier();

  // --- 3. 主循环 (Steady State) ---
  // 每轮处理一对 odd/even tile，不使用动态 stage 下标。
  // v91 沿用 v78 的第三组 B 寄存器 b_pref：
  //   3a load odd  -> stage1 / b_odd，同时提前 prefetch next even B -> b_pref；
  //   3b compute even，A 在 K32#0/K32#1 中间预读下一 M16；
  //   3c wait odd ready；
  //   3d load even -> stage0，B 复用 b_pref；
  //   3e compute odd，A 在 K32#0/K32#1 中间预读下一 M16；
  //   3f wait even ready，并把 b_pref 复制成下一轮的 b_even。
  for (int k_tile = 1; k_tile < k_tiles - 1; k_tile += 2) {
    // 3a. 加载当前 odd tile 到 stage1/odd，同时把 next even tile 的 B
    //     提前放进第三组寄存器。3a 后每个 wave 新发：
    //       odd A(8) + odd B(4) + pref B(4) = 16 个 future vm op
    //     这样 compute even 前可以用 vmcnt(16)，只要求上一轮的 even B 已经完成。
    {
      const int k_base = k_tile * BlockK;
      const char* a_ptr =
          input + static_cast<int64_t>(m_block_start) * size_k +
          static_cast<int64_t>(k_base);
      const int lds_addr = a_stage1_base_addr;
      load_a_m128_k64_buffer_to_rowmajor_lds(a_ptr, size_k, lds_addr, thread_id);

#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int logical_n = logical_n_base + n_frag * kMmacCols;
        const int n_band = logical_n / kMmacCols;
        const int n_inner = logical_n & (kMmacCols - 1);
        const int b_k = k_base + lane_group * 16;
        const int b_k_band = b_k / kK64Tile;
        const int b_k_mid = (b_k & (kK64Tile - 1)) / 16;
        const int b_k_inner = b_k & 15;
        const int b_tile = (b_k_band * b_n_tiles + n_band) * 4 + b_k_mid;
        const int b_offset =
            expert_b_base + (b_tile * kMmacCols + n_inner) * 16 + b_k_inner;
        buffer_load_i32x4(packed_weight_w6, B_ODD_AT(n_frag).i32x4, b_offset);
      }

      const int pref_k_base = (k_tile + 1) * BlockK;
#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int logical_n = logical_n_base + n_frag * kMmacCols;
        const int n_band = logical_n / kMmacCols;
        const int n_inner = logical_n & (kMmacCols - 1);
        const int b_k = pref_k_base + lane_group * 16;
        const int b_k_band = b_k / kK64Tile;
        const int b_k_mid = (b_k & (kK64Tile - 1)) / 16;
        const int b_k_inner = b_k & 15;
        const int b_tile = (b_k_band * b_n_tiles + n_band) * 4 + b_k_mid;
        const int b_offset =
            expert_b_base + (b_tile * kMmacCols + n_inner) * 16 + b_k_inner;
        buffer_load_i32x4(packed_weight_w6, B_PREF_AT(n_frag).i32x4, b_offset);
      }
    }

    // 3b. 计算已经就绪的 even tile，从 stage0/even 读 A。
    {
      // A LDS -> A regs 使用 2 槽滚动；进入当前 M16 的第一段 MMAC 前
      // 预读下一 M16，让 K32#0 和 K32#1 都参与覆盖 LDS load 延迟。
      LOAD_A_M16_FROM_LDS(a_stage0_base_addr, warp_m_start/kMmacRows, a_regs[0]);
      wait_lgkmcnt_zero();
      wait_vmcnt16_zero();

#pragma unroll
      for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int compute_slot = m_loop & 1;
        const int load_slot = compute_slot ^ 1;

        if (m_loop + 1 < kMSubtiles) {
          LOAD_A_M16_FROM_LDS(a_stage0_base_addr, warp_m_start/kMmacRows + m_loop + 1,
                                 a_regs[load_slot]);
        }

#pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
          MMAC_I8_K32(acc[m_loop][n_frag],
                       a_regs[compute_slot].int8x8[0],
                       B_EVEN_AT(n_frag).int8x8[0]);
        }

#pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
          MMAC_I8_K32(acc[m_loop][n_frag],
                       a_regs[compute_slot].int8x8[1],
                       B_EVEN_AT(n_frag).int8x8[1]);
        }

        if (m_loop + 1 < kMSubtiles) {
          wait_lgkmcnt_zero();
        }
      }
    }

    // 3c. 等待 odd tile 的 A buffer_load 写入 LDS；允许 odd B 和 pref B 继续在飞。
    wait_vmcnt8_barrier();

    // 3d. 加载当前 even tile 到 stage0/even。B 已经在 3a 提前进入 b_pref。
    {
      const int k_base = (k_tile + 1) * BlockK;
      const char* a_ptr =
          input + static_cast<int64_t>(m_block_start) * size_k +
          static_cast<int64_t>(k_base);
      const int lds_addr = a_stage0_base_addr;
      load_a_m128_k64_buffer_to_rowmajor_lds(a_ptr, size_k, lds_addr, thread_id);
    }

    // 3e. 计算已经就绪的 odd tile，从 stage1/odd 读 A。
    {
      LOAD_A_M16_FROM_LDS(a_stage1_base_addr, warp_m_start/kMmacRows, a_regs[0]);
      wait_lgkmcnt_zero();
      // 3d 之后，允许 newer 的 pref B(4) + next even A(8) 继续在飞；
      // 只要求 older 的 odd B 已经进入寄存器。
      wait_vmcnt12_zero();

#pragma unroll
      for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int compute_slot = m_loop & 1;
        const int load_slot = compute_slot ^ 1;

        if (m_loop + 1 < kMSubtiles) {
          LOAD_A_M16_FROM_LDS(a_stage1_base_addr, warp_m_start/kMmacRows + m_loop + 1,
                                 a_regs[load_slot]);
        }

#pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
          MMAC_I8_K32(acc[m_loop][n_frag],
                       a_regs[compute_slot].int8x8[0],
                       B_ODD_AT(n_frag).int8x8[0]);
        }

#pragma unroll
        for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
          MMAC_I8_K32(acc[m_loop][n_frag],
                       a_regs[compute_slot].int8x8[1],
                       B_ODD_AT(n_frag).int8x8[1]);
        }

        if (m_loop + 1 < kMSubtiles) {
          wait_lgkmcnt_zero();
        }
      }
    }

    // 3f. b_pref 比 even A 更早发射，因此这里用 vmcnt(0)+barrier 同时收住
    //     pref B 和 even A。随后把 pref B 变成下一轮的 b_even。
    wait_vmcnt_barrier();
    b_even0 = b_pref0;
    b_even1 = b_pref1;
    b_even2 = b_pref2;
    b_even3 = b_pref3;
  }

  // --- 4. 收尾 (Tail) ---
  // 主循环退出时，最后一个 even tile 已经在 stage0/even 中就绪。
  // 这里先发最后一个 odd tile 的 load，再计算 even tile，最后等待并计算 odd tile。
  {
    const int tail_odd_tile = k_tiles - 1;
    const int k_base = tail_odd_tile * BlockK;
    const char* a_ptr =
        input + static_cast<int64_t>(m_block_start) * size_k +
        static_cast<int64_t>(k_base);
    const int lds_addr = a_stage1_base_addr;
    load_a_m128_k64_buffer_to_rowmajor_lds(a_ptr, size_k, lds_addr, thread_id);

#pragma unroll
    for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
      const int logical_n = logical_n_base + n_frag * kMmacCols;
      const int n_band = logical_n / kMmacCols;
      const int n_inner = logical_n & (kMmacCols - 1);
      const int b_k = k_base + lane_group * 16;
      const int b_k_band = b_k / kK64Tile;
      const int b_k_mid = (b_k & (kK64Tile - 1)) / 16;
      const int b_k_inner = b_k & 15;
      const int b_tile = (b_k_band * b_n_tiles + n_band) * 4 + b_k_mid;
      const int b_offset =
          expert_b_base + (b_tile * kMmacCols + n_inner) * 16 + b_k_inner;
      buffer_load_i32x4(packed_weight_w6, B_ODD_AT(n_frag).i32x4, b_offset);
    }
  }

  // 4a. 计算最后一个 even tile。
  {
    LOAD_A_M16_FROM_LDS(a_stage0_base_addr, warp_m_start/kMmacRows, a_regs[0]);
    wait_lgkmcnt_zero();

#pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
      const int compute_slot = m_loop & 1;
      const int load_slot = compute_slot ^ 1;

      if (m_loop + 1 < kMSubtiles) {
        LOAD_A_M16_FROM_LDS(a_stage0_base_addr, warp_m_start/kMmacRows + m_loop + 1,
                               a_regs[load_slot]);
      }

#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        MMAC_I8_K32(acc[m_loop][n_frag],
                     a_regs[compute_slot].int8x8[0],
                     B_EVEN_AT(n_frag).int8x8[0]);
      }

#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        MMAC_I8_K32(acc[m_loop][n_frag],
                     a_regs[compute_slot].int8x8[1],
                     B_EVEN_AT(n_frag).int8x8[1]);
      }

      if (m_loop + 1 < kMSubtiles) {
        wait_lgkmcnt_zero();
      }
    }
  }

  // 4b. 等待最后一个 odd tile 的 A buffer_load 到达，再从 stage1/odd 计算。
  wait_vmcnt4_barrier();
  {
    LOAD_A_M16_FROM_LDS(a_stage1_base_addr, warp_m_start/kMmacRows, a_regs[0]);
    wait_lgkmcnt_zero();
    wait_vmcnt_zero();

#pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
      const int compute_slot = m_loop & 1;
      const int load_slot = compute_slot ^ 1;

      if (m_loop + 1 < kMSubtiles) {
        LOAD_A_M16_FROM_LDS(a_stage1_base_addr, warp_m_start/kMmacRows + m_loop + 1,
                               a_regs[load_slot]);
      }

#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        MMAC_I8_K32(acc[m_loop][n_frag],
                     a_regs[compute_slot].int8x8[0],
                     B_ODD_AT(n_frag).int8x8[0]);
      }

#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        MMAC_I8_K32(acc[m_loop][n_frag],
                     a_regs[compute_slot].int8x8[1],
                     B_ODD_AT(n_frag).int8x8[1]);
      }

      if (m_loop + 1 < kMSubtiles) {
        wait_lgkmcnt_zero();
      }
    }
  }

  // --- 5. 写回全局内存 ---
  // N 维在 launch 侧保证按 BN=256 对齐；先缓存本 wave 负责列的 scale，
  // 避免每个 M subtile 重复加载同一组 weight_scale。
  float col_scale[kNFragments][kMmacVgprs];
#pragma unroll
  for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
    const int col_base =
        n_block_start + warp_n_group * kNFragments * kMmacCols +
        n_frag * kMmacCols + lane_group * kMmacVgprs;
#pragma unroll
    for (int elem = 0; elem < kMmacVgprs; ++elem) {
      col_scale[n_frag][elem] =
          weight_scale[static_cast<int64_t>(expert_id_s) * size_n +
                       col_base + elem];
    }
  }

  // In the contiguous W8A8 int8 scene, M is padded to 256 and every expert
  // occupies contiguous rows. Therefore the last row of this BM128 block is
  // enough to distinguish the fast full-block store path from the padding tail.
  const bool full_block_valid = (m_indices[m_block_start + BlockM - 1] >= 0);

  if (full_block_valid) {
#pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
      const int row = m_block_start + warp_m_start + m_loop * kMmacRows + lane_row;
      const float row_scale = input_scale[row];
#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int col_base =
            n_block_start + warp_n_group * kNFragments * kMmacCols +
            n_frag * kMmacCols + lane_group * kMmacVgprs;
        const int col = col_base;
        const float value0 = static_cast<float>(acc[m_loop][n_frag][0]) * row_scale *
                             col_scale[n_frag][0];
        const float value1 = static_cast<float>(acc[m_loop][n_frag][1]) * row_scale *
                             col_scale[n_frag][1];
        const float value2 = static_cast<float>(acc[m_loop][n_frag][2]) * row_scale *
                             col_scale[n_frag][2];
        const float value3 = static_cast<float>(acc[m_loop][n_frag][3]) * row_scale *
                             col_scale[n_frag][3];
        *reinterpret_cast<uint64_t*>(output + static_cast<int64_t>(row) *
                                             size_n + col) =
            pack_bf16x4(value0, value1, value2, value3);
      }
    }
  } else {
#pragma unroll
    for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
      const int row = m_block_start + warp_m_start + m_loop * kMmacRows + lane_row;
      if (m_indices[row] < 0) {
        continue;
      }
      const float row_scale = input_scale[row];
#pragma unroll
      for (int n_frag = 0; n_frag < kNFragments; ++n_frag) {
        const int col_base =
            n_block_start + warp_n_group * kNFragments * kMmacCols +
            n_frag * kMmacCols + lane_group * kMmacVgprs;
        const int col = col_base;
        const float value0 = static_cast<float>(acc[m_loop][n_frag][0]) * row_scale *
                             col_scale[n_frag][0];
        const float value1 = static_cast<float>(acc[m_loop][n_frag][1]) * row_scale *
                             col_scale[n_frag][1];
        const float value2 = static_cast<float>(acc[m_loop][n_frag][2]) * row_scale *
                             col_scale[n_frag][2];
        const float value3 = static_cast<float>(acc[m_loop][n_frag][3]) * row_scale *
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
  TORCH_CHECK(requested == 0 || requested == 4,
              "contiguous int8 BN256 only supports N_LOOP_NUM=4; pass 0 or 4");
  TORCH_CHECK(size_n % kBlockN == 0,
              "N must be divisible by BLOCK_N=256 for contiguous int8 BN256");
  return 4;
}

template <int NLoopNum>
void launch_i8_contig_ll_nloop(torch::Tensor input,
                                torch::Tensor b_qweight,
                                torch::Tensor output,
                                torch::Tensor a_scale,
                                torch::Tensor b_scale,
                                torch::Tensor m_indices) {
  static_assert(NLoopNum == 4,
                "contiguous int8 BN256 supports N_LOOP_NUM=4 only");
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 256;
  constexpr int kBlockK = 64;
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
  TORCH_CHECK(size_m % 256 == 0,
              "v91 scene path expects M padded/aligned to 256");
  TORCH_CHECK(size_k % (2 * kBlockK) == 0,
              "K must be divisible by 128 for v91 two-stage A pipeline");
  TORCH_CHECK(size_k >= 4 * kBlockK,
              "v91 scene path currently requires K >= 256; K=128 needs a dedicated two-tile tail path");
  TORCH_CHECK(size_n % kBlockN == 0,
              "N must be divisible by BLOCK_N=256 for contiguous int8 BN256");
  TORCH_CHECK(input.scalar_type() == at::kChar,
              "contiguous int8 only supports the int8_e4m3fn input path");

  const dim3 block(kNumWarps * 64, 1, 1);
  const dim3 grid(ceil_div(size_n, kBlockN),
                  size_m / kBlockM, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  i8_contig_ll_kernel<kBlockM, kBlockN, kBlockK, kNumWarps, NLoopNum>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const char*>(input.data_ptr()),
          reinterpret_cast<const char*>(b_qweight.data_ptr()),
          a_scale.data_ptr<float>(),
          b_scale.data_ptr<float>(),
          m_indices.data_ptr<int32_t>(),
          reinterpret_cast<bfloat16_t*>(output.data_ptr()),
          size_m,
          size_n,
          size_k);
}

bool current_device_is_supported_gfx() {
  int device_id = 0;
  hipError_t err = hipGetDevice(&device_id);
  TORCH_CHECK(err == hipSuccess, "failed to get current HIP device");

  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, device_id);
  TORCH_CHECK(err == hipSuccess, "failed to get HIP device properties");

  const std::string arch(prop.gcnArchName);
  return arch.rfind("gfx938", 0) == 0 || arch.rfind("gfx936", 0) == 0 ||
         arch.rfind("gfx92a", 0) == 0 ||
         arch.rfind("gfx928", 0) == 0;
}


torch::Tensor i8_contig_ll(torch::Tensor input,
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
  TORCH_CHECK(current_device_is_supported_gfx(),
              "i8_contig_ll_v92 requires gfx938/gfx936/gfx92a/gfx928");
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
              "contiguous int8 expects a_scale as [M] or [M,1]");
  TORCH_CHECK(a_scale.size(0) == input.size(0),
              "a_scale M dimension must match input");
  TORCH_CHECK(b_scale.dim() == 2 ||
                  (b_scale.dim() == 3 && b_scale.size(2) == 1),
              "contiguous int8 expects b_scale as [E,N] or [E,N,1]");
  TORCH_CHECK(m_indices.numel() == input.size(0),
              "m_indices length must match M");
  TORCH_CHECK(b_qweight.size(3) == 4 && b_qweight.size(4) == 16 &&
                  b_qweight.size(5) == 16,
              "packed weight trailing dims must be [4,16,16]");
  TORCH_CHECK(b_scale.size(0) == b_qweight.size(0),
              "b_scale expert dimension must match packed weight");
  TORCH_CHECK(b_scale.size(1) == b_qweight.size(2) * b_qweight.size(4),
              "b_scale N dimension must match packed weight");
  TORCH_CHECK(block_m == 128 && block_n == 256,
              "contiguous int8 is compiled only for BM=128, BN=256");

  torch::Tensor a_scale_kernel =
      a_scale.dim() == 2 ? a_scale.view({a_scale.size(0)}) : a_scale;
  torch::Tensor b_scale_kernel =
      b_scale.dim() == 3 ? b_scale.view({b_scale.size(0), b_scale.size(1)})
                         : b_scale;

  const int resolved_n_loop =
      resolve_n_loop_num(static_cast<int>(output.size(1)),
                         static_cast<int>(n_loop_num));
  (void)resolved_n_loop;
  launch_i8_contig_ll_nloop<4>(input, b_qweight, output, a_scale_kernel, b_scale_kernel,
                                m_indices);
  return output;
}

}  // namespace i8_contig_ll_v92

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
  m.def("m_grouped_w8a8_gemm_nt_contiguous_gfx936("
        "Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, "
        "Tensor b_scale, Tensor m_indices, int block_m, int block_n, "
        "int n_loop_num) -> Tensor");
  m.impl("m_grouped_w8a8_gemm_nt_contiguous_gfx936", torch::kCUDA,
         &i8_contig_ll_v92::i8_contig_ll);
}
