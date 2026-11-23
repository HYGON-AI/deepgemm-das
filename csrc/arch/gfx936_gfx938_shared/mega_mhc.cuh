// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// mega_mhc 的 gfx936/gfx938 共用实现。
// Python 接口在 deepgemm/mega_mhc.py；host 契约和 workspace 在 mega_mhc_dispatch.cu。
// 本文件集中保存转换、投影、epilogue 与发射逻辑；入口在 mega_mhc.cu，MMAC 的
// 架构差异由下面的 __gfx936__ / __gfx938__ 预处理分支选择。
// 默认计算路径为 main + reduce；MEGA_MHC_LAUNCH=fused 选择融合路径。
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#include "mega_mhc.h"

namespace deepgemm::mega_mhc {

// ---- 公共类型、转换与参数 ----

// ---------------------------------------------------------------------------
// 问题常量（官方契约）
// ---------------------------------------------------------------------------
constexpr int kHcMult = 4;
constexpr int kNumRoutes = kHcMult;                       // 4
constexpr int kNumHCOutputs = kHcMult * (kHcMult + 2);    // 24
constexpr int kNumMixes = kNumRoutes + kNumRoutes + kNumRoutes * kNumRoutes;  // 24
constexpr int kHiddenPerSF = 128;                         // 一个 int32 装 4 个 UE8M0 字节（gran_k 32）
constexpr int kSfGran = 32;

// ---------------------------------------------------------------------------
// launch 形状与资源分块
// ---------------------------------------------------------------------------
// 每个 CTA 的两个 MMAC M tile 复用同一份 packed-Fn；A tile 按 route 分块存入 LDS。
constexpr int kBlockM = 32;        // 每个 CTA 的 token 数 = kMGroups 个 MMAC M tile
constexpr int kMGroups = kBlockM / 16;
constexpr int kThreadsPerToken = 256 / kBlockM;   // 负责一个 token 的 post-mix 线程数
constexpr int kChunkH = 32;        // 每次迭代、每条 route 处理的 hidden 列数
constexpr int kMmacK = 16;         // MMAC 的 K
constexpr int kNumThreads = 256;   // 4 个 wave × 64 lane
constexpr int kNumWaves = kNumThreads / 64;
constexpr int kChunkK = kNumRoutes * kChunkH;               // 每次迭代128个k
constexpr int kNumKTiles = kChunkK / kMmacK;                // 每次迭代8个K16块
constexpr int kSlotsPerToken = 32;                          // 24 个投影 + sq_r + sq_u + 补齐
constexpr int kProjSlots = kNumHCOutputs;                   // 0..23
constexpr int kSqRSlot = kNumHCOutputs;                     // 24
constexpr int kSqUSlot = kNumHCOutputs + 1;                 // 25
constexpr int kNumSlotsUsed = kNumHCOutputs + 2;            // 26
constexpr int kMaxEpilogueSlots = 64;                       // 每次在共享缓冲中暂存的 slot 数
constexpr int kMaxPartialBlock = kMaxEpilogueSlots * kSlotsPerToken;  // 单位：float

using bf16 = __hip_bfloat16;
using fp32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using bf16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bf16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using i32x4 = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;

// BF16 与 FP32 的位级表示转换；向量载荷先整体读入，再按元素解包。
union Bf16x4Bits {
  bf16x4 v;
  uint16_t s[4];
};

union F32x4Parts {
  fp32x4 v;
  float f[4];
};

union Bf16Bits {
  uint16_t u;
  bf16 b;
};

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ bf16 bf16_from_bits(uint16_t bits) {
  Bf16Bits c;
  c.u = bits;
  return c.b;
}

__device__ __forceinline__ uint16_t bf16_to_bits(bf16 value) {
  Bf16Bits c;
  c.b = value;
  return c.u;
}

// ---------------------------------------------------------------------------
// 转换
// ---------------------------------------------------------------------------

// FP32 转 BF16，采用就近舍入到偶数（RNE）。
__device__ __forceinline__ uint16_t f32_to_bf16_rne_bits(float x) {
  uint32_t bits = __builtin_bit_cast(uint32_t, x);
  if (~bits & 0x7f800000u) {
    bits += 0x7fffu + ((bits >> 16) & 1u);
  } else if (bits & 0xffffu) {
    bits |= 0x10000u;                        // 保持 signalling NaN 仍是 NaN
  }
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ bf16 f32_to_bf16_rne(float x) {
  return bf16_from_bits(f32_to_bf16_rne_bits(x));
}

__device__ __forceinline__ float bf16_to_f32(bf16 x) {
  return bf16_bits_to_f32(bf16_to_bits(x));
}

// 软件 E4M3FN 编码：RNE 舍入、次正规数与带符号零；超出编码范围时返回 NaN。
__device__ __forceinline__ uint8_t f32_to_e4m3fn(float x) {
  uint32_t b = __builtin_bit_cast(uint32_t, x);
  uint32_t sign = (b >> 24) & 0x80u;
  uint32_t ax = b & 0x7fffffffu;
  if (ax > 0x7f800000u) return static_cast<uint8_t>(sign | 0x7fu);
  int32_t e = static_cast<int32_t>(ax >> 23) - 127;
  uint32_t mant = ax & 0x7fffffu;
  int32_t E = e + 7;
  if (E >= 1) {
    uint32_t m3 = mant >> 20;
    uint32_t rest = mant & 0xfffffu;
    if (rest > 0x80000u || (rest == 0x80000u && (m3 & 1u))) {
      m3 += 1u;
      if (m3 == 8u) { m3 = 0u; E += 1; }
    }
    if (E > 15) return static_cast<uint8_t>(sign | 0x7fu);      // 溢出 -> NaN
    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(E) << 3) | m3);
  }
  // 次正规数的量子为 2^-9，右移量为 21-E。移位数不小于32时结果直接舍入为带符号零。
  uint32_t full = (1u << 23) | mant;
  const int32_t rem_bits = 21 - E;
  if (rem_bits >= 32) return static_cast<uint8_t>(sign);
  uint32_t base = full >> rem_bits;
  uint32_t rem = full & ((1u << rem_bits) - 1u);
  const uint32_t halfway = 1u << (rem_bits - 1);
  if (rem > halfway || (rem == halfway && (base & 1u))) base += 1u;
  if (base >= 8u) return static_cast<uint8_t>(sign | (1u << 3));
  return static_cast<uint8_t>(sign | base);
}

// UE8M0：取 >= |x| 的最小 2 的幂的指数（参考实现的 `ceil_to_ue8m0`），
// 以原始 8 位指数字段返回。
__device__ __forceinline__ uint8_t ceil_to_ue8m0_exponent(float x) {
  uint32_t bits = __builtin_bit_cast(uint32_t, x) & 0x7fffffffu;
  int32_t exp = static_cast<int32_t>(bits >> 23);
  if (bits & 0x7fffffu) exp += 1;
  if (exp < 1) exp = 1;
  if (exp > 254) exp = 254;
  return static_cast<uint8_t>(exp);
}

// ---------------------------------------------------------------------------
// MMAC
// ---------------------------------------------------------------------------
__device__ __forceinline__ fp32x4 mmac_bf16_16x16x16(const bf16x4& a, const bf16x4& b,
                                                     fp32x4 acc) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__)
  return __builtin_hcu_mmac_f32_16x16x16_bf16(a, b, acc);
#elif defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, acc, false, false);
#else
  return acc;
#endif
}

// 从 shared memory 一次读 8 个连续 bf16（4 个 dword），拆成 MMAC 消费的两个 K = 16 半段。
__device__ __forceinline__ bf16x8 ds_read_b128_bf16(const bf16* base, int byte_offset) {
  uint32_t raw[4];
  const int addr = static_cast<int>(reinterpret_cast<size_t>(base)) + byte_offset;
  asm volatile("ds_read_b128 %0, %1\n" : "=v"(raw[0]), "=v"(raw[1]), "=v"(raw[2]),
               "=v"(raw[3]) : "v"(addr) : "memory");
  bf16x8 out;
  const uint32_t* src = raw;
  uint16_t* dst = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
  for (int i = 0; i < 8; ++i) dst[i] = reinterpret_cast<const uint16_t*>(src)[i];
  return out;
}

__device__ __forceinline__ void wait_lgkmcnt(int n) {
  asm volatile("s_waitcnt lgkmcnt(%0)\n" : : "n"(n) : "memory");
}

__device__ __forceinline__ void wait_vmcnt(int n) {
  asm volatile("s_waitcnt vmcnt(%0)\n" : : "n"(n) : "memory");
}

// ---------------------------------------------------------------------------
// 参数 / workspace 结构体
// ---------------------------------------------------------------------------
struct MainParams {
  const bf16* __restrict__ x;               // [T, H]
  const bf16* __restrict__ residual;        // [T, 4, H]
  const float* __restrict__ post_mix;       // [T, 4]
  const float* __restrict__ comb_mix;       // [T, 4, 4]（输入路 i -> 输出路 j）
  const float* __restrict__ prev_mix;       // [T, 4] 或空（normal）
  const bf16* __restrict__ fn_packed;       // 24×4H×2 个 BF16，每个 k 组内 hi4/lo4 交错
  int64_t fn_half;                          // 每一半的元素数（24 * 4H）
  bf16* __restrict__ new_residual;          // [T, 4, H]
  bf16* __restrict__ u_out;                 // [T, H] 未归一化的 collapse（仅 shifted）
  float* __restrict__ partial;              // [T, num_slots, 32]
  int num_tokens;
  int hidden;
  int num_splits;
  int num_slots;
};

struct ReduceParams {
  const float* __restrict__ partial;        // [T, num_slots, 32]
  const float* __restrict__ mix_scales;     // [3]
  const float* __restrict__ mix_bases;      // [24]
  const bf16* __restrict__ rmsnorm_weight;  // [H]
  const bf16* __restrict__ new_residual;    // [T, 4, H]（normal 时作为 Z 的来源）
  bf16* __restrict__ u_io;                  // [T, H]：先写入，再原地归一化
  float* __restrict__ new_prev_mix;         // [T, 4] 或空
  float* __restrict__ new_post_mix;         // [T, 4]
  float* __restrict__ new_comb_mix;         // [T, 4, 4]
  uint8_t* __restrict__ y_fp8;              // [T, H] 或空
  int* __restrict__ y_gemm_sf;              // [H/128, align(T,4)] 列主序，或空
  int* __restrict__ y_routed_sf;            // [T, H/128] 连续，或空
  int* __restrict__ y_shared_sf;            // 行被置换的布局，或空
  int num_tokens;
  int hidden;
  int num_slots;
  int gemm_sf_stride1;                      // 每个 SF 列的元素数
  int shared_sf_block_m;
  int shared_sf_stride1;
  int shared_sf_offset;
  float hc_norm_eps;
  float hc_pre_eps;
  float hc_post_scale;
  float sinkhorn_eps;
  int num_sinkhorn_iters;
  float rmsnorm_eps;
  float rmsnorm_scale;
  bool is_shifted;
  int flags;                                // bit0：输出 FP8 与 SF
};

// 共享专家 SF 布局的行置换（对应官方的 `extra_sf_rows`）。
__device__ __forceinline__ int extra_sf_row(int token, int block_m) {
  const int in_block = token % block_m;
  const int block = token / block_m;
  const int align_block = ((block_m + 127) / 128) * 128;
  return block * align_block + (in_block / 128) * 128 + (in_block % 32) * 4 +
         ((in_block % 128) / 32);
}

// ---- 门控与 Sinkhorn ----

__device__ __forceinline__ float sigmoidf_exact(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Lane 并行的 Sinkhorn（16 个 lane，元素下标 = row * 4 + col）。
//
// 与参考实现同样的算术顺序（`softmax(-1)`、`+eps`、`/(列和+eps)`，然后
// `(iters-1)` 轮 `[/(行和+eps), /(列和+eps)]`），但 16 个元素分散在 16 个 lane 上，
// 于是行/列求和变成 2 次 shuffle 归约，而且不同元素的除法可以并行。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float shfl_sum_row4(float v) {
  v += __shfl_xor(v, 1);
  v += __shfl_xor(v, 2);
  return v;
}

__device__ __forceinline__ float shfl_sum_col4(float v) {
  v += __shfl_xor(v, 4);
  v += __shfl_xor(v, 8);
  return v;
}

__device__ __forceinline__ void sinkhorn_lanes(float* m, int iters, float eps, int lane) {
  float v = m[lane];
  float mx = v;
  mx = fmaxf(mx, __shfl_xor(mx, 1));
  mx = fmaxf(mx, __shfl_xor(mx, 2));
  const float e = expf(v - mx);
  v = e / shfl_sum_row4(e) + eps;
  v = v / (shfl_sum_col4(v) + eps);
  for (int it = 1; it < iters; ++it) {
    v = v / (shfl_sum_row4(v) + eps);
    v = v / (shfl_sum_col4(v) + eps);
  }
  m[lane] = v;
}

// 原地做 4x4 块上的行 softmax + 行/列交替归一化。
__device__ __forceinline__ void sinkhorn_4x4(float* m, int iters, float eps) {
  // 行 softmax
  for (int r = 0; r < kNumRoutes; ++r) {
    float mx = m[r * kNumRoutes];
    for (int c = 1; c < kNumRoutes; ++c) mx = fmaxf(mx, m[r * kNumRoutes + c]);
    float sum = 0.0f;
    for (int c = 0; c < kNumRoutes; ++c) {
      const float e = expf(m[r * kNumRoutes + c] - mx);
      m[r * kNumRoutes + c] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (int c = 0; c < kNumRoutes; ++c) m[r * kNumRoutes + c] *= inv;
  }
  for (int r = 0; r < kNumRoutes; ++r)
    for (int c = 0; c < kNumRoutes; ++c) m[r * kNumRoutes + c] += eps;
  // 第一轮列归一化
  for (int c = 0; c < kNumRoutes; ++c) {
    float sum = 0.0f;
    for (int r = 0; r < kNumRoutes; ++r) sum += m[r * kNumRoutes + c];
    const float denom = sum + eps;
    for (int r = 0; r < kNumRoutes; ++r) m[r * kNumRoutes + c] /= denom;
  }
  for (int it = 1; it < iters; ++it) {
    for (int r = 0; r < kNumRoutes; ++r) {
      float sum = 0.0f;
      for (int c = 0; c < kNumRoutes; ++c) sum += m[r * kNumRoutes + c];
      const float denom = sum + eps;
      for (int c = 0; c < kNumRoutes; ++c) m[r * kNumRoutes + c] /= denom;
    }
    for (int c = 0; c < kNumRoutes; ++c) {
      float sum = 0.0f;
      for (int r = 0; r < kNumRoutes; ++r) sum += m[r * kNumRoutes + c];
      const float denom = sum + eps;
      for (int r = 0; r < kNumRoutes; ++r) m[r * kNumRoutes + c] /= denom;
    }
  }
}

// 由归约后的投影值与 HC 平方和算出 L[o]。
__device__ __forceinline__ float mix_logit(float proj, float hc_sqr_sum, int o,
                                           int hidden, const float* mix_scales,
                                           const float* mix_bases, float hc_norm_eps) {
  const float rstd = rsqrtf(hc_sqr_sum / static_cast<float>(kNumRoutes * hidden) +
                            hc_norm_eps);
  const int cls = o < kNumRoutes ? 0 : (o < 2 * kNumRoutes ? 1 : 2);
  return proj * rstd * mix_scales[cls] + mix_bases[o];
}

// ---- Fn 打包与投影 ----

// 一个 K = 16 块的 A fragment：a[i] = A[m = row][k = 4*col + i]。
__device__ __forceinline__ bf16x4 load_a_fragment(const bf16* z_tile, int row, int col,
                                                  int k_tile) {
  const int k_local = k_tile * kMmacK + col * 4;
  return *reinterpret_cast<const bf16x4*>(z_tile + row * kChunkK + k_local);
}

struct BF16Pair {
  bf16x4 hi;
  bf16x4 lo;
};

// 从预打包权重读取 K=16、N=16 的 BF16 hi/lo 分片。
// o_row 是本 lane 的权重行；超过24个有效输出的填充行返回零。
__device__ __forceinline__ BF16Pair load_b_fragment_packed(
    const bf16* __restrict__ fn_packed, int o_row, int col, int64_t global_k,
    int64_t fn_half, bool valid) {
  BF16Pair out;
  out.hi = bf16x4{0, 0, 0, 0};
  out.lo = bf16x4{0, 0, 0, 0};
  if (!valid) return out;
  // 每组按 hi4/lo4 交错存放，共16字节；两个 K16 分片从相邻地址读取。
  const int64_t group = static_cast<int64_t>(o_row) * (fn_half / 24) + global_k + col * 4;
  const bf16* base = fn_packed + 2 * group;
  out.hi = *reinterpret_cast<const bf16x4*>(base);
  out.lo = *reinterpret_cast<const bf16x4*>(base + 4);
  return out;
}

// 打包 kernel：fp32 的 fn -> BF16 的 hi 与残差 lo，装进一个 2 * |fn| 的缓冲。
__global__ void pack_fn_kernel(const float* __restrict__ fn, bf16* __restrict__ packed,
                               int64_t numel) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (i >= numel) return;
  const float4 raw = *reinterpret_cast<const float4*>(fn + i);
  const float v[4] = {raw.x, raw.y, raw.z, raw.w};
  bf16x4 hi;
  bf16x4 lo;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint16_t h = f32_to_bf16_rne_bits(v[j]);
    hi[j] = h;
    lo[j] = f32_to_bf16_rne_bits(v[j] - bf16_bits_to_f32(h));
  }
  // 交错存放：第 i/4 组装到输出元素 [2i, 2i+8)，先 hi4 再 lo4。
  *reinterpret_cast<bf16x4*>(packed + 2 * i) = hi;
  *reinterpret_cast<bf16x4*>(packed + 2 * i + 4) = lo;
}

// 两次 MMAC（先 hi 后 lo）累加进同一个累加器。
__device__ __forceinline__ fp32x4 project_k16(const bf16x4& a, const BF16Pair& b,
                                              fp32x4 acc) {
  acc = mmac_bf16_16x16x16(a, b.hi, acc);
  acc = mmac_bf16_16x16x16(a, b.lo, acc);
  return acc;
}

// ---- main / reduce kernel ----

constexpr int kSfGroup = 32;          // 每个 UE8M0 scale 覆盖的元素数

// 一个打包好的 UE8M0 字，按调用方要求的 SF 布局写出
// （列主序 GEMM、连续的 routed、或共享专家的行置换）。
__device__ __forceinline__ void store_sf_word(const ReduceParams& p, int token, int sf_col,
                                              uint32_t packed) {
  if (p.y_gemm_sf != nullptr) {
    p.y_gemm_sf[static_cast<int64_t>(sf_col) * p.gemm_sf_stride1 + token] =
        static_cast<int>(packed);
  }
  if (p.y_routed_sf != nullptr) {
    p.y_routed_sf[static_cast<int64_t>(token) * (p.hidden / kHiddenPerSF) + sf_col] =
        static_cast<int>(packed);
  }
  if (p.y_shared_sf != nullptr) {
    const int row = extra_sf_row(token, p.shared_sf_block_m);
    p.y_shared_sf[static_cast<int64_t>(row) +
                  static_cast<int64_t>(sf_col) * p.shared_sf_stride1] =
        static_cast<int>(packed);
  }
}

// ---------------------------------------------------------------------------
// 单 token 的 epilogue（mix 门控、collapse、RMSNorm、可选 FP8/SF）
// ---------------------------------------------------------------------------
// 共享缓冲由调用方传入，这样 split 与 fused 两条路径共用同一份代码
// （也共用同一套 LDS 预算纪律）。
__device__ void epilogue_one_token(const ReduceParams& p, int t, int tid,
                                   float* l_mix, float* l_pre, float* l_sq,
                                   float* l_red, float* l_sf_exp, float* l_part,
                                   float* l_comb, int* l_flag) {
  // ---- 按固定的 slot 顺序归约各 split 的 partial ------------------------------
  // 每个 token 的 partial 按 [slot][32] 存放，所有线程连续搬运到共享缓冲。
  // 按 kMaxEpilogueSlots 分块并以固定 slot 顺序归约，保证同一配置下结果可重复。
  float sums[kNumSlotsUsed];
#pragma unroll
  for (int s = 0; s < kNumSlotsUsed; ++s) sums[s] = 0.0f;
  const float* base = p.partial + static_cast<int64_t>(t) * p.num_slots * kSlotsPerToken;
  for (int slot0 = 0; slot0 < p.num_slots; slot0 += kMaxEpilogueSlots) {
    const int chunk = min(kMaxEpilogueSlots, p.num_slots - slot0);
    const int total = chunk * kSlotsPerToken;
    for (int i = tid; i < total; i += kNumThreads)
      l_part[i] = base[static_cast<int64_t>(slot0) * kSlotsPerToken + i];
    __syncthreads();
    if (tid < kNumSlotsUsed) {
      float sum = 0.0f;
      for (int slot = 0; slot < chunk; ++slot)
        sum += l_part[slot * kSlotsPerToken + tid];
      sums[tid] += sum;
    }
    __syncthreads();
  }
  if (tid < kNumSlotsUsed) {
    if (tid < kNumHCOutputs) l_mix[tid] = sums[tid];
    else l_sq[tid - kNumHCOutputs] = sums[tid];
  }
  __syncthreads();

  const float rstd =
      rsqrtf(l_sq[0] / static_cast<float>(kNumRoutes * p.hidden) + p.hc_norm_eps);
  if (tid < kNumHCOutputs) {
    // 官方的角色划分：lane 0-3 是 pre，4-7 是 post，8-23 是 4x4 的 comb logits。
    const int o = tid;
    const int cls = o < kNumRoutes ? 0 : (o < 2 * kNumRoutes ? 1 : 2);
    const float logit = l_mix[o] * rstd * p.mix_scales[cls] + p.mix_bases[o];
    if (o < kNumRoutes) {
      const float pre = sigmoidf_exact(logit) + p.hc_pre_eps;
      l_pre[o] = pre;
      if (p.new_prev_mix != nullptr) p.new_prev_mix[t * kNumRoutes + o] = pre;
    } else if (o < 2 * kNumRoutes) {
      p.new_post_mix[t * kNumRoutes + (o - kNumRoutes)] =
          sigmoidf_exact(logit) * p.hc_post_scale;
    } else {
      l_comb[o - 2 * kNumRoutes] = logit;
    }
  }
  __syncthreads();
  if (tid < kNumRoutes * kNumRoutes) {
    // 每个 lane 处理4×4矩阵的一个元素；行内 xor=1/2，列内 xor=4/8。
    sinkhorn_lanes(l_comb, p.num_sinkhorn_iters, p.sinkhorn_eps, tid);
    p.new_comb_mix[t * (kNumRoutes * kNumRoutes) + tid] = l_comb[tid];
  }
  __syncthreads();

  // ---- 先 collapse，再做 RMSNorm ----------------------------------------------
  // 每个线程处理8个连续BF16元素，使用两组8字节向量读写。
  constexpr int kVec = 8;
  float norm_scale;
  if (p.is_shifted) {
    // main pass 已经产出了按 BF16 舍入的 collapse 及其平方和
    norm_scale = rsqrtf(l_sq[1] / static_cast<float>(p.hidden) + p.rmsnorm_eps);
  } else {
    // normal 消费的是**本次**调用生成的 pre 门控，所以这里从按 BF16 舍入的 Z
    // 现场构建 collapse；它经过 shifted 路径用的同一个 BF16 缓冲暂存，
    // 从而保持舍入边界完全一致。
    float local = 0.0f;
    const int64_t z_row = static_cast<int64_t>(t) * kNumRoutes * p.hidden;
    const int64_t y_row = static_cast<int64_t>(t) * p.hidden;
    for (int h0 = tid * kVec; h0 < p.hidden; h0 += kNumThreads * kVec) {
      Bf16x4Bits z0[kNumRoutes];
      Bf16x4Bits z1[kNumRoutes];
#pragma unroll
      for (int j = 0; j < kNumRoutes; ++j) {
        const bf16* src = p.new_residual + z_row + static_cast<int64_t>(j) * p.hidden + h0;
        z0[j].v = *reinterpret_cast<const bf16x4*>(src);
        z1[j].v = *reinterpret_cast<const bf16x4*>(src + 4);
      }
      uint16_t out[kVec];
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        const int half = e / 4;
        const int idx = e % 4;
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < kNumRoutes; ++j)
          v += bf16_bits_to_f32(half == 0 ? z0[j].s[idx] : z1[j].s[idx]) * l_pre[j];
        const uint16_t bits = f32_to_bf16_rne_bits(v);
        out[e] = bits;
        const float ur = bf16_bits_to_f32(bits);
        local += ur * ur;
      }
      bf16* dst = p.u_io + y_row + h0;
      Bf16x4Bits lo4;
      Bf16x4Bits hi4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        lo4.s[e] = out[e];
        hi4.s[e] = out[e + 4];
      }
      *reinterpret_cast<bf16x4*>(dst) = lo4.v;
      *reinterpret_cast<bf16x4*>(dst + 4) = hi4.v;
    }
    l_red[tid] = local;
    __syncthreads();
    for (int s = kNumThreads / 2; s > 0; s >>= 1) {
      if (tid < s) l_red[tid] += l_red[tid + s];
      __syncthreads();
    }
    norm_scale = rsqrtf(l_red[0] / static_cast<float>(p.hidden) + p.rmsnorm_eps);
    __syncthreads();
  }

  const bool want_fp8 = (p.flags & 1) != 0;
  {
    const int64_t y_row = static_cast<int64_t>(t) * p.hidden;
    for (int h0 = tid * kVec; h0 < p.hidden; h0 += kNumThreads * kVec) {
      bf16* row = p.u_io + y_row + h0;
      Bf16x4Bits u0;
      Bf16x4Bits u1;
      Bf16x4Bits w0;
      Bf16x4Bits w1;
      u0.v = *reinterpret_cast<const bf16x4*>(row);
      u1.v = *reinterpret_cast<const bf16x4*>(row + 4);
      w0.v = *reinterpret_cast<const bf16x4*>(p.rmsnorm_weight + h0);
      w1.v = *reinterpret_cast<const bf16x4*>(p.rmsnorm_weight + h0 + 4);
      uint16_t out[kVec];
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        const int half = e / 4;
        const int idx = e % 4;
        const float u = bf16_bits_to_f32(half == 0 ? u0.s[idx] : u1.s[idx]);
        const float w = bf16_bits_to_f32(half == 0 ? w0.s[idx] : w1.s[idx]);
        out[e] = f32_to_bf16_rne_bits((u * norm_scale) * (w * p.rmsnorm_scale));
      }
      // Y 总是要写：它要么是 y_bf16 输出，要么是 FP8 量化回读的暂存区，
      // 并且是原地替换掉那份暂存的 collapse。
      Bf16x4Bits lo4;
      Bf16x4Bits hi4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        lo4.s[e] = out[e];
        hi4.s[e] = out[e + 4];
      }
      *reinterpret_cast<bf16x4*>(row) = lo4.v;
      *reinterpret_cast<bf16x4*>(row + 4) = hi4.v;
    }
  }

  // ---- E4M3FN + UE8M0 --------------------------------------------------------
  // 每 32 个连续 lane 处理一个 32 元素组；四个组按固定顺序经 shared memory
  // 拼成一个 int32 的 SF 字。
  if (want_fp8) {
    __syncthreads();
    const int group = tid / kSfGroup;              // 0..7
    const int lane_in = tid % kSfGroup;            // 0..31
    const int num_passes = p.hidden / kNumThreads;  // host 保证可整除
    for (int pass = 0; pass < num_passes; ++pass) {
      const int base = pass * kNumThreads;
      const int h = base + tid;
      const float y = bf16_to_f32(p.u_io[static_cast<int64_t>(t) * p.hidden + h]);
      float amax = fabsf(y);
#pragma unroll
      for (int mask = 1; mask < kSfGroup; mask <<= 1)
        amax = fmaxf(amax, __shfl_xor(amax, mask));
      amax = fmaxf(amax, 1e-4f);
      const float sf = amax / 448.0f;
      const uint8_t exp = ceil_to_ue8m0_exponent(sf);
      const float scale = __uint_as_float(static_cast<uint32_t>(exp) << 23);
      p.y_fp8[static_cast<int64_t>(t) * p.hidden + h] =
          f32_to_e4m3fn(y * (1.0f / scale));
      if (lane_in == 0) l_sf_exp[group] = static_cast<float>(exp);
      __syncthreads();
      if (tid < 2) {
        const int g0 = tid * 4;
        const uint32_t packed =
            static_cast<uint32_t>(static_cast<uint8_t>(l_sf_exp[g0])) |
            (static_cast<uint32_t>(static_cast<uint8_t>(l_sf_exp[g0 + 1])) << 8) |
            (static_cast<uint32_t>(static_cast<uint8_t>(l_sf_exp[g0 + 2])) << 16) |
            (static_cast<uint32_t>(static_cast<uint8_t>(l_sf_exp[g0 + 3])) << 24);
        const int sf_col = (base / kHiddenPerSF) + tid;
        if (sf_col < p.hidden / kHiddenPerSF) store_sf_word(p, t, sf_col, packed);
      }
      __syncthreads();
    }
  }
  (void)l_flag;
}

// ---------------------------------------------------------------------------
// main kernel：post-mix + 投影（`arrive` != null 时再跑融合 epilogue）
// ---------------------------------------------------------------------------
template <bool kFused>
__global__ void __launch_bounds__(kNumThreads)
mega_mhc_main_kernel(const MainParams p, const ReduceParams rp, int* arrive) {
  // A tile、投影累加暂存和平方和分别使用独立的共享缓冲。
  __shared__ bf16 z_tile[kBlockM][kChunkK];              // 8 KiB
  // 投影分两批暂存，每批两个 wave；平方和先在 token 对应的 lane 组内归约。
  __shared__ float acc_smem[2][2][256];                   // 4 KiB
  __shared__ float sq_tok[2][kBlockM];                    // 256 B
  // 仅 fused 实例化需要 epilogue 暂存，split 实例化不为这些缓冲分配 LDS。
  __shared__ float l_mix[kFused ? kNumHCOutputs : 1];
  __shared__ float l_pre[kFused ? kNumRoutes : 1];
  __shared__ float l_sq[kFused ? 2 : 1];
  __shared__ float l_red[kFused ? kNumThreads : 1];
  __shared__ float l_sf_exp[kFused ? 8 : 1];
  __shared__ float l_part[kFused ? kMaxPartialBlock : 1];
  __shared__ float l_comb[kFused ? kNumRoutes * kNumRoutes : 1];
  __shared__ int l_flag;

  const int m_block = blockIdx.x;
  const int split = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int row = lane & 15;
  const int col = lane >> 4;

  const int m_base = m_block * kBlockM;
  const int tokens = min(kBlockM, p.num_tokens - m_base);
  const int h_per_split = p.hidden / p.num_splits;
  const int h_begin = split * h_per_split;
  const int chunks = h_per_split / kChunkH;

  // post-mix 的所有权：每个线程负责一个 token 和 4 个连续 hidden 列
  const int own_token = tid / kThreadsPerToken;
  const int own_col = (tid % kThreadsPerToken) * 4;
  const bool own_active = own_token < tokens;

  float post[kNumRoutes];
  float comb[kNumRoutes][kNumRoutes];
  float pre[kNumRoutes];
#pragma unroll
  for (int j = 0; j < kNumRoutes; ++j) {
    post[j] = 0.0f;
    pre[j] = 0.0f;
#pragma unroll
    for (int i = 0; i < kNumRoutes; ++i) comb[i][j] = 0.0f;
  }
  if (own_active) {
    const int64_t token = m_base + own_token;
#pragma unroll
    for (int j = 0; j < kNumRoutes; ++j) {
      post[j] = p.post_mix[token * kNumRoutes + j];
      pre[j] = p.prev_mix ? p.prev_mix[token * kNumRoutes + j] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kNumRoutes; ++i)
#pragma unroll
      for (int j = 0; j < kNumRoutes; ++j)
        comb[i][j] = p.comb_mix[token * (kNumRoutes * kNumRoutes) + i * kNumRoutes + j];
  }

  float sqr_acc = 0.0f;
  float squ_acc = 0.0f;
  // 每个 (M 组, N tile) 一个累加器：同一份 B fragment 喂两个 M 组。
  fp32x4 acc[kMGroups][2];
#pragma unroll
  for (int m = 0; m < kMGroups; ++m) {
    acc[m][0] = fp32x4{0.0f, 0.0f, 0.0f, 0.0f};
    acc[m][1] = fp32x4{0.0f, 0.0f, 0.0f, 0.0f};
  }

  for (int c = 0; c < chunks; ++c) {
    // ---- 本 chunk 的 post-mix -----------------------------------------------
    // `h_local` 是该 chunk 的全局列偏移；shared memory 里的 A tile 只装这一个
    // chunk，所以它的下标是 chunk 局部的（`own_col`）。
    const int h_local = c * kChunkH + own_col;
    if (own_active) {
      const int64_t token = m_base + own_token;
      const int64_t h = h_begin + h_local;
      Bf16x4Bits res[kNumRoutes];
      uint16_t xbits[4];
      {
        Bf16x4Bits tmp;
#pragma unroll
        for (int r = 0; r < kNumRoutes; ++r) {
          tmp.v = *reinterpret_cast<const bf16x4*>(
              p.residual + (token * kNumRoutes + r) * p.hidden + h);
          res[r] = tmp;
        }
        tmp.v = *reinterpret_cast<const bf16x4*>(p.x + token * p.hidden + h);
#pragma unroll
        for (int i = 0; i < 4; ++i) xbits[i] = tmp.s[i];
      }

      float zf[kNumRoutes][4];
#pragma unroll
      for (int j = 0; j < kNumRoutes; ++j) {
        const float xf[4] = {bf16_bits_to_f32(xbits[0]), bf16_bits_to_f32(xbits[1]),
                             bf16_bits_to_f32(xbits[2]), bf16_bits_to_f32(xbits[3])};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float v = post[j] * xf[i];
#pragma unroll
          for (int r = 0; r < kNumRoutes; ++r)
            v += comb[r][j] * bf16_bits_to_f32(res[r].s[i]);
          zf[j][i] = v;
        }
      }

      Bf16x4Bits zbuf[kNumRoutes];
#pragma unroll
      for (int j = 0; j < kNumRoutes; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint16_t bits = f32_to_bf16_rne_bits(zf[j][i]);
          zbuf[j].s[i] = bits;
          const float zr = bf16_bits_to_f32(bits);
          sqr_acc += zr * zr;
          z_tile[own_token][j * kChunkH + own_col + i] = bf16_from_bits(bits);
        }
        *reinterpret_cast<bf16x4*>(p.new_residual + (token * kNumRoutes + j) * p.hidden + h) =
            zbuf[j].v;
      }

      if (p.prev_mix != nullptr) {
        Bf16x4Bits u;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float v = 0.0f;
#pragma unroll
          for (int j = 0; j < kNumRoutes; ++j)
            v += pre[j] * bf16_bits_to_f32(zbuf[j].s[i]);
          const uint16_t bits = f32_to_bf16_rne_bits(v);
          u.s[i] = bits;
          const float ur = bf16_bits_to_f32(bits);
          squ_acc += ur * ur;
        }
        *reinterpret_cast<bf16x4*>(p.u_out + token * p.hidden + h) = u.v;
      }
    } else {
      // 保持 A tile 里补齐的行是零，以免 MMAC 出非有限值
#pragma unroll
      for (int j = 0; j < kNumRoutes; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          z_tile[own_token][j * kChunkH + own_col + i] = bf16_from_bits(0);
    }
    __syncthreads();

    // ---- 本 chunk 的 16 个 k 块投影，按 wave 分工 -----------------------------
    for (int kt = wave; kt < kNumKTiles; kt += kNumWaves) {
      // kt 枚举 (route, route 内的 k tile)；后者取决于 kChunkH，
      // 所以不能硬编码成每条 route 4 个 tile。
      constexpr int kTilesPerRoute = kChunkH / kMmacK;
      const int route = kt / kTilesPerRoute;
      const int in_route = kt % kTilesPerRoute;
      const int64_t gk = static_cast<int64_t>(route) * p.hidden + h_begin +
                         c * kChunkH + in_route * kMmacK;
      // 两个 B tile 各读一次，被所有 M 组复用（这正是提高 kBlockM 的全部意义）：
      // 2 次读取喂 2 * kMGroups 条 MMAC。
      const BF16Pair b0 =
          load_b_fragment_packed(p.fn_packed, row, col, gk, p.fn_half, true);
      const BF16Pair b1 = load_b_fragment_packed(p.fn_packed, 16 + row, col, gk, p.fn_half,
                                                 (16 + row) < kNumHCOutputs);
#pragma unroll
      for (int m = 0; m < kMGroups; ++m) {
        const bf16x4 a = load_a_fragment(&z_tile[m * 16][0], row, col, kt);
        acc[m][0] = project_k16(a, b0, acc[m][0]);
        acc[m][1] = project_k16(a, b1, acc[m][1]);
      }
    }
    __syncthreads();
  }

  // ---- 发布 partial ----------------------------------------------------------
  // 每个 token 由同一 wave 中连续的8个 lane 归约平方和，xor=1/2/4不跨组。
  {
    float sr = sqr_acc;
    float su = squ_acc;
#pragma unroll
    for (int off = 1; off < kThreadsPerToken; off <<= 1) {
      sr += __shfl_xor(sr, off, 64);
      su += __shfl_xor(su, off, 64);
    }
    if ((tid % kThreadsPerToken) == 0) {
      sq_tok[0][own_token] = sr;
      sq_tok[1][own_token] = su;
    }
  }
  // chunk循环结束后按M组发布累加器，wave求和顺序固定为((w0+w1)+w2)+w3。
#pragma unroll
  for (int m = 0; m < kMGroups; ++m) {
    F32x4Parts a0;
    F32x4Parts a1;
    a0.v = acc[m][0];
    a1.v = acc[m][1];
    float part[2][4];
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      if ((wave >> 1) == pass) {
        const int slot = wave & 1;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc_smem[slot][0][lane * 4 + e] = a0.f[e];
          acc_smem[slot][1][lane * 4 + e] = a1.f[e];
        }
      }
      __syncthreads();
      if (wave == 0) {
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float v = acc_smem[0][tile][lane * 4 + e] + acc_smem[1][tile][lane * 4 + e];
            // pass 0 计算 w0+w1；pass 1 依次累加 w2 和 w3。
            part[tile][e] = pass == 0 ? v : (part[tile][e] + acc_smem[0][tile][lane * 4 + e]) +
                                              acc_smem[1][tile][lane * 4 + e];
          }
        }
      }
      __syncthreads();
    }
    if (wave == 0) {
      // 投影 partial：固定的 wave 顺序 0..3，HCU C 分片 n = e*4 + col
      const int token = m_base + m * 16 + row;
#pragma unroll
      for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float v = part[tile][e];
          const int n = e * 4 + col;
          if (token < p.num_tokens) {
            const int o = tile * 16 + n;
            if (o < kNumHCOutputs) {
              p.partial[(static_cast<int64_t>(token) * p.num_slots + split) * kSlotsPerToken + o] = v;
            }
          }
        }
      }
    }
  }
  if (wave == 0) {
    // 每 token 的平方和：每个 token 只写一个已预先归约的值（见上面的 shuffle）。
    if (tid < kBlockM) {
      const int token = m_base + tid;
      if (token < p.num_tokens) {
        float* slot = p.partial +
                      (static_cast<int64_t>(token) * p.num_slots + split) * kSlotsPerToken;
        slot[kSqRSlot] = sq_tok[0][tid];
        slot[kSqUSlot] = sq_tok[1][tid];
      }
    }
  }

  // ---- 融合 epilogue：该 m-block 最后到达的那个 split 负责执行 ----------------
  // 整块都放在 `if constexpr` 之下，这样 split 实例化根本不会引用 epilogue 的
  // 暂存（那些数组在它里面只有 1 个元素）。
  if constexpr (kFused) {
    if (arrive == nullptr) return;
    // 各线程先发布 residual、collapse 与 partial 写入，整个 CTA 会合后再更新到达计数。
    // 消费者据此读取所有 split 的结果，不能在其它 wave 发布完成前递增计数。
    __threadfence();                     // 以 agent 作用域释放本线程的写入
    __syncthreads();                     // ... 所有 wave 都已 fence
    if (tid == 0) {
      l_flag = (atomicAdd(arrive + m_block, 1) == p.num_splits - 1) ? 1 : 0;
    }
    __syncthreads();
    if (l_flag == 0) return;
    __threadfence();                     // acquire：所有生产者的 partial 都可见了
    for (int tt = 0; tt < tokens; ++tt) {
      epilogue_one_token(rp, m_base + tt, tid, l_mix, l_pre, l_sq, l_red, l_sf_exp, l_part,
                         l_comb, &l_flag);
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// 默认 split 路径的 epilogue kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kNumThreads)
mega_mhc_reduce_kernel(const ReduceParams p) {
  __shared__ float l_mix[kNumHCOutputs];
  __shared__ float l_pre[kNumRoutes];
  __shared__ float l_sq[2];
  __shared__ float l_red[kNumThreads];
  __shared__ float l_sf_exp[8];
  __shared__ float l_part[kMaxPartialBlock];
  __shared__ float l_comb[kNumRoutes * kNumRoutes];
  __shared__ int l_flag;
  epilogue_one_token(p, blockIdx.x, threadIdx.x, l_mix, l_pre, l_sq, l_red, l_sf_exp,
                     l_part, l_comb, &l_flag);
}

// ---- 架构共用的 launch ----

inline void mega_mhc_launch_common(const MegaMhcLaunchArgs& a) {
  hipStream_t stream = reinterpret_cast<hipStream_t>(a.stream);

  MainParams main{};
  main.x = reinterpret_cast<const bf16*>(a.x);
  main.residual = reinterpret_cast<const bf16*>(a.residual);
  main.post_mix = reinterpret_cast<const float*>(a.post_mix);
  main.comb_mix = reinterpret_cast<const float*>(a.comb_mix);
  main.prev_mix = reinterpret_cast<const float*>(a.prev_mix);
  main.fn_packed = reinterpret_cast<const bf16*>(a.fn_packed);
  main.fn_half = a.fn_half;
  main.new_residual = reinterpret_cast<bf16*>(a.new_residual);
  main.u_out = reinterpret_cast<bf16*>(a.u_io);
  main.partial = reinterpret_cast<float*>(a.partial);
  main.num_tokens = a.num_tokens;
  main.hidden = a.hidden;
  main.num_splits = a.num_splits;
  main.num_slots = a.num_slots;

  ReduceParams reduce{};
  reduce.partial = reinterpret_cast<const float*>(a.partial);
  reduce.mix_scales = reinterpret_cast<const float*>(a.mix_scales);
  reduce.mix_bases = reinterpret_cast<const float*>(a.mix_bases);
  reduce.rmsnorm_weight = reinterpret_cast<const bf16*>(a.rmsnorm_weight);
  reduce.new_residual = reinterpret_cast<const bf16*>(a.new_residual);
  reduce.u_io = reinterpret_cast<bf16*>(a.u_io);
  reduce.new_prev_mix = reinterpret_cast<float*>(a.new_prev_mix);
  reduce.new_post_mix = reinterpret_cast<float*>(a.new_post_mix);
  reduce.new_comb_mix = reinterpret_cast<float*>(a.new_comb_mix);
  reduce.y_fp8 = reinterpret_cast<uint8_t*>(a.y_fp8);
  reduce.y_gemm_sf = reinterpret_cast<int*>(a.y_gemm_sf);
  reduce.y_routed_sf = reinterpret_cast<int*>(a.y_routed_sf);
  reduce.y_shared_sf = reinterpret_cast<int*>(a.y_shared_sf);
  reduce.num_tokens = a.num_tokens;
  reduce.hidden = a.hidden;
  reduce.num_slots = a.num_slots;
  reduce.gemm_sf_stride1 = a.gemm_sf_stride1;
  reduce.shared_sf_block_m = a.shared_sf_block_m;
  reduce.shared_sf_stride1 = a.shared_sf_stride1;
  reduce.hc_norm_eps = a.hc_norm_eps;
  reduce.hc_pre_eps = a.hc_pre_eps;
  reduce.hc_post_scale = a.hc_post_scale;
  reduce.sinkhorn_eps = a.sinkhorn_eps;
  reduce.num_sinkhorn_iters = a.num_sinkhorn_iters;
  reduce.rmsnorm_eps = a.rmsnorm_eps;
  reduce.rmsnorm_scale = a.rmsnorm_scale;
  reduce.is_shifted = a.is_shifted;
  reduce.flags = a.flags;

  // 把 fn 打包成 BF16 的 hi/lo。已经用 `mega_mhc_prepare_fn` 备好权重的调用方
  // 会把 pack_fn 置 0，于是这一步整个跳过。
  if (a.pack_fn) {
    constexpr int kPackThreads = 256;
    const int64_t blocks = (a.fn_half / 4 + kPackThreads - 1) / kPackThreads;
    pack_fn_kernel<<<dim3(static_cast<unsigned>(blocks), 1, 1), kPackThreads, 0, stream>>>(
        reinterpret_cast<const float*>(a.fn), reinterpret_cast<bf16*>(a.fn_packed),
        a.fn_half);
  }

  const int num_m_blocks = (a.num_tokens + kBlockM - 1) / kBlockM;
  dim3 grid_main(num_m_blocks, a.num_splits, 1);
  if (a.fused) {
    int* arrive = reinterpret_cast<int*>(a.arrive);
    mega_mhc_main_kernel<true><<<grid_main, kNumThreads, 0, stream>>>(main, reduce, arrive);
  } else {
    mega_mhc_main_kernel<false><<<grid_main, kNumThreads, 0, stream>>>(main, reduce, nullptr);
    mega_mhc_reduce_kernel<<<dim3(a.num_tokens, 1, 1), kNumThreads, 0, stream>>>(reduce);
  }
}

// 只打包的入口，供 `mega_mhc_prepare_fn` 使用（不发射 main/reduce）。
inline void pack_fn_only(const MegaMhcLaunchArgs& a) {
  constexpr int kPackThreads = 256;
  const hipStream_t stream = reinterpret_cast<hipStream_t>(a.stream);
  const int64_t blocks = (a.fn_half / 4 + kPackThreads - 1) / kPackThreads;
  pack_fn_kernel<<<dim3(static_cast<unsigned>(blocks), 1, 1), kPackThreads, 0, stream>>>(
      reinterpret_cast<const float*>(a.fn), reinterpret_cast<bf16*>(a.fn_packed), a.fn_half);
}

}  // namespace deepgemm::mega_mhc
