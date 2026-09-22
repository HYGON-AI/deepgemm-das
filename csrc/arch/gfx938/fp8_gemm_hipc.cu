// FP8_e4m3 -> BF16 GEMM on gfx938, warp-specialized producer-consumer.
//
// Block tile:  256M x 256N x 128K   (matches hipblasLt v0's MT256x256x128)
// WG shape:    12 waves x 64 = 768 threads
//    Producers (wave 0..3): buffer_load_dwordx4 Direct-to-LDS with XOR K swizzle
//    Consumers (wave 4..11): 8 waves, each owns 256M x 32N output stripe
//
// LDS budget on gfx938 = 64 KB / WG.
//     A stage: 4 slabs x 64M x 128K =  32 KB
//     2 stages                      =  64 KB
// B is streamed into VGPR ping-pong (32 VGPR / lane) — never touches LDS.
//
// Sync: WG-wide s_barrier. Producer and consumer waves execute the SAME
// number of s_barriers in the SAME order; between barriers producer runs
// buffer_load Direct-to-LDS while consumer runs DS_READ + buffer_load_B + MMAC.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <type_traits>

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "fp8_gemm_hipc.h"
#include "gemm.h"  // deepgemm::fp8_gemm_hipc declaration

using bhalf_t   = __hip_bfloat16;
using half_t    = __half;
using fp8_e4m3  = __hip_fp8_e4m3;

using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using vec4_int8 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;
using vec8_int8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using vec4_int  = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using vec2_bf16 = __attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short;

union union_vec4_fp32 {
    vec4_fp32 f32;
    vec2_fp32 f32x2[2];
};
union union_vec16_fp8 {
    int8_t   i8[16];
    vec8_int8 i8x8[2];
    vec4_int8 i8x4[4];
    vec4_fp32 f32x4;
    vec4_int  i32x4;
    int32_t   i32[4];
};
template<typename Element>
union union_vec2_f16x2 {
    vec2_fp32  f32;
    vec2_bf16  f16x2[2];
    Element    f16[4];
};

// ============================================================================
// Traits
// ============================================================================
struct GemmTraitsPC {
    using Element      = fp8_e4m3;
    using Element_O    = bhalf_t;
    using ElementAccum = float;

    static constexpr int kBlockM = 256;
    static constexpr int kBlockN = 256;
    static constexpr int kStageK = 128;

    static constexpr int kWaveN  = 32;   // unique N strip per consumer
    static constexpr int kPassK  = 64;   // one ds_read_a / mmac pass along K

    static constexpr int kNConsumerWaves = 8;
    static constexpr int kNProducerWaves = 4;
    static constexpr int kNWaves         = kNConsumerWaves + kNProducerWaves;
    static constexpr int kNThreads       = kNWaves * 64;

    static constexpr int kProdSlabM = kBlockM / kNProducerWaves;  // 64

    // A LDS: 4 slabs x (64M x 128K) x 2 stages = 64 KB.
    // XOR-swizzled dense layout (avoids 16-way bank conflict on ds_read_b128):
    //   addr = row*128 + (k16 ^ (row&7))*16
    // Plain row-major used 128B row stride (= 32 dwords = 0 mod 32 banks) so
    // all 16 rows of a micro-tile hit the same bank (max_bank=16). XOR drops
    // that to the theoretical minimum max_bank=8 for a wave-wide b128.
    static constexpr int STAGES           = 2;
    static constexpr size_t A_slab_bytes  = kProdSlabM * kStageK;            // 8192
    static constexpr size_t A_stage_bytes = kNProducerWaves * A_slab_bytes;  // 32768
    static constexpr int    kSmemSize     = STAGES * A_stage_bytes;         // 65536
};

// ============================================================================
// Intrinsic wrappers
// ============================================================================
__device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

namespace flash {

__forceinline__ __device__ void wait_all_warp_arrived() {
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_barrier\n");
    __builtin_hcu_sched_barrier(0);
}

template<bool Sync>
__forceinline__ __device__ void wait_all_buffer_data_arrived(int wait_count = 0) {
    __builtin_hcu_sched_barrier(0);
    if constexpr (Sync) {
        asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier\n" :: "n"(wait_count));
    } else {
        asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"(wait_count));
    }
    __builtin_hcu_sched_barrier(0);
}

template<bool Sync>
__forceinline__ __device__ void wait_lds_data_arrived(int wait_count = 0) {
    __builtin_hcu_sched_barrier(0);
    if constexpr (Sync) {
        asm volatile("s_waitcnt lgkmcnt(%0)\n\ts_barrier\n" :: "n"(wait_count));
    } else {
        asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(wait_count));
    }
    __builtin_hcu_sched_barrier(0);
}

// s_setprio N (N = 0..3): hint to the CU wave scheduler that this wave should
// be preferred (higher PRIO) or deferred (lower PRIO) when picking which wave
// to issue an instruction from next cycle. Default wave priority is 1.
// Typical use: PRIO=3 around an mmac chain to shield it from other waves'
// memory-op issue slots; PRIO=0 in producer waves so they never steal issue
// cycles from mmac-heavy consumers. Bracketed with sched_barrier(0) so the
// compiler can't reorder instructions across the priority change.
template<int PRIO>
__forceinline__ __device__ void set_prio() {
    static_assert(PRIO >= 0 && PRIO <= 3, "s_setprio operand must be 0..3");
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_setprio %0" :: "n"(PRIO));
    __builtin_hcu_sched_barrier(0);
}

} // namespace flash

__device__ __forceinline__ vec4_fp32 mmac_4interleave_b8(
    const vec8_int8& v1, const vec8_int8& v2, const vec4_fp32& v3
) {
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(v1, v2, v3, 1, 0);
}

__host__ __device__ __forceinline__ bhalf_t float2bfloat16_nonan(float f) {
    bhalf_t ret;
#if 0 /* disabled by -frewrite-includes */
#if defined(__gfx938__)
#endif
#endif /* disabled by -frewrite-includes */
#if 1 /* evaluated by -frewrite-includes */
    *(unsigned short*)&ret = __builtin_hcu_cvt_bf16_f32(f, false, false);
#else
    union { float fp32; unsigned int u32; } u = {f};
    u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
    *(unsigned short*)&ret = (u.u32 >> 16);
#endif
    return ret;
}

__host__ __device__ __forceinline__ vec2_bf16 DownCastPair_f32_bf16(const vec2_fp32& source) {
#if 0 /* disabled by -frewrite-includes */
#if defined(__gfx938__) || defined(__gfx92a__)
#endif
#endif /* disabled by -frewrite-includes */
#if 1 /* evaluated by -frewrite-includes */
    auto result = __builtin_hcu_cvt_pk_bf16_f32(source[0], source[1], false);
    return *(vec2_bf16*)(&result);
#else
    vec2_bf16 result;
    result[0] = *(unsigned short*)&float2bfloat16_nonan(source[0]);
    result[1] = *(unsigned short*)&float2bfloat16_nonan(source[1]);
    return result;
#endif
}

// fp32 -> fp16 packed pair. Same interface / layout as DownCastPair_f32_bf16;
// the returned 2×16-bit value is just re-interpreted as fp16 downstream.
// Caller must ensure the accumulator magnitude fits in fp16 (max ~65504) —
// bf16 has fp32-wide dynamic range but fp16 does not.
//
// __builtin_hcu_cvt_pk_f16_f32(srcA, srcB, clamp, omod):
//   clamp: bool imm — false = pass through, true = saturate to [0.0, 1.0]
//   omod : int  imm — output modifier: 0=×1, 1=×2, 2=×4, 3=×0.5. We want ×1.
__host__ __device__ __forceinline__ vec2_bf16 DownCastPair_f32_fp16(const vec2_fp32& source) {
    auto result = __builtin_hcu_cvt_pk_f16_f32(source[0], source[1],
                                               /*clamp=*/false, /*omod=*/0);
    return *(vec2_bf16*)(&result);
}

// ============================================================================
// Buffer-load / ds_read helpers (adapted from test_gemm_64x64x64)
// ============================================================================
typedef int32_t dword2_t __attribute__((ext_vector_type(2)));
typedef int32_t dword4_t __attribute__((ext_vector_type(4)));
typedef int32_t buffer_rsrc_t __attribute__((ext_vector_type(4)));

template <bool cache_swizzle_enable = true>
__device__ __forceinline__ buffer_rsrc_t make_buffer_rsrc(const void* ptr, int stride, uint32_t size_bytes) {
    buffer_rsrc_t r;
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    r[0] = (int32_t)(addr & 0xffffffff);
    r[1] = (int32_t)(addr >> 32);
    r[2] = (int32_t)size_bytes;
    r[3] = 0x00020000;

    if (cache_swizzle_enable && (((stride ^ (stride - 1)) == 0) || ((stride & 128) == 0))) {
        uint32_t val = 1u << 30; // CACHE_SWIZZLE_ENABLE
        val |= (stride & 0x3fff) << 16;
        r[1] |= val;
    }

    return r;
}

// 8× dwordx4 Direct-to-LDS, ONE asm block, 8 distinct offen VGPRs (hipblasLt-style).
// Each load must already be coalesced across the wave — do NOT use one shared
// voffset with imm K-steps when lanes own different rows (stride=lda → 64
// scattered 16B txs and multi-µs gaps on the wave timeline).
// m0 += 1024 between loads (one 64×16B LDS panel; hipblasLt pads +0x1000).
//
// s_col_base is a uniform SGPR added to every effective address (like the B
// consumer path). Callers put the stage-varying K col offset (k_stage*kStageK)
// here so the 8 per-lane VGPR offsets can be precomputed once outside the
// stage loop. Mask-for-OOB must use 0x80000000 (bit 31) rather than
// 0xFFFFFFFF: bit 31 + s_col_base(<~few KB) does not wrap in 32-bit and stays
// > num_records → hardware OOB → load writes 0 to LDS.
__device__ __forceinline__ void buffer_load_dwordx4_x8_to_lds(
    buffer_rsrc_t rsrc,
    uint32_t voff0, uint32_t voff1, uint32_t voff2, uint32_t voff3,
    uint32_t voff4, uint32_t voff5, uint32_t voff6, uint32_t voff7,
    uint32_t lds_m0, uint32_t s_col_base) {
    asm volatile(
        "s_mov_b32 m0, %8\n\t"
        "buffer_load_dwordx4 %0, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %1, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %2, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %3, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %4, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %5, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %6, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %7, %9, %10 offen lds\n"
        :
        : "v"(voff0), "v"(voff1), "v"(voff2), "v"(voff3),
          "v"(voff4), "v"(voff5), "v"(voff6), "v"(voff7),
          "s"(lds_m0), "s"(rsrc), "s"(s_col_base)
        : "memory", "m0");
}

// SGPR-soffset variant: 8 loads share ONE per-lane VGPR voffset (the XOR-
// swizzled base) and use 8 uniform SGPR soffsets (one per row-group of 8).
// Effective addr per lane = rsrc.base + voff + s_soff[i]. Caller precomputes
// s_soff[i] = k_stage*kStageK + i*(8*lda), which is stage-varying but uniform
// across lanes — perfect for SGPR. Kills 16 VALU (8 v_add for v_off_pre + 8
// v_or for v_off_last mask) that the previous voff-per-load variant needed;
// the 8 s_add for s_soff[] live in the stage loop as SALU (SALU does not
// contend with consumer MMAC on the same SIMD).
//
// K-tail masking is applied by producer_body: fully out-of-bounds 16-byte
// chunks use bit 31 in voff to trigger hardware OOB zero-fill. A chunk that
// straddles K is loaded normally, then its invalid suffix is cleared in LDS.
__device__ __forceinline__ void buffer_load_dwordx4_x8_to_lds_soff(
    buffer_rsrc_t rsrc,
    uint32_t voff,
    uint32_t lds_m0,
    uint32_t s_soff0, uint32_t s_soff1, uint32_t s_soff2, uint32_t s_soff3,
    uint32_t s_soff4, uint32_t s_soff5, uint32_t s_soff6, uint32_t s_soff7) {
    asm volatile(
        "s_mov_b32 m0, %1\n\t"
        "buffer_load_dwordx4 %0, %2, %3 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %4 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %5 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %6 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %7 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %8 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %9 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %10 offen lds\n"
        :
        : "v"(voff), "s"(lds_m0), "s"(rsrc),
          "s"(s_soff0), "s"(s_soff1), "s"(s_soff2), "s"(s_soff3),
          "s"(s_soff4), "s"(s_soff5), "s"(s_soff6), "s"(s_soff7)
        : "memory", "m0");
}


// 4× buffer_load_dwordx4 into VGPRs (consumer B path). Each lane addresses via:
//   - four per-lane VGPR offsets (static across stages, incl. K-tail mask OR'd in)
//   - one uniform SGPR soffset (col_base = stage * kStageK)
// Effective address = rsrc_base + v_off + s_col_base.
// Mask trick: caller ORs 0x80000000 (bit 31) into v_off for lanes whose whole
// chunk is past K. Bit 31 + s_col_base (<~2K) does not wrap in 32-bit and stays
// > num_records → hardware OOB → load returns 0. (Cannot use 0xFFFFFFFF: it
// wraps to a small in-bounds value when added to a non-zero soffset.)
__device__ __forceinline__ void buffer_load_dwordx4_x4_soff(
    buffer_rsrc_t rsrc,
    uint32_t v_off_n0_kh0, uint32_t v_off_n0_kh1,
    uint32_t v_off_n1_kh0, uint32_t v_off_n1_kh1,
    uint32_t s_col_base,
    dword4_t& o0, dword4_t& o1, dword4_t& o2, dword4_t& o3) {
    asm volatile(
        "buffer_load_dwordx4 %0, %4, %8, %9 offen offset:0\n\t"
        "buffer_load_dwordx4 %1, %5, %8, %9 offen offset:0\n\t"
        "buffer_load_dwordx4 %2, %6, %8, %9 offen offset:0\n\t"
        "buffer_load_dwordx4 %3, %7, %8, %9 offen offset:0\n"
        : "=&v"(o0), "=&v"(o1), "=&v"(o2), "=&v"(o3)
        : "v"(v_off_n0_kh0), "v"(v_off_n0_kh1),
          "v"(v_off_n1_kh0), "v"(v_off_n1_kh1),
          "s"(rsrc), "s"(s_col_base)
        : "memory");
}

// 16× ds_read_b128 off ONE base VGPR + imm (hipblasLt v242+offset pattern).
// Stride 0x800 = 16 rows × 128B — matches XOR layout where (row&7) is invariant
// under +16 row steps, so M-tiles are pure base + m*2048. No v_add between reads.
//
// EXTRA is a compile-time offset added to every imm (used to fold the LDS
// K-stage parity offset — (stage_id & 1) * A_stage_bytes = 0 or 32768 — into
// the ds_read imm slot, keeping the per-lane base VGPR fully loop-invariant.
template<int EXTRA = 0>
__device__ __forceinline__ void ds_read_b128_x16_imm(
    uint32_t base,
    dword4_t& o0,  dword4_t& o1,  dword4_t& o2,  dword4_t& o3,
    dword4_t& o4,  dword4_t& o5,  dword4_t& o6,  dword4_t& o7,
    dword4_t& o8,  dword4_t& o9,  dword4_t& o10, dword4_t& o11,
    dword4_t& o12, dword4_t& o13, dword4_t& o14, dword4_t& o15) {
    static_assert(EXTRA >= 0 && EXTRA + 30720 <= 65535,
                  "ds_read_b128 imm offset overflows 16-bit unsigned field");
    asm volatile(
        "ds_read_b128 %0,  %16 offset:%c17\n\t"
        "ds_read_b128 %1,  %16 offset:%c18\n\t"
        "ds_read_b128 %2,  %16 offset:%c19\n\t"
        "ds_read_b128 %3,  %16 offset:%c20\n\t"
        "ds_read_b128 %4,  %16 offset:%c21\n\t"
        "ds_read_b128 %5,  %16 offset:%c22\n\t"
        "ds_read_b128 %6,  %16 offset:%c23\n\t"
        "ds_read_b128 %7,  %16 offset:%c24\n\t"
        "ds_read_b128 %8,  %16 offset:%c25\n\t"
        "ds_read_b128 %9,  %16 offset:%c26\n\t"
        "ds_read_b128 %10, %16 offset:%c27\n\t"
        "ds_read_b128 %11, %16 offset:%c28\n\t"
        "ds_read_b128 %12, %16 offset:%c29\n\t"
        "ds_read_b128 %13, %16 offset:%c30\n\t"
        "ds_read_b128 %14, %16 offset:%c31\n\t"
        "ds_read_b128 %15, %16 offset:%c32\n"
        : "=&v"(o0),  "=&v"(o1),  "=&v"(o2),  "=&v"(o3),
          "=&v"(o4),  "=&v"(o5),  "=&v"(o6),  "=&v"(o7),
          "=&v"(o8),  "=&v"(o9),  "=&v"(o10), "=&v"(o11),
          "=&v"(o12), "=&v"(o13), "=&v"(o14), "=&v"(o15)
        : "v"(base),
          "n"(EXTRA + 0),     "n"(EXTRA + 2048),  "n"(EXTRA + 4096),  "n"(EXTRA + 6144),
          "n"(EXTRA + 8192),  "n"(EXTRA + 10240), "n"(EXTRA + 12288), "n"(EXTRA + 14336),
          "n"(EXTRA + 16384), "n"(EXTRA + 18432), "n"(EXTRA + 20480), "n"(EXTRA + 22528),
          "n"(EXTRA + 24576), "n"(EXTRA + 26624), "n"(EXTRA + 28672), "n"(EXTRA + 30720)
        : "memory");
}

// 16 dwords = 64 B = 16 VGPR / lane. Consumer B buffer: one K-stage of
// 128K x 32N per wave. Layout: d4[n * 2 + k_half] holds a 16B chunk; the
// same 16B is aliased as v8[(n*2+k_half)*2 + kk] to feed two 8B mmac K-chunks.
union union_vec16_dword {
    dword4_t   d4[4];
    vec8_int8  v8[8];
    float      f[16];
};

// Read a lane's per-row fp32 scales from LDS (staged there by the producer's
// idle tail). 16 A-scales at stride 64 B (one per 16-row sub-tile) + 8 B-scales
// as two contiguous 16 B groups. One shared base VGPR per side + compile-time
// imm offsets → zero VALU in the read (it runs right after the last MMAC where
// VALU would contend on the same SIMD). Caller waits lgkmcnt(0) afterwards.
__device__ __forceinline__ void ds_read_scale_pc(
    uint32_t sa_base, uint32_t sb_base,
    float& a0,  float& a1,  float& a2,  float& a3,
    float& a4,  float& a5,  float& a6,  float& a7,
    float& a8,  float& a9,  float& a10, float& a11,
    float& a12, float& a13, float& a14, float& a15,
    dword4_t& b0, dword4_t& b1)
{
    asm volatile(
        "ds_read_b32 %0,  %18 offset:0\n\t"
        "ds_read_b32 %1,  %18 offset:64\n\t"
        "ds_read_b32 %2,  %18 offset:128\n\t"
        "ds_read_b32 %3,  %18 offset:192\n\t"
        "ds_read_b32 %4,  %18 offset:256\n\t"
        "ds_read_b32 %5,  %18 offset:320\n\t"
        "ds_read_b32 %6,  %18 offset:384\n\t"
        "ds_read_b32 %7,  %18 offset:448\n\t"
        "ds_read_b32 %8,  %18 offset:512\n\t"
        "ds_read_b32 %9,  %18 offset:576\n\t"
        "ds_read_b32 %10, %18 offset:640\n\t"
        "ds_read_b32 %11, %18 offset:704\n\t"
        "ds_read_b32 %12, %18 offset:768\n\t"
        "ds_read_b32 %13, %18 offset:832\n\t"
        "ds_read_b32 %14, %18 offset:896\n\t"
        "ds_read_b32 %15, %18 offset:960\n\t"
        "ds_read_b128 %16, %19 offset:0\n\t"
        "ds_read_b128 %17, %19 offset:64\n\t"
        : "=&v"(a0), "=&v"(a1), "=&v"(a2), "=&v"(a3),
          "=&v"(a4), "=&v"(a5), "=&v"(a6), "=&v"(a7),
          "=&v"(a8), "=&v"(a9), "=&v"(a10), "=&v"(a11),
          "=&v"(a12), "=&v"(a13), "=&v"(a14), "=&v"(a15),
          "=&v"(b0), "=&v"(b1)
        : "v"(sa_base), "v"(sb_base)
        : "memory");
}

// ============================================================================
// Reusable epilogue: LDS-staged writeback for MMAC 16x16 accumulators
// ============================================================================
// Native MMAC output puts wave-64 across 16 rows (lane & 15 = row-in-tile,
// lane >> 4 = col-quad).  A direct global store therefore issues ~16
// cache-line txns per store.  Staging through LDS and reading back with a
// "wave-64 covers 4 rows x 128 B" layout drops that to 4 txns / store.
//
// LDS scratch requirement per wave (bf16 elements):
//   (WARP_M / kEpiHalvesPerWarp) * WARP_N
// V5: 128 * 32 = 4096 bf16 = 8 KB per consumer wave.
// 8 consumer waves x 8 KB = 64 KB = full LDS budget (producers are done
// by the time we hit the epilogue, so their share of LDS is free too).
//
// XOR-swizzle: 8B unit col XOR (row & 15) — fans 16 rows across bank
// groups on both store and load sides.  The two dwords inside an 8B unit
// stay physically adjacent, so the load side reads back with ds_read_b64.
// ============================================================================

constexpr int kEpiHalvesPerWarpPC = 2;
constexpr int kEpiElemsPerLanePC   = 8;   // 16 B / lane → global_store_dwordx4

// Coalesced reload layout derived from WARP_N (bf16 elems / row):
//   WARP_N=32  →  4 lanes/row, 16 rows/wave-pass
template<int WARP_N>
struct EpiCoalescePC {
    static constexpr int kColsPerRow = WARP_N / kEpiElemsPerLanePC;
    static constexpr int kRowsPerPass = 64 / kColsPerRow;
    static constexpr int kLog2ColsPerRow =
        (kColsPerRow == 16) ? 4 :
        (kColsPerRow ==  8) ? 3 :
        (kColsPerRow ==  4) ? 2 :
        (kColsPerRow ==  2) ? 1 : -1;
    static_assert(WARP_N % kEpiElemsPerLanePC == 0, "WARP_N must be multiple of 8");
    static_assert(kLog2ColsPerRow >= 0, "WARP_N/8 must be power of 2");
    static_assert(64 % kColsPerRow == 0, "wave must tile rows evenly");
};

template<typename Element_O>
struct union_vec8_bf16_pc {
    union {
        vec2_fp32 f32x2[2];
        Element_O bf16[8];
    };
};

// XOR-swizzled LDS byte offset for one bf16 pair (== 1 dword).
// Slab layout: rows at stride kSlabN bf16 (no padding), 8B units
// XOR-permuted by (row & 15) to fan 16 rows across bank groups.
template<int kSlabN>
__device__ __forceinline__ int epi_lds_offset_bytes_pc(int row_in_slab, int col_bf16) {
    constexpr int kRowStrideBytes = kSlabN * 2;
    constexpr int kUnitsPerRow = kSlabN / 4;   // 8B units per row (power of 2)
    static_assert((kUnitsPerRow & (kUnitsPerRow - 1)) == 0,
                  "WARP_N/4 must be a power of 2");
    int col_q = col_bf16 >> 2;                 // 8B-unit index within the row
    int swizzled_q = col_q ^ (row_in_slab & (kUnitsPerRow - 1));
    return row_in_slab * kRowStrideBytes + swizzled_q * 8 + (col_bf16 & 3) * 2;
}

template<int WARP_M, int WARP_N>
__device__ __forceinline__ int epi_lds_slab_bytes_per_wave_pc() {
    constexpr int kRowsPerHalf = WARP_M / kEpiHalvesPerWarpPC;
    return kRowsPerHalf * WARP_N * int(sizeof(uint16_t));
}

// Store half of the MMAC accumulator (rows in [half*kRowsPerHalf,
// (half+1)*kRowsPerHalf) of the wave tile) into the wave's LDS slab.
template<typename Element_O, int WARP_M, int WARP_N>
__device__ __forceinline__ void epi_store_half_to_lds_pc(
    Element_O* lds_slab, int lane_id, int half,
    const union_vec2_f16x2<Element_O>* acc_bf16_flat)
{
    static_assert(sizeof(Element_O) == 2, "helper assumes 2-byte output element");
    constexpr int kSubM = WARP_M / 16;
    constexpr int kSubN = WARP_N / 16;
    constexpr int kRowsPerHalf = WARP_M / kEpiHalvesPerWarpPC;
    constexpr int kSubMPerHalf = kRowsPerHalf / 16;

    const int lane_and_15 = lane_id & 15;
    const int lane_lsr_4  = lane_id >> 4;

    #pragma unroll
    for (int mm_local = 0; mm_local < kSubMPerHalf; ++mm_local) {
        int mm_global   = half * kSubMPerHalf + mm_local;
        int row_in_slab = mm_local * 16 + lane_and_15;
        #pragma unroll
        for (int nn = 0; nn < kSubN; ++nn) {
            int col_base = nn * 16 + lane_lsr_4 * 4;
            int flat_idx = mm_global * kSubN + nn;
            #pragma unroll
            for (int dw_pair = 0; dw_pair < 2; ++dw_pair) {
                int col = col_base + dw_pair * 2;
                int off = epi_lds_offset_bytes_pc<WARP_N>(row_in_slab, col);
                *(vec2_bf16*)((char*)lds_slab + off)
                    = acc_bf16_flat[flat_idx].f16x2[dw_pair];
            }
        }
    }
}

// Read the wave's LDS slab back and write it to global with coalesced stores.
template<typename Element_O, int WARP_M, int WARP_N>
__device__ __forceinline__ void epi_load_lds_and_store_global_pc(
    const Element_O* lds_slab, Element_O* d_ptr, int ldd,
    int row_base_global, int col_base_global, int row_offset_in_wave,
    int M, int N, int lane_id)
{
    using C = EpiCoalescePC<WARP_N>;
    constexpr int kRowsPerHalf = WARP_M / kEpiHalvesPerWarpPC;
    constexpr int kPasses      = kRowsPerHalf / C::kRowsPerPass;

    const int row_in_pass = lane_id >> C::kLog2ColsPerRow;
    const int col_in_wave = (lane_id & (C::kColsPerRow - 1)) * kEpiElemsPerLanePC;
    const int col_global  = col_base_global + col_in_wave;

    // Phase A: batched LDS reads into VGPR buffer.
    union_vec8_bf16_pc<Element_O> buf[kPasses];
    #pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        int row_in_slab = pass * C::kRowsPerPass + row_in_pass;
        // Two ds_read_b64 (8B) per lane instead of four 4B reads — the
        // 8B-granularity XOR keeps each 8B unit physically contiguous.
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            int col = col_in_wave + q * 4;
            int off = epi_lds_offset_bytes_pc<WARP_N>(row_in_slab, col);
            *(int64_t*)&buf[pass].bf16[q * 4]
                = *(const int64_t*)((const char*)lds_slab + off);
        }
    }
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_hcu_sched_barrier(0);

    // Phase B: back-to-back global stores.
    bool all_rows_in_bounds = true;
    #pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        int row_global = row_base_global + row_offset_in_wave
                       + pass * C::kRowsPerPass + row_in_pass;
        if (row_global >= M) { all_rows_in_bounds = false; break; }
    }
    bool cols_in_bounds = (col_global + kEpiElemsPerLanePC <= N);

    if (all_rows_in_bounds && cols_in_bounds) {
        #pragma unroll
        for (int pass = 0; pass < kPasses; ++pass) {
            int row_global = row_base_global + row_offset_in_wave
                           + pass * C::kRowsPerPass + row_in_pass;
            *(union_vec8_bf16_pc<Element_O>*)(d_ptr + row_global * ldd + col_global)
                = buf[pass];
        }
    } else {
        #pragma unroll
        for (int pass = 0; pass < kPasses; ++pass) {
            int row_global = row_base_global + row_offset_in_wave
                           + pass * C::kRowsPerPass + row_in_pass;
            if (row_global >= M) continue;
            if (cols_in_bounds) {
                *(union_vec8_bf16_pc<Element_O>*)(d_ptr + row_global * ldd + col_global)
                    = buf[pass];
            } else {
                #pragma unroll
                for (int i = 0; i < kEpiElemsPerLanePC; ++i) {
                    if (col_global + i < N) {
                        d_ptr[row_global * ldd + col_global + i] = buf[pass].bf16[i];
                    }
                }
            }
        }
    }
}

// bf16/fp16 epilogue: cast fp32 accumulators to Element_O, stage through LDS
// in kEpiHalvesPerWarpPC passes, coalesced global stores. Each consumer wave
// owns its own LDS slab (no s_barrier needed — only lgkmcnt).
// Runtime-uniform modifiers, each dispatched with WHOLE-loop bodies (never
// mixed into a single unrolled loop — see the if-conversion warning below):
//   has_scale (params.scale_a_ptr != nullptr) : per-element sa*sb multiply
//   alpha != 1  : uniform fp32 scale of acc_o (in place)
//   beta  != 0  : add beta * D_old (C == old in-place output) in fp32 before
//                 the single downcast.
template<typename Element_O, int WARP_M, int WARP_N>
__device__ __forceinline__ void gemm_epilogue_writeback_via_lds_pc(
    Element_O* d_ptr, int ldd, int M, int N,
    int row_base_global, int col_base_global,
    int consumer_id, int lane_id,
    union_vec4_fp32* acc_o_flat,
    Element_O* lds_scratch_base,
    const union_vec16_dword* scale_regs,
    bool has_scale,
    float alpha, float beta)
{
    using C = EpiCoalescePC<WARP_N>;
    constexpr int kSubM     = WARP_M / 16;
    constexpr int kSubN     = WARP_N / 16;
    constexpr int kTotalSub = kSubM * kSubN;
    constexpr int kRowsPerHalf = WARP_M / kEpiHalvesPerWarpPC;

    static_assert(WARP_M % kEpiHalvesPerWarpPC == 0,
                  "WARP_M must be divisible by kEpiHalvesPerWarpPC");
    static_assert(kRowsPerHalf % C::kRowsPerPass == 0,
                  "kRowsPerHalf must be divisible by kRowsPerPass");

    // Cast fp32 -> Element_O (bf16 or fp16) in registers (whole warp tile).
    // NEVER put a runtime uniform `if` (has_scale / alpha / beta) inside the
    // fully-unrolled loops — aicc if-converts it into predication and the
    // default path pays the skipped work's VALU tax (~5-6 us/tile for scale).
    // Each modifier is a branch around WHOLE loop bodies instead.
    union_vec2_f16x2<Element_O> acc_bf16[kTotalSub];

    // alpha: uniform whole-tile multiply of the fp32 accumulator (only runs
    // when alpha != 1). acc_o_flat is dead after the epilogue, so in place.
    if (alpha != 1.0f) {
        const vec4_fp32 alpha4 = {alpha, alpha, alpha, alpha};
        #pragma unroll
        for (int i = 0; i < kTotalSub; ++i) {
            acc_o_flat[i].f32 = acc_o_flat[i].f32 * alpha4;
        }
    }

    auto downcast_store = [&](int i, int v, const vec2_fp32& scaled) {
        if constexpr (std::is_same_v<Element_O, bhalf_t>) {
            acc_bf16[i].f16x2[v] = DownCastPair_f32_bf16(scaled);
        } else {
            acc_bf16[i].f16x2[v] = DownCastPair_f32_fp16(scaled);
        }
    };

    // beta != 0 bodies: add beta * C (C = old in-place output) as fp32 BEFORE
    // the single downcast. C reads are guarded per element (edge tiles -> 0;
    // the coalesced global store later drops OOB elements anyway). This whole
    // path only exists for beta != 0 — the default beta==0 kernels do not
    // touch the output until the store phase.
    if (has_scale) {
        if (beta != 0.0f) {
            #pragma unroll
            for (int i = 0; i < kTotalSub; ++i) {
                const int m_local = i / kSubN;
                const int n_local = i % kSubN;
                const int row_g = row_base_global + m_local * 16 + (lane_id & 15);
                const int col_g = col_base_global + n_local * 16 + (lane_id >> 4) * 4;
                const float sa = scale_regs[0].f[m_local];
                const float* sb = scale_regs[1].f;
                float c[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[j] = (row_g < M && col_g + j < N)
                               ? (float)d_ptr[(int64_t)row_g * ldd + col_g + j]
                               : 0.0f;
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    vec2_fp32 scaled = acc_o_flat[i].f32x2[v];
                    scaled[0] *= sa * sb[n_local * 4 + (v * 2)];
                    scaled[1] *= sa * sb[n_local * 4 + (v * 2) + 1];
                    scaled[0] += beta * c[v * 2];
                    scaled[1] += beta * c[v * 2 + 1];
                    downcast_store(i, v, scaled);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kTotalSub; ++i) {
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    vec2_fp32 scaled = acc_o_flat[i].f32x2[v];
                    const int m_local = i / kSubN;
                    const int n_local = i % kSubN;
                    const float sa = scale_regs[0].f[m_local];
                    const float* sb = scale_regs[1].f;
                    scaled[0] *= sa * sb[n_local * 4 + (v * 2)];
                    scaled[1] *= sa * sb[n_local * 4 + (v * 2) + 1];
                    if constexpr (std::is_same_v<Element_O, bhalf_t>) {
                        acc_bf16[i].f16x2[v] = DownCastPair_f32_bf16(scaled);
                    } else {
                        acc_bf16[i].f16x2[v] = DownCastPair_f32_fp16(scaled);
                    }
                }
            }
        }
    } else {
        if (beta != 0.0f) {
            #pragma unroll
            for (int i = 0; i < kTotalSub; ++i) {
                const int m_local = i / kSubN;
                const int n_local = i % kSubN;
                const int row_g = row_base_global + m_local * 16 + (lane_id & 15);
                const int col_g = col_base_global + n_local * 16 + (lane_id >> 4) * 4;
                float c[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[j] = (row_g < M && col_g + j < N)
                               ? (float)d_ptr[(int64_t)row_g * ldd + col_g + j]
                               : 0.0f;
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    vec2_fp32 scaled = acc_o_flat[i].f32x2[v];
                    scaled[0] += beta * c[v * 2];
                    scaled[1] += beta * c[v * 2 + 1];
                    downcast_store(i, v, scaled);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kTotalSub; ++i) {
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    vec2_fp32 scaled = acc_o_flat[i].f32x2[v];
                    if constexpr (std::is_same_v<Element_O, bhalf_t>) {
                        acc_bf16[i].f16x2[v] = DownCastPair_f32_bf16(scaled);
                    } else {
                        acc_bf16[i].f16x2[v] = DownCastPair_f32_fp16(scaled);
                    }
                }
            }
        }
    }

    // Per-wave LDS slab (bytes).
    Element_O* lds_slab = reinterpret_cast<Element_O*>(
        reinterpret_cast<char*>(lds_scratch_base)
        + consumer_id * epi_lds_slab_bytes_per_wave_pc<WARP_M, WARP_N>());

    // Two-pass stage.
    #pragma unroll
    for (int half = 0; half < kEpiHalvesPerWarpPC; ++half) {
        epi_store_half_to_lds_pc<Element_O, WARP_M, WARP_N>(
            lds_slab, lane_id, half, acc_bf16);

        __builtin_hcu_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)\n");
        __builtin_hcu_sched_barrier(0);

        epi_load_lds_and_store_global_pc<Element_O, WARP_M, WARP_N>(
            lds_slab, d_ptr, ldd,
            row_base_global, col_base_global,
            /*row_offset_in_wave=*/half * kRowsPerHalf,
            M, N, lane_id);

        if (half + 1 < kEpiHalvesPerWarpPC) {
            __builtin_hcu_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_hcu_sched_barrier(0);
        }
    }
}

// ============================================================================
// fp32-output epilogue. fp32 is the MMAC accumulator's own precision, so this
// is just: apply alpha / sa*sb / beta in fp32 in place, then stage the raw
// acc_o through LDS for coalesced global stores — no downcast, no extra
// registers. Same LDS-coalescing trick as the 2-byte path, fp32 geometry:
//   * one 16 B dwordx4 per lane per store   → 4 floats / lane
//   * XOR-swizzle unit = 16 B (4 floats) by row&7 → whole b128 unit accesses
//   * row = WARP_N floats = 128 B (a full cache line)
// A wave writes WARP_M = 256 rows in 4 slices of 64 rows → slab = 64*128 =
// 8 KB/wave, same per-wave budget as the 2-byte path (64 KB WG total).
// ============================================================================
template<int WARP_N, typename Element_O>
struct EpiGeomF32 {
    static constexpr int kE              = sizeof(Element_O);       // 4
    static constexpr int kElemsPerLane   = 16 / kE;                 // 4
    static constexpr int kColsPerRow     = WARP_N / kElemsPerLane;  // 8
    static constexpr int kRowsPerPass    = 64 / kColsPerRow;        // 8
    static constexpr int kLog2ColsPerRow =
        (kColsPerRow == 16) ? 4 :
        (kColsPerRow ==  8) ? 3 :
        (kColsPerRow ==  4) ? 2 :
        (kColsPerRow ==  2) ? 1 : -1;
    static constexpr int kUnitElems      = 4;                       // elems / XOR unit
    static constexpr int kUnitsPerRow    = WARP_N / kUnitElems;     // 8
    static constexpr int kRowBytes       = WARP_N * kE;             // 128
    static constexpr int kUnitBytes      = kUnitElems * kE;         // 16
    static_assert(WARP_N % kElemsPerLane == 0, "WARP_N multiple of elems/lane");
    static_assert(kLog2ColsPerRow >= 0, "kColsPerRow power of 2");
    static_assert(64 % kColsPerRow == 0, "wave tiles rows evenly");
    static_assert((kUnitsPerRow & (kUnitsPerRow - 1)) == 0, "units/row power of 2");
};

template<int WARP_M, int WARP_N, typename Element_O>
struct EpiPassF32 {
    using G = EpiGeomF32<WARP_N, Element_O>;
    static constexpr int kBudgetBytes  = 8192;
    static constexpr int kPasses       = (WARP_M * G::kRowBytes + kBudgetBytes - 1) / kBudgetBytes;
    static constexpr int kRowsInFlight = WARP_M / kPasses;
    static_assert(WARP_M % kPasses == 0, "kPasses must divide WARP_M");
    static_assert(kRowsInFlight % G::kRowsPerPass == 0,
                  "rows-in-flight multiple of reload rows/pass");
};

template<typename Element_O>
struct EpiUnitF32 {
    Element_O v[4];   // one 16 B XOR unit (4 floats)
};

template<typename Element_O>
struct EpiRow16F32 {
    Element_O v[16 / sizeof(Element_O)];   // one coalesced dwordx4 (16 B)
};

// XOR-swizzled LDS byte offset for one element. Slab: rows at stride
// WARP_N*kE bytes, 4-element (16 B) units XOR-permuted by (row & 7).
template<int WARP_N, typename Element_O>
__device__ __forceinline__ int epi_lds_offset_bytes_f32(int row_in_slab, int col) {
    using G = EpiGeomF32<WARP_N, Element_O>;
    int col_q = col >> 2;
    int swizzled_q = col_q ^ (row_in_slab & (G::kUnitsPerRow - 1));
    return row_in_slab * G::kRowBytes
         + swizzled_q * G::kUnitBytes
         + (col & 3) * G::kE;
}

template<int WARP_N, typename Element_O>
__device__ __forceinline__ void epi_unit_read_f32(
    const Element_O* lds_slab, int row_in_slab, int col, void* dst) {
    const int off = epi_lds_offset_bytes_f32<WARP_N, Element_O>(row_in_slab, col);
    *(dword4_t*)dst = *(const dword4_t*)((const char*)lds_slab + off);  // ds_read_b128
}

template<int WARP_N, typename Element_O>
__device__ __forceinline__ void epi_unit_write_f32(
    Element_O* lds_slab, int row_in_slab, int col, const void* src) {
    const int off = epi_lds_offset_bytes_f32<WARP_N, Element_O>(row_in_slab, col);
    *(dword4_t*)((char*)lds_slab + off) = *(const dword4_t*)src;        // ds_write_b128
}

// Stage one slice (kRowsInFlight rows) from the in-register 4-float units
// (indexed by flat m*n sub-tile, the acc_o layout) into the wave's LDS slab.
template<typename Element_O, int WARP_M, int WARP_N>
__device__ __forceinline__ void epi_store_slice_f32(
    Element_O* lds_slab, int lane_id, int slice,
    const EpiUnitF32<Element_O>* out)
{
    using P = EpiPassF32<WARP_M, WARP_N, Element_O>;
    constexpr int kSubN = WARP_N / 16;
    constexpr int kSliceRows    = P::kRowsInFlight;
    constexpr int kSubMPerSlice = kSliceRows / 16;

    const int lane_and_15 = lane_id & 15;
    const int lane_lsr_4  = lane_id >> 4;

    #pragma unroll
    for (int mm_local = 0; mm_local < kSubMPerSlice; ++mm_local) {
        int mm_global   = slice * kSubMPerSlice + mm_local;
        int row_in_slab = mm_local * 16 + lane_and_15;
        #pragma unroll
        for (int nn = 0; nn < kSubN; ++nn) {
            int col_base = nn * 16 + lane_lsr_4 * 4;
            int flat_idx = mm_global * kSubN + nn;
            epi_unit_write_f32<WARP_N, Element_O>(
                lds_slab, row_in_slab, col_base, out[flat_idx].v);
        }
    }
}

// Read the wave's LDS slab back (16 B / lane / pass = one b128 unit) and write
// it to global with coalesced dwordx4 stores (8 rows per wave-pass).
template<typename Element_O, int WARP_M, int WARP_N>
__device__ __forceinline__ void epi_load_lds_and_store_global_f32(
    const Element_O* lds_slab, Element_O* d_ptr, int ldd,
    int row_base_global, int col_base_global, int row_offset_in_wave,
    int M, int N, int lane_id)
{
    using G = EpiGeomF32<WARP_N, Element_O>;
    using P = EpiPassF32<WARP_M, WARP_N, Element_O>;
    constexpr int kColsPerRow     = G::kColsPerRow;
    constexpr int kRowsPerPass    = G::kRowsPerPass;
    constexpr int kLog2ColsPerRow = G::kLog2ColsPerRow;
    constexpr int kElemsPerLane   = G::kElemsPerLane;
    constexpr int kSliceRows      = P::kRowsInFlight;
    constexpr int kPasses         = kSliceRows / kRowsPerPass;
    static_assert(kPasses * kRowsPerPass == kSliceRows, "reload must tile rows evenly");

    const int row_in_pass = lane_id >> kLog2ColsPerRow;
    const int col_in_wave = (lane_id & (kColsPerRow - 1)) * kElemsPerLane;
    const int col_global  = col_base_global + col_in_wave;

    // Phase A: batched LDS reads into VGPR buffer.
    EpiRow16F32<Element_O> buf[kPasses];
    #pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        int row_in_slab = pass * kRowsPerPass + row_in_pass;
        epi_unit_read_f32<WARP_N, Element_O>(
            lds_slab, row_in_slab, col_in_wave, buf[pass].v);
    }
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_hcu_sched_barrier(0);

    // Phase B: back-to-back global stores.
    bool all_rows_in_bounds = true;
    #pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        int row_global = row_base_global + row_offset_in_wave
                       + pass * kRowsPerPass + row_in_pass;
        if (row_global >= M) { all_rows_in_bounds = false; break; }
    }
    bool cols_in_bounds = (col_global + kElemsPerLane <= N);

    if (all_rows_in_bounds && cols_in_bounds) {
        #pragma unroll
        for (int pass = 0; pass < kPasses; ++pass) {
            int row_global = row_base_global + row_offset_in_wave
                           + pass * kRowsPerPass + row_in_pass;
            *(EpiRow16F32<Element_O>*)(d_ptr + row_global * ldd + col_global)
                = buf[pass];
        }
    } else {
        #pragma unroll
        for (int pass = 0; pass < kPasses; ++pass) {
            int row_global = row_base_global + row_offset_in_wave
                           + pass * kRowsPerPass + row_in_pass;
            if (row_global >= M) continue;
            if (cols_in_bounds) {
                *(EpiRow16F32<Element_O>*)(d_ptr + row_global * ldd + col_global)
                    = buf[pass];
            } else {
                #pragma unroll
                for (int i = 0; i < kElemsPerLane; ++i) {
                    if (col_global + i < N) {
                        d_ptr[row_global * ldd + col_global + i] = buf[pass].v[i];
                    }
                }
            }
        }
    }
}

// Shared fp32 slice pipeline: stage `unit_src` in kPasses slices, reload +
// store to global between slices.
template<typename Element_O, int WARP_M, int WARP_N>
__device__ __forceinline__ void epi_writeback_slices_f32(
    Element_O* lds_slab, Element_O* d_ptr, int ldd, int M, int N,
    int row_base_global, int col_base_global,
    int consumer_id, int lane_id,
    const EpiUnitF32<Element_O>* unit_src)
{
    using P = EpiPassF32<WARP_M, WARP_N, Element_O>;
    constexpr int kSliceRows = P::kRowsInFlight;
    constexpr int kPasses    = P::kPasses;

    #pragma unroll
    for (int slice = 0; slice < kPasses; ++slice) {
        epi_store_slice_f32<Element_O, WARP_M, WARP_N>(
            lds_slab, lane_id, slice, unit_src);

        __builtin_hcu_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)\n");
        __builtin_hcu_sched_barrier(0);

        epi_load_lds_and_store_global_f32<Element_O, WARP_M, WARP_N>(
            lds_slab, d_ptr, ldd,
            row_base_global, col_base_global,
            /*row_offset_in_wave=*/slice * kSliceRows,
            M, N, lane_id);

        if (slice + 1 < kPasses) {
            __builtin_hcu_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_hcu_sched_barrier(0);
        }
    }
}

// Same shape as gemm_epilogue_writeback_via_lds_pc but for fp32 output: no
// downcast, so modifiers write acc_o in place; whole-body branching per
// modifier follows the same if-conversion caveat.
template<typename Element_O, int WARP_M, int WARP_N>
__device__ __forceinline__ void gemm_epilogue_writeback_f32(
    Element_O* d_ptr, int ldd, int M, int N,
    int row_base_global, int col_base_global,
    int consumer_id, int lane_id,
    union_vec4_fp32* acc_o_flat,
    Element_O* lds_scratch_base,
    const union_vec16_dword* scale_regs,
    bool has_scale,
    float alpha, float beta)
{
    static_assert(sizeof(Element_O) == 4, "fp32 epilogue is for float output only");
    using G = EpiGeomF32<WARP_N, Element_O>;
    using P = EpiPassF32<WARP_M, WARP_N, Element_O>;
    constexpr int kSubM     = WARP_M / 16;
    constexpr int kSubN     = WARP_N / 16;
    constexpr int kTotalSub = kSubM * kSubN;
    constexpr int kSliceRows = P::kRowsInFlight;

    if (alpha != 1.0f) {
        const vec4_fp32 alpha4 = {alpha, alpha, alpha, alpha};
        #pragma unroll
        for (int i = 0; i < kTotalSub; ++i) {
            acc_o_flat[i].f32 = acc_o_flat[i].f32 * alpha4;
        }
    }

    auto acc_writeback = [&](int i, int v, const vec2_fp32& scaled) {
        acc_o_flat[i].f32x2[v] = scaled;
    };

    if (has_scale) {
        if (beta != 0.0f) {
            #pragma unroll
            for (int i = 0; i < kTotalSub; ++i) {
                const int m_local = i / kSubN;
                const int n_local = i % kSubN;
                const int row_g = row_base_global + m_local * 16 + (lane_id & 15);
                const int col_g = col_base_global + n_local * 16 + (lane_id >> 4) * 4;
                const float sa = scale_regs[0].f[m_local];
                const float* sb = scale_regs[1].f;
                float c[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[j] = (row_g < M && col_g + j < N)
                               ? (float)d_ptr[(int64_t)row_g * ldd + col_g + j]
                               : 0.0f;
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    vec2_fp32 scaled = acc_o_flat[i].f32x2[v];
                    scaled[0] *= sa * sb[n_local * 4 + (v * 2)];
                    scaled[1] *= sa * sb[n_local * 4 + (v * 2) + 1];
                    scaled[0] += beta * c[v * 2];
                    scaled[1] += beta * c[v * 2 + 1];
                    acc_writeback(i, v, scaled);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kTotalSub; ++i) {
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    vec2_fp32 scaled = acc_o_flat[i].f32x2[v];
                    const int m_local = i / kSubN;
                    const int n_local = i % kSubN;
                    const float sa = scale_regs[0].f[m_local];
                    const float* sb = scale_regs[1].f;
                    scaled[0] *= sa * sb[n_local * 4 + (v * 2)];
                    scaled[1] *= sa * sb[n_local * 4 + (v * 2) + 1];
                    acc_writeback(i, v, scaled);
                }
            }
        }
    } else {
        if (beta != 0.0f) {
            #pragma unroll
            for (int i = 0; i < kTotalSub; ++i) {
                const int m_local = i / kSubN;
                const int n_local = i % kSubN;
                const int row_g = row_base_global + m_local * 16 + (lane_id & 15);
                const int col_g = col_base_global + n_local * 16 + (lane_id >> 4) * 4;
                float c[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    c[j] = (row_g < M && col_g + j < N)
                               ? (float)d_ptr[(int64_t)row_g * ldd + col_g + j]
                               : 0.0f;
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    vec2_fp32 scaled = acc_o_flat[i].f32x2[v];
                    scaled[0] += beta * c[v * 2];
                    scaled[1] += beta * c[v * 2 + 1];
                    acc_writeback(i, v, scaled);
                }
            }
        } else {
            // beta == 0, no scale: the accumulator is the output as-is.
        }
    }

    // Per-wave LDS slab (bytes): kSliceRows rows x 128 B = 8 KB.
    Element_O* lds_slab = reinterpret_cast<Element_O*>(
        reinterpret_cast<char*>(lds_scratch_base)
        + consumer_id * (kSliceRows * G::kRowBytes));

    // acc_o IS the fp32 output — stage it directly (zero extra registers).
    epi_writeback_slices_f32<Element_O, WARP_M, WARP_N>(
        lds_slab, d_ptr, ldd, M, N, row_base_global, col_base_global,
        consumer_id, lane_id,
        reinterpret_cast<const EpiUnitF32<Element_O>*>(acc_o_flat));
}

// ============================================================================
// Producer: buffer_load_dwordx4 Direct-to-LDS with XOR K swizzle.
// Per panel i: lane L loads A[row=i*8+(L>>3)][K=((L&7)^(L>>3))*16 :+16]
// HW writes it to m0+16*L = row*128 + (k16^(row&7))*16.
// has_scale (params.scale_a_ptr != nullptr, uniform across the WG) is a
// RUNTIME decision: with scales the tail stages sa/sb into the dead A stage
// (3 barriers); without, the tail is a plain 2-barrier idle. Producers and
// consumers read the same pointer, so barrier counts stay matched.
// ============================================================================
__device__ __forceinline__ void producer_body(
    const DeepGemm_params& params,
    int prod_id,
    fp8_e4m3* a_lds_all,
    int by, int bx,
    int lane_id
) {
    constexpr int kBlockM    = GemmTraitsPC::kBlockM;
    constexpr int kBlockN    = GemmTraitsPC::kBlockN;
    constexpr int kStageK    = GemmTraitsPC::kStageK;
    constexpr int kProdSlabM = GemmTraitsPC::kProdSlabM;

    const int K   = params.k;
    const int lda = params.a_stride;
    const int a_row_start = by * kBlockM + prod_id * kProdSlabM;

    buffer_rsrc_t a_rsrc = make_buffer_rsrc(
        reinterpret_cast<fp8_e4m3*>(params.a_ptr), lda,
        (uint32_t)(params.m * lda));

    const int n_stages = ceil_div(K, kStageK);
    const int row_group = lane_id >> 3;
    const int k_in_panel = ((lane_id & 7) ^ row_group) * 16;

    const uint32_t stride_8rows = (uint32_t)(8 * lda);

    // ONE per-lane VGPR shared by all 8 loads. All row-group offsets (i*8*lda)
    // and stage col_base (k_stage*kStageK) ride the SGPR soffset instead of
    // being baked into 8 different VGPRs. This kills 16 VALU from the previous
    // startup (8 v_add for v_off_pre[] + 8 v_or for v_off_last[]).
    // Startup cost: ~4 VALU here (row_group shift, XOR swizzle, mul-add).
    const uint32_t v_off = (uint32_t)((a_row_start + row_group) * lda + k_in_panel);

    // 8 loop-invariant SGPRs holding i*8*lda (uniform → SALU only).
    uint32_t s_stride[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) s_stride[i] = (uint32_t)i * stride_8rows;

    constexpr uint32_t A_stage_bytes_ct = (uint32_t)GemmTraitsPC::A_stage_bytes;
    const uint32_t prod_slab_off = (uint32_t)(prod_id * (int)GemmTraitsPC::A_slab_bytes);

    // ---- Scale staging addresses, computed ONCE in the prologue. ----
    // The producer's tail overlaps the consumers' last-stage MMAC, and VALU
    // there would steal MMAC issue slots on the same SIMD. Compute every
    // per-lane address here (before any MMAC runs) so the tail issues only
    // buffer_load + ds_write with zero VALU. Skipped (branch false) when the
    // runtime has_scale is false.
    const bool has_scale = (params.scale_a_ptr != nullptr);
    uint32_t ga = 0, gb = 0, la = 0;
    buffer_rsrc_t sa_rsrc{}, sb_rsrc{};
    if (has_scale) {
        const uint32_t scale_lds_base =
            (uint32_t)((n_stages & 1) * (int)GemmTraitsPC::A_stage_bytes);
        const uint32_t tid4 = ((uint32_t)lane_id << 2) + (uint32_t)(prod_id * 256);
        ga = (uint32_t)(by * kBlockM * 4) + tid4;   // scale_a byte offset
        gb = (uint32_t)(bx * kBlockN * 4) + tid4;   // scale_b byte offset
        // la is produced by an opaque asm so the compiler CANNOT rematerialize
        // it in the tail (it would otherwise re-derive la = scale_lds_base +
        // tid4 with a VALU right next to the consumers' last MMAC). lb is
        // folded into the ds_write immediate offset (la + 1024) instead.
        asm volatile("v_add_u32_e32 %0, %1, %2" : "=v"(la) : "v"(tid4), "v"(scale_lds_base));
        sa_rsrc = make_buffer_rsrc<false>(
            reinterpret_cast<const float*>(params.scale_a_ptr), 4, (uint32_t)(params.m * 4));
        sb_rsrc = make_buffer_rsrc<false>(
            reinterpret_cast<const float*>(params.scale_b_ptr), 4, (uint32_t)(params.n * 4));
    }

    const int k_last_stage = n_stages - 1;
    const int k_left_last  = K - k_last_stage * kStageK;
    const int tail_chunk   = k_left_last / 16;
    const int tail_bytes   = k_left_last % 16;
    const uint32_t v_off_tail =
        (k_left_last < kStageK && k_in_panel >= k_left_last)
            ? (v_off | 0x80000000u)
            : v_off;

    // The x8 load maps each lane to eight rows, 1024 bytes apart in LDS.
    // Clear the suffix of the one chunk that can straddle K after the VMEM
    // loads complete. Other tail chunks are already hardware zero-filled.
    auto zero_tail = [&](int s) {
        if (tail_bytes == 0 || k_in_panel != tail_chunk * 16) return;
        const uint32_t stage_base = (uint32_t)reinterpret_cast<size_t>(a_lds_all)
                                   + prod_slab_off + (uint32_t)s * A_stage_bytes_ct;
        const uint32_t zero = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t addr = stage_base + i * 1024 + lane_id * 16 + tail_bytes;
            int bytes = 16 - tail_bytes;
            while ((addr & 3) != 0 && bytes > 0) {
                asm volatile("ds_write_b8 %0, %1" :: "v"(addr), "v"(zero)
                             : "memory");
                ++addr;
                --bytes;
            }
            while (bytes >= 4) {
                asm volatile("ds_write_b32 %0, %1" :: "v"(addr), "v"(zero)
                             : "memory");
                addr += 4;
                bytes -= 4;
            }
            while (bytes > 0) {
                asm volatile("ds_write_b8 %0, %1" :: "v"(addr), "v"(zero)
                             : "memory");
                ++addr;
                --bytes;
            }
        }
        // s_barrier below synchronizes wave arrival, but does not replace the
        // writer-side LDS completion wait required before consumers read it.
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    };

    auto issue_stage = [&](int k_stage, int s, uint32_t v_off_use) {
        const uint32_t s_col_base = (uint32_t)(k_stage * kStageK);
        const uint32_t lds_m0 = prod_slab_off + s * A_stage_bytes_ct;
        buffer_load_dwordx4_x8_to_lds_soff(
            a_rsrc, v_off_use, lds_m0,
            s_col_base + s_stride[0], s_col_base + s_stride[1],
            s_col_base + s_stride[2], s_col_base + s_stride[3],
            s_col_base + s_stride[4], s_col_base + s_stride[5],
            s_col_base + s_stride[6], s_col_base + s_stride[7]);
    };

    // ============================================================================
    // Double-barrier pipeline. Total barriers = 2 * n_stages + 3 (has_scale)
    // or 2 * n_stages + 2 (no scale).
    //   Prologue (2 barriers):     issue A(0);  BARRIER;  BARRIER
    //   Main k = 1..n-2 (2 each):  issue A(k);  BARRIER; wait vmcnt; BARRIER
    //   Peeled tail (2 barriers):  issue A(n-1) with v_off_tail; BARRIER;
    //                              wait vmcnt; zero_tail; BARRIER
    //   Scales (3 or 2 barriers):  BARRIER; [stage scales to LDS]; BARRIER; BARRIER
    // Caller passes v_off or v_off_tail explicitly — no ternary inside
    // issue_stage, so the main loop stays truly zero-VALU (no v_cndmask).
    // ============================================================================

    // ---------- Prologue: issue stage 0. ----------
    // If n_stages == 1 stage 0 IS the K-tail, so it uses v_off_tail; the
    // n_stages == 1 branch is a scalar (uniform) check → compiler emits an
    // SCC branch, not a VGPR select.
    if (n_stages == 1) {
        issue_stage(0, /*s=*/0, v_off_tail);
    } else {
        issue_stage(0, /*s=*/0, v_off);
    }
    flash::wait_all_warp_arrived();                      // BARRIER prologue.mid
    flash::wait_all_buffer_data_arrived<false>(0);
    if (n_stages == 1) zero_tail(/*s=*/0);
    flash::wait_all_warp_arrived();                      // BARRIER prologue.end

    // ---------- Main: issue stages 1..n-2 (all full-K, use v_off). ----------
    #pragma unroll 1
    for (int k_stage = 0; k_stage + 2 < n_stages; ++k_stage) {
        const int stage_to_issue = k_stage + 1;
        const int s = stage_to_issue & 1;
        issue_stage(stage_to_issue, s, v_off);
        flash::wait_all_warp_arrived();                  // BARRIER k.mid
        flash::wait_all_buffer_data_arrived<false>(0);
        flash::wait_all_warp_arrived();                  // BARRIER k.end
    }

    // ---------- Peeled last issue: stage n-1 with v_off_tail. ----------
    if (n_stages >= 2) {
        const int s = k_last_stage & 1;
        issue_stage(k_last_stage, s, v_off_tail);
        flash::wait_all_warp_arrived();                  // BARRIER k.mid
        flash::wait_all_buffer_data_arrived<false>(0);
        zero_tail(s);
        flash::wait_all_warp_arrived();                  // BARRIER k.end
    }

    if (has_scale) {
        // ---------- Tail: stage n-1 idle + scale staging (3 barriers) ----------
        // The A buffer that held A(n_stages-2) (index n_stages&1) is dead here.
        // Stage the per-row fp32 scales into its first 2 KB: scale_a[by*256..+255]
        // then scale_b[bx*256..+255]. Consumers ds_read them in the epilogue, so
        // the global fetch latency lives entirely in this idle producer window.
        // All addresses were computed in the prologue — this tail is zero VALU.
        // (has_scale is exactly params.scale_a_ptr != nullptr, so no identity
        // "ones" fallback is needed here — the branch is taken iff scales exist.)
        flash::wait_all_warp_arrived();                  // BARRIER (n-1).mid
        float sa, sb;
        const uint32_t zero = 0;
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:0\n\t"
                     : "=&v"(sa) : "v"(ga), "s"(sa_rsrc), "s"(zero) : "memory");
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:0\n\t"
                     : "=&v"(sb) : "v"(gb), "s"(sb_rsrc), "s"(zero) : "memory");
        asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
        asm volatile("ds_write_b32 %0, %1\n\t" :: "v"(la), "v"(sa) : "memory");
        asm volatile("ds_write_b32 %0, %1 offset:1024\n\t" :: "v"(la), "v"(sb) : "memory");
        asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
        flash::wait_all_warp_arrived();                  // BARRIER (n-1).end
        flash::wait_all_warp_arrived();                  // BARRIER epi.sync
    } else {
        // ---------- Tail: stage n-1 idle 2 barriers (no scale) ----------
        flash::wait_all_warp_arrived();                  // BARRIER (n-1).mid
        flash::wait_all_warp_arrived();                  // BARRIER (n-1).end
    }
}

// ============================================================================
// Consumer body
// ============================================================================
template<typename Element_O>
__device__ __forceinline__ void consumer_body(
    const DeepGemm_params& params,
    int consumer_id,
    fp8_e4m3* a_lds_all,
    int by, int bx,
    int lane_id
) {
    constexpr int kBlockM     = GemmTraitsPC::kBlockM;     // 256
    constexpr int kBlockN     = GemmTraitsPC::kBlockN;     // 256
    constexpr int kStageK     = GemmTraitsPC::kStageK;     // 128
    constexpr int kWaveN      = GemmTraitsPC::kWaveN;      // 32
    constexpr int kPassK      = GemmTraitsPC::kPassK;      // 64
    constexpr int kMmacM      = 16;
    constexpr int kMmacN      = 16;
    constexpr int kMmacK      = 32;
    constexpr int kSubTilesM  = kBlockM / kMmacM;            // 16
    constexpr int kSubTilesN  = kWaveN / kMmacN;             // 2
    constexpr int kSubTilesKPass = kPassK / kMmacK;          // 2

    const int nj = consumer_id;
    const int row_in_tile = lane_id & 15;
    const int col_group   = lane_id >> 4;
    const int n_stages = ceil_div(params.k, kStageK);

    // Runtime scale selection mirrors the producer — every wave reads the same
    // params.scale_a_ptr, so the branch (and therefore the barrier counts) stay
    // uniform across the workgroup.
    const bool has_scale = (params.scale_a_ptr != nullptr);

    const int M = params.m, N = params.n;
    const int ldb = params.b_stride, ldd = params.d_stride;
    const int b_col_start = bx * kBlockN + nj * kWaveN;

    buffer_rsrc_t b_rsrc = make_buffer_rsrc(
        reinterpret_cast<fp8_e4m3*>(params.b_ptr), ldb, (uint32_t)(N * ldb));

    // Acc: 256M x 32N → 128 VGPR
    union_vec4_fp32 acc_o[kSubTilesM * kSubTilesN] = {};
    // A: full 256M x 64K for one K-half → 16 x 16B = 64 VGPR
    union_vec16_fp8 a_regs[kSubTilesM];
    // B: explicit double buffer, 16 VGPR per buffer = 32 VGPR / lane.
    // 128K x 32N packed as d4[n * 2 + k_half] (4 x 16B); each 16B is
    // aliased as v8[(n*2+k_half)*2 + kk] to feed the two 8B mmac K-chunks.
    // Every use below indexes b_regs with a compile-time literal so aicc
    // keeps the two buffers in independent VGPRs (past attempts with a
    // runtime `if(buf)` alias both onto the same registers and clobber
    // the live B → we hand-unroll the K-stage loop by parity instead).
    union_vec16_dword b_regs[2];

    // Precomputed loop-invariant per-lane LDS base VGPRs for the two K-halves.
    // Layout: base(kh) = a_lds_all + row_in_tile*kStageK + (k16(kh) ^ r7)*16
    //   where k16(kh=0) = col_group in [0..3], k16(kh=1) = col_group+4 in [4..7].
    // The stage-parity offset ((stage_id & 1) * A_stage_bytes = 0 or 32768) is
    // folded into the ds_read imm via the EXTRA template arg, so a_base_h{0,1}
    // stays loop-invariant — 0 VALU per stage instead of ~4-6 for the old
    // a_base(k_stage, k_half) lambda. 32768 + 30720 = 63488 < 65535 fits in
    // ds_read_b128's 16-bit unsigned imm offset.
    const int r7 = row_in_tile & 7;
    const uint32_t a_base_h0 = (uint32_t)(
        reinterpret_cast<size_t>(a_lds_all)
        + row_in_tile * kStageK
        + (col_group ^ r7) * 16);
    const uint32_t a_base_h1 = (uint32_t)(
        reinterpret_cast<size_t>(a_lds_all)
        + row_in_tile * kStageK
        + ((col_group + 4) ^ r7) * 16);

    auto ds_read_a_half = [&]<int STAGE_OFF>(uint32_t base) {
        ds_read_b128_x16_imm<STAGE_OFF>(
            base,
            reinterpret_cast<dword4_t&>(a_regs[0].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[1].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[2].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[3].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[4].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[5].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[6].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[7].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[8].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[9].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[10].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[11].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[12].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[13].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[14].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[15].i32x4));
    };

    // mmac over one K-half using B from buffer BUF (compile-time literal).
    // Reads b_regs[BUF].v8[(n*2 + k_half)*2 + kk].
    auto mmac_half = [&]<int BUF>(int k_half) {
        // kk outermost so consecutive mmacs write DIFFERENT acc_o registers.
        // Old (m,n,kk) order had 4-deep RAW chain per acc (~8c pass-through);
        // new order spreads writes across kSubTilesM*kSubTilesN = 16 acc_o,
        // hiding pass-through behind 15 independent mmacs.
        //
        // First mmac peeled out so set_prio<3> fires *after* one issued mmac
        // and set_prio<0> after the very last one — no per-iteration branch.

        // (kk=0, m=0, n=0)
        acc_o[0].f32 = mmac_4interleave_b8(
            a_regs[0].i8x8[0],
            b_regs[BUF].v8[(0 * 2 + k_half) * 2 + 0],
            acc_o[0].f32);
        flash::set_prio<3>();

        // remainder of (kk=0, m=0): n = 1..kSubTilesN-1
        #pragma unroll
        for (int n = 1; n < kSubTilesN; ++n) {
            acc_o[n].f32 = mmac_4interleave_b8(
                a_regs[0].i8x8[0],
                b_regs[BUF].v8[(n * 2 + k_half) * 2 + 0],
                acc_o[n].f32);
        }
        // remainder of kk=0: m = 1..kSubTilesM-1
        #pragma unroll
        for (int m = 1; m < kSubTilesM; ++m) {
            #pragma unroll
            for (int n = 0; n < kSubTilesN; ++n) {
                acc_o[m * kSubTilesN + n].f32 = mmac_4interleave_b8(
                    a_regs[m].i8x8[0],
                    b_regs[BUF].v8[(n * 2 + k_half) * 2 + 0],
                    acc_o[m * kSubTilesN + n].f32);
            }
        }
        // kk = 1..kSubTilesKPass-1: full m x n
        #pragma unroll
        for (int kk = 1; kk < kSubTilesKPass; ++kk) {
            #pragma unroll
            for (int m = 0; m < kSubTilesM; ++m) {
                #pragma unroll
                for (int n = 0; n < kSubTilesN; ++n) {
                    acc_o[m * kSubTilesN + n].f32 = mmac_4interleave_b8(
                        a_regs[m].i8x8[kk],
                        b_regs[BUF].v8[(n * 2 + k_half) * 2 + kk],
                        acc_o[m * kSubTilesN + n].f32);
                }
            }
        }
        flash::set_prio<0>();
    };

    // Static per-lane VGPR offsets for B: n * ldb + col_group * 16 + k_half*kPassK.
    // SGPR soffset holds the stage-varying col_base. Effective addr = rsrc +
    // v_off + s_col_base.
    //
    // No OOB mask on B: producer_body guarantees that A LDS is zero for every
    // k >= K, using hardware zero-fill for complete chunks and explicit LDS
    // clearing for the one chunk that can straddle K. Therefore B values read
    // past K multiply by zero. This avoids per-stage B-tail masking.
    const int n0 = b_col_start + 0 * kMmacN + row_in_tile;
    const int n1 = b_col_start + 1 * kMmacN + row_in_tile;
    const uint32_t v_off_n0_kh0 = (uint32_t)(n0 * ldb + col_group * 16 + 0 * kPassK);
    const uint32_t v_off_n0_kh1 = (uint32_t)(n0 * ldb + col_group * 16 + 1 * kPassK);
    const uint32_t v_off_n1_kh0 = (uint32_t)(n1 * ldb + col_group * 16 + 0 * kPassK);
    const uint32_t v_off_n1_kh1 = (uint32_t)(n1 * ldb + col_group * 16 + 1 * kPassK);

    // 4 x buffer_load_dwordx4: full 128K x 32N stripe into b_regs[BUF] in
    // one asm block. Layout matches d4[n * 2 + k_half].
    auto buffer_load_b = [&]<int BUF>(int stage_id) {
        const uint32_t s_col_base = (uint32_t)(stage_id * kStageK);
        buffer_load_dwordx4_x4_soff(
            b_rsrc,
            v_off_n0_kh0, v_off_n0_kh1,
            v_off_n1_kh0, v_off_n1_kh1,
            s_col_base,
            b_regs[BUF].d4[0],   // n=0, k_half=0
            b_regs[BUF].d4[1],   // n=0, k_half=1
            b_regs[BUF].d4[2],   // n=1, k_half=0
            b_regs[BUF].d4[3]);  // n=1, k_half=1
    };

    // ============================================================================
    // Two personas, ONE morphology — C' is C shifted by 1/4 stage.
    //
    // A stage is 4 quarters: Q0..Q3. Both personas execute the same sequence
    // per stage but pack them into barrier intervals with a 1-quarter offset:
    //
    //   Q0(k): ds_read A(k, 0)
    //   Q1(k): mmac(k, 0)                    reads A(k,0) + B(k)[BUF=CUR]
    //   Q2(k): ds_read A(k, 1); load B(k+1)  B goes into BUF=CUR^1
    //   Q3(k): mmac(k, 1)                    reads A(k,1) + B(k)[BUF=CUR]
    //
    //   C  : (Q0,Q1) | BAR | (Q2,Q3) | BAR   -- mmac at end of each interval
    //   C' : (Q3_prev,Q0) | BAR | (Q1,Q2) | BAR   -- mmac at start of interval
    //         Q3 of stage -1 skipped (IS_FIRST); Q3 of stage n-1 owed as epilogue.
    //
    // The 1-quarter offset staggers mmac vs ds_read across the two personas
    // within every barrier window — MMAC unit and LDS/vmem pipes both busy.
    //
    // (The two bodies below spell out the Q sequence inline — a Q-lambda
    // refactor read cleaner but wrapping mmac_half in another lambda cost
    // ~3% TFLOPS in codegen; kept inline for perf.)
    //
    // Prologue (both): load B(0); BARRIER; s_waitcnt vmcnt; BARRIER
    // ============================================================================

    // C body: (Q0,Q1) | BAR k.mid | (Q2,Q3) | BAR k.end
    auto stage_body_C = [&]<int CUR>(int k_stage) {
        constexpr int NXT = CUR ^ 1;
        constexpr int STAGE_OFF = CUR * (int)GemmTraitsPC::A_stage_bytes;
        // Q0: ds_read A(k,0)
        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h0);
        // Q1: mmac(k,0)
        flash::wait_lds_data_arrived<false>(0);
        flash::wait_all_buffer_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(0);
        flash::wait_all_warp_arrived();                  // BARRIER k.mid

        // Q2: ds_read A(k,1) then prefetch B(k+1).
        // A goes first: its LDS latency gates Q3's mmac (via lgkmcnt(0)),
        // whereas B(k+1) has a full stage to arrive so a few issue-slots
        // of extra head start don't help. Measured +~4% TFLOPS vs B-first.
        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h1);
        if (k_stage + 1 < n_stages) {
            buffer_load_b.template operator()<NXT>(k_stage + 1);
        }
        // Q3: mmac(k,1)
        flash::wait_lds_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(1);
        flash::wait_all_warp_arrived();                  // BARRIER k.end
    };

    // C' body: (Q3_prev,Q0) | BAR k.mid | (Q1,Q2) | BAR k.end
    // NXT here doubles as "prev stage's CUR" (both = CUR^1).
    auto stage_body_Cp = [&]<int CUR, bool IS_FIRST>(int k_stage) {
        constexpr int NXT = CUR ^ 1;
        constexpr int STAGE_OFF = CUR * (int)GemmTraitsPC::A_stage_bytes;
        // Q3_prev: mmac(k-1,1) using prev stage's CUR (skipped for stage 0)
        if constexpr (!IS_FIRST) {
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<NXT>(1);
        }
        // Q0: ds_read A(k,0)
        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h0);
        flash::wait_all_warp_arrived();                  // BARRIER k.mid

        // Q1: mmac(k,0)
        flash::wait_lds_data_arrived<false>(0);
        flash::wait_all_buffer_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(0);
        // Q2: ds_read A(k,1) then prefetch B(k+1). See C body for rationale.
        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h1);
        if (k_stage + 1 < n_stages) {
            buffer_load_b.template operator()<NXT>(k_stage + 1);
        }
        flash::wait_all_warp_arrived();                  // BARRIER k.end
    };

    // ---------- Prologue ----------
    buffer_load_b.template operator()<0>(0);
    flash::wait_all_warp_arrived();                      // BARRIER prologue.mid

    flash::wait_all_buffer_data_arrived<false>(0);
    flash::wait_all_warp_arrived();                      // BARRIER prologue.end

    const bool is_C_prime = consumer_id >= 4;

    // ---------- Main loop ----------
    if (!is_C_prime) {
        #pragma unroll
        for (int k = 0; k < n_stages; k += 2) {
            stage_body_C.template operator()<0>(k);
            if (k + 1 < n_stages) {
                stage_body_C.template operator()<1>(k + 1);
            }
        }
    } else {
        // C' owes a trailing Q3 whose BUF depends on which CUR the last stage
        // used. We split by n_stages parity so both the trailing Q3 AND the
        // branch-entry b_regs liveness stay compile-time — a runtime BUF (or
        // hoisting the branch below the loop) forces both b_regs[0] and
        // b_regs[1] live through the tail, costing ~30% TFLOPS.
        if ((n_stages & 1) != 0) {
            // Odd n_stages: loop covers all stages; last CUR=0 → tail is Q3<0>.
            stage_body_Cp.template operator()<0, true>(0);
            #pragma unroll
            for (int k = 1; k < n_stages - 1; k += 2) {
                stage_body_Cp.template operator()<1, false>(k);
                stage_body_Cp.template operator()<0, false>(k + 1);
            }
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<0>(1);         // epilogue Q3
        } else {
            // Even n_stages: loop covers n-1 stages; one more with CUR=1,
            //                then last CUR=1 → tail is Q3<1>.
            stage_body_Cp.template operator()<0, true>(0);
            #pragma unroll
            for (int k = 1; k < n_stages - 1; k += 2) {
                stage_body_Cp.template operator()<1, false>(k);
                stage_body_Cp.template operator()<0, false>(k + 1);
            }
            stage_body_Cp.template operator()<1, false>(n_stages - 1);
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<1>(1);         // epilogue Q3
        }
    }

    // b_regs are dead after the final MMAC. When has_scale (runtime), the
    // producer staged the per-row fp32 scales (sa[16], sb[8]) into the dead A
    // buffer's first 2 KB during its idle tail; read them back via LDS and
    // fence with one extra WG-wide barrier before the epilogue reuses LDS as
    // scratch. The branch is uniform — the extra barrier is executed by every
    // wave iff the producer ran the 3-barrier scale tail.
    if (has_scale) {
        const uint32_t scale_lds_base =
            (uint32_t)((n_stages & 1) * (int)GemmTraitsPC::A_stage_bytes);
        const uint32_t sa_base = scale_lds_base + (uint32_t)(row_in_tile * 4);
        const uint32_t sb_base = scale_lds_base + 1024u
                               + (uint32_t)((nj * 32 + col_group * 4) * 4);
        ds_read_scale_pc(
            sa_base, sb_base,
            b_regs[0].f[0],  b_regs[0].f[1],  b_regs[0].f[2],  b_regs[0].f[3],
            b_regs[0].f[4],  b_regs[0].f[5],  b_regs[0].f[6],  b_regs[0].f[7],
            b_regs[0].f[8],  b_regs[0].f[9],  b_regs[0].f[10], b_regs[0].f[11],
            b_regs[0].f[12], b_regs[0].f[13], b_regs[0].f[14], b_regs[0].f[15],
            b_regs[1].d4[0], b_regs[1].d4[1]);
        flash::wait_lds_data_arrived<true>(0);   // s_waitcnt lgkmcnt(0); s_barrier
    }

    // ---------- Epilogue: LDS-staged coalesced writeback ----------
    // A LDS is free after the final barrier — reuse the whole 64 KB as
    // per-wave scratch (8 KB / consumer wave). Fork on output dtype: bf16/fp16
    // downcast + 2-pass path, fp32 identity + 4-slice path (see the two
    // epilogue definitions above for the layouts).
    Element_O* d_ptr = reinterpret_cast<Element_O*>(params.d_ptr);
    Element_O* epi_lds = reinterpret_cast<Element_O*>(a_lds_all);
    if constexpr (sizeof(Element_O) == 2) {
        gemm_epilogue_writeback_via_lds_pc<Element_O, kBlockM, kWaveN>(
            d_ptr, ldd, M, N,
            /*row_base_global=*/by * kBlockM,
            /*col_base_global=*/b_col_start,
            consumer_id, lane_id,
            acc_o, epi_lds, b_regs, has_scale,
            params.alpha, params.beta);
    } else {
        gemm_epilogue_writeback_f32<Element_O, kBlockM, kWaveN>(
            d_ptr, ldd, M, N,
            /*row_base_global=*/by * kBlockM,
            /*col_base_global=*/b_col_start,
            consumer_id, lane_id,
            acc_o, epi_lds, b_regs, has_scale,
            params.alpha, params.beta);
    }
}

// ============================================================================
// Kernel entry
// ============================================================================
// Select one batch's base pointers. The kernel grid is (tiles_n, tiles_m,
// batch), so blockIdx.z identifies the batch. All row/column tile math inside
// producer_body/consumer_body is unchanged — it operates relative to the
// returned per-batch bases. Batch strides are element counts (0 = the buffer
// is shared across batches, e.g. a scale reused by every batch).
template<typename Element_O>
__device__ __forceinline__ DeepGemm_params batch_params_for_block(
    const DeepGemm_params& params) {
    DeepGemm_params batch_params = params;
    const int64_t batch_idx = blockIdx.z;
    if (batch_idx == 0) return batch_params;   // common single-GEMM fast path
    batch_params.a_ptr = reinterpret_cast<fp8_e4m3*>(params.a_ptr)
                       + batch_idx * params.a_batch_stride;
    batch_params.b_ptr = reinterpret_cast<fp8_e4m3*>(params.b_ptr)
                       + batch_idx * params.b_batch_stride;
    batch_params.d_ptr = reinterpret_cast<Element_O*>(params.d_ptr)
                       + batch_idx * params.d_batch_stride;
    if (params.scale_a_ptr != nullptr) {
        batch_params.scale_a_ptr = reinterpret_cast<float*>(params.scale_a_ptr)
                                 + batch_idx * params.scale_a_batch_stride;
    }
    if (params.scale_b_ptr != nullptr) {
        batch_params.scale_b_ptr = reinterpret_cast<float*>(params.scale_b_ptr)
                                 + batch_idx * params.scale_b_batch_stride;
    }
    return batch_params;
}

template<typename Element_O>
__device__ void compute_fp8_gemm_hipc(const DeepGemm_params& params) {
    using Element = GemmTraitsPC::Element;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int warp_id = __builtin_hcu_readfirstlane(threadIdx.x / 64);
    const int lane_id = threadIdx.x & 63;

    extern __shared__ Element smem[];
    Element* a_lds_all = smem;

    const DeepGemm_params batch_params =
        batch_params_for_block<Element_O>(params);

    if (warp_id < GemmTraitsPC::kNProducerWaves) {
        producer_body(batch_params, warp_id, a_lds_all, by, bx, lane_id);
    } else {
        int consumer_id = warp_id - GemmTraitsPC::kNProducerWaves;
        consumer_body<Element_O>(batch_params, consumer_id, a_lds_all, by, bx, lane_id);
    }
}

// amdgpu_waves_per_eu(1, 1): pin BOTH min AND max to 1.
// The single-arg form (1) only sets the minimum — the compiler still tries
// to hit peak occupancy (8 waves/EU on gfx938) and starts spilling as soon
// as VGPR use crosses ~32/lane. Consumer B ping-pong (b_regs[2] = 32 VGPR)
// pushes total to acc(128) + A(64) + B(32) ≈ 224 VGPR, comfortably under
// the 256 cap but well past the occupancy break-even → without a max the
// compiler spills every extra buffer. 768-thread WG already burns 12 wave
// slots per CU, so more waves/EU is not achievable anyway.
template<typename Element_O>
__global__ void
__attribute__((amdgpu_flat_work_group_size(1, GemmTraitsPC::kNThreads),
               amdgpu_waves_per_eu(1, 1)))
fp8_gemm_hipc_kernel_gfx938(DeepGemm_params params) {
    compute_fp8_gemm_hipc<Element_O>(params);
}

// One kernel per output dtype; scaled vs identity is a runtime branch (see
// run_fp8_gemm_hipc_impl).
template __global__ void fp8_gemm_hipc_kernel_gfx938<bhalf_t>(DeepGemm_params);
template __global__ void fp8_gemm_hipc_kernel_gfx938<half_t>(DeepGemm_params);
template __global__ void fp8_gemm_hipc_kernel_gfx938<float>(DeepGemm_params);

template<typename Element_O>
void run_fp8_gemm_hipc_impl(DeepGemm_params& params, hipStream_t stream) {
    int block_y = (params.m + GemmTraitsPC::kBlockM - 1) / GemmTraitsPC::kBlockM;
    int block_x = (params.n + GemmTraitsPC::kBlockN - 1) / GemmTraitsPC::kBlockN;
    // Grid z = batch: each blockIdx.z picks one batch's base pointers via
    // batch_params_for_block.
    const int batch_count = params.batch > 0 ? params.batch : 1;
    dim3 grid(block_x, block_y, batch_count);
    dim3 block(GemmTraitsPC::kNThreads);
    // Scaled vs identity is a runtime branch on params.scale_a_ptr — uniform
    // across the WG, so barrier counts match. No per-scale kernel copies.
    fp8_gemm_hipc_kernel_gfx938<Element_O>
        <<<grid, block, GemmTraitsPC::kSmemSize, stream>>>(params);
}

// Public entry point. See fp8_gemm_hipc.h for API semantics and the mat_a↔mat_b /
// scale_a↔scale_b swap inherited from deepgemm::fp8_gemm. This file just
// wires the torch tensors into DeepGemm_params, applies shape/scale checks,
// and dispatches on output dtype.
//
// Scale tensors may be shared by every batch ([rows]) or per-batch
// ([batch, rows]); an empty/undefined scale = identity (null ptr in
// DeepGemm_params, kernel skips the multiply).
void run_fp8_gemm_hipc(
                    torch::Tensor mat_a,
                    torch::Tensor mat_b,
                    torch::Tensor scale_a,
                    torch::Tensor scale_b,
                    torch::Tensor output,
                    const int64_t m,
                    const int64_t n,
                    const int64_t k,
                    const int64_t batch,
                    const std::string& /*transpose_flag*/,
                    const torch::Tensor& alpha,
                    const torch::Tensor& beta,
                    const std::optional<torch::Tensor>& /*bias*/) {
    // Roles after the swap: mat_b = true A (n rows, → p.m); mat_a = true B
    // (m rows, → p.n); output = [n, m]. scale_b scales true A (n per batch),
    // scale_a scales true B (m per batch).
    TORCH_CHECK(batch >= 1 && batch <= 65535,
                "batch must be in [1, 65535] (gridDim.z limit), got ", batch);

    auto check_tile = [&](const torch::Tensor& t, const char* name,
                          int64_t rows, int64_t cols) {
        if (batch == 1) {
            TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
                        name, " must be [", rows, ",", cols,
                        "] for batch=1, got ", t.sizes());
        } else {
            TORCH_CHECK(t.dim() == 3 && t.size(0) == batch &&
                            t.size(1) == rows && t.size(2) == cols,
                        name, " must be [", batch, ",", rows, ",", cols,
                        "], got ", t.sizes());
        }
    };
    check_tile(mat_b, "mat_b (true A)", n, k);
    check_tile(mat_a, "mat_a (true B)", m, k);
    check_tile(output, "output", n, m);

    // An "unset" scale is signaled by an undefined or 0-element tensor
    // (treated as identity, skipped in the kernel). Any set scale must be
    // contiguous fp32 with `rows` (shared by all batches) or `batch*rows`
    // (per-batch) elements. Reject a single-sided scale.
    const bool has_sa = scale_a.defined() && scale_a.numel() > 0;
    const bool has_sb = scale_b.defined() && scale_b.numel() > 0;
    TORCH_CHECK(has_sa == has_sb,
                "scale_a and scale_b must both be set or both empty");

    int64_t scale_a_batch_stride = 0;   // internal-B scale = arg scale_a (m rows)
    int64_t scale_b_batch_stride = 0;   // internal-A scale = arg scale_b (n rows)
    auto check_scale = [&](const torch::Tensor& s, const char* name,
                           int64_t rows, int64_t& batch_stride) {
        if (!(s.defined() && s.numel() > 0)) return;
        TORCH_CHECK(s.scalar_type() == torch::kFloat32, name, " must be float32");
        if (s.numel() == rows) {
            batch_stride = 0;              // shared by every batch
        } else {
            TORCH_CHECK(s.numel() == batch * rows,
                        name, " must have ", rows, " (shared) or ", batch * rows,
                        " (per-batch) elements, got ", s.numel());
            batch_stride = rows;           // per-batch [batch, rows]
        }
    };
    check_scale(scale_a, "scale_a", m, scale_a_batch_stride);
    check_scale(scale_b, "scale_b", n, scale_b_batch_stride);

    DeepGemm_params p{};
    p.a_ptr    = mat_b.data_ptr();          // true A
    p.b_ptr    = mat_a.data_ptr();          // true B
    p.d_ptr    = output.data_ptr();
    p.m        = static_cast<int>(n);       // internal A rows (= true M)
    p.n        = static_cast<int>(m);       // internal B rows (= true N)
    p.k        = static_cast<int>(k);
    p.a_stride = static_cast<int>(k);
    p.b_stride = static_cast<int>(k);
    p.d_stride = static_cast<int>(m);       // output inner dim (= true N)
    // Scale-pointer mapping mirrors the mat_a↔mat_b swap: p.a_ptr = mat_b, so
    // p.scale_a_ptr (for internal-A rows, size p.m) = arg scale_b, and
    // p.scale_b_ptr (for internal-B rows, size p.n) = arg scale_a.
    p.scale_a_ptr = has_sb ? scale_b.data_ptr() : nullptr;
    p.scale_b_ptr = has_sa ? scale_a.data_ptr() : nullptr;
    // Batch layout (element strides of the contiguous per-batch tensors).
    p.batch                = static_cast<int>(batch);
    p.a_batch_stride       = static_cast<int64_t>(n) * k;   // true A [B,M,K]
    p.b_batch_stride       = static_cast<int64_t>(m) * k;   // true B [B,N,K]
    p.d_batch_stride       = static_cast<int64_t>(n) * m;   // output [B,M,N]
    p.scale_a_batch_stride = scale_b_batch_stride;          // 0 = shared
    p.scale_b_batch_stride = scale_a_batch_stride;
    // Epilogue scalars: single fp32 value for the whole (batched) GEMM.
    // alpha=1 / beta=0 (the defaults) keep the exact original epilogue path.
    p.alpha = (alpha.defined() && alpha.numel() > 0) ? alpha.item<float>() : 1.0f;
    p.beta  = (beta.defined()  && beta.numel()  > 0) ? beta.item<float>()  : 0.0f;
    p.cu_count = 1;

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // Dispatch on the output tensor's dtype. All dtypes share the same
    // producer/consumer/mmac layout; the epilogue differs only in the final
    // fp32→output step (bf16/fp16 downcast, fp32 identity). fp16 callers must
    // ensure |A·B^T| < ~65504 (fp16 max) — bf16/fp32 tolerate the fp32 range.
    switch (output.scalar_type()) {
        case torch::kBFloat16:
            run_fp8_gemm_hipc_impl<bhalf_t>(p, stream);
            break;
        case torch::kHalf:
            run_fp8_gemm_hipc_impl<half_t>(p, stream);
            break;
        case torch::kFloat32:
            run_fp8_gemm_hipc_impl<float>(p, stream);
            break;
        default:
            TORCH_CHECK(false,
                "producer-consumer fp8 gemm output must be bfloat16, float16 or float32, got ",
                output.scalar_type());
    }
}

// ============================================================================
// deepgemm::fp8_gemm_hipc — op-level entry bound to Python as op.fp8_gemm_hipc.
// ============================================================================
// Thin namespaced forwarder over the raw run_fp8_gemm_hipc above so the pybind
// module (deepgemm_ops.cpp) can reference &deepgemm::fp8_gemm_hipc the same
// way it references &deepgemm::fp8_gemm. Signature / semantics are identical
// (incl. the mat_a<->mat_b and scale_a<->scale_b swap documented above).
namespace deepgemm {
void fp8_gemm_hipc(
                    torch::Tensor mat_a,
                    torch::Tensor mat_b,
                    torch::Tensor scale_a,
                    torch::Tensor scale_b,
                    torch::Tensor output,
                    const int64_t m,
                    const int64_t n,
                    const int64_t k,
                    const int64_t batch,
                    const std::string& transpose_flag,
                    const torch::Tensor& alpha,
                    const torch::Tensor& beta,
                    const std::optional<torch::Tensor>& bias) {
    ::run_fp8_gemm_hipc(mat_a, mat_b, scale_a, scale_b, output,
                        m, n, k, batch, transpose_flag, alpha, beta, bias);
}
}  // namespace deepgemm
