// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// int8 variant of m_grouped_fp8_gemm_nt_contiguous_wasp_pair32/bm128_bn256_nloop4.inc.cuh
// Identical logic (producer/consumer wasp split, pair32 B payload layout, 2-stage
// A pipeline, per-M-row / per-N-col scale dequant); only the mmac instruction and
// accumulator type differ: i32 accumulation via mmac_i32_16x16x32_i8.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

namespace gfx92a::m_grouped_int8_gemm_nt_contiguous_wasp_pair32_bm128_bn256_nloop4{

using bhalf_t   = __hip_bfloat16;

struct DeepGemm_params {
    void* __restrict__ a_ptr;
    void* __restrict__ b_ptr;
    void* __restrict__ d_ptr;
    void* __restrict__ scale_a_ptr;
    void* __restrict__ scale_b_ptr;
    const int32_t* __restrict__ m_indices;
    int m;
    int n;
    int k;
    int a_stride;
    int b_stride;
    int d_stride;
    int experts;
    float alpha;
    float beta;
};

using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using vec4_int8 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;
using vec8_int8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using vec4_int  = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using vec2_bf16 = __attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short;

union union_vec16_i8 {
    int8_t   i8[16];
    vec8_int8 i8x8[2];
    vec4_int8 i8x4[4];
    vec4_fp32 f32x4;
    vec4_int  i32x4;
    int32_t   i32[4];
};


struct GemmTraitsPC {
    using Element      = int8_t;

    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 256;
    static constexpr int kStageK = 128;

    static constexpr int kWaveN  = 32;

    static constexpr int kNConsumerWaves = 8;
    static constexpr int kNProducerWaves = 4;
    static constexpr int kNWaves         = kNConsumerWaves + kNProducerWaves;
    static constexpr int kNThreads       = kNWaves * 64;

    static constexpr int kProdSlabM = kBlockM / kNProducerWaves;


    static constexpr int STAGES           = 2;
    static constexpr size_t A_slab_bytes  = kProdSlabM * kStageK;
    static constexpr size_t A_stage_bytes = kNProducerWaves * A_slab_bytes;
    static constexpr int    kSmemSize     = STAGES * A_stage_bytes;
};


__device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ int pair32_lane_id_after_vgpr_resize() {
    return threadIdx.x & 63;
}

namespace flash {

// Named ebarrier arrive: this wave arrives at ebarrier `ID`; the barrier
// releases when `WAVE_CNT` waves have arrived in total.
template<int ID, int WAVE_CNT>
__forceinline__ __device__ void s_ebarrier_arrive() {
    static_assert(ID >= 0 && ID <= 15, "ebarrier_id must fit the HW range");
    static_assert(WAVE_CNT >= 1, "wave_cnt must be at least 1");
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_ebarrier_arrive %0, %1\n" :: "n"(ID), "n"(WAVE_CNT));
    __builtin_hcu_sched_barrier(0);
}

// Named ebarrier wait: block until ebarrier `ID` has collected all arrivals.
template<int ID>
__forceinline__ __device__ void s_ebarrier_wait() {
    static_assert(ID >= 0 && ID <= 15, "ebarrier_id must fit the HW range");
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_ebarrier_wait %0\n" :: "n"(ID));
    __builtin_hcu_sched_barrier(0);
}

// Named ebarrier sync: arrive at `ID` and wait for the other `WAVE_CNT - 1`
// waves (equivalent to the pseudocode's s_ebarrier_sync).
template<int ID, int WAVE_CNT>
__forceinline__ __device__ void s_ebarrier_sync() {
    static_assert(ID >= 0 && ID <= 15, "ebarrier_id must fit the HW range");
    static_assert(WAVE_CNT >= 1, "wave_cnt must be at least 1");
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_ebarrier_sync %0, %1"
                 :: "n"(ID), "n"(WAVE_CNT));
    __builtin_hcu_sched_barrier(0);
}

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


template<int PRIO>
__forceinline__ __device__ void set_prio() {
    static_assert(PRIO >= 0 && PRIO <= 3, "s_setprio operand must be 0..3");
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_setprio %0" :: "n"(PRIO));
    __builtin_hcu_sched_barrier(0);
}

}

__device__ __forceinline__ vec4_int mmac_4interleave_i8(
    const vec8_int8& v1, const vec8_int8& v2, const vec4_int& v3
) {
    return __builtin_hcu_mmac_i32_16x16x32_i8(v1, v2, v3);
}

__host__ __device__ __forceinline__ vec2_bf16 DownCastPair_f32_bf16(const vec2_fp32& source) {
    auto result = __builtin_hcu_cvt_pk_bf16_f32(source[0], source[1], false);
    return *(vec2_bf16*)(&result);
}


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
        uint32_t val = 1u << 30;
        val |= (stride & 0x3fff) << 16;
        r[1] |= val;
    }

    return r;
}


__device__ __forceinline__ void buffer_load_dwordx4_x4_to_lds_soff(
    buffer_rsrc_t rsrc,
    uint32_t voff,
    uint32_t lds_m0,
    uint32_t s_soff0, uint32_t s_soff1, uint32_t s_soff2, uint32_t s_soff3) {
    asm volatile(
        "s_mov_b32 m0, %1\n\t"
        "buffer_load_dwordx4 %0, %2, %3 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %4 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %5 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %6 offen lds\n"
        :
        : "v"(voff), "s"(lds_m0), "s"(rsrc),
          "s"(s_soff0), "s"(s_soff1), "s"(s_soff2), "s"(s_soff3)
        : "memory", "m0");
}


__device__ __forceinline__ void buffer_load_i32x4_pair32(
    const char* base, dword4_t& value, int byte_offset) {
    __builtin_hcu_sched_barrier(0);
    using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    intx4 resource;
    *reinterpret_cast<uint64_t*>(&resource) = reinterpret_cast<uint64_t>(base);
    resource[1] += 0x40800000;
    resource[2] = 0x80000000;
    resource[3] = 0x00020000;
    asm volatile(
        "s_nop 2\n\t"
        "buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n"
        : "=v"(value), "+v"(byte_offset), "+s"(resource)
        :
        : "memory");
}


template<int EXTRA = 0>
__device__ __forceinline__ void ds_read_b128_x8_imm(
    uint32_t base,
    dword4_t& o0,  dword4_t& o1,  dword4_t& o2,  dword4_t& o3,
    dword4_t& o4,  dword4_t& o5,  dword4_t& o6,  dword4_t& o7) {
    static_assert(EXTRA >= 0 && EXTRA + 14336 <= 65535,
                  "ds_read_b128 imm offset overflows 16-bit unsigned field");
    asm volatile(
        "ds_read_b128 %0, %8  offset:%c9\n\t"
        "ds_read_b128 %1, %8  offset:%c10\n\t"
        "ds_read_b128 %2, %8  offset:%c11\n\t"
        "ds_read_b128 %3, %8  offset:%c12\n\t"
        "ds_read_b128 %4, %8  offset:%c13\n\t"
        "ds_read_b128 %5, %8  offset:%c14\n\t"
        "ds_read_b128 %6, %8  offset:%c15\n\t"
        "ds_read_b128 %7, %8  offset:%c16\n"
        : "=&v"(o0), "=&v"(o1), "=&v"(o2), "=&v"(o3),
          "=&v"(o4), "=&v"(o5), "=&v"(o6), "=&v"(o7)
        : "v"(base),
          "n"(EXTRA + 0),     "n"(EXTRA + 2048),  "n"(EXTRA + 4096),  "n"(EXTRA + 6144),
          "n"(EXTRA + 8192),  "n"(EXTRA + 10240), "n"(EXTRA + 12288), "n"(EXTRA + 14336)
        : "memory");
}


union union_vec16_dword {
    dword4_t   d4[4];
    vec8_int8  v8[8];
    float      f[16];
};


__device__ __forceinline__ void ds_read_scale_pc(
    uint32_t sa_base, uint32_t sb_base,
    float& a0,  float& a1,  float& a2,  float& a3,
    float& a4,  float& a5,  float& a6,  float& a7,
    dword4_t& b0, dword4_t& b1)
{
    asm volatile(
        "ds_read_b32 %0, %10 offset:0\n\t"
        "ds_read_b32 %1, %10 offset:64\n\t"
        "ds_read_b32 %2, %10 offset:128\n\t"
        "ds_read_b32 %3, %10 offset:192\n\t"
        "ds_read_b32 %4, %10 offset:256\n\t"
        "ds_read_b32 %5, %10 offset:320\n\t"
        "ds_read_b32 %6, %10 offset:384\n\t"
        "ds_read_b32 %7, %10 offset:448\n\t"
        "ds_read_b128 %8, %11 offset:0\n\t"
        "ds_read_b128 %9, %11 offset:16\n"
        : "=&v"(a0), "=&v"(a1), "=&v"(a2), "=&v"(a3),
          "=&v"(a4), "=&v"(a5), "=&v"(a6), "=&v"(a7),
          "=&v"(b0), "=&v"(b1)
        : "v"(sa_base), "v"(sb_base)
        : "memory");
}


using bf16x8_bits =
    __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
union bf16x8_pair_bits {
    bf16x8_bits u16x8;
    vec2_bf16 pair[4];
};

template<typename Element_O>
__device__ __forceinline__ void pair32_writeback_bf16(
    Element_O* output,
    int ldd,
    int M,
    int N,
    int row_base,
    int col_base,
    int consumer_id,
    int lane_id,
    const vec4_int* acc_o,
    const union_vec16_dword* scale_regs,
    float alpha) {
    static_assert(sizeof(Element_O) == 2,
                  "pair32 WASP currently writes a 2-byte output");
    const int lane_row = lane_id & 15;
    const int lane_group = lane_id >> 4;
    const int col = col_base + consumer_id * GemmTraitsPC::kWaveN
                  + lane_group * 8;
    (void)M;
    (void)N;

#pragma unroll
    for (int m_loop = 0; m_loop < GemmTraitsPC::kBlockM / 16; ++m_loop) {
        const int row = row_base + m_loop * 16 + lane_row;
        const float row_scale = scale_regs[0].f[m_loop];
        bf16x8_pair_bits out_bits;
        const vec2_fp32 row_alpha = {row_scale * alpha, row_scale * alpha};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            vec2_fp32 scaled = {
                static_cast<float>(acc_o[m_loop * 2][r]),
                static_cast<float>(acc_o[m_loop * 2 + 1][r]),
            };
            const vec2_fp32 col_scale = {
                scale_regs[1].f[2 * r],
                scale_regs[1].f[2 * r + 1],
            };
            scaled = scaled * row_alpha;
            scaled = scaled * col_scale;
            out_bits.pair[r] = DownCastPair_f32_bf16(scaled);
        }
        *reinterpret_cast<bf16x8_bits*>(
            output + static_cast<int64_t>(row) * ldd + col) = out_bits.u16x8;
        __builtin_hcu_sched_barrier(0);
    }
}


__device__ __forceinline__ void producer_body(
    const DeepGemm_params& params,
    int prod_id,
    int8_t* a_lds_all,
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
        reinterpret_cast<int8_t*>(params.a_ptr), lda,
        (uint32_t)(params.m * lda));

    const int n_stages = ceil_div(K, kStageK);
    const int row_group = lane_id >> 3;
    const int k_in_panel = ((lane_id & 7) ^ row_group) * 16;

    const uint32_t stride_8rows = (uint32_t)(8 * lda);


    const uint32_t v_off = (uint32_t)((a_row_start + row_group) * lda + k_in_panel);


    uint32_t s_stride[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) s_stride[i] = (uint32_t)i * stride_8rows;

    constexpr uint32_t A_stage_bytes_ct = (uint32_t)GemmTraitsPC::A_stage_bytes;
    const uint32_t prod_slab_off = (uint32_t)(prod_id * (int)GemmTraitsPC::A_slab_bytes);

    const int32_t expert_id =
        __builtin_hcu_readfirstlane(params.m_indices[by * kBlockM]);
    const uint32_t scale_lds_base =
        (uint32_t)((n_stages & 1) * (int)GemmTraitsPC::A_stage_bytes);
    // sa: kProdSlabM (=32) slots per producer, guarded by lane_id < kProdSlabM
    const uint32_t tid4_sa = (uint32_t)(prod_id * kProdSlabM * 4)
                           + (uint32_t)(lane_id & (kProdSlabM - 1)) * 4;
    // sb: all 64 lanes of the wave, 64 slots per producer (BN=256 / 4 producers)
    const uint32_t tid4_sb = (uint32_t)(lane_id * 4) + (uint32_t)(prod_id * 256);
    const uint32_t ga_sa = (uint32_t)(by * kBlockM * 4) + tid4_sa;
    const uint32_t ga_sb = (uint32_t)(bx * kBlockN * 4) + tid4_sb;
    uint32_t la = 0;

    asm volatile("v_add_u32_e32 %0, %1, %2" : "=v"(la) : "v"(tid4_sb), "v"(scale_lds_base));
    buffer_rsrc_t sa_rsrc = make_buffer_rsrc<false>(
        reinterpret_cast<const float*>(params.scale_a_ptr), 4, (uint32_t)(params.m * 4));
    buffer_rsrc_t sb_rsrc = make_buffer_rsrc<false>(
        reinterpret_cast<const float*>(params.scale_b_ptr)
            + static_cast<int64_t>(expert_id) * params.n,
        4, (uint32_t)(params.n * 4));

    const int k_last_stage = n_stages - 1;
    const int k_left_last  = K - k_last_stage * kStageK;
    const int tail_chunk   = k_left_last / 16;
    const int tail_bytes   = k_left_last % 16;
    const uint32_t v_off_tail =
        (k_left_last < kStageK && k_in_panel >= k_left_last)
            ? (v_off | 0x80000000u)
            : v_off;


    auto zero_tail = [&](int s) {
        if (tail_bytes == 0 || k_in_panel != tail_chunk * 16) return;
        const uint32_t stage_base = (uint32_t)reinterpret_cast<size_t>(a_lds_all)
                                   + prod_slab_off + (uint32_t)s * A_stage_bytes_ct;
        const uint32_t zero = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
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


        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    };

    auto issue_stage = [&](int k_stage, int s, uint32_t v_off_use) {
        const uint32_t s_col_base = (uint32_t)(k_stage * kStageK);
        const uint32_t lds_m0 = prod_slab_off + s * A_stage_bytes_ct;
        buffer_load_dwordx4_x4_to_lds_soff(
            a_rsrc, v_off_use, lds_m0,
            s_col_base + s_stride[0], s_col_base + s_stride[1],
            s_col_base + s_stride[2], s_col_base + s_stride[3]);
    };


    if (n_stages == 1) {
        issue_stage(0, 0, v_off_tail);
    } else {
        issue_stage(0, 0, v_off);
    }
    flash::wait_all_warp_arrived();
    flash::wait_all_buffer_data_arrived<false>(0);
    if (n_stages == 1) zero_tail(0);
    flash::wait_all_warp_arrived();


    #pragma unroll 1
    for (int k_stage = 0; k_stage + 2 < n_stages; ++k_stage) {
        const int stage_to_issue = k_stage + 1;
        const int s = stage_to_issue & 1;
        issue_stage(stage_to_issue, s, v_off);
        // flash::wait_all_warp_arrived();
        flash::wait_all_buffer_data_arrived<false>(0);
        flash::wait_all_warp_arrived();
        // flash::s_ebarrier_sync<1,12>();
    }


    if (n_stages >= 2) {
        const int s = k_last_stage & 1;
        issue_stage(k_last_stage, s, v_off_tail);
        // flash::wait_all_warp_arrived();
        flash::wait_all_buffer_data_arrived<false>(0);
        zero_tail(s);
        flash::wait_all_warp_arrived();
        // flash::s_ebarrier_sync<1,12>();
    }

    // flash::wait_all_warp_arrived();
    float sa, sb;
    const uint32_t zero = 0;
    if (lane_id < kProdSlabM) {
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:0\n\t"
                     : "=&v"(sa) : "v"(ga_sa), "s"(sa_rsrc), "s"(zero) : "memory");
    }
    asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:0\n\t"
                 : "=&v"(sb) : "v"(ga_sb), "s"(sb_rsrc), "s"(zero) : "memory");
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    if (lane_id < kProdSlabM) {
        asm volatile("ds_write_b32 %0, %1\n\t" :: "v"(scale_lds_base + tid4_sa), "v"(sa) : "memory");
    }
    asm volatile("ds_write_b32 %0, %1 offset:512\n\t" :: "v"(la), "v"(sb) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
    flash::wait_all_warp_arrived();
    // flash::s_ebarrier_sync<1,12>();
    flash::wait_all_warp_arrived();
}


template<typename Element_O>
__device__ __forceinline__ void consumer_body(
    const DeepGemm_params& params,
    int consumer_id,
    int8_t* a_lds_all,
    int by, int bx,
    int lane_id
) {
    constexpr int kBlockM     = GemmTraitsPC::kBlockM;
    constexpr int kBlockN     = GemmTraitsPC::kBlockN;
    constexpr int kStageK     = GemmTraitsPC::kStageK;
    constexpr int kWaveN      = GemmTraitsPC::kWaveN;
    constexpr int kMmacM      = 16;
    constexpr int kMmacN      = 16;
    constexpr int kSubTilesM  = kBlockM / kMmacM;
    constexpr int kSubTilesN  = kWaveN / kMmacN;

    const int row_in_tile = lane_id & 15;
    const int col_group   = lane_id >> 4;
    const int n_stages = ceil_div(params.k, kStageK);

    const int M = params.m, N = params.n;
    const int ldd = params.d_stride;
    const int b_col_start = bx * kBlockN;
    const int32_t expert_id =
        __builtin_hcu_readfirstlane(params.m_indices[by * kBlockM]);
    const int b_n32_tiles = N / 32;


    vec4_int acc_o[kSubTilesM * kSubTilesN] = {};


    union_vec16_i8 a_regs[kSubTilesM];


    union_vec16_dword b_regs[2];


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
        ds_read_b128_x8_imm<STAGE_OFF>(
            base,
            reinterpret_cast<dword4_t&>(a_regs[0].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[1].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[2].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[3].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[4].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[5].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[6].i32x4),
            reinterpret_cast<dword4_t&>(a_regs[7].i32x4));
    };


    auto mmac_half = [&]<int BUF>(int k_half) {
        const int b_base = k_half * 4;
#pragma unroll
        for (int m = 0; m < kSubTilesM; ++m) {
            acc_o[m * 2] = mmac_4interleave_i8(
                a_regs[m].i8x8[0], b_regs[BUF].v8[b_base + 0],
                acc_o[m * 2]);
            acc_o[m * 2] = mmac_4interleave_i8(
                a_regs[m].i8x8[1], b_regs[BUF].v8[b_base + 1],
                acc_o[m * 2]);
            acc_o[m * 2 + 1] = mmac_4interleave_i8(
                a_regs[m].i8x8[0], b_regs[BUF].v8[b_base + 2],
                acc_o[m * 2 + 1]);
            acc_o[m * 2 + 1] = mmac_4interleave_i8(
                a_regs[m].i8x8[1], b_regs[BUF].v8[b_base + 3],
                acc_o[m * 2 + 1]);
        }
    };


    auto buffer_load_b = [&]<int BUF>(int stage_id) {
        const int load_col_group = pair32_lane_id_after_vgpr_resize() >> 4;
        const int n32_band = bx * (kBlockN / 32) + consumer_id;
        const int64_t tile0 =
            (static_cast<int64_t>(expert_id) * (params.k / 64) +
             stage_id * 2) * b_n32_tiles + n32_band;
        const int64_t row_offset =
            (((tile0 * 2) * 4 + load_col_group) * 16 + row_in_tile) * 16;
        constexpr int kPairStrideBytes = 4 * 16 * 16;

        buffer_load_i32x4_pair32(
            reinterpret_cast<const char*>(params.b_ptr),
            b_regs[BUF].d4[0],
            static_cast<int>(row_offset));
        buffer_load_i32x4_pair32(
            reinterpret_cast<const char*>(params.b_ptr),
            b_regs[BUF].d4[1],
            static_cast<int>(row_offset + kPairStrideBytes));

        const int64_t tile1 = tile0 + b_n32_tiles;
        const int64_t row_offset1 =
            (((tile1 * 2) * 4 + load_col_group) * 16 + row_in_tile) * 16;
        buffer_load_i32x4_pair32(
            reinterpret_cast<const char*>(params.b_ptr),
            b_regs[BUF].d4[2],
            static_cast<int>(row_offset1));
        buffer_load_i32x4_pair32(
            reinterpret_cast<const char*>(params.b_ptr),
            b_regs[BUF].d4[3],
            static_cast<int>(row_offset1 + kPairStrideBytes));
    };


    auto stage_body_C = [&]<int CUR>(int k_stage) {
        constexpr int NXT = CUR ^ 1;
        constexpr int STAGE_OFF = CUR * (int)GemmTraitsPC::A_stage_bytes;

        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h0);

        flash::wait_lds_data_arrived<false>(0);
        flash::wait_all_buffer_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(0);

        // flash::wait_all_warp_arrived();

        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h1);
        if (k_stage + 1 < n_stages) {
            buffer_load_b.template operator()<NXT>(k_stage + 1);
        }

        flash::wait_lds_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(1);
        flash::wait_all_warp_arrived();
        // flash::s_ebarrier_sync<1,12>();
    };


    auto stage_body_Cp = [&]<int CUR, bool IS_FIRST>(int k_stage) {
        constexpr int NXT = CUR ^ 1;
        constexpr int STAGE_OFF = CUR * (int)GemmTraitsPC::A_stage_bytes;

        if constexpr (!IS_FIRST) {
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<NXT>(1);
        }

        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h0);

        // flash::wait_all_warp_arrived();

        flash::wait_lds_data_arrived<false>(0);
        flash::wait_all_buffer_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(0);

        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h1);
        if (k_stage + 1 < n_stages) {
            buffer_load_b.template operator()<NXT>(k_stage + 1);
        }
        flash::wait_all_warp_arrived();
        // flash::s_ebarrier_sync<1,12>();
    };


    buffer_load_b.template operator()<0>(0);
    flash::wait_all_warp_arrived();

    flash::wait_all_buffer_data_arrived<false>(0);
    flash::wait_all_warp_arrived();

    const bool is_C_prime = consumer_id >= 4;


    if (!is_C_prime) {
        #pragma unroll
        for (int k = 0; k < n_stages; k += 2) {
            stage_body_C.template operator()<0>(k);
            if (k + 1 < n_stages) {
                stage_body_C.template operator()<1>(k + 1);
            }
        }
    } else {


        if ((n_stages & 1) != 0) {

            stage_body_Cp.template operator()<0, true>(0);
            #pragma unroll
            for (int k = 1; k < n_stages - 1; k += 2) {
                stage_body_Cp.template operator()<1, false>(k);
                stage_body_Cp.template operator()<0, false>(k + 1);
            }
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<0>(1);
        } else {


            stage_body_Cp.template operator()<0, true>(0);
            #pragma unroll
            for (int k = 1; k < n_stages - 1; k += 2) {
                stage_body_Cp.template operator()<1, false>(k);
                stage_body_Cp.template operator()<0, false>(k + 1);
            }
            stage_body_Cp.template operator()<1, false>(n_stages - 1);
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<1>(1);
        }
    }

    const uint32_t scale_lds_base =
        (uint32_t)((n_stages & 1) * (int)GemmTraitsPC::A_stage_bytes);
    const uint32_t sa_base = scale_lds_base + (uint32_t)(row_in_tile * 4);
    const uint32_t sb_base = scale_lds_base + 512u
                           + (uint32_t)((consumer_id * 32 + col_group * 8) * 4);
    ds_read_scale_pc(
        sa_base, sb_base,
        b_regs[0].f[0],  b_regs[0].f[1],  b_regs[0].f[2],  b_regs[0].f[3],
        b_regs[0].f[4],  b_regs[0].f[5],  b_regs[0].f[6],  b_regs[0].f[7],
        b_regs[1].d4[0], b_regs[1].d4[1]);
    flash::wait_lds_data_arrived<true>(0);


    Element_O* d_ptr = reinterpret_cast<Element_O*>(params.d_ptr);
    const int writeback_lane_id = pair32_lane_id_after_vgpr_resize();
    pair32_writeback_bf16<Element_O>(
        d_ptr, ldd, M, N,
        by * kBlockM,
        b_col_start,
        consumer_id, writeback_lane_id,
        acc_o, b_regs, params.alpha);
}


template<typename Element_O>
__device__ __forceinline__ void compute_int8_pair32_wasp(
    const DeepGemm_params& params) {
    using Element = GemmTraitsPC::Element;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int warp_id = __builtin_hcu_readfirstlane(threadIdx.x / 64);
    const int lane_id = threadIdx.x & 63;

    extern __shared__ Element smem[];
    if (warp_id < GemmTraitsPC::kNProducerWaves) {
        producer_body(params, warp_id, smem, by, bx, lane_id);
    } else {
        consumer_body<Element_O>(
            params, warp_id - GemmTraitsPC::kNProducerWaves,
            smem, by, bx, lane_id);
    }
}

template<typename Element_O>
__global__ __launch_bounds__(256, 1)
void int8_pair32_wasp_kernel(DeepGemm_params params) {
    if (blockIdx.x * GemmTraitsPC::kBlockN >= params.n ||
        blockIdx.y * GemmTraitsPC::kBlockM >= params.m) {
        return;
    }
    if (params.m_indices[blockIdx.y * GemmTraitsPC::kBlockM] < 0) {
        return;
    }
    compute_int8_pair32_wasp<Element_O>(params);
}

torch::Tensor run(
    torch::Tensor input,
    torch::Tensor packed_weight,
    torch::Tensor output,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    torch::Tensor m_indices,
    int64_t block_m,
    int64_t block_n,
    int64_t n_loop_num) {
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    TORCH_CHECK(input.is_cuda() && packed_weight.is_cuda() &&
                    output.is_cuda() && input_scale.is_cuda() &&
                    weight_scale.is_cuda() && m_indices.is_cuda(),
                "all tensors must be CUDA/HIP tensors");
    TORCH_CHECK(input.is_contiguous() && packed_weight.is_contiguous() &&
                    output.is_contiguous() && input_scale.is_contiguous() &&
                    weight_scale.is_contiguous() && m_indices.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kChar,
                "input must be int8");
    TORCH_CHECK(packed_weight.scalar_type() == at::kChar,
                "packed_weight must be int8");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16,
                "WASP pair32 currently supports BF16 output");
    TORCH_CHECK(input_scale.scalar_type() == at::kFloat &&
                    weight_scale.scalar_type() == at::kFloat,
                "scales must be float32");
    TORCH_CHECK(m_indices.scalar_type() == at::kInt,
                "m_indices must be int32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M,K]");
    TORCH_CHECK(packed_weight.dim() == 7,
                "packed_weight must be 7D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [M,N]");
    TORCH_CHECK(m_indices.dim() == 1, "m_indices must be 1D");
    TORCH_CHECK(input_scale.dim() == 1 ||
                    (input_scale.dim() == 2 && input_scale.size(1) == 1),
                "input_scale must be 1D [M] or 2D [M,1]");
    TORCH_CHECK(weight_scale.dim() == 2 ||
                    (weight_scale.dim() == 3 && weight_scale.size(2) == 1),
                "weight_scale must be 2D [E,N] or 3D [E,N,1]");
    TORCH_CHECK(packed_weight.size(3) == 2 &&
                    packed_weight.size(4) == 4 &&
                    packed_weight.size(5) == 16 &&
                    packed_weight.size(6) == 16,
                "packed weight trailing dimensions must be [2,4,16,16]");
    TORCH_CHECK(block_m == 128 && block_n == 256 &&
                    (n_loop_num == 0 || n_loop_num == 4),
                "WASP pair32 is compiled for BM=128, BN=256, N_LOOP_NUM=4");

    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const int E = static_cast<int>(packed_weight.size(0));
    const int N = static_cast<int>(packed_weight.size(2)) * 32;
    TORCH_CHECK(packed_weight.size(1) * 64 == K,
                "packed K does not match input K");
    TORCH_CHECK(output.size(0) == M && output.size(1) == N,
                "output must be [M,N]");
    TORCH_CHECK(input_scale.size(0) == M &&
                    weight_scale.size(0) == E &&
                    weight_scale.size(1) == N &&
                    m_indices.numel() == M,
                "scale/index dimensions do not match input");
    TORCH_CHECK(M % GemmTraitsPC::kBlockM == 0 &&
                    N % GemmTraitsPC::kBlockN == 0 &&
                    K % GemmTraitsPC::kStageK == 0,
                "WASP pair32 currently requires M/N divisible by 128/256 and K by 128");

    DeepGemm_params params{};
    params.a_ptr = reinterpret_cast<void*>(input.data_ptr());
    params.b_ptr = reinterpret_cast<void*>(packed_weight.data_ptr());
    params.d_ptr = reinterpret_cast<void*>(output.data_ptr());
    params.scale_a_ptr = reinterpret_cast<void*>(input_scale.data_ptr());
    params.scale_b_ptr = reinterpret_cast<void*>(weight_scale.data_ptr());
    params.m_indices = m_indices.data_ptr<int32_t>();
    params.m = M;
    params.n = N;
    params.k = K;
    params.a_stride = K;
    params.b_stride = 0;
    params.d_stride = N;
    params.experts = E;
    params.alpha = 1.0f;
    params.beta = 0.0f;

    const dim3 grid(N / GemmTraitsPC::kBlockN,
                    M / GemmTraitsPC::kBlockM, 1);
    const dim3 block(GemmTraitsPC::kNThreads, 1, 1);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    hipLaunchKernelGGL(
        (int8_pair32_wasp_kernel<bhalf_t>),
        grid, block, GemmTraitsPC::kSmemSize, stream, params);
    return output;
}

}
