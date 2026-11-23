// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// BM128 x BN512 x BK64, 512 threads. Select 16/32/64/128 rows per CTA.
// Only waves supplying those A rows load A; all eight waves compute N64.
// A-to-LDS completion and B completion use separate waits.
// Each wave computes up to M128 x N64. Process one N32 band across K64 at a time,
// keeping eight expanded B registers and replacing eight shifts by four 64-bit shifts.
// B keeps the native HIPC W4 shuffle [E,K/64,N/32,4,32,8].
// The signed INT4 nibbles are expanded to INT8 x16; B scales are divided by 16.
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

namespace gfx936_gfx938_shared::m_grouped_w4a8_gemm_nt_contiguous_wasp_bm128_bn512_bk64 {
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
    static constexpr int kBlockM=128, kBlockN=512, kStageK=64;
    static constexpr int kNThreads=512;
    static constexpr int kSmemSize=2*kBlockM*kStageK;
};
__device__ __forceinline__ vec4_int mmac_4interleave_i8(
    const vec8_int8& v1, const vec8_int8& v2, const vec4_int& v3
) {
    return __builtin_hcu_mmac_i32_16x16x32_i8(v1, v2, v3);
}

__host__ __device__ __forceinline__ vec2_bf16 DownCastPair_f32_bf16(const vec2_fp32& source) {
#if defined(__gfx92a__) || defined(__gfx938__)
    auto result = __builtin_hcu_cvt_pk_bf16_f32(source[0], source[1], false);
    return *(vec2_bf16*)(&result);
#else
    const __hip_bfloat16 lo_bf16(source[0]);
    const __hip_bfloat16 hi_bf16(source[1]);
    const __hip_bfloat16_raw lo_raw = lo_bf16;
    const __hip_bfloat16_raw hi_raw = hi_bf16;
    uint32_t packed = static_cast<uint32_t>(lo_raw.x) |
                      (static_cast<uint32_t>(hi_raw.x) << 16);
    return *(vec2_bf16*)(&packed);
#endif
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



using bf16x8_bits=__attribute__((__vector_size__(8*sizeof(uint16_t)))) uint16_t;
union bf16x8_pair_bits { bf16x8_bits u16x8; vec2_bf16 pair[4]; };

__device__ __forceinline__ void barrier() {
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_barrier\n" ::: "memory");
    __builtin_hcu_sched_barrier(0);
}

template<typename Element_O, int M_TILES>
__device__ __forceinline__
void compute_tile(DeepGemm_params p, int expert) {
    const int bx=blockIdx.x, by=blockIdx.y;
    const int tid=threadIdx.x, lane=tid&63;
    const int wave=__builtin_hcu_readfirstlane(tid/64);
    const int wm=0, wn=wave;
    const int row=lane&15, cg=lane>>4;
    extern __shared__ int8_t smem[];
    const uint32_t lds_base=uint32_t(reinterpret_cast<uintptr_t>(smem));
    buffer_rsrc_t ar=make_buffer_rsrc(reinterpret_cast<int8_t*>(p.a_ptr),p.k,uint32_t(p.m)*uint32_t(p.k));
    const int arow=by*128+tid/4;
    const uint32_t av=uint32_t(arow)*uint32_t(p.k)+uint32_t(((tid%4)^((tid/4)&3))*16);
    const char* be=reinterpret_cast<const char*>(p.b_ptr)+int64_t(expert)*p.n*p.k/2;
    buffer_rsrc_t br=make_buffer_rsrc<false>(be,1,0x80000000u);
    br[1]+=0x40800000;
    const int n32=p.n/32;
    vec4_int acc[M_TILES][4];
    dword4_t packed[2][2];
    union Reg8 { dword2_t d2; vec8_int8 i8; } bd[4];
    union Reg16 { dword4_t d4; vec8_int8 i8[2]; } a[M_TILES];

    auto load_a=[&](int stage) {
        const uint32_t m0=(stage&1)*8192+wave*1024;
        const uint32_t off=stage*64;
        if (wave < M_TILES) asm volatile("s_mov_b32 m0,%1\n\tbuffer_load_dwordx4 %0,%2,%3 offen lds\n"
                     : : "v"(av),"s"(m0),"s"(ar),"s"(off) : "m0","memory");
        __builtin_hcu_sched_barrier(0);
    };
    const uint32_t b_lane=uint32_t(cg*256+row*16);
    auto load_b=[&]<int BUF>(int stage) {
        const uint32_t soff=__builtin_hcu_readfirstlane(uint32_t((stage*n32+bx*16+wn*2)*1024));
        auto* r=packed[BUF];
        asm volatile("s_nop 2\n\t"
                     "buffer_load_dwordx4 %0,%2,%3,%4 offen offset:0\n\t"
                     "buffer_load_dwordx4 %1,%2,%3,%4 offen offset:1024\n"
                     : "=&v"(r[0]),"=&v"(r[1]) : "v"(b_lane),"s"(br),"s"(soff) : "memory");
        __builtin_hcu_sched_barrier(0);
    };
    auto wait_b_a=[&]<int BUF>() {
        auto* r=packed[BUF];
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n" : "+v"(r[0]),"+v"(r[1]) : : "memory");
        __builtin_hcu_sched_barrier(0);
    };
    auto read_a=[&]<int BUF>() {
        const uint32_t off=lds_base+BUF*8192+row*64+((cg^(row&3))*16);
        auto* r=a;
        if constexpr (M_TILES == 8) {
            asm volatile(
                "ds_read_b128 %0, %8 offset:0\n\t"
                "ds_read_b128 %1, %8 offset:1024\n\t"
                "ds_read_b128 %2, %8 offset:2048\n\t"
                "ds_read_b128 %3, %8 offset:3072\n\t"
                "ds_read_b128 %4, %8 offset:4096\n\t"
                "ds_read_b128 %5, %8 offset:5120\n\t"
                "ds_read_b128 %6, %8 offset:6144\n\t"
                "ds_read_b128 %7, %8 offset:7168\n\t"
                : "=&v"(r[0].d4), "=&v"(r[1].d4), "=&v"(r[2].d4), "=&v"(r[3].d4), "=&v"(r[4].d4), "=&v"(r[5].d4), "=&v"(r[6].d4), "=&v"(r[7].d4)
                : "v"(off) : "memory");
        }
        else if constexpr (M_TILES == 4) {
            asm volatile(
                "ds_read_b128 %0, %4 offset:0\n\t"
                "ds_read_b128 %1, %4 offset:1024\n\t"
                "ds_read_b128 %2, %4 offset:2048\n\t"
                "ds_read_b128 %3, %4 offset:3072\n\t"
                : "=&v"(r[0].d4), "=&v"(r[1].d4), "=&v"(r[2].d4), "=&v"(r[3].d4)
                : "v"(off) : "memory");
        }
        else if constexpr (M_TILES == 2) {
            asm volatile(
                "ds_read_b128 %0, %2 offset:0\n\t"
                "ds_read_b128 %1, %2 offset:1024\n\t"
                : "=&v"(r[0].d4), "=&v"(r[1].d4)
                : "v"(off) : "memory");
        }
        else if constexpr (M_TILES == 1) {
            asm volatile(
                "ds_read_b128 %0, %1 offset:0\n\t"
                : "=&v"(r[0].d4)
                : "v"(off) : "memory");
        }
    };
    auto decode=[&]<int BUF,int NG>() {
#pragma unroll
        for(int nt=0;nt<2;++nt) {
            union Pair { uint64_t q; uint32_t d[2]; } src,shift;
            src.d[0]=uint32_t(packed[BUF][NG][nt*2]);
            src.d[1]=uint32_t(packed[BUF][NG][nt*2+1]);
            asm volatile("v_lshlrev_b64 %0,4,%1\n" : "=v"(shift.q) : "v"(src.q));
            bd[nt*2].d2={int32_t(src.d[0]&0xf0f0f0f0u),int32_t(shift.d[0]&0xf0f0f0f0u)};
            bd[nt*2+1].d2={int32_t(src.d[1]&0xf0f0f0f0u),int32_t(shift.d[1]&0xf0f0f0f0u)};
        }
    };
    auto mma_group=[&]<int NG,int G,bool FIRST>() {
#pragma unroll
        for(int mi=G*((M_TILES+1)/2);mi<(G+1)*((M_TILES+1)/2) && mi<M_TILES;++mi) {
#pragma unroll
            for(int nt=0;nt<2;++nt) {
                if constexpr(FIRST) {
                    vec4_int zero={};auto* c=acc;auto* ar=a;auto* br=bd;
                    asm volatile("v_mmac_i32_16x16x32_i8 %0,%1,%2,%3\n"
                                 : "=&v"(c[mi][NG*2+nt]) : "v"(ar[mi].i8[0]),"v"(br[nt*2].i8),"v"(zero));
                } else {
                    acc[mi][NG*2+nt]=mmac_4interleave_i8(a[mi].i8[0],bd[nt*2].i8,acc[mi][NG*2+nt]);
                }
                acc[mi][NG*2+nt]=mmac_4interleave_i8(a[mi].i8[1],bd[nt*2+1].i8,acc[mi][NG*2+nt]);
            }
        }
    };
    const int stages=p.k/64;
    load_a(0); load_b.template operator()<0>(0); wait_b_a.template operator()<0>(); barrier();
    auto step=[&]<int CUR,bool FIRST>(int stage) {
        constexpr int NXT=CUR^1;
        read_a.template operator()<CUR>();
        // The previous stage waited for A, leaving up to two B loads pending.
        // Issue LDS reads before draining B to overlap the independent requests.
        auto* r=packed[CUR];
        asm volatile("s_waitcnt vmcnt(0)\n" : "+v"(r[0]),"+v"(r[1]) : : "memory");
        __builtin_hcu_sched_barrier(0);
        decode.template operator()<CUR,0>();
        asm volatile("s_waitcnt lgkmcnt(%0)\n" : : "n"(M_TILES/2) : "memory");
        __builtin_hcu_sched_barrier(0);
        if (stage+1<stages) {
            // Keep this order: vmcnt(2) must complete the older A-to-LDS load.
            load_a(stage+1);
            load_b.template operator()<NXT>(stage+1);
        }
        mma_group.template operator()<0,0,FIRST>();
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __builtin_hcu_sched_barrier(0);
        mma_group.template operator()<0,1,FIRST>();
        decode.template operator()<CUR,1>();
        mma_group.template operator()<1,0,FIRST>();
        mma_group.template operator()<1,1,FIRST>();
        if (stage+1<stages) {
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(0)\n" ::: "memory");
            __builtin_hcu_sched_barrier(0);
        }
        barrier();
    };
    step.template operator()<0,true>(0);
#pragma unroll 1
    for (int k=1;k<stages;k+=2) {
        step.template operator()<1,false>(k);
        if(k+1<stages) step.template operator()<0,false>(k+1);
    }
    const float* sa=reinterpret_cast<const float*>(p.scale_a_ptr);
    const float* sb=reinterpret_cast<const float*>(p.scale_b_ptr)+int64_t(expert)*p.n;
    Element_O* out=reinterpret_cast<Element_O*>(p.d_ptr);
    const uint64_t d_span=uint64_t(128)*p.d_stride*sizeof(Element_O);
    const bool use_buffer_store=d_span<=0xffffffffull;
    buffer_rsrc_t dr=make_buffer_rsrc<false>(out+int64_t(by)*128*p.d_stride,1,uint32_t(d_span));
    float bs[2][8];
#pragma unroll
    for(int nh=0;nh<2;++nh) {
#pragma unroll
        for(int i=0;i<8;++i)bs[nh][i]=sb[bx*512+wn*64+nh*32+cg*8+i];
    }
#pragma unroll
    for(int mi=0;mi<M_TILES;++mi) {
        const int mr=by*128+mi*16+row;
        const float as=sa[mr];
#pragma unroll
        for(int nh=0;nh<2;++nh) {
            bf16x8_pair_bits bits;
#pragma unroll
            for(int r=0;r<4;++r) {
                vec2_fp32 v={float(acc[mi][nh*2][r]),float(acc[mi][nh*2+1][r])};
                v=v*vec2_fp32{as,as};
                v=v*vec2_fp32{bs[nh][2*r],bs[nh][2*r+1]};
                bits.pair[r]=DownCastPair_f32_bf16(v);
            }
            if(use_buffer_store) {
                const uint32_t d_off=(uint32_t(mi*16+row)*uint32_t(p.d_stride)+uint32_t(bx*512+wn*64+nh*32+cg*8))*uint32_t(sizeof(Element_O));
                asm volatile("buffer_store_dwordx4 %0,%1,%2,0 offen\n"
                             : : "v"(bits.u16x8),"v"(d_off),"s"(dr) : "memory");
            } else {
                *reinterpret_cast<bf16x8_bits*>(out+int64_t(mr)*p.d_stride+bx*512+wn*64+nh*32+cg*8)=bits.u16x8;
            }
        }
    }
}
// All decisions are CTA-uniform and use device-side routing metadata.
// Expert segments are padded to 128, with valid rows followed by -1 padding.
template<typename Element_O>
__global__ __launch_bounds__(512,2)
void w4a8_wasp_kernel(DeepGemm_params p) {
    const int base=int(blockIdx.y)*128;
    const int expert=__builtin_hcu_readfirstlane(p.m_indices[base]);
    if (expert<0) return;
    if (__builtin_hcu_readfirstlane(p.m_indices[base+64])==expert) {
        compute_tile<Element_O,8>(p,expert);
    } else if (__builtin_hcu_readfirstlane(p.m_indices[base+32])==expert) {
        compute_tile<Element_O,4>(p,expert);
    } else if (__builtin_hcu_readfirstlane(p.m_indices[base+16])==expert) {
        compute_tile<Element_O,2>(p,expert);
    } else {
        compute_tile<Element_O,1>(p,expert);
    }
}

bool pair32_wasp_is_gfx936_gfx938() {
    int device_id = 0;
    TORCH_CHECK(hipGetDevice(&device_id) == hipSuccess,
                "failed to get HIP device");
    hipDeviceProp_t prop{};
    TORCH_CHECK(hipGetDeviceProperties(&prop, device_id) == hipSuccess,
                "failed to query HIP device properties");
    const std::string arch_name(prop.gcnArchName);
    return arch_name.rfind("gfx936", 0) == 0 ||
           arch_name.rfind("gfx938", 0) == 0;
}

torch::Tensor run(
    torch::Tensor input,
    torch::Tensor packed_weight,
    torch::Tensor output,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    torch::Tensor m_indices) {
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    TORCH_CHECK(pair32_wasp_is_gfx936_gfx938(),
                "unsupported GPU architecture");
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
                "W4A8 WASP mode 1006 currently supports BF16 output");
    TORCH_CHECK(input_scale.scalar_type() == at::kFloat &&
                    weight_scale.scalar_type() == at::kFloat,
                "scales must be float32");
    TORCH_CHECK(m_indices.scalar_type() == at::kInt,
                "m_indices must be int32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M,K]");
    TORCH_CHECK(packed_weight.dim() == 3,
                "packed_weight must be HIPC W4 bytes [E,N/32,K*16]");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [M,N]");
    TORCH_CHECK(m_indices.dim() == 1, "m_indices must be 1D");
    TORCH_CHECK(input_scale.dim() == 1 ||
                    (input_scale.dim() == 2 && input_scale.size(1) == 1),
                "input_scale must be 1D [M] or 2D [M,1]");
    TORCH_CHECK(weight_scale.dim() == 2 ||
                    (weight_scale.dim() == 3 && weight_scale.size(2) == 1),
                "weight_scale must be 2D [E,N] or 3D [E,N,1]");
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const int E = static_cast<int>(packed_weight.size(0));
    const int N = static_cast<int>(output.size(1));
    TORCH_CHECK(packed_weight.numel() == static_cast<int64_t>(E) * N * K / 2,
                "packed HIPC W4 weight must contain E*N*K/2 bytes");
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
                "W4A8 WASP mode 1006 requires M divisible by 128, N by 512, K by 64, and experts padded to 128 rows");

    if (M == 0 || N == 0) return output;
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
    params.alpha = 1.0f;  // Caller supplies natural B scales /16, as in mode 1002.
    params.beta = 0.0f;

    const dim3 grid(N / GemmTraitsPC::kBlockN,
                    M / GemmTraitsPC::kBlockM, 1);
    const dim3 block(GemmTraitsPC::kNThreads, 1, 1);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    hipLaunchKernelGGL(
        (w4a8_wasp_kernel<bhalf_t>),
        grid, block, GemmTraitsPC::kSmemSize, stream, params);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "W4A8 WASP kernel launch failed");
    return output;
}

}
