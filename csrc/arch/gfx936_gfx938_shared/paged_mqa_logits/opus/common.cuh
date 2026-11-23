// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <type_traits>
#include <algorithm>
namespace deepgemm::opus {
constexpr int kWaveSize=64;
__host__ __device__ constexpr int ceil_div(int x,int y){return x/y+(x%y!=0);}
constexpr int constexpr_ceil_div(int x,int y){return x/y+(x%y!=0);}
typedef __fp16 fp16VecType;
typedef fp16VecType fp16x2 __attribute__((ext_vector_type(2)));
typedef fp16VecType fp16x4 __attribute__((ext_vector_type(4)));
typedef fp16VecType fp16x8 __attribute__((ext_vector_type(8)));

using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;

typedef uint8_t fp8_t;
typedef fp8_t fp8x2 __attribute__((ext_vector_type(2)));
typedef fp8_t fp8x4 __attribute__((ext_vector_type(4)));
typedef fp8_t fp8x8 __attribute__((ext_vector_type(8)));
typedef fp8_t fp8x16 __attribute__((ext_vector_type(16)));

typedef short v4bh __attribute__((ext_vector_type(4)));

typedef float fp32x2 __attribute__((ext_vector_type(2)));
typedef float fp32x4 __attribute__((ext_vector_type(4)));
typedef float fp32x8 __attribute__((ext_vector_type(8)));

typedef int v4i __attribute__((ext_vector_type(4)));

typedef union
{
    fp16x8 data;
    struct
    {
        fp16x4 front;
        fp16x4 rear;
    };
} f16x8_t;

typedef union
{
    fp8x16 data;
    struct
    {
        fp8x8 front;
        fp8x8 rear;
    };
} f8x16_t;

// H8 reuses a 16-row MMAC tile while keeping the LDS Q/weight caches compact.
// Invalid rows never read LDS and enter MMAC/reduction as deterministic zero.
template <int kNumHeads, int kQTile, typename Operand, typename Element>
inline __device__ __forceinline__ Operand load_padded_q_operand(const Element* smem_q,
                                                                const int offset,
                                                                const int head_idx) {
    if constexpr (kNumHeads < kQTile) {
        if (head_idx >= kNumHeads) {
            Operand zero = {};
            return zero;
        }
    }
    return *reinterpret_cast<const Operand*>(&smem_q[offset]);
}

template <int kNumHeads, int kQTile>
inline __device__ __forceinline__ float load_padded_weight(const float* smem_weights, const int head_idx) {
    if constexpr (kNumHeads < kQTile) {
        return head_idx < kNumHeads ? smem_weights[head_idx] : 0.0f;
    }
    return smem_weights[head_idx];
}

#define OPUS_MQA_DIRECT_LDS_WORDx4 16
#define OPUS_MQA_DIRECT_LDS_WORDx2 8
#define OPUS_MQA_DIRECT_LDS_WORD 4

#define OPUS_MQA_WAIT_VMCNT_LDS(X)               \
    __builtin_amdgcn_sched_barrier(0);  \
    asm volatile(                       \
    "s_waitcnt vmcnt(%0)\n\t"           \
    "s_barrier\n"                       \
    :: "I"(X)                           \
    :);                                 \
    __builtin_amdgcn_sched_barrier(0);

#define OPUS_MQA_CLEAR_ACC(acc, x)               \
    for (int i = 0; i < x; ++i) {       \
        acc[i] = {0.0f};                \
    }

#define OPUS_MQA_CLEAR_ACC_2D(acc_2d, x, y)      \
    for (int j = 0; j < x; ++j) {       \
        OPUS_MQA_CLEAR_ACC(acc_2d[j], y);        \
    }

inline __device__ void buffer_load_lds_x4(v4i* desc, uint8_t* smem_ptr, int offset) {
    #if defined(__gfx936__) || defined(__gfx938__)
    __builtin_amdgcn_raw_buffer_load_lds(*desc,
                                         // Cast address type to which compiler can recognize as lds type.
                                         *(__attribute__((address_space(3))) int**) (&smem_ptr),
                                         OPUS_MQA_DIRECT_LDS_WORDx4,
                                         offset,
                                         0,
                                         0,
                                         0);
    #endif
}

template <typename Element, int kHeadDim, int kKvTile>
inline __device__ __forceinline__ void buffer_load_page_tile_lds(const Element* page,
                                                                 const int row_in_page,
                                                                 const int lane_idx,
                                                                 Element* smem_tile) {
    constexpr int kAlignment = 16 / sizeof(Element);
    constexpr int kLoadIter = kKvTile * kHeadDim / (kWaveSize * kAlignment);

    long glob_kv_desc[2];
    glob_kv_desc[0] = *reinterpret_cast<const long*>(&page);
    glob_kv_desc[1] = (static_cast<long>(0x20000) << 32) | 0xFFFFFFFE;
    auto* src = reinterpret_cast<v4i*>(glob_kv_desc);

    #pragma unroll
    for (int load_iter = 0; load_iter < kLoadIter; ++load_iter) {
        const int tile_linear_elem = (load_iter * kWaveSize + lane_idx) * kAlignment;
        const int page_linear_elem = row_in_page * kHeadDim + tile_linear_elem;
        #if defined(__gfx936__) || defined(__gfx938__)
        auto* lds_ptr = reinterpret_cast<uint8_t*>(smem_tile + load_iter * kWaveSize * kAlignment);
        buffer_load_lds_x4(src, lds_ptr, page_linear_elem * sizeof(Element));
        #else
        *reinterpret_cast<f16x8_t*>(smem_tile + tile_linear_elem) =
            *reinterpret_cast<const f16x8_t*>(page + page_linear_elem);
        #endif
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<bool is_half = true>
inline __device__ void builtin_b16_mmac(const fp16x4& reg_a, const fp16x4& reg_b, fp32x4& reg_c) {
    #if defined(__gfx938__)
    if constexpr (is_half) {
        reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a, reg_b, reg_c, false, false);
    } else {
        reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c, false, false);
    }
    #elif defined(__gfx936__)
    if constexpr (is_half) {
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16f16(reg_a, reg_b, reg_c);
    } else {
        reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*) &reg_a, *(v4bh*) &reg_b, reg_c);
    }
    #endif
}

template<bool is_e4m3 = true>
inline __device__ void builtin_fp8_mmac(const fp8x8& reg_a, const fp8x8& reg_b, fp32x4& reg_c) {
    #if defined(__gfx938__)
    if constexpr (is_e4m3) {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(*(intx2*) &reg_a,*(intx2*) &reg_b, reg_c, false, false);
    } else {
        reg_c = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(*(intx2*) &reg_a,*(intx2*) &reg_b, reg_c, false, false);
    }
    #endif
}


#define OPUS_MQA_WAIT_STAGE(X) do { __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(%0)"::"I"(X):"memory"); if constexpr(!WaveLocal) __syncthreads(); __builtin_amdgcn_sched_barrier(0); } while(0)
template<bool Repacked>
__device__ __forceinline__ void load_s1(const uint8_t* kv,const int* tables,int first,int stage,int lane,int len,uint8_t* dst){
    const int n=first+lane/4;
    const int offset=n<len?(Repacked?(tables[n]/64)*8448+(tables[n]%64)*128:tables[n]*132)+stage*64+((lane%4)^((lane/8)%4))*16:-1;
    long desc[2];desc[0]=reinterpret_cast<long>(kv);desc[1]=(long(0x20000)<<32)|0xFFFFFFFE;
    buffer_load_lds_x4(reinterpret_cast<v4i*>(desc),dst,offset);
}


// Preserve the public 4-byte Q alignment contract. The common aligned case
// keeps vector loads; offset views use a memcpy with no 16-byte assumption.
__device__ __forceinline__ f8x16_t load_q16(const uint8_t* p) {
    if(reinterpret_cast<uintptr_t>(p)%16==0)return *reinterpret_cast<const f8x16_t*>(p);
    f8x16_t v;__builtin_memcpy(&v,p,16);return v;
}
__device__ __forceinline__ unsigned short fp8_half(unsigned char b){
 const unsigned m=b&127,sign=(b&128)<<8;
 if(m>=8)return sign|(m==127?0x7fff:((m<<7)+0x2000));
 __half x=__float2half(float(m)*(1.0f/512.0f));
 return sign|__builtin_bit_cast(unsigned short,x);
}
// Decode exactly, including E4M3FN subnormals, without global scratch.
// Two independent 16-bit fields hold sign/magnitude. A packed FP16
// subtraction normalizes the eight E4M3 subnormal magnitudes exactly:
// (2^-6 + m*2^-9) - 2^-6. Every result is normal in FP16.
// Masks select normal/subnormal/NaN fields without per-byte EXEC branches.
__device__ __forceinline__ uint32_t decode_fp8_pair(uint32_t pair) {
    const uint32_t mag=pair&0x007f007fu;
    const uint32_t normal=(mag<<7)+0x20002000u;
    const uint32_t sub_bits=(mag<<7)+0x24002400u;
    const __half2 sub=__hsub2(__builtin_bit_cast(__half2,sub_bits),
                             __builtin_bit_cast(__half2,uint32_t(0x24002400u)));
    const uint32_t exp=mag&0x00780078u;
    const uint32_t normal_mask=(((exp+0x00780078u)&0x00800080u)>>7)*0xffffu;
    const uint32_t not_nan=mag^0x007f007fu;
    const uint32_t finite_mask=(((not_nan+0x007f007fu)&0x00800080u)>>7)*0xffffu;
    const uint32_t value=(normal&normal_mask)|(__builtin_bit_cast(uint32_t,sub)&~normal_mask);
    return ((pair&0x00800080u)<<8)|(value&finite_mask)|(0x7fff7fffu&~finite_mask);
}
__device__ __forceinline__ f16x8_t decode_fp8x8_value(intx2 words) {
    using u32x4=uint32_t __attribute__((ext_vector_type(4)));
    u32x4 result;
    #pragma unroll
    for(int j=0;j<2;++j) {
        const uint32_t bits=words[j];
        result[j*2]=decode_fp8_pair((bits&0xffu)|((bits&0xff00u)<<8));
        result[j*2+1]=decode_fp8_pair(((bits>>16)&0xffu)|((bits>>8)&0xff0000u));
    }
    return __builtin_bit_cast(f16x8_t,result);
}
static __global__ void convert_q(const uint8_t* src,uint16_t* dst,int size){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size)dst[i]=fp8_half(src[i]);
}
// One lane converts one aligned logical 16-byte key fragment. The load helper
// also supports storage views with only the public four-byte alignment.
template<int S>
__global__ void convert_cache_vector(const uint8_t* src,uint8_t* dst,int tokens){
    const int fragment=blockIdx.x*blockDim.x+threadIdx.x;
    if(fragment>=tokens*8)return;
    const int token=fragment/8,d=(fragment%8)*16;
    const int srcpos=(token/S)*S*132+(token%S)*128+d;
    const auto bytes=load_q16(src+srcpos);
    using u16x16=uint16_t __attribute__((ext_vector_type(16)));
    u16x16 values;
    #pragma unroll
    for(int j=0;j<16;++j)values[j]=fp8_half(bytes.data[j]);
    *reinterpret_cast<u16x16*>(dst+int64_t(token)*256+d*2)=values;
    if(d==0)*reinterpret_cast<float*>(dst+int64_t(tokens)*256+int64_t(token)*4)=
        *reinterpret_cast<const float*>(src+(token/S)*S*132+S*128+(token%S)*4);
}
static __global__ void pack_s1(const uint32_t* src,uint32_t* dst,int tokens){
    const int word=blockIdx.x*blockDim.x+threadIdx.x;
    if(word>=tokens*33)return;
    const int token=word/33,j=word%33;
    const int pos=(token/64)*2112+(j<32?(token%64)*32+j:2048+token%64);
    dst[pos]=src[word];
}
static __global__ void check_groups(const int* context,const int* tables,int* flags,int* compact_context,int* compact_tables,
 int B,int N,int S,int tw,int QT){
 const int g=blockIdx.y,q0=g*QT,tid=threadIdx.x,ctw=(N-1)/64+1;
 const int chunk=blockIdx.x,chunks=(N-1)/(S*1024)+1;
 __shared__ int bad;
 const int len=context[q0],p0=tables[int64_t(q0)*tw];
 if(tid==0)bad=(q0+QT>B || (p0*S)%64!=0);
 __syncthreads();
 for(int q=0;q<QT&&q0+q<B;++q){
  if(context[q0+q]!=len){if(tid==0)atomicExch(&bad,1);continue;}
  for(int i=chunk*1024+tid;i<ceil_div(len,S) && i<(chunk+1)*1024;i+=blockDim.x)
   if(tables[int64_t(q0+q)*tw+i]!=(S==64?tables[int64_t(q0)*tw+i]:int64_t(p0)+i))atomicExch(&bad,1);
 }
 __syncthreads();
 if(tid==0)flags[g*chunks+chunk]=bad?-1:p0*S;
 if(chunk==0){
  if(tid==0)compact_context[g]=len;
  for(int i=tid;i<ctw;i+=blockDim.x)compact_tables[int64_t(g)*ctw+i]=S==64?(i<ceil_div(len,S)?tables[int64_t(q0)*tw+i]:0):p0*S/64+i;
 }
}

} // namespace deepgemm::opus
