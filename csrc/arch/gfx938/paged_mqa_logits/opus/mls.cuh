// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "paged_mqa_logits/persistent_work.cuh"
#include "../../../gfx936_gfx938_shared/paged_mqa_logits/opus/common.cuh"
namespace deepgemm::opus {
#if defined(__gfx938__)
// MLS writes canonical B8 tiles directly into LDS. Descriptor SGPRs produced
// by readfirstlane require five wait states before the VMEM instruction.
__device__ __forceinline__ void mls_load(const uint8_t* src,uint8_t* dst) {
    using u4 = uint32_t __attribute__((ext_vector_type(4)));
    const auto address=reinterpret_cast<uintptr_t>(src);
    u4 desc={uint32_t(__builtin_amdgcn_readfirstlane(uint32_t(address))),
             uint32_t(__builtin_amdgcn_readfirstlane(uint32_t(address>>32)&0xffff)),128,0};
    const int offset=__builtin_amdgcn_readfirstlane(int(reinterpret_cast<uintptr_t>(dst)))|0x80000000u;
    asm volatile("s_nop 4\n\tmatrix_load_128x16_b8 %0, %1, moffset:0 t lds"
                 ::"s"(desc),"s"(offset):"memory");
}
__device__ __forceinline__ f8x16_t mls_read(uint8_t* p) {
    const int offset=__builtin_amdgcn_readfirstlane(int(reinterpret_cast<uintptr_t>(p)));
    v4i value;
    asm volatile("s_add_u32 m0, %1, 0x80000000\n\ts_nop 0\n\t"
                 "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x3 col:0x1 alt:0x0"
                 :"=v"(value):"s"(offset):"memory");
    return __builtin_bit_cast(f8x16_t,value);
}
template<int Control> __device__ __forceinline__ float mls_dpp(float value) {
    return __builtin_bit_cast(float,__builtin_amdgcn_mov_dpp(
        __builtin_bit_cast(int,value),Control,0xf,0xf,true));
}
#define OPUS_MLS_WAIT(OP) do { __builtin_amdgcn_sched_barrier(0); asm volatile(OP:::"memory"); __builtin_amdgcn_sched_barrier(0); } while(0)
template<int H,int R,int G=1,int I=4>
__device__ __forceinline__ void paged_mqa_mls_device(const uint8_t* q,const uint8_t* k,
    const float* w,const int* ctx,const int* tb,float* out,int N,int tw,int batch, bool clean_logits) {
    static_assert(I==1 || I==2 || I==4);
    constexpr int QI=H*G/16, T=64*I;
    const int tid=threadIdx.x,wave=tid/64,lane=tid%64;
    const int r=blockIdx.x%R,first=(blockIdx.x/R)*T,len=ctx[batch];
    const int64_t output_row=int64_t(batch*R*G+r)*N;
    if(first>=len) {
        if(clean_logits) {
            if(tid<T && first+tid<N) {
                for(int g=0;g<G;++g)out[output_row+g*N+first+tid]=-INFINITY;
            }
        }
        return;
    }
    extern __shared__ __align__(1024) uint8_t smem[];
    const uint8_t* query=q+int64_t(batch*R*G+r)*H*128;
    if(wave<QI)mls_load(query+wave*2048,smem+wave*2048);
    // Issue weights and page metadata while the cooperative Q load is in flight.
    float wr[QI];
    #pragma unroll
    for(int h=0;h<QI;++h)wr[h]=w[(batch*R*G+r)*H+h*16+lane%16];
    const bool valid=first+wave*16*I<len;
    const int page=valid?__builtin_amdgcn_readfirstlane(tb[int64_t(batch)*tw+first/64+wave*I/4]):0;
    const uint8_t* kp=k+int64_t(page)*8448;
    OPUS_MLS_WAIT("s_waitcnt vmcnt(0)");
    __syncthreads();
    f8x16_t qr[QI][2];
    #pragma unroll
    for(int h=0;h<QI;++h) {
        qr[h][0]=mls_read(smem+h*2048);qr[h][1]=mls_read(smem+h*2048+1024);
    }
    OPUS_MLS_WAIT("s_waitcnt lgkmcnt(0)");
    __syncthreads(); // all waves have consumed Q before their KV tiles reuse LDS

    uint8_t* tile=smem+wave*2048;
    float* result=reinterpret_cast<float*>(smem+8192);
    const float scale=valid?reinterpret_cast<const float*>(kp+8192)[(wave*16*I%64+lane%(16*I))]:0.f;
    if(valid)mls_load(kp+(wave*I%4)*2048,tile);
    OPUS_MLS_WAIT("s_waitcnt vmcnt(0)");
    #pragma unroll
    for(int i=0;i<I;++i) {
        f8x16_t kr[2]={};
        if(valid){kr[0]=mls_read(tile);kr[1]=mls_read(tile+1024);}
        OPUS_MLS_WAIT("s_waitcnt lgkmcnt(0)");
        // Wave-private tile: DS reads must finish before overwrite, but no
        // workgroup barrier is needed. Prefetch overlaps MMAC and reduction.
        if(i<I-1&&valid)mls_load(kp+(wave*I%4+i+1)*2048,tile);
        fp32x4 sum[G]={};
        #pragma unroll
        for(int h=0;h<QI;++h) {
            fp32x4 acc={};
            #pragma unroll
            for(int half=0;half<2;++half) {
                acc=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                    __builtin_bit_cast(intx2,qr[h][half].front),__builtin_bit_cast(intx2,kr[half].front),acc,true,false);
                acc=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                    __builtin_bit_cast(intx2,qr[h][half].rear),__builtin_bit_cast(intx2,kr[half].rear),acc,true,false);
            }
            #pragma unroll
            for(int c=0;c<4;++c)sum[h/(H/16)][c]+=fmaxf(acc[c],0.0f)*wr[h];
        }
        #pragma unroll
        for(int g=0;g<G;++g) {
            #pragma unroll
            for(int c=0;c<4;++c) {
                sum[g][c]+=mls_dpp<0xb1>(sum[g][c]);sum[g][c]+=mls_dpp<0x4e>(sum[g][c]);
                sum[g][c]+=__shfl_xor(sum[g][c],4,16);sum[g][c]+=mls_dpp<0x128>(sum[g][c]);
                if(lane%16==0)result[g*256+wave*16*I+i*16+(lane/16)*4+c]=sum[g][c];
            }
        }
        if(i<I-1) { OPUS_MLS_WAIT("s_waitcnt vmcnt(0)"); }
    }
    OPUS_MLS_WAIT("s_waitcnt lgkmcnt(0)");
    // Each wave reads only the result range it produced. This is also
    // required for N64/N128: a tid-based read would cross wave ownership.
    const int out_idx=wave*16*I+lane;
    const int n=first+out_idx;
    if(lane<16*I && n<N && (clean_logits||n<len))for(int g=0;g<G;++g)out[output_row+g*N+n]=n<len-R+r+1 ? result[g*256+out_idx]*scale : -INFINITY;
}

template<int H,int R,int G=1,int I=4>
__device__ __forceinline__ void paged_mqa_mls_persistent(const uint8_t* q,const uint8_t* k,
    const float* w,const int* ctx,const int* tb,float* out,int N,int tw,int batch, int work_first, int work_last) {
    static_assert(R == 1, "Persistent MLS supports R=1");
    static_assert(I==1 || I==2 || I==4);
    constexpr int QI=H*G/16, T=64*I;
    const int tid=threadIdx.x,wave=tid/64,lane=tid%64;
    const int r=0,len=ctx[batch];
    int first=work_first*T;
    const int64_t output_row=int64_t(batch*R*G+r)*N;
    if(first>=len) return;
    extern __shared__ __align__(1024) uint8_t smem[];
    const uint8_t* query=q+int64_t(batch*R*G+r)*H*128;
    if(wave<QI)mls_load(query+wave*2048,smem+wave*2048);
    // Issue weights and page metadata while the cooperative Q load is in flight.
    float wr[QI];
    #pragma unroll
    for(int h=0;h<QI;++h)wr[h]=w[(batch*R*G+r)*H+h*16+lane%16];
    OPUS_MLS_WAIT("s_waitcnt vmcnt(0)");
    __syncthreads();
    f8x16_t qr[QI][2];
    #pragma unroll
    for(int h=0;h<QI;++h) {
        qr[h][0]=mls_read(smem+h*2048);qr[h][1]=mls_read(smem+h*2048+1024);
    }
    OPUS_MLS_WAIT("s_waitcnt lgkmcnt(0)");
    __syncthreads(); // all waves have consumed Q before their KV tiles reuse LDS

    // Register Q and weights are invariant for the request.
    while (true) {
        const bool valid=first+wave*16*I<len;
        const int page=valid?__builtin_amdgcn_readfirstlane(tb[int64_t(batch)*tw+first/64+wave*I/4]):0;
        const uint8_t* kp=k+int64_t(page)*8448;
        uint8_t* tile=smem+wave*2048;
        float* result=reinterpret_cast<float*>(smem+8192);
        const float scale=valid?reinterpret_cast<const float*>(kp+8192)[(wave*16*I%64+lane%(16*I))]:0.f;
        if(valid)mls_load(kp+(wave*I%4)*2048,tile);
        OPUS_MLS_WAIT("s_waitcnt vmcnt(0)");
        #pragma unroll
        for(int i=0;i<I;++i) {
            f8x16_t kr[2]={};
            if(valid){kr[0]=mls_read(tile);kr[1]=mls_read(tile+1024);}
            OPUS_MLS_WAIT("s_waitcnt lgkmcnt(0)");
            // Wave-private tile: DS reads must finish before overwrite, but no
            // workgroup barrier is needed. Prefetch overlaps MMAC and reduction.
            if(i<I-1&&valid)mls_load(kp+(wave*I%4+i+1)*2048,tile);
            fp32x4 sum[G]={};
            #pragma unroll
            for(int h=0;h<QI;++h) {
                fp32x4 acc={};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    acc=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                        __builtin_bit_cast(intx2,qr[h][half].front),__builtin_bit_cast(intx2,kr[half].front),acc,true,false);
                    acc=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
                        __builtin_bit_cast(intx2,qr[h][half].rear),__builtin_bit_cast(intx2,kr[half].rear),acc,true,false);
                }
                #pragma unroll
                for(int c=0;c<4;++c)sum[h/(H/16)][c]+=fmaxf(acc[c],0.0f)*wr[h];
            }
            #pragma unroll
            for(int g=0;g<G;++g) {
                #pragma unroll
                for(int c=0;c<4;++c) {
                    sum[g][c]+=mls_dpp<0xb1>(sum[g][c]);sum[g][c]+=mls_dpp<0x4e>(sum[g][c]);
                    sum[g][c]+=__shfl_xor(sum[g][c],4,16);sum[g][c]+=mls_dpp<0x128>(sum[g][c]);
                    if(lane%16==0)result[g*256+wave*16*I+i*16+(lane/16)*4+c]=sum[g][c];
                }
            }
            if(i<I-1) { OPUS_MLS_WAIT("s_waitcnt vmcnt(0)"); }
        }
        OPUS_MLS_WAIT("s_waitcnt lgkmcnt(0)");
        // Each wave reads only the result range it produced. This is also
        // required for N64/N128: a tid-based read would cross wave ownership.
        const int out_idx=wave*16*I+lane;
        const int n=first+out_idx;
        if(lane<16*I && n<N && (n<len))for(int g=0;g<G;++g)out[output_row+g*N+n]=n<len-R+r+1 ? result[g*256+out_idx]*scale : -INFINITY;

        // The exclusive token end may exceed INT_MAX; compare tile indices.
        if (++work_first >= work_last) break;
        first = work_first*T;
    }
}

#undef OPUS_MLS_WAIT
#endif
template<int H,int R,int I=4>
__global__ __launch_bounds__(256) void paged_mqa_mls_single(const uint8_t* q,const uint8_t* k,const float* w,
 const int* ctx,const int* tb,float* out,int N,int tw, bool clean_logits, int batch_size) {
#if defined(__gfx938__)
    if constexpr (R==1) {
        if(!clean_logits) {
            deepgemm::PagedMqaWorkRange work(ctx, N, 64*I, batch_size);
            int work_batch, work_first, work_last;
            while(work.next(work_batch,work_first,work_last)) {
                paged_mqa_mls_persistent<H,1,1,I>(q,k,w,ctx,tb,out,N,tw,work_batch,work_first,work_last);
                __syncthreads();
            }
            return;
        }
    }
    paged_mqa_mls_device<H,R,1,I>(q,k,w,ctx,tb,out,N,tw,blockIdx.y,clean_logits);
#endif
}

__global__ __launch_bounds__(256) void paged_mqa_mls_grouped(const uint8_t* q,const uint8_t* k,const float* w,
 const int* ctx,const int* tb,float* out,int B,int N,int tw,const int* flags,const int* cctx,const int* ctb, bool clean_logits) {
#if defined(__gfx938__)
    const int g=blockIdx.y,chunks=(N-1)/(64*1024)+1;
    bool bad=false;
    for(int j=threadIdx.x%64;j<chunks;j+=64)bad|=flags[g*chunks+j]<0;
    if(__ballot(bad)==0) {
        paged_mqa_mls_device<32,1,2,4>(q,k,w,cctx,ctb,out,N,(N-1)/64+1,g,clean_logits);
    } else {
        for(int i=0;i<2&&g*2+i<B;++i) {
            paged_mqa_mls_device<32,1,1,4>(q,k,w,ctx,tb,out,N,tw,g*2+i,clean_logits);
            __syncthreads();
        }
    }
#endif
}
} // namespace deepgemm::opus
