// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "common.cuh"
namespace deepgemm::opus {
template<int H,int R>
__global__ __launch_bounds__(256) void paged_mqa_direct_half(const uint16_t* q,const uint8_t* k,
 const float* w,const int* ctx,const int* tb,float* out,int N,int tw, bool clean_logits) {
    constexpr int QI=H/16;
    int lane=threadIdx.x%64,wave=threadIdx.x/64,b=blockIdx.y,r=blockIdx.x%R;
    int first=(blockIdx.x/R)*256,len=ctx[b];
    int64_t output_row=int64_t(b*R+r)*N;
    if(first>=len) {
        if(clean_logits) {
            if(first+threadIdx.x<N)out[output_row+first+threadIdx.x]=-INFINITY;
        }
        return;
    }
    f16x8_t qr[QI][4];float wr[QI][4];
    #pragma unroll
    for(int h=0;h<QI;++h) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            qr[h][j]=*reinterpret_cast<const f16x8_t*>(q+int64_t(b*R+r)*H*128+(h*16+lane%16)*128+(lane/16)*8+j*32);
            wr[h][j]=w[(b*R+r)*H+h*16+j*4+lane/16];
        }
    }
    const bool valid=first+wave*64<len;
    int page=valid?tb[int64_t(b)*tw+first/64+wave]:0;
    const uint8_t* kp=k+int64_t(page)*8448;
    #pragma unroll
    for(int i=0;i<4;++i) {
        intx2 raw[4];
        #pragma unroll
        for(int j=0;j<4;++j)__builtin_memcpy(&raw[j],kp+(i*16+lane%16)*128+lane/16*8+j*32,8);
        fp32x4 acc[QI]={};
        #pragma unroll
        for(int j=0;j<4;++j) {
            auto kr=decode_fp8x8_value(raw[j]);
            #pragma unroll
            for(int h=0;h<QI;++h) {
                builtin_b16_mmac(kr.front,qr[h][j].front,acc[h]);
                builtin_b16_mmac(kr.rear,qr[h][j].rear,acc[h]);
            }
        }
        float sum=0;
        #pragma unroll
        for(int h=0;h<QI;++h) {
            #pragma unroll
            for(int c=0;c<4;++c)sum+=fmaxf(acc[h][c],0.f)*wr[h][c];
        }
        sum+=__shfl_down(sum,16);sum+=__shfl_down(sum,32);
        const int n=first+wave*64+i*16+lane;
        if(lane<16&&n<N&&(clean_logits||n<len))out[output_row+n]=n<len-R+r+1 ? sum*reinterpret_cast<const float*>(kp+8192)[i*16+lane] : -INFINITY;
    }
}
} // namespace deepgemm::opus
