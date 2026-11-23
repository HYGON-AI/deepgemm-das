// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "fp16.cuh"
#include "direct_half.cuh"

namespace deepgemm::opus {

template<int H,int S,int R>
__global__ __launch_bounds__(256) void paged_mqa_fp16_single(const uint16_t* q,const uint16_t* k,const float* w,
 const int* ctx,const int* tb,float* out,int B,int N,int tw,int tokens, bool clean_logits) {
    const float* scales=(const float*)((const uint8_t*)k+int64_t(tokens)*256);
    grouped_half_device<uint16_t,R,H,128,S,S*4,4,64,1>(q,k,w,B,0,S*256,N,tw,ctx,out,tb,scales,blockIdx.y,clean_logits);
}
template<int QT,int S>
__global__ __launch_bounds__(256) void paged_mqa_half_grouped(const uint16_t* q,const uint16_t* k,const float* w,
 const int* ctx,const int* tb,float* out,int B,int N,int tw,int tokens,const int* flags,const int* cctx,const int* ctb, bool clean_logits){
 const int g=blockIdx.y;const float* scales=(const float*)((const uint8_t*)k+int64_t(tokens)*256);
 const int chunks=(N-1)/(S*1024)+1;
 bool bad=false;
 for(int j=threadIdx.x%64;j<chunks;j+=64)bad|=flags[g*chunks+j]<0;
 if(__ballot(bad)==0){
    grouped_half_device<uint16_t,1,32*QT,128,64,256,4,64,QT>(q,k,w,B,0,16384,N,((N-1)/64+1),cctx,out,ctb,scales,g,clean_logits);
 }else{
    for(int i=0;i<QT&&g*QT+i<B;++i){
      grouped_half_device<uint16_t,1,32,128,S,S*4,4,64,1>(q,k,w,B,0,S*256,N,tw,ctx,out,tb,scales,g*QT+i,clean_logits);
      __syncthreads();
    }
 }
}

} // namespace deepgemm::opus
