// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "fp8.cuh"
#include "mls.cuh"

namespace deepgemm::opus {

// ID6: every request has its own page table. No sharing assumption is made.
template<int H,int S,int R>
__global__ __launch_bounds__(256) void paged_mqa_fp8_single(const uint8_t* q,const uint8_t* k,const float* w,
 const int* ctx,const int* tb,float* out,int B,int N,int tw, bool clean_logits) {
    grouped_device<uint8_t,R,H,128,S,S*4,4,64,true,1,false>(q,k,(const float*)(k+S*128),w,B,0,S*132,N,tw,ctx,out,tb,blockIdx.y,clean_logits);
}
template<int QT,int S>
__global__ __launch_bounds__(256) void paged_mqa_grouped(const uint8_t* q,const uint8_t* k,const float* w,
 const int* ctx,const int* tb,float* out,int B,int N,int tw,const int* flags,const int* cctx,const int* ctb, bool clean_logits){
 const int g=blockIdx.y;
 const int chunks=(N-1)/(S*1024)+1;
 bool bad=false;
 for(int j=threadIdx.x%64;j<chunks;j+=64)bad|=flags[g*chunks+j]<0;
 if(__ballot(bad)==0){
    grouped_device<uint8_t,1,32*QT,128,64,256,4,64,true,QT,true>(q,k,(const float*)(k+8192),w,B,0,8448,N,((N-1)/64+1),cctx,out,ctb,g,clean_logits);
 }else{
    for(int i=0;i<QT&&g*QT+i<B;++i){
      grouped_device<uint8_t,1,32,128,S,S*4,4,64,true,1,true>(q,k,(const float*)(k+S*128),w,B,0,S*132,N,tw,ctx,out,tb,g*QT+i,clean_logits);
      __syncthreads();
    }
 }
}
} // namespace deepgemm::opus
