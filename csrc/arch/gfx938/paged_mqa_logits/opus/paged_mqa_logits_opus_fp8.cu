// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "paged_mqa_logits/cta_schedule.h"
#include <ATen/hip/HIPContext.h>
#include "paged_mqa_logits/opus/paged_mqa_logits.h"
#include "paged_mqa_logits.cuh"
#include "kernels.cuh"
#include <algorithm>

namespace deepgemm::opus {
namespace {
// All temporaries belong to the current PyTorch stream/capture pool. Preparation
// runs on every invocation; no cache contents or device page IDs are cached on CPU.
void launch_fp8(const torch::Tensor& q, const torch::Tensor& cache,
                const torch::Tensor& weights, const torch::Tensor& context,
                const torch::Tensor& tables, torch::Tensor& output,
                int max_len, int kernel_id, int64_t launch_tokens, bool clean_logits)
{
    const int batch=q.size(0), next=q.size(1), heads=q.size(2), page_size=cache.size(1);
    const int tokens=cache.size(0)*page_size, table_width=tables.size(1);
    const auto stream=at::hip::getCurrentHIPStream();
    const bool grouped=kernel_id==7;
    const uint8_t* qp=static_cast<const uint8_t*>(q.data_ptr());
    const uint8_t* kp=cache.data_ptr<uint8_t>();
    const bool mls_aligned=((reinterpret_cast<uintptr_t>(qp)|reinterpret_cast<uintptr_t>(kp))%16)==0;
    if(!grouped && page_size==64 && mls_aligned) {
        // Small grids favor less serial KV work per wave; large grids reuse Q.
        const int64_t elements=int64_t(batch)*next*max_len;
        const int tile=elements<=16384 ? 64 : elements<=32768 ? 128 : 256;
        const dim3 grid(((launch_tokens+tile-1)/tile)*next,batch);
#define MLS_TILE(H,R,I) hipLaunchKernelGGL((paged_mqa_mls_single<H,R,I>), \
    deepgemm::paged_mqa_persistent_grid<R>(grid,(paged_mqa_mls_single<H,R,I>),256,9216,q.get_device(),clean_logits), \
    dim3(256),9216,stream,qp,kp, \
    weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),output.data_ptr<float>(), \
    max_len,table_width,clean_logits,batch)
#define MLS(H,R) if(tile==64) { MLS_TILE(H,R,1); } else if(tile==128) { MLS_TILE(H,R,2); } else { MLS_TILE(H,R,4); }
#define MLS_NEXT(H) if(next==1) { MLS(H,1); } else if(next==2) { MLS(H,2); } else { MLS(H,4); }
        if(heads==32) { MLS_NEXT(32) } else { MLS_NEXT(64) }
#undef MLS_NEXT
#undef MLS
#undef MLS_TILE
        return;
    }
    torch::Tensor kwork, flags, compact_context, compact_tables;
    if(grouped && page_size==1) {
        kwork=torch::empty({((int64_t(tokens)+63)/64)*64*132},cache.options());
        hipLaunchKernelGGL(pack_s1,dim3((int64_t(tokens)*33+255)/256),dim3(256),0,stream,
            reinterpret_cast<const uint32_t*>(kp),reinterpret_cast<uint32_t*>(kwork.data_ptr()),tokens);
        kp=kwork.data_ptr<uint8_t>();
    }
    if(grouped) {
        const int groups=(batch+1)/2, chunks=(max_len-1)/(page_size*1024)+1;
        const int compact_width=(max_len-1)/64+1;
        flags=torch::empty({int64_t(groups)*chunks},context.options());
        compact_context=torch::empty({groups},context.options());
        compact_tables=torch::empty({groups,compact_width},tables.options());
        hipLaunchKernelGGL(check_groups,dim3(chunks,groups),dim3(256),0,stream,
            context.data_ptr<int>(),tables.data_ptr<int>(),flags.data_ptr<int>(),
            compact_context.data_ptr<int>(),compact_tables.data_ptr<int>(),
            batch,max_len,page_size,table_width,2);
        const dim3 grid((launch_tokens+255)/256,groups);
#define GROUP_FP8(S) hipLaunchKernelGGL((paged_mqa_grouped<2,S>),grid,dim3(256),16384,stream, \
    qp,kp,weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),output.data_ptr<float>(), \
    batch,max_len,table_width,flags.data_ptr<int>(),compact_context.data_ptr<int>(),compact_tables.data_ptr<int>(),clean_logits)
        if(page_size==64 && mls_aligned) {
            hipLaunchKernelGGL((paged_mqa_mls_grouped),grid,dim3(256),10240,stream,qp,kp,weights.data_ptr<float>(),
                context.data_ptr<int>(),tables.data_ptr<int>(),output.data_ptr<float>(),batch,max_len,table_width,
                flags.data_ptr<int>(),compact_context.data_ptr<int>(),compact_tables.data_ptr<int>(),clean_logits);
        } else { if(page_size==1) { GROUP_FP8(1); } else { GROUP_FP8(64); } }
#undef GROUP_FP8
    } else {
        const dim3 grid(((launch_tokens+255)/256)*next,batch);
        const size_t smem=16384;
#define SINGLE(H,S,R) hipLaunchKernelGGL((paged_mqa_fp8_single<H,S,R>),grid,dim3(256),smem,stream, \
        qp,kp,weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),output.data_ptr<float>(), \
        batch,max_len,table_width,clean_logits)
#define NEXT(H,S) if(next==1) { SINGLE(H,S,1); } else if(next==2) { SINGLE(H,S,2); } else { SINGLE(H,S,4); }
        if(heads==32) { if(page_size==1) { NEXT(32,1) } else { NEXT(32,64) } }
        else { if(page_size==1) { NEXT(64,1) } else { NEXT(64,64) } }
#undef NEXT
#undef SINGLE
    }
}
} // namespace

void launch_paged_mqa_logits_fp8(const torch::Tensor& q, const torch::Tensor& cache,
                                 const torch::Tensor& weights, const torch::Tensor& context,
                                 const torch::Tensor& tables, torch::Tensor& output,
                                 int max_len, int kernel_id, bool clean_logits) {
    const auto batch = q.size(0), next = q.size(1), heads = q.size(2);
    // Metadata only: no device-value read or synchronization during graph capture.
    const int64_t launch_tokens=clean_logits ? max_len :
        std::min<int64_t>(max_len,tables.size(1)*cache.size(1));
    const dim3 grid(static_cast<unsigned>((launch_tokens+63)/64),static_cast<unsigned>(batch*next));
    const auto stream=at::hip::getCurrentHIPStream();
    if(kernel_id>=6) {
        launch_fp8(q,cache,weights,context,tables,output,static_cast<int>(max_len),
                   static_cast<int>(kernel_id),launch_tokens,clean_logits);
    }
    else if(kernel_id>=2) {
#define LAUNCH_DPP(H,P) hipLaunchKernelGGL((opus_kernels::paged_mqa_dpp<H,P>),grid,dim3(256),0,stream, \
        static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(), \
        weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(), \
        output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)),clean_logits)
        if(kernel_id==2) {
            if(heads==32) { LAUNCH_DPP(32,false); } else { LAUNCH_DPP(64,false); }
        } else {
            if(heads==32) { LAUNCH_DPP(32,true); } else { LAUNCH_DPP(64,true); }
        }
#undef LAUNCH_DPP
    }
    else if(kernel_id==1 && heads==32)
        hipLaunchKernelGGL((opus_kernels::paged_mqa_id0<32,true>),grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)),clean_logits);
    else if(kernel_id==1 && heads==64)
        hipLaunchKernelGGL((opus_kernels::paged_mqa_id0<64,true>),grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)),clean_logits);
    else if(heads==32)
        hipLaunchKernelGGL((opus_kernels::paged_mqa_id0<32,false>),grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)),clean_logits);
    else
        hipLaunchKernelGGL((opus_kernels::paged_mqa_id0<64,false>),grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)),clean_logits);
}

} // namespace deepgemm::opus
