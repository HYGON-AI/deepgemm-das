// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "paged_mqa_logits/opus/paged_mqa_logits.h"
#include "kernels.cuh"
#include <algorithm>
#include <climits>
#include <cstring>
#include <optional>

namespace deepgemm::opus {
namespace {
// All temporaries belong to the current PyTorch stream/capture pool. Preparation
// runs on every invocation; no cache contents or device page IDs are cached on CPU.
void launch_half(const torch::Tensor& q, const torch::Tensor& cache,
                 const torch::Tensor& weights, const torch::Tensor& context,
                 const torch::Tensor& tables, torch::Tensor& output,
                 int max_len, int kernel_id, bool clean_logits)
{
    const int batch=q.size(0), next=q.size(1), heads=q.size(2), page_size=cache.size(1);
    const int tokens=cache.size(0)*page_size, table_width=tables.size(1);
    // Use only tensor metadata so capture/replay can change device lengths.
    const int64_t launch_tokens=clean_logits ? max_len :
        std::min<int64_t>(max_len,int64_t(table_width)*page_size);
    const auto stream=at::hip::getCurrentHIPStream();
    const bool grouped=kernel_id==7;
    const uint8_t* qp=static_cast<const uint8_t*>(q.data_ptr());
    const uint8_t* kp=cache.data_ptr<uint8_t>();
    if(!grouped && page_size==64 && !(batch>=32 && int64_t(tokens)<=2LL*max_len) &&
       !(batch==1 && max_len<=8192)) {
        auto converted_q=torch::empty(q.sizes(),q.options().dtype(at::kHalf));
        hipLaunchKernelGGL(convert_q,dim3((q.numel()+255)/256),dim3(256),0,stream,qp,
            reinterpret_cast<uint16_t*>(converted_q.data_ptr()),static_cast<int>(q.numel()));
        const dim3 grid(((launch_tokens+255)/256)*next,batch);
#define FUSED(H,R) hipLaunchKernelGGL((paged_mqa_direct_half<H,R>),grid,dim3(256),0,stream, \
    reinterpret_cast<const uint16_t*>(converted_q.data_ptr()),kp,weights.data_ptr<float>(), \
    context.data_ptr<int>(),tables.data_ptr<int>(),output.data_ptr<float>(),max_len,table_width,clean_logits)
#define FUSED_NEXT(H) if(next==1) { FUSED(H,1); } else if(next==2) { FUSED(H,2); } else { FUSED(H,4); }
        if(heads==32) { FUSED_NEXT(32) } else { FUSED_NEXT(64) }
#undef FUSED_NEXT
#undef FUSED
        return;
    }
    torch::Tensor qwork, kwork, flags, compact_context, compact_tables;
    {
        qwork=torch::empty(q.sizes(),q.options().dtype(at::kHalf));
        kwork=torch::empty({((int64_t(tokens)+63)/64)*64*260},cache.options());
        hipLaunchKernelGGL(convert_q,dim3((q.numel()+255)/256),dim3(256),0,stream,
            qp,reinterpret_cast<uint16_t*>(qwork.data_ptr()),static_cast<int>(q.numel()));
        if(page_size==64) {
            hipLaunchKernelGGL((convert_cache_vector<64>),dim3((int64_t(tokens)*8+255)/256),dim3(256),0,stream,
                kp,kwork.data_ptr<uint8_t>(),tokens);
        } else {
            hipLaunchKernelGGL((convert_cache_vector<1>),dim3((int64_t(tokens)*8+255)/256),dim3(256),0,stream,
                kp,kwork.data_ptr<uint8_t>(),tokens);
        }
        qp=static_cast<const uint8_t*>(qwork.data_ptr());
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
#define GROUP_FP16(S) hipLaunchKernelGGL((paged_mqa_half_grouped<2,S>),grid,dim3(256),16640,stream, \
    reinterpret_cast<const uint16_t*>(qp),reinterpret_cast<const uint16_t*>(kp),weights.data_ptr<float>(), \
    context.data_ptr<int>(),tables.data_ptr<int>(),output.data_ptr<float>(),batch,max_len,table_width,tokens, \
    flags.data_ptr<int>(),compact_context.data_ptr<int>(),compact_tables.data_ptr<int>(),clean_logits)
        if(page_size==1) { GROUP_FP16(1); } else { GROUP_FP16(64); }
#undef GROUP_FP16
    } else {
        const dim3 grid(((launch_tokens+255)/256)*next,batch);
        const size_t smem=heads==64 ? 16640 : 16384;
#define SINGLE(H,S,R) \
    hipLaunchKernelGGL((paged_mqa_fp16_single<H,S,R>),grid,dim3(256),smem,stream, \
        reinterpret_cast<const uint16_t*>(qp),reinterpret_cast<const uint16_t*>(kp),weights.data_ptr<float>(), \
        context.data_ptr<int>(),tables.data_ptr<int>(),output.data_ptr<float>(),batch,max_len,table_width,tokens,clean_logits)
#define NEXT(H,S) if(next==1) { SINGLE(H,S,1); } else if(next==2) { SINGLE(H,S,2); } else { SINGLE(H,S,4); }
        if(heads==32) { if(page_size==1) { NEXT(32,1) } else { NEXT(32,64) } }
        else { if(page_size==1) { NEXT(64,1) } else { NEXT(64,64) } }
#undef NEXT
#undef SINGLE
    }
}
} // namespace

torch::Tensor paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& cache,
                               const torch::Tensor& weights, const torch::Tensor& context,
                               const torch::Tensor& tables, std::optional<torch::Tensor> supplied_output,
                               int64_t max_len, int64_t kernel_id, bool clean_logits)
{
    TORCH_CHECK(q.is_cuda(), "paged_mqa_logits: q must be on GPU");
    for(const auto& t : {q,cache,weights,context,tables}) {
        TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous() && !t.requires_grad(),
                    "paged_mqa_logits: tensors must be contiguous inference tensors on the same GPU");
    }
    TORCH_CHECK(q.dim()==4 && q.scalar_type()==at::ScalarType::Float8_e4m3fn,
                "paged_mqa_logits: q must be E4M3FN [B,R,H,128]");
    const auto batch=q.size(0), next=q.size(1), heads=q.size(2);
    TORCH_CHECK(batch>0 && (next==1 || next==2 || next==4) &&
                (heads==32 || heads==64) && q.size(3)==128 && batch*next<=65535,
                "paged_mqa_logits: requires B>0, B*R<=65535, R=1/2/4, H=32/64, D=128");
    TORCH_CHECK(cache.scalar_type()==at::kByte && cache.dim()==4 && cache.size(0)>0 &&
                (cache.size(1)==1 || cache.size(1)==64) && cache.size(2)==1 && cache.size(3)==132 &&
                cache.size(0)<=INT_MAX/(cache.size(1)*132),
                "paged_mqa_logits: cache must be uint8 [P,S,1,132], S=1/64, P*S*132<=INT_MAX");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr())%4==0 &&
                reinterpret_cast<uintptr_t>(cache.data_ptr())%4==0,
                "paged_mqa_logits: Q/cache storage must be 4-byte aligned");
    TORCH_CHECK(weights.scalar_type()==at::kFloat && weights.dim()==2 &&
                weights.size(0)==batch*next && weights.size(1)==heads,
                "paged_mqa_logits: weights must be FP32 [B*R,H]");
    TORCH_CHECK(context.scalar_type()==at::kInt && context.dim()==1 && context.size(0)==batch,
                "paged_mqa_logits: context must be int32 [B]");
    TORCH_CHECK(tables.scalar_type()==at::kInt && tables.dim()==2 && tables.size(0)==batch &&
                tables.size(1)>0 && tables.size(1)<=INT_MAX,
                "paged_mqa_logits: tables must be int32 [B,T]");
    TORCH_CHECK(max_len>0 && max_len<=INT_MAX,"paged_mqa_logits: max_len must be a positive int32 value");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(q));
    // Cache per calling thread/device; never infer a tensor's architecture from
    // GPU_ARCHS or from a different current device.
    static thread_local int cached_device=-1;
    static thread_local bool supported=false;
    static thread_local bool is_gfx936=false;
    const int device=q.get_device();
    if(cached_device!=device) {
        hipDeviceProp_t prop{};
        TORCH_CHECK(hipGetDeviceProperties(&prop,device)==hipSuccess,"cannot query HIP device");
        is_gfx936=std::strncmp(prop.gcnArchName,"gfx936",6)==0;
        supported=is_gfx936 || std::strncmp(prop.gcnArchName,"gfx938",6)==0;
        cached_device=device;
    }
    TORCH_CHECK(supported,"paged_mqa_logits: supported architectures are gfx936/gfx938");
    if(kernel_id==-1) {
        // For gfx938 no-clean, small storage favors persistent tiles. A large
        // shared prefix can still favor grouped; preserve its allocation band.
        // Allocation metadata only: retain clean/S1 and every explicit ID.
        const int64_t physical_tokens=cache.size(0)*cache.size(1);
        const bool prefer_persistent=!is_gfx936 && !clean_logits && cache.size(1)==64 &&
            ((reinterpret_cast<uintptr_t>(q.data_ptr()) |
              reinterpret_cast<uintptr_t>(cache.data_ptr())) % 16)==0 &&
            (physical_tokens<=max_len/2 || physical_tokens>max_len);
        kernel_id=(heads==32 && next==1 && batch>=32 && max_len>=(is_gfx936?4096:16384) &&
                   physical_tokens<=2*max_len && !prefer_persistent) ? 7 : 6;
    }
    TORCH_CHECK(kernel_id>=0 && kernel_id<=7,"paged_mqa_logits: kernelId must be in [0,7]");
    TORCH_CHECK(kernel_id>=6 || cache.size(1)==1,"paged_mqa_logits: ID0-5 require S=1");
    TORCH_CHECK(kernel_id!=7 || (heads==32 && next==1),"paged_mqa_logits: ID7 requires H=32,R=1");
    TORCH_CHECK(is_gfx936 ? kernel_id>=6 : (kernel_id<=3 || kernel_id>=6),
                "paged_mqa_logits: gfx936 requires ID6-7; gfx938 ID0-3/6-7");
    // Eager allocation and metadata validation share this native entry. The
    // supplied-output path retains storage-alias and inference-only checks.
    torch::Tensor output=supplied_output ? *supplied_output :
        torch::empty({batch*next,max_len},q.options().dtype(at::kFloat));
    TORCH_CHECK(output.is_cuda() && output.device()==q.device() && output.is_contiguous() && !output.requires_grad() &&
                output.scalar_type()==at::kFloat && output.dim()==2 && output.size(0)==batch*next &&
                output.size(1)==max_len,
                "paged_mqa_logits: output must be contiguous FP32 [B*R,max_len] on input GPU");
    if(supplied_output)for(const auto& t : {q,cache,weights,context,tables})
        TORCH_CHECK(!output.is_alias_of(t),"paged_mqa_logits: output must not alias inputs");
    if(clean_logits && kernel_id<2)output.fill_(-INFINITY);
    if (is_gfx936) {
        launch_half(q, cache, weights, context, tables, output,
                    static_cast<int>(max_len), static_cast<int>(kernel_id), clean_logits);
    } else {
#ifdef DEEPGEMM_ENABLE_GFX938
        launch_paged_mqa_logits_fp8(q, cache, weights, context, tables, output,
                                    static_cast<int>(max_len), static_cast<int>(kernel_id), clean_logits);
#else
        TORCH_CHECK(false, "paged_mqa_logits: OPUS FP8 kernels were not built for gfx938");
#endif
    }
    const auto err=hipGetLastError();
    TORCH_CHECK(err==hipSuccess,"paged_mqa_logits launch: ",hipGetErrorString(err));
    return output;
}

} // namespace deepgemm::opus
