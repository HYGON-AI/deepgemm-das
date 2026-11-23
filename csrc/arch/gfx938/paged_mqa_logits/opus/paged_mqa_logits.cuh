// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#endif

namespace deepgemm::opus::opus_kernels {

// Four waves, logical GEMM [H,128] x [64,128]^T. Each CTA owns all heads
// for one query and token tile; no global head workspace or atomic reduction.
template<int Heads, bool WaveReduce=false>
__global__ __launch_bounds__(256)
void paged_mqa_id0(const unsigned char* q, const unsigned char* cache,
                   const float* weights, const int* context, const int* tables,
                   float* output, int next_n, int max_len, int table_width, bool clean_logits)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx938__)
    using namespace ::opus;
    using ::opus::fp8_t;
    constexpr int BN=64, BK=32, LD=36, EM=Heads/32;
    const int tid=thread_id_x(), wave=tid/64, lane=tid%64;
    const int row=block_id_y(), batch=row/next_n, query=row%next_n;
    const int token0=block_id_x()*BN;
    const int length=context[batch];
    const int valid_end=length-next_n+query+1;
    if(length < next_n || length > max_len || length > table_width || token0 >= valid_end) {
        // ID0/1 used to rely on the host fill for entirely causal tiles.
        // No-clean must still initialize the causal positions inside context.
        if(!clean_logits) {
            if(tid<BN && token0+tid<max_len && token0+tid<length)
                output[static_cast<long long>(row)*max_len+token0+tid]=-__builtin_inff();
        }
        return;
    }
    __shared__ fp8_t qa[Heads*LD];
    __shared__ fp8_t kb[BN*LD];
    // ID0 retains the original full-head scratch. ID1 reduces each 16-head
    // lane group before writing LDS, leaving only Heads/16 partial rows.
    constexpr int PartialRows=WaveReduce ? Heads/16 : Heads;
    __shared__ float scores[PartialRows*BN];
    __shared__ float scales[BN];
    __shared__ int pages[BN];
    if(tid<BN) {
        const int n=token0+tid;
        const int p=n<length ? tables[static_cast<long long>(batch)*table_width+n] : 0;
        pages[tid]=p;
        scales[tid]=n<length ? *reinterpret_cast<const float*>(cache+static_cast<long long>(p)*132+128) : 0;
    }
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto gq=make_gmem(reinterpret_cast<const fp8_t*>(q)+static_cast<long long>(row)*Heads*128);
    auto gk=make_gmem(reinterpret_cast<const fp8_t*>(cache));
    auto sa=make_smem(qa), sb=make_smem(kb);
    auto mma=make_tiled_mma<fp8_t,fp8_t,fp32_t>(
        seq<EM,2,1>{},seq<2,2,1>{},seq<16,16,32>{},mmac_adaptor_hcu{});
    auto pa=::opus::make_tuple(wave/2,lane%mma.grpm_a,0,lane/mma.grpm_a);
    auto pb=::opus::make_tuple(wave%2,lane%mma.grpn_b,0,lane/mma.grpn_b);
    auto ra=partition_layout_a<8>(mma,::opus::make_tuple(LD,1_I),pa);
    auto rb=partition_layout_b<8>(mma,::opus::make_tuple(LD,1_I),pb);
    typename decltype(mma)::vtype_c acc{0};
    for(int k0=0;k0<128;k0+=BK) {
        static_for<Heads/32>([&](auto i) {
            const int head=tid/8+i.value*32, kv=tid%8;
            auto uq=make_layout<4>(::opus::make_tuple(number<Heads>{},8_I,4_I),
                ::opus::make_tuple(128,4_I,1_I),::opus::make_tuple(head,kv,underscore{}));
            auto us=make_layout<4>(::opus::make_tuple(number<Heads>{},8_I,4_I),
                ::opus::make_tuple(LD,4_I,1_I),::opus::make_tuple(head,kv,underscore{}));
            auto value=load<4>(gq,uq,k0);
            store<4>(sa,value,us);
        });
        static_for<2>([&](auto i) {
            const int n=tid/8+i.value*32, kv=tid%8;
            fp8x4_t value{};
            if(token0+n<length) value=load<4>(gk,pages[n]*132+k0+kv*4);
            store<4>(sb,value,n*LD+kv*4);
        });
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        auto a=load<8>(sa,ra);
        auto b=load<8>(sb,rb);
        s_waitcnt_lgkmcnt(0_I);
        acc=mma(a,b,acc);
        __builtin_amdgcn_s_barrier();
    }
    auto pc=::opus::make_tuple(wave/2,lane%mma.grpn_c,wave%2,lane/mma.grpn_c);
    auto uc=partition_layout_c<1>(mma,::opus::make_tuple(BN,1_I),pc);
    auto offsets=layout_to_offsets<1>(uc);
    static_for<EM*2*4>([&](auto i) {
        const int offset=offsets[i];
        const int head=offset/BN, n=offset%BN;
        const float score=__builtin_fmaxf(acc[i.value]*scales[n],0.0f);
        float value=score*weights[static_cast<long long>(row)*Heads+head];
        if constexpr(WaveReduce) {
            // HCU adaptor: m=lane%16, n=4*s+lane/16. XOR 1/2/4/8
            // therefore reduces heads while preserving the logical token.
            value+=__shfl_xor(value,1,64);
            value+=__shfl_xor(value,2,64);
            value+=__shfl_xor(value,4,64);
            value+=__shfl_xor(value,8,64);
            if(lane%16==0) scores[(head/16)*BN+n]=value;
        } else {
            scores[offset]=value;
        }
    });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    if(tid<BN && token0+tid<max_len && (clean_logits || token0+tid<length)) {
        float sum=0;
        for(int head=0;head<PartialRows;head++) sum+=scores[head*BN+tid];
        output[static_cast<long long>(row)*max_len+token0+tid]=
            token0+tid<valid_end ? sum : -__builtin_inff();
    }
#endif
#endif
}
// Tuned N64 path: DPP head reduction and complete invalid-tail ownership.
// Prefetch=false retains K32 staging for medium grids. Prefetch=true gathers
// D128 into registers and stages K64 twice, reducing CTA rendezvous.
template<int Heads, bool Prefetch>
__global__ __launch_bounds__(256)
void paged_mqa_dpp(const unsigned char* q, const unsigned char* cache,
                   const float* weights, const int* context, const int* tables,
                   float* output, int next_n, int max_len, int table_width, bool clean_logits)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx938__)
    using namespace ::opus;
    using ::opus::fp8_t;
    constexpr int BK=Prefetch ? 64 : 32, BN=64, LD=BK+4, EM=Heads/32;
    constexpr int Columns=BK/4, GroupRows=256/Columns;
    constexpr int QLoads=Heads/GroupRows, KLoads=BN/GroupRows, Stages=128/BK;
    const int tid=thread_id_x(), wave=tid/64, lane=tid%64;
    const int row=block_id_y(), batch=row/next_n, query=row%next_n;
    const int token0=block_id_x()*BN;
    const int length=context[batch];
    const int valid_end=length-next_n+query+1;
    if(length < next_n || length > max_len || length > table_width || token0 >= valid_end) {
        if(tid<BN && token0+tid<max_len && (clean_logits || token0+tid<length))
            output[static_cast<long long>(row)*max_len+token0+tid]=-__builtin_inff();
        return;
    }
    __shared__ fp8_t qa[Heads*LD];
    __shared__ fp8_t kb[BN*LD];
    constexpr int PartialRows=Heads/16;
    __shared__ float scores[PartialRows*BN];
    __shared__ float scales[BN];
    __shared__ int pages[BN];
    if(tid<BN) {
        const int n=token0+tid;
        const int p=n<length ? tables[static_cast<long long>(batch)*table_width+n] : 0;
        pages[tid]=p;
        scales[tid]=n<length ? *reinterpret_cast<const float*>(cache+static_cast<long long>(p)*132+128) : 0;
    }
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto gq=make_gmem(reinterpret_cast<const fp8_t*>(q)+static_cast<long long>(row)*Heads*128);
    auto gk=make_gmem(reinterpret_cast<const fp8_t*>(cache));
    auto sa=make_smem(qa), sb=make_smem(kb);
    auto mma=make_tiled_mma<fp8_t,fp8_t,fp32_t>(
        seq<EM,2,1>{},seq<2,2,1>{},seq<16,16,32>{},mmac_adaptor_hcu{});
    auto pa=::opus::make_tuple(wave/2,lane%mma.grpm_a,0,lane/mma.grpm_a);
    auto pb=::opus::make_tuple(wave%2,lane%mma.grpn_b,0,lane/mma.grpn_b);
    auto ra=partition_layout_a<8>(mma,::opus::make_tuple(LD,1_I),pa);
    auto rb=partition_layout_b<8>(mma,::opus::make_tuple(LD,1_I),pb);
    typename decltype(mma)::vtype_c acc{0};
    // Keep four-byte global loads: the packed 132-byte page stride and
    // public storage contract do not guarantee eight-byte alignment.
    // Compile-time stage indices keep prefetched values in registers.
    fp8x4_t pre_q[Stages][QLoads], pre_k[Stages][KLoads];
    if constexpr(Prefetch) {
        static_for<Stages>([&](auto stage) {
            static_for<QLoads>([&](auto i) {
                const int head=tid/Columns+i.value*GroupRows, kv=tid%Columns;
                pre_q[stage.value][i.value]=load<4>(gq,head*128+stage.value*BK+kv*4);
            });
            static_for<KLoads>([&](auto i) {
                const int n=tid/Columns+i.value*GroupRows, kv=tid%Columns;
                fp8x4_t value{};
                if(token0+n<length) value=load<4>(gk,pages[n]*132+stage.value*BK+kv*4);
                pre_k[stage.value][i.value]=value;
            });
        });
        s_waitcnt_vmcnt(0_I);
    }
    auto compute_stage=[&](auto stage) {
        const int k0=stage*BK;
        static_for<QLoads>([&](auto i) {
            const int head=tid/Columns+i.value*GroupRows, kv=tid%Columns;
            auto uq=make_layout<4>(::opus::make_tuple(number<Heads>{},number<Columns>{},4_I),
                ::opus::make_tuple(128,4_I,1_I),::opus::make_tuple(head,kv,underscore{}));
            auto us=make_layout<4>(::opus::make_tuple(number<Heads>{},number<Columns>{},4_I),
                ::opus::make_tuple(LD,4_I,1_I),::opus::make_tuple(head,kv,underscore{}));
            fp8x4_t value;
            if constexpr(Prefetch) value=pre_q[stage][i.value];
            else value=load<4>(gq,uq,k0);
            store<4>(sa,value,us);
        });
        static_for<KLoads>([&](auto i) {
            const int n=tid/Columns+i.value*GroupRows, kv=tid%Columns;
            fp8x4_t value{};
            if constexpr(Prefetch) value=pre_k[stage][i.value];
            else if(token0+n<length) value=load<4>(gk,pages[n]*132+k0+kv*4);
            store<4>(sb,value,n*LD+kv*4);
        });
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        static_for<BK/32>([&](auto k) {
            auto ska=make_smem(qa+k.value*32), skb=make_smem(kb+k.value*32);
            auto a=load<8>(ska,ra);
            auto b=load<8>(skb,rb);
            s_waitcnt_lgkmcnt(0_I);
            acc=mma(a,b,acc);
        });
        __builtin_amdgcn_s_barrier();
    };
    if constexpr(Prefetch) {
        static_for<Stages>([&](auto k) { compute_stage(k); });
    } else {
        for(int k=0;k<Stages;k++) compute_stage(k);
    }
    auto pc=::opus::make_tuple(wave/2,lane%mma.grpn_c,wave%2,lane/mma.grpn_c);
    auto uc=partition_layout_c<1>(mma,::opus::make_tuple(BN,1_I),pc);
    auto offsets=layout_to_offsets<1>(uc);
    static_for<EM*2*4>([&](auto i) {
        const int offset=offsets[i];
        const int head=offset/BN, n=offset%BN;
        const float score=__builtin_fmaxf(acc[i.value]*scales[n],0.0f);
        float value=score*weights[static_cast<long long>(row)*Heads+head];
        // HCU fragment: m=lane%16, n=4*s+lane/16. These DPP controls
        // sum one 16-head group at lane 15, without LDS-backed shuffles.
        value+=mov_dpp(value,number<0xb1>{});  // quad permutation XOR 1
        value+=mov_dpp(value,number<0x4e>{});  // quad permutation XOR 2
        value+=mov_dpp(value,number<0x114>{}); // row shift right 4
        value+=mov_dpp(value,number<0x118>{}); // row shift right 8
        if(lane%16==15) scores[(head/16)*BN+n]=value;
    });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    if(tid<BN && token0+tid<max_len && (clean_logits || token0+tid<length)) {
        float sum=0;
        for(int head=0;head<PartialRows;head++) sum+=scores[head*BN+tid];
        output[static_cast<long long>(row)*max_len+token0+tid]=
            token0+tid<valid_end ? sum : -__builtin_inff();
    }
#endif
#endif
}
} // namespace deepgemm::opus::opus_kernels
