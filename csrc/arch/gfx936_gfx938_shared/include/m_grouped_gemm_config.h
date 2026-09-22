#pragma once
#include <unordered_map>
#include <functional>
#include <cstdint>
#include "hip/hip_bf16.h"

namespace deepgemm {

// 类型别名
using bhalf_t = __hip_bfloat16;

///////////////////////////////////////////////////////////////////////GroupGemm params///////////////////////////////////////////////////////////////////////////////////////

// GroupGemmParams结构体定义 - 用于 grouped gemm masked 的参数
template <typename T, typename OutputType = bhalf_t>
struct GroupGemmParams {
    GroupGemmParams(
        const T* ptr_A,               // input
        const T* ptr_B,               // weight  
        OutputType* ptr_C,            // output
        float* ptr_A_scale,           // input scale
        float* ptr_B_scale,           // weight scale
        const int32_t* masked_m,      // tokens per expert
        uint32_t experts_num,         // num of experts
        uint32_t size_m,              // input size m
        uint32_t size_n,              // output size n
        uint32_t size_k,              // input size k
        uint32_t expected_m_per_group // expected m per group
    )
        : ptr_A(ptr_A),
          ptr_B(ptr_B),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
          ptr_B_scale(ptr_B_scale),
          masked_m(masked_m),
          experts_num(experts_num),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          expected_m_per_group(expected_m_per_group)
    {}

    const T* ptr_A;                    // input
    const T* ptr_B;                    // weight
    OutputType* ptr_C;                 // output
    float* ptr_A_scale;                // input scale
    float* ptr_B_scale;                // weight scale
    const int32_t* masked_m;           // tokens per expert
    uint32_t experts_num;              // num of experts
    uint32_t size_m;                   // input size m
    uint32_t size_n;                   // output size n
    uint32_t size_k;                   // input size k
    uint32_t expected_m_per_group;     // expected m per group
};

///////////////////////////////////////////////////////////////////////launch function declarations///////////////////////////////////////////////////////////////////////////////////////

// 函数声明 - CUDA 版本 decode
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_decode(const GroupGemmParams<T, OutputType>& params);

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_prefill(const GroupGemmParams<T, OutputType>& params);


template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent(const GroupGemmParams<T, OutputType>& params);


template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_bypass(const GroupGemmParams<T, OutputType>& params);

///////////////////////////////////////////////////////////////////////kernel maps///////////////////////////////////////////////////////////////////////////////////////

// 定义函数类型
using GroupGemmKernelFunc = std::function<void(const GroupGemmParams<int8_t, bhalf_t>&)>;
// GroupGemm decode 映射
// 配置格式: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES

static std::unordered_map<int, GroupGemmKernelFunc> kernel_maps_groupgemm1_decode = {
    //////////////////////////////////////////////////////////////// BLOCK_SIZE_K = 64 ////////////////////////////////////////////////////////////////
    // 基础配置 BLOCK_M=16, BLOCK_N=32, BLOCK_K=64
    //{0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    
    {0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 64, 16, 16,  64, 2, int8_t, bhalf_t>(p); }},
    {1, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {2, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 256, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {3, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向两个warp
    {4, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 16,  64, 2, int8_t, bhalf_t>(p); }},
    {5, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {6, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {7, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向重复算
    {8, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {9, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {10, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {11, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    // MN 16 64 N方向两个warp+重复算两次
    {12, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {13, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {14, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 256, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {15, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 512, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    // /////////////////////////////////////////////////// warpk 128/////////////////////////////////////////////////////////////
    {16, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 128, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {17, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 256, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {18, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 512, 16, 16, 128, 2,int8_t, bhalf_t>(p); }}, //8提升明显
    {19, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 1024, 16, 16, 128,2, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向两个warp
    {20, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {21, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {22, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {23, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 16, 128,2, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向重复算
    {24, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {25, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {26, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {27, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 32, 128,2,  int8_t, bhalf_t>(p); }},
    // 16 warp
    {28, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {29, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }}, //8提升明显
    {30, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 16, 128,2,  int8_t, bhalf_t>(p); }},
    {31, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 32, 128,2,  int8_t, bhalf_t>(p); }},

    {32, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {33, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 384, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {34, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 384, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {35, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {36, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {37, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {38, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {39, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},

    {40, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {41, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 512, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {42, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 256, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {43, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 256, 32, 16, 64, 2, int8_t, bhalf_t>(p); }},

    // MN 16 64 N方向两个warp+重复算两次
    {44, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {45, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 256, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {46, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 512, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {47, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 1024, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},

    {48, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16,  64, 128, 2, int8_t, bhalf_t>(p); }},
    {49, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 256, 16,  64, 128, 2, int8_t, bhalf_t>(p); }},
    {50, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 512, 16,  64, 128, 2, int8_t, bhalf_t>(p); }},
    {51, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 1024, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},

    {52, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 256, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {53, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 512, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {54, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 1024, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {55, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 2048, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},

    {56, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 32, 256, 2, int8_t, bhalf_t>(p); }}, 
    {57, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {58, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {59, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 2048, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    
// stages 4
    {60, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {61, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 128, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {62, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 256, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {63, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 512, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向两个warp
    {64, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {65, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {66, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {67, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向重复算
    {68, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {69, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {70, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {71, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    // MN 4, 16 64 N方向两个warp+重复算两次
    {72, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {73, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {74, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 256, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {75, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 512, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    // /////////////////////////////////////////////////// warpk 128/////////////////////////////////////////////////////////////
    {76, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 128, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {77, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 256, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {78, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 512, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向两个warp
    {79, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {80, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {81, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
     // MN 16 32  N方向重复算
    {82, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 32, 128, 4, int8_t, bhalf_t>(p); }},
    {83, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 256, 16, 32, 128, 4, int8_t, bhalf_t>(p); }},
    {84, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 32, 128, 4, int8_t, bhalf_t>(p); }},

    
    //launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 16, 16, 64, int8_t, bhalf_t>(params);
    {100 + 0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 16, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 1, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 16, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 2, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 16, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 3, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 32, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 4, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 32, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 5, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 64, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 6, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 64, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 7, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 8, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 9, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 10, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 11, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 12, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 13, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 14, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 15, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 16, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 17, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {100 + 18, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 128, 64, int8_t, bhalf_t>(p); }},
    {100 + 19, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 20, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 21, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {100 + 22, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 128, 64, int8_t, bhalf_t>(p); }},
    {100 + 23, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 24, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 25, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 64, 64, int8_t, bhalf_t>(p); }},
    {100 + 26, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 128, 64, int8_t, bhalf_t>(p); }},

    // persistent 不过LDS版本
    {200 + 0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 64, 16, 16,  64, 2, int8_t, bhalf_t>(p); }},
    {200 + 1, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 2, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 256, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 3, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 4, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 16,  64, 2, int8_t, bhalf_t>(p); }},
    {200 + 5, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 6, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 7, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 8, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 9, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 10, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 11, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 12, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 13, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 14, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 256, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 15, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 512, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 16, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 128, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {200 + 17, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 256, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {200 + 18, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 512, 16, 16, 128, 2,int8_t, bhalf_t>(p); }}, //8提升明显
    {200 + 19, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 1024, 16, 16, 128,2, int8_t, bhalf_t>(p); }},
    {200 + 20, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {200 + 21, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {200 + 22, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 128, 2,int8_t, bhalf_t>(p); }},
    {200 + 23, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 16, 128,2, int8_t, bhalf_t>(p); }},
    {200 + 24, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 25, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 26, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 27, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 32, 128,2,  int8_t, bhalf_t>(p); }},
    {200 + 28, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 29, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }}, //8提升明显
    {200 + 30, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 16, 128,2,  int8_t, bhalf_t>(p); }},
    {200 + 31, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 32, 128,2,  int8_t, bhalf_t>(p); }},
    {200 + 32, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 33, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 384, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 34, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 384, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 35, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 36, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 37, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 38, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 39, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 40, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 41, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 512, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 42, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 256, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 43, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 256, 32, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 44, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 45, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 256, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 46, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 512, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 47, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 1024, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 48, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16,  64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 49, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 256, 16,  64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 50, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 512, 16,  64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 51, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 1024, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 52, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 256, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {200 + 53, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 512, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {200 + 54, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 1024, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {200 + 55, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 2048, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {200 + 56, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 32, 256, 2, int8_t, bhalf_t>(p); }}, 
    {200 + 57, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {200 + 58, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {200 + 59, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 2048, 16, 32, 256, 2, int8_t, bhalf_t>(p); }},
    {200 + 60, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 61, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 128, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 62, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 256, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 63, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 512, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 64, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 65, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 66, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 67, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 68, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 69, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 70, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 71, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 72, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 73, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 74, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 256, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 75, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 512, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 76, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 128, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 77, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 256, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 78, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 512, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 79, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 80, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 81, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 82, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 32, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 83, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 256, 16, 32, 128, 4, int8_t, bhalf_t>(p); }},
    {200 + 84, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 32, 128, 4, int8_t, bhalf_t>(p); }},
    // A和B都过LDS版本
        // {500, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {501, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {502, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {503, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {504, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {506, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 256, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {509, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {510, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {511, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 256, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {513, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {514, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {515, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {516, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {517, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {518, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {519, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {520, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {521, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {522, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {523, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {524, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {525, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {526, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {527, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {528, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {529, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {530, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {531, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {532, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {533, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {535, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {536, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {539, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 32, 16, 32, int8_t, bhalf_t>(p); }},
        // {540, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 64, 16, 32, int8_t, bhalf_t>(p); }},
        // {541, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
        // {542, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 32, 32, int8_t, bhalf_t>(p); }},
        // {543, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 256, 128, 32, 64, 32, int8_t, bhalf_t>(p); }},
        // {544, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 128, 64, 64, 32, int8_t, bhalf_t>(p); }},
        // {545, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 16, 32, int8_t, bhalf_t>(p); }},
        // {546, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 128, 64, 32, 32, int8_t, bhalf_t>(p); }},
        // {547, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
        // {548, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
        // {549, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 64, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
        // {550, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 256, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {551, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 256, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {552, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 256, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {553, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 256, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {554, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 256, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {556, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {559, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 256, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {560, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 256, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {561, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 256, 256, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {563, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 256, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {564, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {565, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {566, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {567, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {568, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {569, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 256, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {570, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {571, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
        // {574, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {575, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {576, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {577, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {578, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {579, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {580, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {581, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {582, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {583, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {584, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
        // {585, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
        // {586, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {587, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {588, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {589, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {590, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {591, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {592, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
        // {593, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
        // {594, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
        // {595, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }}
    
 
};

static std::unordered_map<int, GroupGemmKernelFunc> kernel_maps_groupgemm2_decode = {
    //////////////////////////////////////////////////////////////// BLOCK_SIZE_K = 64 ////////////////////////////////////////////////////////////////
    // 基础配置 BLOCK_M=16, BLOCK_N=32, BLOCK_K=64
    //{0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    
    {0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {1, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向多个warp 2-8
    {2, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {3, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {4, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {5, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {6, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {7, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {8, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {9, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {10, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {11, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {12, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {13, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {14, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {15, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    // warpk 128
    {16, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {17, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {18, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {19, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {20, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {21, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {22, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {23, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {24, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {25, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {26, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {27, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {28, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},

    {29, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {30, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {31, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},

    {32, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {33, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {34, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 384, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {35, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},

    {36, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {37, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {38, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {39, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},

    {40, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {41, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {42, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {43, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},

    {44, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {45, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 1024, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {46, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {47, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},

    {48, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 64, 64, 2, int8_t, bhalf_t>(p); }},
    {49, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 64, 64, 2, int8_t, bhalf_t>(p); }},
    {50, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {51, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},

    {52, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},
    {53, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},
    {54, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},
    {55, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},

    {56, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {57, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {58, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},
    {59, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},

// stage4
    {60, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 16, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {61, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向多个warp 2-8
    {62, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {63, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {64, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {65, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {66, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {67, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    // MN 16 32  N方向重复算 + 多个warp
    {68, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {69, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {70, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {71, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 64, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {72, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {73, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {74, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {75, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 128, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    // warpk 128
    {76, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {77, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {78, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {79, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},

    {80, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 64, 64, 4, int8_t, bhalf_t>(p); }},
    {81, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 64, 64, 4, int8_t, bhalf_t>(p); }},
    {82, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 512, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {83, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 256, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},

    {84, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode<16, 128, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},

    // add
    {100 + 0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 16, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 1, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 16, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 2, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 16, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 3, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 32, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 4, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 32, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 5, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 64, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 6, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 64, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 7, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 8, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 9, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 10, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 11, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 12, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 13, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 14, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 64, 64, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 15, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 16, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 17, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {100 + 18, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<16, 128, 64, 16, 128, 64, int8_t, bhalf_t>(p); }},
    {100 + 19, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 20, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 21, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {100 + 22, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 16, 128, 64, int8_t, bhalf_t>(p); }},
    {100 + 23, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {100 + 24, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {100 + 25, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 64, 64, int8_t, bhalf_t>(p); }},
    {100 + 26, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 128, 64, 32, 128, 64, int8_t, bhalf_t>(p); }},

    // persistent
    {200 + 0, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 1, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 2, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 3, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 4, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 5, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 6, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 7, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 128, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 8, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 9, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 10, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 11, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 12, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 13, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 14, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 15, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 128, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 16, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 17, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 18, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 19, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 20, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 21, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 22, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 23, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 24, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 25, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 26, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 27, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 28, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 29, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 30, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 31, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 32, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 33, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 34, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 384, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 35, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 384, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 36, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 37, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 38, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 39, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 40, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 1024, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 41, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 1024, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 42, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 43, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 512, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 44, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 45, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 1024, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 46, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 128, 16, 32, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 47, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 128, 16, 16, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 48, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 64, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 49, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 64, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 50, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 32, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 51, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 16, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 52, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 53, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 54, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 55, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 128, 64, 2, int8_t, bhalf_t>(p); }},
    {200 + 56, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 57, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 128, 16, 64, 128, 2, int8_t, bhalf_t>(p); }},
    {200 + 58, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 59, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 60, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 16, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 61, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 62, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 63, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 64, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 65, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 66, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 67, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 68, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 69, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 70, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 71, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 64, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 72, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 73, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 74, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 75, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 128, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 76, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 32, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 77, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 78, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 79, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 80, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 64, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 81, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 64, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 82, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 512, 64, 16, 32, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 83, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 256, 64, 16, 16, 64, 4, int8_t, bhalf_t>(p); }},
    {200 + 84, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent<16, 128, 64, 16, 128, 64, 4, int8_t, bhalf_t>(p); }},
#if 0
    {500, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {501, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {502, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {503, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {504, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {506, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 256, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {509, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {510, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {511, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 256, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {513, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {514, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {515, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {516, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {517, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {518, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {519, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {520, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {521, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {522, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {523, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {524, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {525, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {526, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {527, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {528, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {529, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {530, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {531, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {532, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {533, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {535, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {536, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {539, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 32, 16, 32, int8_t, bhalf_t>(p); }},
    {540, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 64, 16, 32, int8_t, bhalf_t>(p); }},
    {541, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
    {542, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 32, 32, int8_t, bhalf_t>(p); }},
    {543, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 256, 128, 32, 64, 32, int8_t, bhalf_t>(p); }},
    {544, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 128, 64, 64, 32, int8_t, bhalf_t>(p); }},
    {545, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 16, 32, int8_t, bhalf_t>(p); }},
    {546, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 128, 64, 32, 32, int8_t, bhalf_t>(p); }},
    {547, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
    {548, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
    {549, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 64, 128, 32, 32, 32, int8_t, bhalf_t>(p); }},
    {550, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 256, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {551, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 256, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {552, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 256, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {553, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 256, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {554, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 256, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {556, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {559, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 256, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {560, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 256, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {561, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 256, 256, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {563, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 256, 256, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {564, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {565, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {566, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {567, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {568, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {569, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 256, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {570, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {571, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 64, 16, 64, int8_t, bhalf_t>(p); }},
    {574, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {575, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {576, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {577, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {578, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {579, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {580, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 128, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {581, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 128, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {582, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {583, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 64, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {584, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 32, 128, 32, 16, 64, int8_t, bhalf_t>(p); }},
    {585, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<128, 64, 128, 32, 32, 64, int8_t, bhalf_t>(p); }},
    {586, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {587, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {588, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {589, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {590, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {591, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {592, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }},
    {593, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<32, 64, 128, 16, 64, 64, int8_t, bhalf_t>(p); }},
    {594, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 16, 128, 16, 16, 64, int8_t, bhalf_t>(p); }},
    {595, [](const GroupGemmParams<int8_t, bhalf_t>& p) { launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<64, 32, 128, 16, 32, 64, int8_t, bhalf_t>(p); }}
#endif 
};


} // namespace deepgemm
