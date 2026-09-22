#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hip_common.h"
#include "asm_m_grouped_w8a8_gemm_nt_contiguous.h"

using half_t  = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))


__global__ void build_grouped_counts_offsets_kernel(
    const int32_t* __restrict__ m_indices,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int size_m,
    int experts_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size_m) return;
    int expert = m_indices[i];
    if (expert < 0 || expert >= experts_num) return;
    atomicAdd(&counts[expert], 1);
    if (i == 0 || m_indices[i - 1] != expert) {
        offsets[expert] = i;
    }
}

template <typename T>
struct __attribute__((packed)) GemmInfo {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    T* d;
    T* c;
    int8_t* a;
    int8_t* b;
    uint32_t strideD1;
    uint32_t strideD2;
    uint32_t strideC1;
    uint32_t strideC2;
    uint32_t strideA1;
    uint32_t strideA2;
    uint32_t strideB1;
    uint32_t strideB2;
    int8_t  alpha[16];
    int8_t   beta[16];
    float* scaleA=nullptr;
    float* scaleB=nullptr;
};

namespace deepgemm {

constexpr int kContiguousPublicMaskedLayoutMode = 2;
enum class ContiguousBigTileLayout {
    kLegacy,
    kMaskedLayout,
};

void _m_grouped_w8a8_gemm_nt_contig_no_pad_asm_impl(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output,
    torch::Tensor a_scale, 
    torch::Tensor b_scale, 
    torch::Tensor m_indices,
    torch::Tensor token_per_expert, 
    uint32_t experts_num) {
    
    std::vector<GemmInfo<bhalf_t>> d_prob_vec;

    const int token_num = input.size(0);
    const int size_n = b_qweight.size(1) * 16;
    const int size_k = input.size(1);
    int32_t lda = size_k;
    int32_t ldb = size_k;
    int32_t ldc = size_n;
    int32_t ldd = size_n;

    float d_alpha = 1.0;
    float d_beta = 0.0;

    int8_t* inp = input.data_ptr<int8_t>();
    int8_t* weight = b_qweight.data_ptr<int8_t>();
    int32_t* m_indices_ptr = m_indices.data_ptr<int32_t>();
    bhalf_t* output_ptr = reinterpret_cast<bhalf_t *>(output.data_ptr());
    float* scaleA = a_scale.data_ptr<float>();
    float* scaleB = b_scale.data_ptr<float>();

    size_t 	size_nk = size_k * size_n;
    size_t stride_b = size_n * size_k;

    int32_t* token_per_expert_ptr = token_per_expert.data_ptr<int32_t>();
    int32_t start_token = 0;
    for (int i = 0, j = 0; i < token_num; ++j) {
        GemmInfo<bhalf_t> d_prob;
        int32_t expertId = m_indices_ptr[i];
        int32_t actual_size_m = token_per_expert_ptr[expertId];
        size_t stride_a = actual_size_m * size_k;
        size_t stride_c = actual_size_m * size_n;
        size_t stride_d = actual_size_m * size_n;
        d_prob.m = size_n;
        d_prob.n = actual_size_m;
        d_prob.batch = 1;
        d_prob.k = size_k;
        d_prob.strideD1 = ldd;
        d_prob.strideD2 = stride_d;
        d_prob.strideC1 = ldc;
        d_prob.strideC2 = stride_c;
        d_prob.strideA1 = ldb;
        d_prob.strideA2 = stride_b;
        d_prob.strideB1 = lda;
        d_prob.strideB2 = stride_a;

        memcpy(d_prob.alpha, &d_alpha, sizeof(float));
        memcpy(d_prob.beta,  &d_beta, sizeof(float));

        d_prob.a = weight + expertId * stride_b;
        //std::cout << std::hex << reinterpret_cast<void*>(d_prob.a) << std::endl;
        d_prob.b = inp + start_token * lda;
        // std::cout << std::hex << reinterpret_cast<void*>(d_prob.b) << std::endl;
        d_prob.c = output_ptr + start_token * ldc;
        d_prob.d = output_ptr + start_token * ldd;
        d_prob.scaleA = scaleB + expertId * ldd; 
        d_prob.scaleB = scaleA + start_token;
        
        start_token += actual_size_m;
        i += actual_size_m;
        d_prob_vec.push_back(d_prob);
    }
    //printf("d_pro_vec size : %d\n", d_prob_vec.size());
    size_t localWorkSize[3] = {768, 1, 1};
    size_t globalWorkSize[3] = {1, 1, 1};

    size_t wgTile = 0;
    for (int i = 0; i < d_prob_vec.size(); ++i) {
        // printf("MNK %d,%d,%d,%d,GM=%d,", d_prob_vec[i].m, d_prob_vec[i].n, 1, d_prob_vec[i].k, i);
        // printf(" wgRight%d\n",wgTile);
        size_t size_m = d_prob_vec[i].m;
        size_t size_n = d_prob_vec[i].n;
        int wg_M = DIVIDE(size_m, 256);
        int wg_N = DIVIDE(size_n, 256);
        wgTile += wg_M * wg_N;
    }
    

	struct __attribute__((packed)){
        uint32_t gemm_count;              //4
        void const* DeviceUserArguments;  //8
        void const* argsPtr;              //8
        uint32_t wgm;                     //4
        unsigned int gsu;                 //4
        int32_t* m_indics;               //8
        int32_t* debug_d;                                
    }hipFunctionArgs;

    uint8_t* d_argsPtr;
    size_t arg_size =  d_prob_vec.size() * sizeof(GemmInfo<bhalf_t>);
    hipMalloc((void**)&d_argsPtr, arg_size);
    hipMemcpy(d_argsPtr, d_prob_vec.data(), arg_size, hipMemcpyHostToDevice);

    hipFunctionArgs.gemm_count = d_prob_vec.size();
    hipFunctionArgs.DeviceUserArguments = d_argsPtr;
    hipFunctionArgs.argsPtr = nullptr;
    hipFunctionArgs.wgm = 1;
    hipFunctionArgs.gsu = 1;
    //hipFunctionArgs.m_indics = m_indices.data_ptr<int32_t>();

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    char func_name_fp[1024], fp16_co_file_tn[1024], bf16_co_file_tn[1024], func_name_bf[1024];
    memset(func_name_fp, 0x0, sizeof(func_name_fp));
    memset(fp16_co_file_tn, 0x0, sizeof(fp16_co_file_tn));
    memset(func_name_bf, 0x0, sizeof(func_name_bf));
    memset(bf16_co_file_tn, 0x0, sizeof(bf16_co_file_tn));
    
    sprintf(func_name_bf, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16", 256, 256, 128);
    sprintf(bf16_co_file_tn, "DeepGemm_W8A8_I8_MARLIN_NOPAD_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16.co", 256, 256, 128);	static AiterAsmKernel deepgemm_w8a8_bf16_tn(func_name_bf, bf16_co_file_tn);

    AiterAsmKernel *impl_ptr = nullptr;
    impl_ptr = &deepgemm_w8a8_bf16_tn;
    impl_ptr->launch_kernel_ext({
        &hipFunctionArgs,
        &size,
        wgTile, 1, 1,
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });
    return;
}


template <
    int BLOCKM,
    int BLOCKN,
    int BLOCKK,
    typename OutputType,
    ContiguousBigTileLayout Layout = ContiguousBigTileLayout::kLegacy,
    int BLOCKDIM = 768>
void _m_grouped_w8a8_gemm_nt_contig_big_tile_v1_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor m_indices,
    uint32_t experts_num,
    bool prebuilt_tables = false) {

    struct HipFunctionArgsV1 {
        uint32_t num_MBlocks;
        uint32_t num_NBlocks;
        OutputType* ptr_C;
        const int8_t* ptr_A;
        const int8_t* ptr_B;
        float* ptr_A_scale;
        float* ptr_B_scale;
        const int32_t* m_indices;
        uint32_t experts_num;
        uint32_t size_m;
        uint32_t size_n;
        uint32_t size_k;
        uint32_t stride_ase;
        uint32_t stride_asm;
        uint32_t stride_ask;
        uint32_t stride_bse;
        uint32_t stride_bsn;
        uint32_t stride_bsk;
        uint32_t numFullBlocks;
        uint32_t wgmRemainder1;
        uint32_t magicNumberWgmRemainder1;
        void* debugBuffer;
    } hipFunctionArgs;

    const int size_m = input.size(0);
    const int size_n = b_scale.size(1);
    const int size_k = input.size(1);
    const int workgroupmapping = 8;

    int device;
    hipDeviceProp_t device_properties;
    HIP_CALL(hipGetDevice(&device));
    HIP_CALL(hipGetDeviceProperties(&device_properties, device));

    size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
    // v4: restore the masked ASM persistent launch style: fixed CU CTAs, each CTA walks logical tiles by +80.
    size_t globalWorkSize[3] = {static_cast<size_t>(device_properties.multiProcessorCount), 1, 1};

    const unsigned smallNumMagicShift = 31;
    unsigned numFullBlocks = globalWorkSize[1] / workgroupmapping;
    unsigned wgmRemainder1 = globalWorkSize[1] % workgroupmapping;
    if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping;
    unsigned magicNumberWgmRemainder1 = ((1L << smallNumMagicShift) / wgmRemainder1 + 1);

    hipFunctionArgs.num_MBlocks = DIVIDE(size_m, BLOCKM);
    hipFunctionArgs.num_NBlocks = DIVIDE(size_n, BLOCKN);
    hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
    hipFunctionArgs.ptr_A = b_qweight.data_ptr<int8_t>();
    hipFunctionArgs.ptr_B = input.data_ptr<int8_t>();
    hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
    hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int32_t* d_counts = nullptr;
    int32_t* d_offsets = nullptr;
    if (prebuilt_tables) {
        TORCH_CHECK(m_indices.numel() >= static_cast<int64_t>(experts_num) * 2,
                    "mode=2001 expects m_indices table layout [counts(E), offsets(E)]");
        d_counts = m_indices.data_ptr<int32_t>();
        d_offsets = m_indices.data_ptr<int32_t>() + experts_num;
    } else {
        // Experimental cache: avoid hipMalloc/hipFree on every GEMM call.
        // Production can replace this with framework-provided workspace.
        static torch::Tensor counts_workspace;
        static torch::Tensor offsets_workspace;
        static int cached_device = -1;
        int current_device = input.get_device();
        auto workspace_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
        if (!counts_workspace.defined() ||
            counts_workspace.numel() < static_cast<int64_t>(experts_num) ||
            cached_device != current_device) {
            counts_workspace = torch::empty({static_cast<int64_t>(experts_num)}, workspace_options);
            offsets_workspace = torch::empty({static_cast<int64_t>(experts_num)}, workspace_options);
            cached_device = current_device;
        }
        d_counts = counts_workspace.data_ptr<int32_t>();
        d_offsets = offsets_workspace.data_ptr<int32_t>();
        HIP_CALL(hipMemsetAsync(d_counts, 0, experts_num * sizeof(int32_t), stream));
        HIP_CALL(hipMemsetAsync(d_offsets, 0, experts_num * sizeof(int32_t), stream));
        int build_block = 256;
        int build_grid = DIVIDE(size_m, build_block);
        hipLaunchKernelGGL(build_grouped_counts_offsets_kernel,
                           dim3(build_grid), dim3(build_block), 0, stream,
                           m_indices.data_ptr<int32_t>(), d_counts, d_offsets,
                           size_m, static_cast<int>(experts_num));
        HIP_CALL(hipGetLastError());
    }

    hipFunctionArgs.m_indices = d_counts;
    hipFunctionArgs.experts_num = experts_num;
    hipFunctionArgs.size_m = size_m;
    hipFunctionArgs.size_n = size_n;
    hipFunctionArgs.size_k = size_k;
    hipFunctionArgs.stride_ase = b_scale.stride(0);
    hipFunctionArgs.stride_asm = b_scale.stride(1);
    hipFunctionArgs.stride_ask = b_scale.dim() > 2 ? b_scale.stride(2) : 1;
    hipFunctionArgs.stride_bse = a_scale.stride(0);
    hipFunctionArgs.stride_bsn = a_scale.dim() > 1 ? a_scale.stride(1) : 1;
    hipFunctionArgs.stride_bsk = 1;
    hipFunctionArgs.numFullBlocks = numFullBlocks;
    hipFunctionArgs.wgmRemainder1 = wgmRemainder1;
    hipFunctionArgs.magicNumberWgmRemainder1 = magicNumberWgmRemainder1;
    hipFunctionArgs.debugBuffer = d_offsets;

    size_t argsSize = sizeof(hipFunctionArgs);
    char funcName[1024], coFile[1024];
    memset(funcName, 0x0, sizeof(funcName));
    memset(coFile, 0x0, sizeof(coFile));
    if (output.scalar_type() == at::ScalarType::BFloat16) {
        sprintf(funcName, "DEEPGEMM_I8_I8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM8_GROUPGEMM_CONTIGUOUS", BLOCKM, BLOCKN, BLOCKK);
        if constexpr (Layout == ContiguousBigTileLayout::kMaskedLayout) {
            sprintf(coFile, "deepgemm_groupgemm_contiguous_w8a8_marlin_masked_layout_%dx%dx%d_TN_BF16_WGM8.co", BLOCKM, BLOCKN, BLOCKK);
        } else {
            sprintf(coFile, "deepgemm_groupgemm_contiguous_w8a8_marlin_%dx%dx%d_TN_BF16_WGM8.co", BLOCKM, BLOCKN, BLOCKK);
        }
    } else {
        TORCH_CHECK(false, "contiguous big tile v1 only supports BFloat16 output");
    }

    static AiterAsmKernel contig_big_tile_v1(funcName, coFile);
    contig_big_tile_v1.launch_kernel_ext({
        &hipFunctionArgs,
        &argsSize,
        globalWorkSize[0],
        globalWorkSize[1],
        globalWorkSize[2],
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });
}

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_w8a8_gemm_nt_contig_asm_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,                   // (M)
    torch::Tensor b_scale,                   // (E, N)
    torch::Tensor m_indices,
    uint32_t experts_num) {
    
    GemmInfo<OutputType> d_prob;

    const int size_m = input.size(0);        //(M, K)
    const int size_n = b_qweight.size(1) * 16;    //(E, N // 16, K * 16)
    const int size_k = input.size(1);

    uint32_t lda = size_k;
    uint32_t ldb = size_k;
    uint32_t ldc = size_n;
    uint32_t ldd = size_n;

    float d_alpha = 1.0;
    float d_beta = 0.0;

    size_t stride_a = size_m * size_k;
    size_t stride_b = size_n * size_k;
    size_t stride_c = size_m * size_n;
    size_t stride_d = size_m * size_n;
    {
    d_prob.m = size_n;
    d_prob.n = size_m;
    d_prob.batch = 1;
    d_prob.k = size_k;
    d_prob.strideD1 = ldd;
    d_prob.strideD2 = stride_d;
    d_prob.strideC1 = ldc;
    d_prob.strideC2 = stride_c;
    d_prob.strideA1 = ldb;
    d_prob.strideA2 = stride_b;
    d_prob.strideB1 = lda;
    d_prob.strideB2 = stride_a;
    memcpy(d_prob.alpha, &d_alpha, sizeof(float));
    memcpy(d_prob.beta,  &d_beta, sizeof(float));
    d_prob.a = b_qweight.data_ptr<int8_t>();
    d_prob.b = input.data_ptr<int8_t>();
    d_prob.c = static_cast<OutputType*>(output.data_ptr());
    d_prob.d = static_cast<OutputType*>(output.data_ptr());
    d_prob.scaleA = b_scale.data_ptr<float>();
    d_prob.scaleB = a_scale.data_ptr<float>();
    }
    
    size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
    size_t globalWorkSize[3] = {1, 1, 1};

    size_t wgTile = 0;
    int wg_M = DIVIDE(size_m, BLOCKM);
    int wg_N = DIVIDE(size_n, BLOCKN);
    wgTile += wg_M * wg_N;

	struct __attribute__((packed)) DirectGemmInfoProbe {
        uint32_t m;
        uint32_t n;
        uint32_t batch;
        uint32_t k;
        uint64_t d;
        uint64_t c;
        uint64_t a;
        uint64_t b;
        uint32_t strideD0;
        uint32_t strideD1;
        uint32_t strideC0;
        uint32_t strideC1;
        uint32_t strideA0;
        uint32_t strideA1;
        uint32_t strideB0;
        uint32_t strideB1;
        float alpha;
        float beta;
        uint64_t scaleA;
        uint64_t scaleB;
        uint64_t scaleAlphaVec;
        uint64_t bias;
        uint32_t biasType;
        uint32_t strideBias;
        float activationAlpha;
        float activationBeta;
        uint32_t activationType;
    };
    static_assert(sizeof(DirectGemmInfoProbe) == 140, "direct 1000 arg record must be 140 bytes");
    static_assert(offsetof(DirectGemmInfoProbe, d) == 16, "direct 1000 d offset mismatch");
    static_assert(offsetof(DirectGemmInfoProbe, b) == 40, "direct 1000 b offset mismatch");
    static_assert(offsetof(DirectGemmInfoProbe, alpha) == 80, "direct 1000 alpha offset mismatch");
    static_assert(offsetof(DirectGemmInfoProbe, scaleA) == 88, "direct 1000 scaleA offset mismatch");
    static_assert(offsetof(DirectGemmInfoProbe, scaleB) == 96, "direct 1000 scaleB offset mismatch");

	struct __attribute__((packed)) DirectHipFunctionArgsProbe {
        uint32_t gemm_count;              //4
        void const* DeviceUserArguments;  //8
        void const* argsPtr;              //8
        uint32_t wgm;                     //4
        unsigned int gsu;                 //4
        int32_t* m_indics;               //8
        int32_t* debug_d;                                
        uint32_t wg_table[1];
        DirectGemmInfoProbe gemm;
    };
    static_assert(offsetof(DirectHipFunctionArgsProbe, wg_table) == 44, "direct 1000 wg table offset mismatch");
    static_assert(offsetof(DirectHipFunctionArgsProbe, gemm) == 48, "direct 1000 gemm record offset mismatch");
    static_assert(sizeof(DirectHipFunctionArgsProbe) == 188, "direct 1000 top-level kernarg size mismatch");

	DirectHipFunctionArgsProbe hipFunctionArgs{};

    hipFunctionArgs.gemm_count = 1;
    hipFunctionArgs.DeviceUserArguments = nullptr;
    hipFunctionArgs.argsPtr = nullptr;
    hipFunctionArgs.wgm = 1;
    hipFunctionArgs.gsu = 1;
    hipFunctionArgs.m_indics = m_indices.data_ptr<int32_t>();
    hipFunctionArgs.debug_d = nullptr;
    hipFunctionArgs.wg_table[0] = wgTile;

    hipFunctionArgs.gemm.m = d_prob.m;
    hipFunctionArgs.gemm.n = d_prob.n;
    hipFunctionArgs.gemm.batch = d_prob.batch;
    hipFunctionArgs.gemm.k = d_prob.k;
    hipFunctionArgs.gemm.d = reinterpret_cast<uint64_t>(d_prob.d);
    hipFunctionArgs.gemm.c = reinterpret_cast<uint64_t>(d_prob.c);
    hipFunctionArgs.gemm.a = reinterpret_cast<uint64_t>(d_prob.a);
    hipFunctionArgs.gemm.b = reinterpret_cast<uint64_t>(d_prob.b);
    hipFunctionArgs.gemm.strideD0 = d_prob.strideD1;
    hipFunctionArgs.gemm.strideD1 = d_prob.strideD2;
    hipFunctionArgs.gemm.strideC0 = d_prob.strideC1;
    hipFunctionArgs.gemm.strideC1 = d_prob.strideC2;
    hipFunctionArgs.gemm.strideA0 = d_prob.strideA1;
    hipFunctionArgs.gemm.strideA1 = d_prob.strideA2;
    hipFunctionArgs.gemm.strideB0 = d_prob.strideB1;
    hipFunctionArgs.gemm.strideB1 = d_prob.strideB2;
    hipFunctionArgs.gemm.alpha = d_alpha;
    hipFunctionArgs.gemm.beta = d_beta;
    hipFunctionArgs.gemm.scaleA = reinterpret_cast<uint64_t>(d_prob.scaleA);
    hipFunctionArgs.gemm.scaleB = reinterpret_cast<uint64_t>(d_prob.scaleB);
    hipFunctionArgs.gemm.scaleAlphaVec = 0;
    hipFunctionArgs.gemm.bias = 0;
    hipFunctionArgs.gemm.biasType = 0;
    hipFunctionArgs.gemm.strideBias = 0;
    hipFunctionArgs.gemm.activationAlpha = 0.0f;
    hipFunctionArgs.gemm.activationBeta = 0.0f;
    hipFunctionArgs.gemm.activationType = 0;

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    char func_name_fp[1024], fp16_co_file_tn[1024], bf16_co_file_tn[1024], func_name_bf[1024];
    memset(func_name_fp, 0x0, sizeof(func_name_fp));
    memset(fp16_co_file_tn, 0x0, sizeof(fp16_co_file_tn));
    memset(func_name_bf, 0x0, sizeof(func_name_bf));
    memset(bf16_co_file_tn, 0x0, sizeof(bf16_co_file_tn));
    
    sprintf(func_name_fp, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_FP16", BLOCKM, BLOCKN, BLOCKK);
    sprintf(func_name_bf, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16", BLOCKM, BLOCKN, BLOCKK);
    sprintf(fp16_co_file_tn, "DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT%dX%dX%d_FP16.co", BLOCKM, BLOCKN, BLOCKK);
    sprintf(bf16_co_file_tn, "DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16.co", BLOCKM, BLOCKN, BLOCKK);

	static AiterAsmKernel deepgemm_w8a8_fp16_tn(func_name_fp, fp16_co_file_tn);
	static AiterAsmKernel deepgemm_w8a8_bf16_tn(func_name_bf, bf16_co_file_tn);

    AiterAsmKernel *impl_ptr = nullptr;
    if (output.scalar_type() == at::ScalarType::BFloat16)
        impl_ptr = &deepgemm_w8a8_bf16_tn;
    else 
        impl_ptr = &deepgemm_w8a8_fp16_tn;
    impl_ptr->launch_kernel_ext({
        &hipFunctionArgs,
        &size,
        wgTile, 1, 1,
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });
    return;
}

torch::Tensor m_grouped_w8a8_gemm_nt_contiguous(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  int mode) {
  
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    uint32_t experts_num = b_qweight.sizes()[0];
    if (mode == kContiguousPublicMaskedLayoutMode) {
        TORCH_CHECK(
            b_qweight.dim() == 3,
            "gfx936 mode=2 expects masked/marlin2 packed weight [E,N/16,K*16]");
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_big_tile_masked_layout_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_big_tile_v1_impl<
                256, 256, 128, scalar_t, ContiguousBigTileLayout::kMaskedLayout>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num, false);
        }
        );
    } else if (mode == 2000 || mode == 2001) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_big_tile_v1_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_big_tile_v1_impl<256, 256, 128, scalar_t>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num, mode == 2001);
        }
        );
    } else if (mode == 1000) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_asm_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_asm_impl<256, 256, 128, scalar_t>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num);
        }
        );
    } else if (mode == 100) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_asm_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_asm_impl<256, 128, 128, scalar_t, 512>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num);
        }
        );
    } else {
        TORCH_CHECK(false, mode, " mode error.");
    }
  return output;
}

torch::Tensor m_grouped_w8a8_gemm_nt_nopad_contiguous(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  torch::Tensor token_per_expert) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    uint32_t experts_num = b_qweight.sizes()[0];
    _m_grouped_w8a8_gemm_nt_contig_no_pad_asm_impl(
        input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert, experts_num
    );
    return output;
  }
}
