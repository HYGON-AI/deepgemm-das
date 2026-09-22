#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <vector>

#include "hip_common.h"
#include "m_grouped_gemm_nt_masked.h"
#include "intrinsic.h"
#include "m_grouped_gemm_config.h"

using half_t  = __half;
using bhalf_t = __hip_bfloat16;

#define DEBUG 0

using half_t = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

/* module function args */
template <typename T>
struct HipFunctionArgs {
  uint32_t num_MBlocks;
  uint32_t num_NBlocks;
  T* ptr_C;                          // output
  const int8_t* ptr_A;              // input
  const int8_t* ptr_B;               // weight
  float* ptr_A_scale;                // input scale
  float* ptr_B_scale;                // weight scale
  // const float* topk_weights;         // topk weights
  // const int32_t* expert_offsets;     // null
  const int32_t* masked_m;       // tokens per expert
  uint32_t experts_num;              // num of experts
  uint32_t size_m;                   // input size m
  uint32_t size_n;                   // output size n
  uint32_t size_k;                   // input size k
  uint32_t stride_ase;               // weight scale stride expert
  uint32_t stride_asm;               // input scale stride m
  uint32_t stride_ask;               // input scale stride k
  uint32_t stride_bse;               // weight scale stride expert
  uint32_t stride_bsn;               // weight scale stride n
  uint32_t stride_bsk;               // weight scale stride k
  uint32_t numFullBlocks;
  uint32_t wgmRemainder1;
  uint32_t magicNumberWgmRemainder1;
  void *debugBuffer;
};

namespace deepgemm {

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_marlin_w8a8_gemm_nt_masked_asm_impl(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m,
  uint32_t expected_m_per_group, 
  uint32_t experts_num,
  const torch::Tensor* expert_offsets = nullptr) {

  const int size_m = input.size(1); 
  const int size_n = b_scale.size(1);
  const int size_k = input.size(2);
  const int workgroupmapping = 8;	//



  /* grid sizes */
  size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
  size_t globalWorkSize[3] = {1, 1, 1};
  const bool compact_layout = expert_offsets != nullptr;
  auto need_size_m = std::min(size_m, static_cast<int>(expected_m_per_group * 1.3));
  if (!compact_layout && BLOCKM == 256 && BLOCKN == 64 && BLOCKK == 128) {
    globalWorkSize[0] = DIVIDE(need_size_m, BLOCKN);
    globalWorkSize[1] = DIVIDE(size_n, BLOCKM);
    globalWorkSize[2] = experts_num;
  } else {
    int device;
    hipDeviceProp_t device_properties;
    HIP_CALL(hipGetDevice(&device));
    HIP_CALL(hipGetDeviceProperties(&device_properties, device));
    globalWorkSize[0] = (BLOCKN == 64 ? 128 : device_properties.multiProcessorCount);
    globalWorkSize[1] = 1;
    globalWorkSize[2] = 1;
  }

  unsigned int problemNumGroupTiles0 = globalWorkSize[0];
  unsigned int problemNumGroupTiles1 = globalWorkSize[1];
  const unsigned smallNumMagicShift = 31; // bozo, review
  unsigned numFullBlocks =  problemNumGroupTiles1 / workgroupmapping; // divide by WorkGroupMapping
  unsigned wgmRemainder1 =  problemNumGroupTiles1 % workgroupmapping;
  if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping;
  unsigned magicNumberWgmRemainder1 = ((1L<<smallNumMagicShift) / wgmRemainder1 + 1);

  HipFunctionArgs<OutputType> hipFunctionArgs;
  hipFunctionArgs.num_MBlocks = DIVIDE(size_m, BLOCKN);
  hipFunctionArgs.num_NBlocks = DIVIDE(size_n, BLOCKM);
  
  hipFunctionArgs.ptr_C = static_cast<OutputType*>(output.data_ptr());
  hipFunctionArgs.ptr_A = b_qweight.data_ptr<int8_t>();
  hipFunctionArgs.ptr_B = input.data_ptr<int8_t>();
  hipFunctionArgs.ptr_A_scale = b_scale.data_ptr<float>();
  hipFunctionArgs.ptr_B_scale = a_scale.data_ptr<float>();
  // hipFunctionArgs.topk_weights = nullptr;
  // hipFunctionArgs.expert_offsets = nullptr;
  hipFunctionArgs.masked_m = masked_m.data_ptr<int32_t>();
  hipFunctionArgs.experts_num = experts_num;
  hipFunctionArgs.size_m = size_m;
  hipFunctionArgs.size_n = size_n;
  hipFunctionArgs.size_k = size_k;
  hipFunctionArgs.stride_ase = b_scale.stride(0);
  hipFunctionArgs.stride_asm = b_scale.stride(1);
  hipFunctionArgs.stride_ask = b_scale.stride(2);
  hipFunctionArgs.stride_bse = a_scale.stride(0);
  hipFunctionArgs.stride_bsn = a_scale.stride(1);
  hipFunctionArgs.stride_bsk = a_scale.stride(2);
  
  hipFunctionArgs.numFullBlocks  = numFullBlocks;
  hipFunctionArgs.wgmRemainder1  = wgmRemainder1;
  hipFunctionArgs.magicNumberWgmRemainder1 = magicNumberWgmRemainder1;
  void * debug_host = nullptr, * debug_device = nullptr;
  #if DEBUG
  std::cout << "_m_grouped_marlin_w8a8_gemm_nt_masked_asm_impl "
          << " size_m: " << size_m
          << " need_size_m: " << need_size_m
          << " size_n: " << size_n
          << " size_k: " << size_k
          << " expected_m_per_group: " << expected_m_per_group
          << " experts_num: " << experts_num
          << std::endl;
  int bytes = 64;
  int debug_size = BLOCKDIM * bytes;
  debug_host = malloc(debug_size);
  hipMalloc((void **)&debug_device, debug_size);
  #endif
  hipFunctionArgs.debugBuffer =
      expert_offsets == nullptr ? debug_device : expert_offsets->data_ptr();
  size_t argsSize = sizeof(hipFunctionArgs);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  char funcName[1024], funcName_alt[1024], w8a8_groupgemm_co_file[1024];
  memset(funcName, 0x0, sizeof(funcName));
  memset(funcName_alt, 0x0, sizeof(funcName_alt));
  memset(w8a8_groupgemm_co_file, 0x0, sizeof(w8a8_groupgemm_co_file));
  
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    if (compact_layout) {
      sprintf(funcName, "DEEPGEMM_I8_I8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM8_GROUPGEMM_COMPACT_MASKED", BLOCKM, BLOCKN, BLOCKK);
      sprintf(w8a8_groupgemm_co_file, "deepgemm_groupgemm_compact_masked_w8a8_marlin_%dx%dx%d_TN_BF16_WGM8.co", BLOCKM, BLOCKN, BLOCKK);
    } else {
      sprintf(funcName, "DEEPGEMM_W8A8_I8_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM8_GROUPGEMM_MASKED", BLOCKM, BLOCKN, BLOCKK);
      sprintf(funcName_alt, "DEEPGEMM_I8_I8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM8_GROUPGEMM_MASKED", BLOCKM, BLOCKN, BLOCKK);
      sprintf(w8a8_groupgemm_co_file, "deepgemm_groupgemm_masked_w8a8_marlin_%dx%dx%d_TN_BF16_WGM8.co", BLOCKM, BLOCKN, BLOCKK);
    }
  } else {
    TORCH_CHECK(false, "moe_groupgemm_w8a8 only supports outputType in [BFloat16]");
  }

  auto launch_kernel = [&](AiterAsmKernel& kernel) {
    kernel.launch_kernel_ext({
      &hipFunctionArgs,
      &argsSize,
      globalWorkSize[0],           // gdx
      globalWorkSize[1],           // gdy
      globalWorkSize[2],           // gdz
      localWorkSize[0],            // bdx: 12 wv64
      localWorkSize[1],            // bdy
      localWorkSize[2],            // bdz
      stream
    });
  };

  if (compact_layout) {
    static AiterAsmKernel compact_kernel(funcName, w8a8_groupgemm_co_file);
    launch_kernel(compact_kernel);
  } else {
    static AiterAsmKernel dense_kernel(funcName, w8a8_groupgemm_co_file, funcName_alt);
    launch_kernel(dense_kernel);
  }
  
  #if DEBUG
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  
  printf("%s:%d gdx:%d gdy:%d gdz:%d bdx:%d bdy:%d bdz:%d argsSize:%ld\n", __FILE__, __LINE__, globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
    localWorkSize[0], localWorkSize[1], localWorkSize[2], argsSize);
  printf("numWorkGroups0:%p\nnumWorkGroups1:%p\nptr_C:%p\nptr_A:%p\nptr_B:%p\nptr_A_scale:%p\nptr_B_scale:%p\nproblem_size:%p\nexperts_num:%p\n"
    "size_m:%p\nsize_n:%p\nsize_k:%p\nstride_ase:%p\nstride_asm:%p\nstride_ask:%p\nstride_bse:%p\nstride_bsn:%p\nstride_bsk:%p\nnumFullBlocks:%p\nwgmRemainder1:%p\nmagicNumberWgmRemainder1:%p\ndebugBuffer%p\n",
    &hipFunctionArgs.numWorkGroups0, &hipFunctionArgs.numWorkGroups1, &hipFunctionArgs.ptr_C,
    &hipFunctionArgs.ptr_A, &hipFunctionArgs.ptr_B, &hipFunctionArgs.ptr_A_scale,
    &hipFunctionArgs.ptr_B_scale,// &hipFunctionArgs.topk_weights, &hipFunctionArgs.expert_offsets, 
    &hipFunctionArgs.masked_m, &hipFunctionArgs.experts_num, &hipFunctionArgs.size_m,
    &hipFunctionArgs.size_n, &hipFunctionArgs.size_k, &hipFunctionArgs.stride_ase, &hipFunctionArgs.stride_asm, 
    &hipFunctionArgs.stride_ask, &hipFunctionArgs.stride_bse, &hipFunctionArgs.stride_bsn,
    &hipFunctionArgs.stride_bsk, &hipFunctionArgs.numFullBlocks, &hipFunctionArgs.wgmRemainder1, &hipFunctionArgs.magicNumberWgmRemainder1, &hipFunctionArgs.debugBuffer
  );
  printf("\ndebug:\n");
  hipMemcpy(debug_host, hipFunctionArgs.debugBuffer, debug_size, hipMemcpyDeviceToHost);
  using DEBUG_TYPE = int8_t;
  // using DEBUG_TYPE = uint32_t;
  // using DEBUG_TYPE = int32_t;
  // using DEBUG_TYPE = float;
  // using DEBUG_TYPE = bhalf_t;
  int split = bytes / sizeof(DEBUG_TYPE);
  DEBUG_TYPE *p_debug = reinterpret_cast<DEBUG_TYPE*>(debug_host);
  for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE); ++i) {
  // for (int i = 0; i < debug_size / sizeof(DEBUG_TYPE) / split; ++i) {
    // if (i % 8 == 0) printf("\n");
    if (i % 16 == 0) printf("\ni:%d ", i / 16);
    // bhalf_t* tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%3d:%10.4f\t", i, static_cast<float>(tmp[i+1]));
    // printf("%3d:%6d\t", i, p_debug[i]);
    printf("%6d\t", p_debug[i]);
    // printf("%10d\t", p_debug[i]);
    // printf("%10.4f\t", p_debug[i]);
    // bhalf_t * tmp = reinterpret_cast<bhalf_t*>(p_debug + i);
    // printf("%10.4f ", static_cast<float>(tmp[0]));
    // int index = split * i;
    // printf("tid:%d %3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   // "%3d %3d %3d %3d %3d %3d %3d %3d "
    //   "%3d %3d %3d %3d %3d %3d %3d %3d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ],
    //   p_debug[index+4 ], p_debug[index+5 ], p_debug[index+6 ], p_debug[index+7 ],
    //   p_debug[index+8 ], p_debug[index+9 ], p_debug[index+10], p_debug[index+11],
    //   p_debug[index+12], p_debug[index+13], p_debug[index+14], p_debug[index+15]
    //   // ,
    //   // p_debug[index+16], p_debug[index+17], p_debug[index+18], p_debug[index+19],
    //   // p_debug[index+20], p_debug[index+21], p_debug[index+22], p_debug[index+23],
    //   // p_debug[index+24], p_debug[index+25], p_debug[index+26], p_debug[index+27],
    //   // p_debug[index+28], p_debug[index+29], p_debug[index+30], p_debug[index+31]
    // );
    // printf("tid:%d %6d %6d %6d %6d\n", i, 
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ]
    // );
    // int8_t * tmp = reinterpret_cast<int8_t*>(p_debug + index);
    // printf("tid:%d %3d %3d %3d %3d 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X 0X%08X %3d %3d %3d %3d\n", i, 
    //   tmp[0], tmp[1], tmp[2], tmp[3],
    //   p_debug[index+0 ], p_debug[index+1 ], p_debug[index+2 ], p_debug[index+3 ], p_debug[index+4 ], p_debug[index+5 ], 
    //   tmp[20], tmp[21], tmp[22], tmp[23] 
    // );
  }
  printf("\n");
  HIP_CALL(hipFree(debug_device));
  free(debug_host);
  #endif
  
  return;
}


// cuda geemnt kernel
template<
    typename Element, 
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M, 
    uint16_t BLOCK_SIZE_N, 
    uint16_t BLOCK_SIZE_K, 
    uint16_t WARP_M, 
    uint16_t WARP_N, 
    uint16_t WARP_K,
    int SIZE_K,
    int STAGE>                                                                                                            
__forceinline__ __device__ void  gemm_nt_marlin_decode(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    union_vec<Element,WARP_K/4> A_reg[][STAGE], // 2 = stage
    union_vec<Element,WARP_K/4> B_reg[][STAGE], // 2 = stage
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int actual_m, 
    const int size_m,
    const int size_n,
    const int size_k,
    const int bidx) {

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
    constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = (warp_id / warp_k_num) % warp_n_num;
    int warp_m_id = (warp_id / warp_k_num) / warp_n_num;



    //warp内部k连续[warpn, 64] warp外部在N方向连续 
    int k_start = warp_k_id * WARP_K;
    //int k_start_b = warp_k_id * WARP_K * size_n; // moe
    int k_start_b = warp_k_id * WARP_K * 16;
    const int stage_offset = warp_k_num * WARP_K;
    //const int stage_offset_b = warp_k_num * WARP_K * size_n; // moe
    const int stage_offset_b = warp_k_num * WARP_K * 16;
    
    // 提前读取token_idx到reg
    int g_row_A[WARP_M / MFMA_M]; 
    int g_row_B[WARP_N / MFMA_N];
    

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        int m_offset = warp_m_id * WARP_M + m_tile * MFMA_M + row_id;
        m_offset = m_offset >= actual_m ? actual_m % BLOCK_SIZE_M : m_offset;
        g_row_A[m_tile] = m_offset * size_k + col_id * 16;
        //g_row_A[m_tile] = (m_tile * MFMA_M + row_id) * size_k + col_id * 16;
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        //g_row_B[n_tile] = (warp_n_id * WARP_N + row_id + n_tile * MFMA_N)* 64 + col_id * 16; //warp内部k连续[warpn, 64] warp外部在N方向连续 
        int n_offset = warp_n_id * WARP_N + row_id + n_tile * MFMA_N;
        g_row_B[n_tile] = (n_offset / 16) * 16 * size_k + (n_offset % 16) * 16 + col_id * 16 * 16;
    }

    
    // --------------------------------------------------------------- prefetch ------------------------------------------------------------------------------------------------
    // read 0
    // load A
    #pragma unroll
    for(int i = 0; i < STAGE - 1; ++i){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

        // load B
        #pragma unroll
        for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * 16 + k_start_b);
            }
        }
        k_start += stage_offset;
        k_start_b += stage_offset_b;
    }
    k_start -= stage_offset;
    k_start_b -= stage_offset_b;
    
    for( ; k_start < size_k - stage_offset * STAGE; k_start += stage_offset * STAGE, k_start_b += stage_offset_b * STAGE){

        #pragma unroll
        for(int i = 0; i < STAGE; ++i){
            ///////////////////////////////////////////////////////////// read 1 ////////////////////////////////////////////////////////////
            // load A
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

            // load B
            #pragma unroll
            for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                #pragma unroll
                for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                    //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * 16 + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

            ///////////////////////////////////////////////////////////// mmac 0 ////////////////////////////////////////////////////////////
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }
        } // stage end

    } // k_loop end
    

    // ------------------------------------------------------------------------最后一次尾处理-------------------------------------------------------//
    if(k_start + stage_offset < size_k){ // size_k / block 为偶数
        //  --------------------------------------------- load mmac ---------------------------------- //
        int i = 0;
        if constexpr (STAGE == 4){ //TODO: STAGE为4 尾处理有四种可能方式 需要size_k来确定loop次数，size_k如果为动态传入会有寄存器溢出 目前写死解决 
            //constexpr int epilogue_tile =  ( 7168/BLOCK_SIZE_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;        
            constexpr int epilogue_tile =  ( SIZE_K/BLOCK_SIZE_K - (STAGE - 1)) % STAGE;   
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * 16 + k_start_b);
                        
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                        *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                        *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        } else{ // two stage
            constexpr int epilogue_tile =  1;      
       
            #pragma unroll
            for(; i < epilogue_tile; i++){
                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){ 
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i+1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

                // load B
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                        //buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i+STAGE-1)%STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i+1) * stage_offset_b + k_tile * 64 * 16 + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

                #pragma unroll
                for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                    #pragma unroll
                    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                        #pragma unroll
                        for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                        *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i].int8t_array[k_tile]), 
                                        *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i].int8t_array[k_tile]), 
                                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                        }
                    }
                }
            }
        }
        //  --------------------------------------------- load mmac ---------------------------------- //

        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int ii = 1; ii < STAGE; ++ii){
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][(i+ii-1)%STAGE].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][(i+ii-1)%STAGE].int8t_array[k_tile]), 
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }

        }

    }
    else{ // todo 不为BLOCK_SIZE_K的倍数
        //---------------------------------------------------- mmac --------------------------------------------------------------------//
        #pragma unroll
        for(int i = 1; i < STAGE; ++i){
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K  +  WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                        C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                                    *(typename vec<Element, 8>::type*)(&A_reg[m_tile][i-1].int8t_array[k_tile]), 
                                    *(typename vec<Element, 8>::type*)(&B_reg[n_tile][i-1].int8t_array[k_tile]), 
                                    C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
                    }
                }
            }
        }
    }
///////////////////////////////////////////////////////////执行规约/////////////////////////////////////////////////////////
    // warp间的规约 
    extern __shared__ int out_smem[]; // 声明lds信息
#if 0
    if constexpr (warp_k_num > 1){
      constexpr int lds_n_offset = warp_k_num * BLOCK_SIZE_N;
      #pragma unroll
      for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
          #pragma unroll
          for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ // todo 连续拷贝还是隔开拷贝 -> 差别不大 n_tile*4 + col_id * 4 *  WARP_N / MFMA_N
              *(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + warp_k_id * BLOCK_SIZE_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
              //*(intx4*)( &out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N + warp_n_id * WARP_N + warp_k_id * BLOCK_N] ) = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];
          }
      }

      __syncthreads();


      if(warp_k_id == 0){
          #pragma unroll
          for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
              #pragma unroll
              for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                  C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = {0, 0, 0, 0};
              }
          }


          #pragma unroll
          for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
              #pragma unroll
              for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                  #pragma unroll
                  for(int k_tile = 0; k_tile < warp_k_num; k_tile++){
                      for(int i = 0; i < 4; ++i){
                          C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += 
                          out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile * 16 + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_SIZE_N + i];
                          //out_smem[(m_tile * MFMA_M + row_id) * lds_n_offset  + n_tile*4 + col_id * 4 *  WARP_N / MFMA_N  + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
                      }
                  }
              }
          }
      }
  }
#else
    if constexpr (warp_k_num > 1){

        constexpr int pading_n = WARP_N + 1;

        if(warp_k_id > 0){ //0和1 只需要 warpk_id为1去拷贝数据
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){ 
                   
                    *(intx4*)(&out_smem[(warp_m_id * WARP_M + m_tile * MFMA_M + row_id) * pading_n  + n_tile * 16 + col_id * 4 + (warp_k_id-1 +  warp_n_id * (warp_k_num-1))  * pading_n * BLOCK_SIZE_M])  = C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile];

                }
            }
        }

        __syncthreads();


        if(warp_k_id == 0){
            #pragma unroll
            for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
                #pragma unroll
                for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
                    #pragma unroll
                    for(int k_tile = 0; k_tile < warp_k_num - 1; k_tile++){
                        intx4 temp = *(intx4*)(&out_smem[(warp_m_id * WARP_M + m_tile * MFMA_M + row_id) * pading_n  + n_tile * 16 + col_id * 4 + (k_tile +  warp_n_id * (warp_k_num-1))  * pading_n * BLOCK_SIZE_M]);
                        #pragma unroll
                        for(int i = 0; i < 4; ++i){
                            C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile][i] += temp[i];
        
                        }
                    }
                }
            }
        }
    }
#endif
}

// gemm_nt_bypass
template<
    typename Element, 
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M, 
    uint16_t BLOCK_SIZE_N, 
    uint16_t BLOCK_SIZE_K, 
    uint16_t WARP_M, 
    uint16_t WARP_N, 
    uint16_t WARP_K,
    int SIZE_K,
    int STAGE>                                                                                                            
__forceinline__ __device__ void  gemm_nt_marlin_bypass(
    const Element* input_ptr, 
    const Element* weight_ptr, 
    Element* A_lds,
    Element* B_lds,
    union_vec<Element,WARP_K/4> A_reg[][STAGE], // 2 = stage
    union_vec<Element,WARP_K/4> B_reg[][STAGE], // 2 = stage
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id, 
    int actual_m, 
    const int size_m,
    const int size_n,
    const int size_k,
    const int bidx) {

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
    int warp_n_id = (warp_id ) % warp_n_num;
    int warp_m_id = (warp_id ) / warp_n_num;


    // 提前读取token_idx到reg
    int g_row_A[(WARP_M / MFMA_M)][(WARP_K / READ_K)]; 
    int lds_row_A[(WARP_M / MFMA_M)][(WARP_K / READ_K)];
    int g_row_B[(WARP_N / MFMA_N)][(WARP_K / READ_K)];
    

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
      #pragma unroll
      for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
        int m_offset = warp_m_id * WARP_M + m_tile * MFMA_M + row_id;
        //int global_offset = (m_offset + bidx * BLOCK_SIZE_M) >= actual_m ? actual_m % BLOCK_SIZE_M : m_offset;
        int global_offset = m_offset;
        g_row_A[m_tile][k_tile] = global_offset * size_k + col_id * 16 + k_tile * READ_K;
        lds_row_A[m_tile][k_tile] = m_offset * WARP_K + col_id * 16 + k_tile * READ_K * BLOCK_SIZE_M;
      }
    }

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
        int n_offset = warp_n_id * WARP_N + row_id + n_tile * MFMA_N;
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
          g_row_B[n_tile][k_tile] = (n_offset / 16) * 16 * size_k + (n_offset % 16) * 16 + col_id * 16 * 16 + k_tile * READ_K * 16;
        }
    }

    // mainloop  不加流水
    for(int k = 0; k < size_k / WARP_K; ++k){
      int A_k_offset = k * WARP_K;
      int B_k_offset = k * WARP_K * 16;
     
      auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + A_k_offset);
      auto B_ptr = weight_ptr + B_k_offset;

      //读取A矩阵到LDS
      int A_block_buffer_load_global_offset = 0; 
      int A_lds_stage_offset_loop = 0;
      buffer_load_lds_tile_pad_sorted_token_dword_lds4(WARP_NUM, size_k, BLOCK_SIZE_M, WARP_K, Element, A_ptr, A_lds, 
      A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_SIZE_K, warp_id, lane_id); 
      
      vmcnt_wait(0);
      //读取B矩阵到REG
      #pragma unroll
      for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
          #pragma unroll
          for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
              buffer_load_reg_dwordx4(B_ptr, B_reg[n_tile][0].int4_array[k_tile], 0, g_row_B[n_tile][k_tile]);
          }
      }

      vmcnt_wait(0);
      //读取A矩阵到REG
      #pragma unroll
      for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
          #pragma unroll
          for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
              A_reg[m_tile][0].int4_array[k_tile] = *(intx4*)(A_lds + lds_row_A[m_tile][k_tile]);
          }
      }
      lgkmcnt_wait_barrier(0);
      //计算
      #pragma unroll
      for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
          #pragma unroll
          for(int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++){
              #pragma unroll
              for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                  C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile] = mmac<Element>(
                              *(typename vec<Element, 8>::type*)(&A_reg[m_tile][0].int8t_array[k_tile]), 
                              *(typename vec<Element, 8>::type*)(&B_reg[n_tile][0].int8t_array[k_tile]), 
                              C_reg[0][m_tile*(WARP_N/MFMA_N) + n_tile]);  
              }
          }
      }
    }  
  
}

// gemm_nt_marlin_prefill
template<
    typename Element, 
    uint16_t WARP_NUM,
    uint16_t BLOCK_M, 
    uint16_t BLOCK_N, 
    uint16_t BLOCK_K, 
    uint16_t WARP_M, 
    uint16_t WARP_N, 
    uint16_t WARP_K,
    int SIZE_K,
    int STAGES>                                                                                                             
__forceinline__ __device__ void  gemm_nt_marlin_prefill(
  const Element* input_ptr, 
  const Element* weight_ptr, 
  Element* A_lds,
  Element* B_lds,
  union_vec<Element, 8> A_reg[][WARP_K/32], // 2 = stage
  union_vec<Element, 8> B_reg[][WARP_K/32], // 2 = stage
  intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
  int warp_id, 
  int actual_m, 
  const int size_m,
  const int size_n,
  const int size_k,
  const int bidx) {


    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    const int warp_n_num = BLOCK_N / WARP_N;
    const int marlin_tile_k = 64;
    const int marlin_iter_num = marlin_tile_k / WARP_K; // only value 2 or 1
    const int shift_value = marlin_iter_num - 1;  //  1 or 0
    const int switch_value = marlin_iter_num >> 1; // 1 or 0
    const int marlin_warp_k = std::max(marlin_tile_k, WARP_K);
    const int seqlen_B_stride = WARP_K * 16;
    constexpr int vec_size = 8;
    int warp_n_id = warp_id % warp_n_num;
    int warp_m_id = warp_id / warp_n_num;

    int precompute_A_lds_offset[(WARP_M/MFMA_M)*(WARP_K/MFMA_K)]; 
    int precompute_B_lds_offset[(WARP_N/MFMA_N)*(WARP_K/MFMA_K)]; 
    typename vec<Element,8>::type *B_lds_v8i8 = (typename vec<Element,8>::type *)(B_lds);
    typename vec<Element,8>::type *A_lds_v8i8 = (typename vec<Element,8>::type *)(A_lds);

    int A_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_M * WARP_K; // A tile拿 1 * 16x64的大小
    int B_lds_stage_offset = STAGES == 1 ? 0 : BLOCK_N * WARP_K; // B tile拿 4 * 32x64的大小,
#if 1
    int stage_id_reg_A = 0;
    int stage_id_reg_B = 0;

    {
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + 0); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + 0); // 配置全局显存信息

        #pragma unroll // 预加载STAGES-1的数据
        for(int stage_id_num=0; stage_id_num<STAGES-1; stage_id_num++){

            if(1) { // 预加载A矩阵
                int A_block_buffer_load_global_offset = stage_id_num * WARP_K; 
                int A_lds_stage_offset_loop = stage_id_num * A_lds_stage_offset; 
                buffer_load_lds_tile_pad_sorted_token(WARP_NUM, size_k, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id, vec_size);                
            }

            if(1) { // 预加载B矩阵
                //int B_block_buffer_load_global_offset = (1 - switch_value) * stage_id_num * seqlen_B_stride + switch_value * ((stage_id_num >> shift_value) * seqlen_B_stride + (stage_id_num & shift_value) * WARP_K * 16); 
                int B_block_buffer_load_global_offset = stage_id_num * seqlen_B_stride;
                int B_lds_stage_offset_loop = stage_id_num * B_lds_stage_offset; 
                buffer_load_lds_tile_pad_weight(WARP_NUM, size_k, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                  vec_size);
                // buffer_load_lds_tile_pad_weight(WARP_NUM, marlin_warp_k, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                //                                             vec_size);
            }
        }
    }

    // 构建A矩阵读取索引
    #pragma unroll 
    for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) {
            // int lds_offset = min_tile_k * 4 + min_tile_n*32 + (lane_id & 15)*16 + ((lane_id & 15)>>2) + ((lane_id/16) & 1)*8 + (lane_id/32); // warp_layout：16x4
            // 线程布局16x4,取16x64的数据,k方向的stride为WARP_K,也正好是WARP_K个byte,换算成dword:WARP_K/4,k方向每个线程取MFMA_K/4=8个元素,即2个dword
            // int lds_col = (min_tile_k * MFMA_K) / 4 + (lane_id/16)*2;
            //int lds_row = (lane_id & 15);
            // int vec_size_dword = vec_size/4/*dword*/;
            // int phase = (lds_row/per_phase) % max_phase;
            // int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
            int lds_offset = warp_m_id * (WARP_M*WARP_K)/4 + (min_tile_m * MFMA_M + row_id)*WARP_K/4/*dword*/ + col_id * 2 + (min_tile_k * MFMA_K) / 4; 
            precompute_A_lds_offset[min_tile_m * (WARP_K/MFMA_K) + min_tile_k] = lds_offset; // 装载了两次,
        }
    }
    // 构建B矩阵读取索引
    #pragma unroll 
    for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) { 
        #pragma unroll
        for(int min_tile_k=0; min_tile_k<(WARP_K/MFMA_K); min_tile_k++) { 
            {
                // int lds_col = (min_tile_k * MFMA_K) / 4 + (lane_id/16)*2;
                // int lds_row = (lane_id & 15);
                // int vec_size_dword = vec_size/4/*dword*/;
                // int phase = (lds_row/per_phase) % max_phase;
                // int col_swizzle = ((lds_col/vec_size_dword) ^ phase) * vec_size_dword;
                //int lds_offset = warp_n_id * (WARP_N*WARP_K)/4 + min_tile_n * (MFMA_N*WARP_K/4) + lds_row * (WARP_K/4);/*dword*/

                int lds_offset = warp_n_id * (WARP_N*WARP_K)/4 + min_tile_n * (MFMA_N*WARP_K/4) + row_id * (WARP_K/4) + col_id * 2 + (min_tile_k * MFMA_K) / 4;/*dword*/
                //int lds_offset = warp_n_id * (WARP_N*WARP_K)/4 + min_tile_n * (MFMA_N*WARP_K/4) + row_id * (16/4) + col_id % 2 + (col_id / 2) *16*4 + (min_tile_k * MFMA_K * 16) / 4;/*dword*/
                precompute_B_lds_offset[min_tile_n*(WARP_K/MFMA_K) + min_tile_k] = lds_offset;
            } 
        }
    }

    for(int k=0; k < size_k/WARP_K - (STAGES - 1); k++){

        int k_offset = k * WARP_K;
        //int b_offset = (1 - switch_value) * k * seqlen_B_stride + switch_value * ((k + (STAGES - 1)) >> shift_value) * seqlen_B_stride;
        int b_offset =  k * seqlen_B_stride;
        auto A_ptr = tcp_cache_swizzle_func<64, Element>(input_ptr + k_offset); // 配置全局显存信息
        auto B_ptr = tcp_cache_swizzle_func<64, Element>(weight_ptr + b_offset); // 配置全局显存信息
        { 
            // 预加载A矩阵->lds
            int A_block_buffer_load_global_offset = (STAGES - 1) * WARP_K; 
            int A_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * A_lds_stage_offset; 
            buffer_load_lds_tile_pad_sorted_token(WARP_NUM, size_k, BLOCK_M, WARP_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                        vec_size);                

            // 预加载B矩阵->lds
            //int B_block_buffer_load_global_offset = (1 - switch_value) * (STAGES - 1) * seqlen_B_stride + switch_value * ((k + (STAGES - 1)) & shift_value) * WARP_K * 16;
            int B_block_buffer_load_global_offset = (STAGES - 1) * seqlen_B_stride;
            int B_lds_stage_offset_loop = ((k+(STAGES - 1)) & (STAGES - 1)) * B_lds_stage_offset; 

            buffer_load_lds_tile_pad_weight(WARP_NUM, size_k, BLOCK_N, WARP_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset_loop, BLOCK_K, warp_id, lane_id,
                                                    vec_size);  
        }

        vmcnt_wait(((BLOCK_N * WARP_K) / (4*64)/WARP_NUM + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES - 1));

        stage_id_reg_A = k & (STAGES-1);
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            { // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1
            }
            { // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/4, B_reg, stage_id_reg_B);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<(WARP_N/MFMA_N); min_tile_n++) {
                    C_reg[0][min_tile_m*(WARP_N / MFMA_N) + min_tile_n] = mmac<Element>(
                        *(typename vec<Element, 8>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                        C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]);        
                }
            }
        }
    }

    ///////////////////////////////////////////////////// for 迭代完 ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for(int stage_id=0; stage_id<STAGES-2; stage_id++){
        vmcnt_wait(((BLOCK_N * WARP_K) / (4*64)/WARP_NUM + (BLOCK_M*WARP_K) / (4*64)/WARP_NUM) * (STAGES-2-stage_id));
        {
            stage_id_reg_A = stage_id+1;
            int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
            int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
            // 这里就开始从lds里面读取出来到vgpr了
            for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
                // 加载A矩阵
                ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
                // 加载B矩阵
                ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/4, B_reg, stage_id_reg_B);
            }
        }

        lgkmcnt_wait_barrier(0);

        #pragma unroll
        for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
                #pragma unroll
                for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n] = mmac<Element>(
                        *(typename vec<Element, 8>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                        *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                        C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
                }
            }
        }
    }

    // 为了最后一次等到vmcnt_wait=0
    vmcnt_wait(0);
    {
        stage_id_reg_A = STAGES-1;
        int A_lds_stage_offset_loop = stage_id_reg_A * A_lds_stage_offset;
        int B_lds_stage_offset_loop = stage_id_reg_A * B_lds_stage_offset;
        // 这里就开始从lds里面读取出来到vgpr了
        for(int k_idx = 0; k_idx < WARP_K / MFMA_K; k_idx++){ 
            // 加载A矩阵
            ds_read2_tile_pad_no_wait(WARP_M, k_idx, WARP_NUM, Element, A_lds_v8i8, precompute_A_lds_offset, A_lds_stage_offset_loop/4, A_reg, stage_id_reg_A); // stage_id_reg_A: 0, 1     
            // 加载B矩阵
            ds_read2_tile_pad_no_wait(WARP_N, k_idx, WARP_NUM, Element, B_lds_v8i8, precompute_B_lds_offset, B_lds_stage_offset_loop/4, B_reg, stage_id_reg_B);
        }
    }

    lgkmcnt_wait_barrier(0);

    #pragma unroll
    for(int min_tile_k=0; min_tile_k<WARP_K/MFMA_K; min_tile_k++) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<(WARP_M/MFMA_M); min_tile_m++) { 
            #pragma unroll
            for(int min_tile_n=0; min_tile_n < (WARP_N/MFMA_N); min_tile_n++) {
                C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n] = mmac<Element>(
                    *(typename vec<Element, 8>::type*)(&A_reg[stage_id_reg_A * (WARP_M/MFMA_M) + min_tile_m][min_tile_k].scalar_array[0]), 
                    *(typename vec<Element, 8>::type*)(&B_reg[min_tile_n][min_tile_k].scalar_array[0]), 
                    C_reg[0][min_tile_m*(WARP_N/MFMA_N) + min_tile_n]); 
            }
        }
    }
    lgkmcnt_wait_barrier(0);

#endif
}

// cuda deepgemm kernel 过lds版本
// cuda deepgemm kernel
template<
  typename Element, 
  typename OutputType, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  int SIZE_K,
  int STAGES
  > 
__global__ void __launch_bounds__(1024) DEEPGEMM_W8A8_I8_PREFILL_MARLIN_TN_GROUPGEMM_MASKED(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  OutputType* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ masked_m,  // [16,1]
  const int32_t experts_num, // 专家数
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k
) {
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int WARP_K_NUM = BLOCK_SIZE_K / WARP_K;
  const int bidx = blockIdx.z; // pid_m  m方向
  const int bidy = blockIdx.y; // pid_n  n方向
  const int bidz = blockIdx.x; // pid_k  e方向
  const int actual_m = ((masked_m[bidz] + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * BLOCK_SIZE_M; //往上取block的整数幂


  if(bidx * BLOCK_SIZE_M >= actual_m) return; // 对于无效的block,直接返回

  const uint32_t input_offset = bidz * size_m * size_k + bidx * BLOCK_SIZE_M * size_k; 
  const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * size_k;
  //const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * 64; // moe
  const uint32_t output_offset = bidz * size_m * size_n + bidx * BLOCK_SIZE_M * size_n + bidy * BLOCK_SIZE_N; 
  const uint32_t input_scale_offset = bidz * size_m + bidx * BLOCK_SIZE_M; 
  const uint32_t weight_scale_offset = bidz * size_n + bidy * BLOCK_SIZE_N; 

  auto g_input = input + input_offset;
  auto g_qweight = qweight + qweight_offset;
  auto g_input_scale = input_scale + input_scale_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  OutputType* g_output = output + output_offset;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;

  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // decode这里没用
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  OutputType* output_lds = (OutputType*)&(smem); // 重复使用lds,留给output_lds

  union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];
  intx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 


  // device gemm todo 
  gemm_nt_marlin_prefill<Element, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, SIZE_K, STAGES>
    (g_input, g_qweight, input_lds, qweight_lds,
     A_reg, B_reg, C_reg, warp_id, actual_m, size_m, size_n, size_k, bidx);

  // write global mem
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
      float a_scale = g_input_scale[warp_m_id * WARP_M + min_tile_m * mfma_m + row_id];
      for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
        #pragma unroll  
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float b_scale = g_weight_scale[warp_n_id * WARP_N + min_tile_n * mfma_n + col_id + reg_id * 4];
          float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale * b_scale; 
          int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + 
          (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16) )/2 * 2/*padding*/;
          output_lds[index] = f32_to_output<OutputType>(value);
          // #if defined(__gfx938__)
          // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16 精度有问题
          // #endif
        }
      }
    }
  }

  __syncthreads();

  
  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      //if(m_idx < actual_m) 
      {
        *reinterpret_cast<vec_bf16_8*>(&g_output[m_idx * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  } 
/////////////////////////////////////////////////src///////////////////////////////////////////////////////////////////

}

// A过lds B读到reg
// 不用slick
template<
  typename Element, 
  typename OutputType, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  int SIZE_K
  > 
__global__ void __launch_bounds__(1024) DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_BYPASS_MASKED(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  OutputType* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ masked_m,  // [16,1]
  const int32_t experts_num, // 专家数
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k
) {
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int STAGES = BLOCK_SIZE_K / WARP_K;
  const int bidx = blockIdx.z; // pid_m  m方向z
  const int bidy = blockIdx.y; // pid_n  n方向
  const int bidz = blockIdx.x; // pid_k  e方向
  const int real_m = masked_m[bidz];
  const int actual_m = ((real_m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * BLOCK_SIZE_M; //往上取block的整数幂


  if(bidx * BLOCK_SIZE_M >= actual_m) {
    return; // 对于无效的block,直接返回
  }

  const uint32_t input_offset = bidz * size_m * size_k + bidx * BLOCK_SIZE_M * size_k; 
  const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * size_k;
  const uint32_t output_offset = bidz * size_m * size_n + bidx * BLOCK_SIZE_M * size_n + bidy * BLOCK_SIZE_N; 
  const uint32_t input_scale_offset = bidz * size_m + bidx * BLOCK_SIZE_M; 
  const uint32_t weight_scale_offset = bidz * size_n + bidy * BLOCK_SIZE_N; 

  auto g_input = input + input_offset;
  auto g_qweight = qweight + qweight_offset;
  auto g_input_scale = input_scale + input_scale_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  OutputType* g_output = output + output_offset;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_m_num = WARP_M / mfma_m;
  int warp_n_id = (warp_id) % warp_n_num;
  int warp_m_id = (warp_id) / warp_n_num;

  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // decode这里没用
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K);
  OutputType* output_lds = (OutputType*)&(smem); // 重复使用lds,留给output_lds
  
  union_vec<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
  union_vec<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];
  intx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 

  gemm_nt_marlin_bypass<Element, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, SIZE_K, STAGES>
    (g_input, g_qweight, input_lds, qweight_lds,
     A_reg, B_reg, C_reg, warp_id, real_m, size_m, size_n, size_k, bidx);
    


  // write global mem
#if 1
 
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){

    float a_scale = g_input_scale[warp_m_id * WARP_M + min_tile_m * mfma_m + row_id];
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll  
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float b_scale = g_weight_scale[warp_n_id * WARP_N + min_tile_n * mfma_n + col_id + reg_id * 4];
        float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale * b_scale; 
        //int m_offset = min_tile_m * mfma_m + warp_m_id * WARP_M + row_id;
        int m_offset = warp_m_id * WARP_M + row_id;
        int index = m_offset * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + reg_id * 4 + col_id + ( m_offset )/2 * 2/*padding*/;
        output_lds[index] = f32_to_output<OutputType>(value);
        // #if defined(__gfx938__)
        // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16 精度有问题
        // #endif
      }
    }

    __syncthreads();

    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    constexpr int WARP_M_LOOP_NUM = WARP_M / mfma_m;
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      if (WARP_M_LOOP_NUM > 1) {
        if(m_idx + bidx * BLOCK_SIZE_M < real_m) {
          int global_m_idx = (m_idx / mfma_m) * WARP_M + (m_idx % mfma_m) + min_tile_m * mfma_m;
          *reinterpret_cast<vec_bf16_8*>(&g_output[global_m_idx * size_n + n_idx * 8]) = 
          //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 ]);
          *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
        }
      } else{
        if(m_idx + bidx * BLOCK_SIZE_M < real_m) {
          *reinterpret_cast<vec_bf16_8*>(&g_output[m_idx * size_n + n_idx * 8]) = 
          *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
        }
      }
    } 
    lgkmcnt_wait_barrier(0);

  } // for min_tile_m
  
    

  
    
#endif
}

// cuda deepgemm kernel 
// A和B都不过LDS  用的slick
template<
  typename Element, 
  typename OutputType, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  int SIZE_K,
  int STAGES
  > 
__global__ void __launch_bounds__(1024) DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_GROUPGEMM_MASKED(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  OutputType* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ masked_m,  // [16,1]
  const int32_t experts_num, // 专家数
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k
) {
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int WARP_K_NUM = BLOCK_SIZE_K / WARP_K;
  const int bidx = blockIdx.z; // pid_m  m方向z
  const int bidy = blockIdx.y; // pid_n  n方向
  const int bidz = blockIdx.x; // pid_k  e方向
  const int real_m = masked_m[bidz];
  const int actual_m = ((real_m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * BLOCK_SIZE_M; //往上取block的整数幂


  if(bidx * BLOCK_SIZE_M >= actual_m) return; // 对于无效的block,直接返回

  const uint32_t input_offset = bidz * size_m * size_k + bidx * BLOCK_SIZE_M * size_k; 
  const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * size_k;
  //const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * 64; // moe
  const uint32_t output_offset = bidz * size_m * size_n + bidx * BLOCK_SIZE_M * size_n + bidy * BLOCK_SIZE_N; 
  const uint32_t input_scale_offset = bidz * size_m + bidx * BLOCK_SIZE_M; 
  const uint32_t weight_scale_offset = bidz * size_n + bidy * BLOCK_SIZE_N; 

  auto g_input = input + input_offset;
  auto g_qweight = qweight + qweight_offset;
  auto g_input_scale = input_scale + input_scale_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;
  OutputType* g_output = output + output_offset;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
  constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;

  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = (warp_id / warp_k_num) % warp_n_num;
  int warp_m_id = (warp_id / warp_k_num) / warp_n_num;

  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // decode这里没用
  Element* qweight_lds = input_lds; // decode这里没用
  OutputType* output_lds = (OutputType*)&(smem); // 重复使用lds,留给output_lds
  
  union_vec<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
  union_vec<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];
  intx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 


  // device gemm todo 
  gemm_nt_marlin_decode<Element, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, SIZE_K, STAGES>
    (g_input, g_qweight, input_lds, qweight_lds,
     A_reg, B_reg, C_reg, warp_id, real_m, size_m, size_n, size_k, bidx);

  // write global mem
  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
      float a_scale = g_input_scale[warp_m_id * WARP_M + min_tile_m * mfma_m + row_id];
      for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
        #pragma unroll  
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float b_scale = g_weight_scale[warp_n_id * WARP_N + min_tile_n * mfma_n + col_id + reg_id * 4];
          float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale * b_scale; 
          int m_offset = min_tile_m * mfma_m + warp_m_id * WARP_M + row_id;
          int index = m_offset * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + reg_id * 4 + col_id + ( m_offset )/2 * 2/*padding*/;
          output_lds[index] = f32_to_output<OutputType>(value);
          // #if defined(__gfx938__)
          // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16 精度有问题
          // #endif
        }
      }
    }
  }

  __syncthreads();

  
  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      if(m_idx < real_m) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[m_idx * size_n + n_idx * 8]) = 
        //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 ]);
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  } 
}

template<
  typename Element, 
  typename OutputType, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  int SIZE_K,
  int STAGES
  > 
__global__ void __launch_bounds__(1024) DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_GROUPGEMM_MASKED_PERSISTENT(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  OutputType* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ masked_m,  // [16,1]
  const int32_t experts_num, // 专家数
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k
) {
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int WARP_K_NUM = BLOCK_SIZE_K / WARP_K;
  for(int e = 0; e < experts_num; e++){
    int real_m = masked_m[e];
    int e_mask = (real_m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int actual_m = e_mask * BLOCK_SIZE_M; //往上取block的整数幂
    int bidx = (e % 2 == 0) ? blockIdx.y : gridDim.y - 1 - blockIdx.y;
    for(; bidx < e_mask; bidx += gridDim.y){
    // for(int bidx = blockIdx.y ; bidx < e_mask; bidx+= gridDim.y){

      const int bidy = blockIdx.x; // pid_n  n方向
      const int bidz = e; // pid_k  e方向

      const uint32_t input_offset = bidz * size_m * size_k + bidx * BLOCK_SIZE_M * size_k; 
      const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * size_k;
      //const uint32_t qweight_offset = bidz * size_n * size_k + bidy * BLOCK_SIZE_N * 64; // moe
      const uint32_t output_offset = bidz * size_m * size_n + bidx * BLOCK_SIZE_M * size_n + bidy * BLOCK_SIZE_N; 
      const uint32_t input_scale_offset = bidz * size_m + bidx * BLOCK_SIZE_M; 
      const uint32_t weight_scale_offset = bidz * size_n + bidy * BLOCK_SIZE_N; 

      auto g_input = input + input_offset;
      auto g_qweight = qweight + qweight_offset;
      auto g_input_scale = input_scale + input_scale_offset;
      auto g_weight_scale = weight_scale + weight_scale_offset;
      OutputType* g_output = output + output_offset;

      int warp_id_vec = threadIdx.x / 64; //warp id in a block
      int warp_id = __builtin_hcu_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
      int lane_id = threadIdx.x & 63; // thread_id
      int row_id = lane_id % 16;
      int col_id = lane_id / 16;
      constexpr int warp_n_num = BLOCK_SIZE_N / WARP_N;
      constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
      int warp_k_id = warp_id % warp_k_num;
      int warp_n_id = warp_id / warp_k_num;

      extern __shared__ Element smem[]; // 声明lds信息
      Element* input_lds = (Element*)&(smem); // decode这里没用
      Element* qweight_lds = input_lds; // decode这里没用
      OutputType* output_lds = (OutputType*)&(smem); // 重复使用lds,留给output_lds
      
      union_vec<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
      union_vec<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];
      intx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 


      // device gemm todo 
      gemm_nt_marlin_decode<Element, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, SIZE_K, STAGES>
        (g_input, g_qweight, input_lds, qweight_lds,
        A_reg, B_reg, C_reg, warp_id, real_m, size_m, size_n, size_k, bidx);

      // write global mem
      if(warp_k_id == 0 || warp_k_num == 1)
      {
        for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
          float a_scale = g_input_scale[min_tile_m * mfma_m + row_id];
          for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
            #pragma unroll  
            for(int reg_id = 0; reg_id < 4; reg_id++){
              float b_scale = g_weight_scale[warp_n_id * WARP_N + min_tile_n * mfma_n + col_id + reg_id * 4];
              float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id]* a_scale * b_scale; 
              int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16) )/2 * 2/*padding*/;
              output_lds[index] = f32_to_output<OutputType>(value);
              // #if defined(__gfx938__)
              // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16 精度有问题
              // #endif
            }
          }
        }
      }

      __syncthreads();

      
      {
        // 最大化利用dwordx4 通用
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
        using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
        int m_idx = threadIdx.x / N_thread;  
        int n_idx = threadIdx.x % N_thread;  
        for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
          if(m_idx < real_m) {
            *reinterpret_cast<vec_bf16_8*>(&g_output[m_idx * size_n + n_idx * 8]) = 
            *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
          }
        } 
      } 
    }
  }
/////////////////////////////////////////////////src///////////////////////////////////////////////////////////////////

}


/////////////////////////////////////////////////////////
// launch 函数实现 - 用于 kernel map 调用
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_decode(const GroupGemmParams<T, OutputType>& params) {
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K) * (BLOCK_SIZE_M / WARP_M);
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const uint32_t size_m = params.size_m;
  const uint32_t size_n = params.size_n;
  const uint32_t size_k = params.size_k;
  
  uint32_t need_size_m = std::min(size_m, (uint32_t)(params.expected_m_per_group * 1.3));
  gridDim.z = DIVIDE(need_size_m, BLOCK_SIZE_M);
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N);
  gridDim.x = params.experts_num;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  const int paddingN = (WARP_N + 1) * (BLOCK_SIZE_N / WARP_N);
  //const int lds_size = std::max(BLOCK_SIZE_M * paddingN * (BLOCK_SIZE_K / WARP_K - 1) * 4, BLOCK_SIZE_M * BLOCK_SIZE_N * 4);
  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;
  
  // 模板化size_k是为了避免四级流水寄存器溢出
  if(params.size_k == 7168){
    DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_GROUPGEMM_MASKED<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 7168,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  } else{
    DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_GROUPGEMM_MASKED<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 2048,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  }
      
}


// prefill 都过lds版本
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_prefill(const GroupGemmParams<T, OutputType>& params) {
  constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const uint32_t size_m = params.size_m;
  const uint32_t size_n = params.size_n;
  const uint32_t size_k = params.size_k;
  const int STAGES = BLOCK_SIZE_K/WARP_K;
  
  uint32_t need_size_m = std::min(size_m, (uint32_t)(params.expected_m_per_group * 1.3));
  gridDim.z = DIVIDE(need_size_m, BLOCK_SIZE_M);
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N);
  gridDim.x = params.experts_num;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  
  // 模板化size_k是为了避免四级流水寄存器溢出
  if(params.size_k == 7168){
    DEEPGEMM_W8A8_I8_PREFILL_MARLIN_TN_GROUPGEMM_MASKED<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 7168,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  } else{
    DEEPGEMM_W8A8_I8_PREFILL_MARLIN_TN_GROUPGEMM_MASKED<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 2048,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  }
      
}

// bypass版本
// prefill 都过lds版本
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_bypass(const GroupGemmParams<T, OutputType>& params) {
  constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const uint32_t size_m = params.size_m;
  const uint32_t size_n = params.size_n;
  const uint32_t size_k = params.size_k;
  constexpr int STAGES = BLOCK_SIZE_K/WARP_K;
  
  uint32_t need_size_m = std::min(size_m, (uint32_t)(params.expected_m_per_group * 1.3));
  gridDim.z = DIVIDE(need_size_m, BLOCK_SIZE_M);
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N);
  gridDim.x = params.experts_num;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  const int bytes_per_element = 2;
  int lds_size_m = BLOCK_SIZE_M / WARP_M * 16; // m方向的warp数 * 16
  int loop = lds_size_m / 2;
  int padding = (loop - 1) * 2 * loop; //累加求和公式推导
  //const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K), (BLOCK_SIZE_M * BLOCK_SIZE_N * 4)  );
  const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K), lds_size_m * BLOCK_SIZE_N * bytes_per_element +  padding * bytes_per_element   );

  // 模板化size_k是为了避免四级流水寄存器溢出
  if(params.size_k == 7168){
    DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_BYPASS_MASKED<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 7168>
    <<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  } else{
    DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_BYPASS_MASKED<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 2048>
    <<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  }
      
}

/////////////////////////////////////////////////////////
// launch 函数实现 - 用于 kernel map 调用
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename OutputType>
void launch_grouped_marlin_w8a8_gemm_nt_masked_decode_persistent(const GroupGemmParams<T, OutputType>& params) {
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const uint32_t size_m = params.size_m;
  const uint32_t size_n = params.size_n;
  const uint32_t size_k = params.size_k;
  
  // 设置kernel参数
  static int HCU_CUS = []() {
                            const char* env = std::getenv("DEEPGEMM_GPU_CUS");
                            return env ? std::strtol(env, nullptr, 10) : 64;
                        }();
  
  const int n_blocks = DIVIDE(size_n, BLOCK_SIZE_N);     // n方向block数
  const int persistent_thread_block = std::lcm(n_blocks, HCU_CUS); // * 2; // 持续化线程块数量设置为n_blocks的整数倍，且不小于2*HCU_CUS
  // std::cout << "zhenggf, DEEPGEMM_GPU_CUS=" << HCU_CUS << std::endl;
  // std::cout << "zhenggf, persistent_thread_block=" << persistent_thread_block << std::endl;
  gridDim.z = 1; // m方向
  gridDim.y = persistent_thread_block/n_blocks;
  gridDim.x = n_blocks; // n方向

  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;
  
  if(params.size_k == 7168){
    DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_GROUPGEMM_MASKED_PERSISTENT<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 7168,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  } else{
    DEEPGEMM_W8A8_I8_DECODE_MARLIN_TN_GROUPGEMM_MASKED_PERSISTENT<int8_t, OutputType, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, 2048,
    STAGES><<<gridDim, blockDim, lds_size, stream>>>(
      (const int8_t*)params.ptr_A, 
      (const int8_t*)params.ptr_B, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale, 
      params.masked_m,
      params.experts_num,
      params.size_m,
      params.size_n,
      params.size_k);
  }
}
/////////////////////////////////////////////////////////



torch::Tensor m_grouped_marlin_w8a8_gemm_nt_masked(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m, 
  int expected_m_per_group,
  int mode) {
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  uint32_t experts_num = b_qweight.sizes()[0];
  const int size_m = input.size(1); 
  const int size_n = b_qweight.size(1) * 16;  // 因为重排的原因 * 16
  const int size_k = input.size(2);
  // debug调试
  // GroupGemmParams<int8_t, bhalf_t> params(
  //   input.data_ptr<int8_t>(),
  //   b_qweight.data_ptr<int8_t>(),
  //   (bhalf_t*)output.data_ptr(),
  //   a_scale.data_ptr<float>(),
  //   b_scale.data_ptr<float>(),
  //   masked_m.data_ptr<int32_t>(),
  //   experts_num,
  //   size_m,
  //   size_n,
  //   size_k,
  //   expected_m_per_group
  // );

  // // // launch_grouped_marlin_w8a8_gemm_nt_masked_prefill<16, 32, 128, 16, 16, 64, int8_t, bhalf_t>(params);
  // launch_grouped_marlin_w8a8_gemm_nt_masked_decode<32, 32, 128, 16, 16, 64, 2, int8_t, bhalf_t>(params);
  // launch_grouped_marlin_w8a8_gemm_nt_masked_bypass<32, 32, 64, 16, 16, 64, int8_t, bhalf_t>(params);
#if 1
  // mode >= 1000: 使用 ASM 版本
  // mode < 1000: 使用 CUDA 版本，通过 kernel map 查找
  if (mode >= 1000) {
    // ASM 版本
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(), "m_groupgemm_w8a8_TN_impl",[&] {
        if (mode == 1000) {
          _m_grouped_marlin_w8a8_gemm_nt_masked_asm_impl<256, 256, 128, scalar_t>(
            input, b_qweight, output, a_scale, b_scale,
            masked_m, expected_m_per_group, experts_num);
        } else if(mode == 1001) {
          _m_grouped_marlin_w8a8_gemm_nt_masked_asm_impl<256, 128, 128, scalar_t, 768>(
            input, b_qweight, output, a_scale, b_scale,
            masked_m, expected_m_per_group, experts_num);
        } else if(mode == 1002) {
          _m_grouped_marlin_w8a8_gemm_nt_masked_asm_impl<256, 64, 128, scalar_t, 512>(
            input, b_qweight, output, a_scale, b_scale,
            masked_m, expected_m_per_group, experts_num);
        } else {
          TORCH_CHECK(false, mode, " ASM mode error.");
        }
      }
    );
  } else {
    TORCH_CHECK(false,
                "m_grouped_marlin_w8a8_gemm_nt_masked mode < 1000 is disabled; "
                "use ASM mode 1000/1001/1002 or the gfx936 low-latency path.");
  }
#endif
  return output;
}

torch::Tensor m_grouped_marlin_w8a8_gemm_nt_compact_masked(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor masked_m,
  torch::Tensor expert_offsets,
  int expected_m_per_group) {

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  TORCH_CHECK(
      input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
          input.scalar_type() == torch::kInt8,
      "compact input must be contiguous CUDA int8 [capacity, hidden]");
  TORCH_CHECK(
      output.is_cuda() && output.is_contiguous() && output.dim() == 2 &&
          output.scalar_type() == torch::kBFloat16,
      "compact output must be contiguous CUDA bfloat16 [capacity, hidden]");
  TORCH_CHECK(
      a_scale.is_cuda() && a_scale.is_contiguous() &&
          a_scale.scalar_type() == torch::kFloat32 &&
          a_scale.size(0) == input.size(0),
      "compact activation scale must be contiguous CUDA float32 and match rows");
  TORCH_CHECK(
      masked_m.is_cuda() && masked_m.is_contiguous() &&
          masked_m.scalar_type() == torch::kInt32 && masked_m.dim() == 1,
      "masked_m must be contiguous CUDA int32 [experts]");
  TORCH_CHECK(
      expert_offsets.is_cuda() && expert_offsets.is_contiguous() &&
          expert_offsets.scalar_type() == torch::kInt32 &&
          expert_offsets.sizes() == masked_m.sizes(),
      "expert_offsets must be contiguous CUDA int32 [experts]");
  TORCH_CHECK(
      b_qweight.is_cuda() && b_qweight.is_contiguous() &&
          b_qweight.size(0) == masked_m.size(0),
      "compact weights and masked_m expert dimensions must match");
  TORCH_CHECK(
      b_scale.is_cuda() && b_scale.is_contiguous() &&
          b_scale.size(0) == masked_m.size(0),
      "compact weight scales and masked_m expert dimensions must match");
  TORCH_CHECK(
      output.size(0) == input.size(0),
      "compact input and output capacities must match");
  TORCH_CHECK(
      expected_m_per_group > 0,
      "expected_m_per_group must be positive");

  const uint32_t experts_num = b_qweight.size(0);
  auto input_3d = input.view({1, input.size(0), input.size(1)});
  auto output_3d = output.view({1, output.size(0), output.size(1)});
  auto scale_3d = a_scale.reshape({1, input.size(0), -1});

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    output.scalar_type(), "m_groupgemm_w8a8_compact_TN_impl", [&] {
      _m_grouped_marlin_w8a8_gemm_nt_masked_asm_impl<
          256, 256, 128, scalar_t>(
          input_3d,
          b_qweight,
          output_3d,
          scale_3d,
          b_scale,
          masked_m,
          expected_m_per_group,
          experts_num,
          &expert_offsets);
    });
  return output;
}


} // namespace deepgemm
