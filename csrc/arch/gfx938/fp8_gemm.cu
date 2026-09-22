#include <hip/hip_runtime.h>
#include "hipblaslt/hipblaslt.h"
#include <hip/hip_bf16.h>
#include <library_types.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <torch/extension.h>

#include <mutex>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <sstream>


hipblasOperation_t get_transpose(char trans){
    if(trans == 'N' || trans == 'n')
        return HIPBLAS_OP_N;           
    else
        return HIPBLAS_OP_T;
}

#define MAX_GPU 16

class HipBlasLtHandleManager {
public:
    static HipBlasLtHandleManager& instance() {
        static HipBlasLtHandleManager mgr;
        return mgr;
    }

    hipblasLtHandle_t getHandleForCurrentDevice() {
        int dev;
        hipGetDevice(&dev);
        std::call_once(*init_flags[dev], [&](){
            auto status = hipblasLtCreate(&handles[dev]);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("hipblasLtCreate failed");
            }
        });
        return handles[dev];
    }

private:
    HipBlasLtHandleManager() {
        int device_count = 0;
        hipGetDeviceCount(&device_count);
        handles.resize(device_count);
        init_flags.resize(device_count);
        for (int i = 0; i < device_count; i++) {
            init_flags[i] = std::make_unique<std::once_flag>();
        }
    }

    ~HipBlasLtHandleManager() {
        for (auto& h : handles) {
            if (h != nullptr) {
                hipblasLtDestroy(h);
            }
        }
    }

    std::vector<hipblasLtHandle_t> handles;
    std::vector<std::unique_ptr<std::once_flag>> init_flags;
};



class BlasltMatmulDesc {
public:
    BlasltMatmulDesc() = delete; 
    BlasltMatmulDesc(hipblasComputeType_t compute_type, hipDataType scale_type) {
        hipblasLtMatmulDescCreate(&desc_, compute_type, scale_type);
    }
    ~BlasltMatmulDesc() {
        hipblasLtMatmulDescDestroy(desc_);
    }
    hipblasLtMatmulDesc_t get() const {
        return desc_;
    }
private:
    hipblasLtMatmulDesc_t desc_;
};

class BlasltMatmulDescManager {
public:
    static BlasltMatmulDesc& get(hipblasComputeType_t compute_type, hipDataType scale_type) {
        using Key = std::pair<int,int>;  
        thread_local std::map<Key, std::unique_ptr<BlasltMatmulDesc>> tls_map;
        Key key{static_cast<int>(compute_type), static_cast<int>(scale_type)};
        auto it = tls_map.find(key);
        if (it == tls_map.end()) {
            auto desc = std::make_unique<BlasltMatmulDesc>(compute_type, scale_type);
            auto& ref = *desc;
            tls_map.emplace(key, std::move(desc));
            return ref;
        }
        return *(it->second);
    }
};



hipDataType getDataType(const at::Tensor& t )
{
    switch (t.scalar_type())
    {
        case at::kHalf:           return HIP_R_16F;
        case at::kBFloat16:       return HIP_R_16BF;
        case at::kFloat:          return HIP_R_32F;
        case at::kDouble:         return HIP_R_64F;
        case at::kChar:           return HIP_R_8I;
        case at::kInt:            return HIP_R_32I;
        case at::kFloat8_e4m3fn:  return HIP_R_8F_E4M3;
        case at::kFloat8_e5m2:    return HIP_R_8F_E5M2;
        default:
            std::cerr << "Unsupported tensor dtype for hipDataType" << std::endl;
            std::exit(EXIT_FAILURE); 
    }
  
}

hipblasComputeType_t getComputeType(const at::Tensor& t)
{
    auto dtype = t.scalar_type();
    if (dtype == at::kFloat)    return HIPBLAS_COMPUTE_32F;
    else if (dtype == at::kInt) return HIPBLAS_COMPUTE_32I;
    else
    {
        std::cerr << "Unsupported tensor dtype for hipblasComputeType" << std::endl;
        std::exit(EXIT_FAILURE);   // 直接终止程序
    }
}


namespace deepgemm {
void fp8_gemm(
                    torch::Tensor mat_a,
                    torch::Tensor mat_b,
                    torch::Tensor scale_a,
                    torch::Tensor scale_b,
                    torch::Tensor output,
                    const int64_t m, 
                    const int64_t n, 
                    const int64_t k, 
                    const int64_t batch,
                    const std::string& transpose_flag,
                    const torch::Tensor& alpha,
                    const torch::Tensor& beta,
                    const std::optional<torch::Tensor>& bias) {
    hipblasLtHandle_t handle = HipBlasLtHandleManager::instance().getHandleForCurrentDevice();
    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);

    int64_t stride_a = m * k;
    int64_t stride_b = n * k;
    int64_t stride_c = m * n;
    int64_t stride_output = m * n;

    size_t a_row = trans_a == HIPBLAS_OP_N ? m : k;
    size_t a_col = trans_a == HIPBLAS_OP_N ? k : m;
    size_t b_row = trans_b == HIPBLAS_OP_N ? k : n;
    size_t b_col = trans_b == HIPBLAS_OP_N ? n : k;
    size_t lda = trans_a==HIPBLAS_OP_N ? m : k;
    size_t ldb = trans_b==HIPBLAS_OP_N ? k : n;
    size_t ldc = m;
    size_t ldd = m;

    hipDataType A_type = getDataType(mat_a);
    hipDataType B_type = getDataType(mat_b);
    hipDataType C_type = getDataType(output);
    hipDataType D_type = getDataType(output);
    hipDataType scale_type=getDataType(scale_a);
    hipblasComputeType_t compute_type=getComputeType(alpha);

    if(A_type == HIP_R_8I &&  B_type == HIP_R_8I ) {
        C_type= HIP_R_32I;
    }

    auto& desc = BlasltMatmulDescManager::get(compute_type, scale_type);
    hipblasLtMatmulDesc_t matmul_desc = desc.get();

    hipblasLtMatrixLayout_t mat_a_layout, mat_b_layout, mat_c_layout, output_layout;
    hipblasLtMatrixLayoutCreate(&mat_a_layout, A_type, a_row, a_col, lda);
    hipblasLtMatrixLayoutCreate(&mat_b_layout, B_type, b_row, b_col, ldb);
    hipblasLtMatrixLayoutCreate(&mat_c_layout, C_type, m, n, ldc);
    hipblasLtMatrixLayoutCreate(&output_layout, D_type, m, n, ldd);

    if (batch > 1) {
        hipblasLtMatrixLayoutSetAttribute(mat_a_layout, 
                HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(mat_a_layout, 
                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, 
                &(stride_a), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(mat_b_layout, 
                HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(mat_b_layout, 
                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, 
                &(stride_b), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(mat_c_layout, 
                HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(mat_c_layout, 
                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, 
                &(stride_c), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(output_layout, 
                HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(output_layout, 
                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, 
                &(stride_output), sizeof(int64_t));
    }


    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t));

    hipblasLtMatmulDescSetAttribute(matmul_desc, 
            HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
            reinterpret_cast<void*>(&scale_a_ptr), 
            sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc,
            HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
            reinterpret_cast<void*>(&scale_b_ptr), 
            sizeof(void*));
    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, (void*)&mode, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, (void*)&mode, sizeof(void*));
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

    if (bias.has_value()) {
        epilogue = HIPBLASLT_EPILOGUE_BIAS;
        void* bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
        hipDataType bias_type = HIP_R_16BF;
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                reinterpret_cast<void*>(&bias_ptr),
                sizeof(void*));
    }
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int32_t));


    hipblasLtMatmulHeuristicResult_t heuristicResult[1];
    hipblasLtMatmulPreference_t pref;
    int returnedAlgoCount = 0;
    hipblasLtMatmulPreferenceCreate(&pref);
    int max_workspace_size=0;
    hipblasLtMatmulPreferenceSetAttribute(
            pref, 
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
            &max_workspace_size, 
            sizeof(max_workspace_size));
    auto ret = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc, 
            mat_a_layout, mat_b_layout, mat_c_layout, output_layout, 
            pref, 1, heuristicResult, &returnedAlgoCount);

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    hipblasLtMatmul(
            handle, matmul_desc, alpha.data_ptr(), 
            mat_a.data_ptr(), mat_a_layout,
            mat_b.data_ptr(), mat_b_layout,
            beta.data_ptr(),
            output.data_ptr(), mat_c_layout,
            output.data_ptr(), output_layout,
            &heuristicResult[0].algo, nullptr, 0, stream);
    hipblasLtMatrixLayoutDestroy(mat_a_layout);
    hipblasLtMatrixLayoutDestroy(mat_b_layout);
    hipblasLtMatrixLayoutDestroy(mat_c_layout);
    hipblasLtMatrixLayoutDestroy(output_layout);
    hipblasLtMatmulPreferenceDestroy(pref);
    return ;
}
}