// fp8_gemm_hipc.h — public API for the warp-specialized producer-consumer
// fp8_e4m3 GEMM in fp8_gemm_hipc.cpp.
//
// Signature matches deepgemm::fp8_gemm (fp8_gemm_hipblaslt.h), which swaps
// the operand slots — internally, arg mat_b is the A operand and arg mat_a
// is the B operand:
//   mat_b  [batch, M, K] fp8_e4m3 row-major   ("A", M rows)
//   mat_a  [batch, N, K] fp8_e4m3 row-major   ("B", N rows)
//   output [batch, M, N] bf16 / fp16 / fp32 row-major
// Scales use the same swap: arg scale_b (size M) scales A's rows, arg
// scale_a (size N) scales B's rows; pass both or leave both empty.
// Semantics:
//   output[b,i,j] = alpha * scale_b[b,i]
//                 * (sum_k mat_b[b,i,k] * mat_a[b,j,k]) * scale_a[b,j]
//                 + beta * output[b,i,j]_old
// alpha/beta are fp32; beta*C reads output in place, matching hipblasLt.
// One launch covers `batch` independent GEMMs (grid dim z). fp16 output
// requires |A·B^T| < 65504; bf16/fp32 hold the full fp32 range. bias is
// accepted for signature compatibility and ignored; transpose_flag = "TN".
#pragma once

#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <optional>
#include <string>

// Raw-pointer params used by the underlying kernel. Not part of the public
// call; kept here so profiling/debug harnesses (and versions.cpp) can build.
struct DeepGemm_params {
    void* __restrict__ a_ptr;
    void* __restrict__ b_ptr;
    void* __restrict__ d_ptr;
    // Per-row scales (fp32). scale_a_ptr[i] scales A[i,:] (i in [0, m)),
    // scale_b_ptr[j] scales B[j,:] (j in [0, n)). Output element
    //   D[i, j] = scale_a_ptr[i] * scale_b_ptr[j] * sum_k A[i,k]*B[j,k].
    // May be nullptr; the launch site treats a null pair as "identity scales"
    // and skips the scale multiply entirely.
    void* __restrict__ scale_a_ptr;
    void* __restrict__ scale_b_ptr;
    int m, n, k;
    int a_stride;
    int b_stride;
    int d_stride;
    // Batched GEMM: one kernel launch covers `batch` independent GEMMs. The
    // grid is (tiles_n, tiles_m, batch); the kernel entry offsets each base
    // pointer by blockIdx.z * <batch stride> (element counts, not bytes).
    // A batch stride of 0 means the same buffer is shared across all batches.
    int batch = 1;
    int64_t a_batch_stride = 0;            // fp8 elements per batch (A)
    int64_t b_batch_stride = 0;            // fp8 elements per batch (B)
    int64_t d_batch_stride = 0;            // Element_O elements per batch (D)
    int64_t scale_a_batch_stride = 0;      // fp32 elems per batch (0 = shared)
    int64_t scale_b_batch_stride = 0;
    // Epilogue scalars (one fp32 value for the whole batched GEMM):
    //   D_new = alpha * scale_a * (A·B^T) * scale_b + beta * D_old
    // (C == D_old: output is read in place, matching the hipblasLt reference
    // that passes output as both C and D). alpha=1 / beta=0, the defaults,
    // take the exact no-alpha / no-beta epilogue path in the kernel.
    float alpha = 1.0f;
    float beta  = 0.0f;
    int cu_count;
};

void run_fp8_gemm_hipc(
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
                    const std::optional<torch::Tensor>& bias);
