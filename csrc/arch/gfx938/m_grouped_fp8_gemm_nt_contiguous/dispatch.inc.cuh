#include <torch/extension.h>

#include "bm64_bn256_bk128_nloop2.inc.cuh"
#include "bm128_bn256_bk64_nloop4.inc.cuh"

namespace m_grouped_fp8_contiguous {

torch::Tensor m_grouped_fp8_gemm_nt_contiguous(torch::Tensor input,
                                                torch::Tensor b_qweight,
                                                torch::Tensor output,
                                                torch::Tensor a_scale,
                                                torch::Tensor b_scale,
                                                torch::Tensor m_indices,
                                                int64_t block_m,
                                                int64_t block_n,
                                                int64_t n_loop_num) {
  if (block_m == 64 && block_n == 256 && n_loop_num == 2) {
    return m_grouped_fp8_gemm_nt_contiguous_bm64_bn256_bk128_nloop2::run(
        input, b_qweight, output, a_scale, b_scale, m_indices, block_m,
        block_n, n_loop_num);
  }

  if (block_m == 128 && block_n == 256 && n_loop_num == 4) {
    return m_grouped_fp8_gemm_nt_contiguous_bm128_bn256_bk64_nloop4::run(
        input, b_qweight, output, a_scale, b_scale, m_indices, block_m,
        block_n, n_loop_num);
  }

  TORCH_CHECK(false,
              "unsupported contiguous FP8 HIPC config: BLOCK_M=", block_m,
              ", BLOCK_N=", block_n, ", N_LOOP_NUM=", n_loop_num);
}

}  // namespace m_grouped_fp8_contiguous
