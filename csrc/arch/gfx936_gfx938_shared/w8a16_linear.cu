// SPDX-License-Identifier: Apache-2.0
// W8A16 linear binding/launcher. Kernel implementations live in
// csrc/include/moe_marlin/w8a16_linear_kernels.h.

#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include "moe_marlin/w8a16_linear_kernels.h"

#ifndef DEEPGEMM_KERNEL_LAUNCH_CHECK
#if defined(C10_HIP_KERNEL_LAUNCH_CHECK)
#define DEEPGEMM_KERNEL_LAUNCH_CHECK C10_HIP_KERNEL_LAUNCH_CHECK
#else
#define DEEPGEMM_KERNEL_LAUNCH_CHECK C10_CUDA_KERNEL_LAUNCH_CHECK
#endif
#endif

namespace deepgemm {

template <typename Element, typename Scale>
__global__ void __launch_bounds__(256)
w8a16_dequant_weight_kernel(
    Element* __restrict__ out,
    const int32_t* __restrict__ b,
    const Scale* __restrict__ scale,
    int total_i32,
    int K_i32)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_i32) return;

    const int n = idx / K_i32;
    const float sc = static_cast<float>(scale[n]);
    const int32_t packed = b[idx];

    w16_vec4_t<Element> val = dequant_int8x4_w16_float_scale<Element>(packed, sc);

    *(reinterpret_cast<w16_vec4_t<Element>*>(out) + idx) = val;
}

static void select_config(int M, int N, int K,
                          int& num_k_warps, int& n_per_warp,
                          int& k_per_warp, int& num_n_warps, int& splitk,
                          int& block_m)
{
    if (num_k_warps <= 0 || n_per_warp <= 0 || num_n_warps <= 0 || splitk <= 0 || block_m <= 0) {
        if (M <= 16) {
            if (num_k_warps <= 0) num_k_warps = 1;
            if (n_per_warp <= 0) n_per_warp = 32;
            if (num_n_warps <= 0) num_n_warps = 4;
            if (splitk <= 0) splitk = 8;
            if (block_m <= 0) block_m = 16;
        } else if (M <= 32) {
            if (num_k_warps <= 0) num_k_warps = 1;
            if (n_per_warp <= 0) n_per_warp = 32;
            if (num_n_warps <= 0) num_n_warps = 4;
            if (splitk <= 0) splitk = 1;
            if (block_m <= 0) block_m = 16;
        } else if (M <= 64) {
            if (num_k_warps <= 0) num_k_warps = (K >= 4096 ? 4 : 1);
            if (n_per_warp <= 0) n_per_warp = 32;
            if (k_per_warp <= 0) k_per_warp = 64;
            if (num_n_warps <= 0) num_n_warps = 2;
            if (splitk <= 0) splitk = 1;
            if (block_m <= 0) block_m = 16;
        } else if (M < 128) {
            if (N <= 2048) {
                if (num_k_warps <= 0) num_k_warps = 1;
                if (n_per_warp <= 0) n_per_warp = 32;
                if (num_n_warps <= 0) num_n_warps = 2;
            } else {
                if (num_k_warps <= 0) num_k_warps = 1;
                if (n_per_warp <= 0) n_per_warp = 64;
                if (num_n_warps <= 0) num_n_warps = 4;
            }
            if (k_per_warp <= 0) k_per_warp = 64;
            if (splitk <= 0) splitk = 1;
            if (block_m <= 0) block_m = 16;
        } else if (M < 192) {
            if (num_k_warps <= 0) num_k_warps = (N <= 2048 ? 2 : 4);
            if (n_per_warp <= 0) n_per_warp = 64;
            if (k_per_warp <= 0) k_per_warp = (N <= 2048 ? 64 : 128);
            if (num_n_warps <= 0) num_n_warps = 2;
            if (splitk <= 0) splitk = 1;
            if (block_m <= 0) block_m = 16;
        } else if (M <= 256) {
            if (num_k_warps <= 0) num_k_warps = (N <= 2048 ? 1 : 2);
            if (n_per_warp <= 0) n_per_warp = 64;
            if (k_per_warp <= 0) k_per_warp = (N <= 2048 ? 64 : 128);
            if (num_n_warps <= 0) num_n_warps = (N <= 2048 ? 4 : 2);
            if (splitk <= 0) splitk = 1;
            if (block_m <= 0) block_m = 32;
        } else if (M <= 1024) {
            if (num_k_warps <= 0) num_k_warps = 2;
            if (n_per_warp <= 0) n_per_warp = 64;
            if (num_n_warps <= 0) num_n_warps = 2;
            if (splitk <= 0) splitk = 1;
            if (block_m <= 0) block_m = 32;
        } else {
            if (num_k_warps <= 0) num_k_warps = 2;
            if (n_per_warp <= 0) n_per_warp = 64;
            if (num_n_warps <= 0) num_n_warps = 2;
            if (splitk <= 0) splitk = 1;
            if (block_m <= 0) block_m = 32;
        }
    }
    if (k_per_warp <= 0) k_per_warp = 128;
}

#define DISPATCH_BM(block_m_val, kernel_call_bm16, kernel_call_bm32, kernel_call_bm64, ...) \
  [&] {                                                                                     \
    if (block_m_val == 16) {                                                                \
      return kernel_call_bm16();                                                            \
    } else if (block_m_val == 32) {                                                         \
      return kernel_call_bm32();                                                            \
    } else {                                                                                \
      return kernel_call_bm64();                                                            \
    }                                                                                       \
  }()

template <typename Element, typename Scale, bool MARLIN_LAYOUT>
static void launch_w8a16_linear(
    Element* out_ptr,
    const Element* a_ptr,
    const int8_t* b_ptr,
    const Scale* s_ptr,
    float* splitk_work_space,
    int* semaphores,
    hipStream_t stream,
    int M,
    int N,
    int K,
    int num_k_warps,
    int n_per_warp,
    int k_per_warp,
    int num_n_warps,
    int splitk,
    int block_m)
{
    splitk = std::max(1, std::min(splitk, std::max(1, K / k_per_warp)));
    KPerWarp_SWITCH(k_per_warp, [&] {
        NPerWarp_SWITCH(n_per_warp, [&] {
            NUM_N_WARPS_SWITCH(num_n_warps, [&] {
                NUM_K_WARPS_SWITCH(num_k_warps, [&] {
                    DISPATCH_BM(block_m,
                        [&] {
                            if (splitk > 1) {
                                dim3 grid(ceil_div(M, 16),
                                          ceil_div(N, NPerWarp * NUM_N_WARPS),
                                          splitk);
                                dim3 block(NUM_K_WARPS * 64, NUM_N_WARPS);
                                int lds_size = smem_bytes(NUM_K_WARPS, NPerWarp, NUM_N_WARPS, MIN(16, M));
                                hipLaunchKernelGGL((gemm_w8a16_int8_splitk_kernel<Element, Scale, MARLIN_LAYOUT, NUM_K_WARPS, NPerWarp, KPerWarp, NUM_N_WARPS>),
                                                   dim3(grid), dim3(block), lds_size, stream,
                                                   out_ptr, a_ptr, b_ptr, s_ptr,
                                                   splitk_work_space, semaphores,
                                                   M, N, K);
                            } else {
                                dim3 grid(ceil_div(M, 16),
                                          ceil_div(N, NPerWarp * NUM_N_WARPS));
                                dim3 block(NUM_K_WARPS * 64, NUM_N_WARPS);
                                int lds_size = smem_bytes(NUM_K_WARPS, NPerWarp, NUM_N_WARPS, MIN(16, M));
                                hipLaunchKernelGGL((gemm_w8a16_int8_kernel<Element, Scale, MARLIN_LAYOUT, NUM_K_WARPS, NPerWarp, KPerWarp, NUM_N_WARPS>),
                                                   dim3(grid), dim3(block), lds_size, stream,
                                                   out_ptr, a_ptr, b_ptr, s_ptr,
                                                   M, N, K);
                            }
                            return 0;
                        },
                        [&] {
                            dim3 grid(ceil_div(M, 32),
                                      ceil_div(N, NPerWarp * NUM_N_WARPS));
                            dim3 block(NUM_K_WARPS * 64, NUM_N_WARPS);
                            int lds_size = bm32_smem_bytes<Element>(NUM_K_WARPS, NPerWarp, NUM_N_WARPS, MIN(32, M), KPerWarp);
                            hipLaunchKernelGGL((gemm_w8a16_int8_bm32_kernel<Element, Scale, MARLIN_LAYOUT, NUM_K_WARPS, NPerWarp, KPerWarp, NUM_N_WARPS>),
                                               dim3(grid), dim3(block), lds_size, stream,
                                               out_ptr, a_ptr, b_ptr, s_ptr,
                                               M, N, K);
                            return 0;
                        },
                        [&] {
                            dim3 grid(ceil_div(M, 64),
                                      ceil_div(N, NPerWarp * NUM_N_WARPS));
                            dim3 block(NUM_K_WARPS * 64, NUM_N_WARPS);
                            int lds_size = smem_bytes(NUM_K_WARPS, NPerWarp, NUM_N_WARPS, MIN(64, M));
                            hipLaunchKernelGGL((gemm_w8a16_int8_bm64_kernel<Element, Scale, MARLIN_LAYOUT, NUM_K_WARPS, NPerWarp, KPerWarp, NUM_N_WARPS>),
                                               dim3(grid), dim3(block), lds_size, stream,
                                               out_ptr, a_ptr, b_ptr, s_ptr,
                                               M, N, K);
                            return 0;
                        }
                    );
                    return 0;
                });
            });
        });
    });
}

torch::Tensor w8a16_linear(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale,
    int num_k_warps,
    int n_per_warp,
    int k_per_warp,
    int num_n_warps,
    int splitk,
    int block_m,
    int weight_layout)
{
    const c10::OptionalDeviceGuard device_guard(device_of(a));
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && scale.is_cuda());
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 || a.scalar_type() == torch::kBFloat16,
                "a must be fp16 or bf16");
    TORCH_CHECK(b.scalar_type() == torch::kInt8, "b must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat16 || scale.scalar_type() == torch::kBFloat16,
                "scale must be fp16 or bf16");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && scale.is_contiguous());

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);
    TORCH_CHECK(b.size(1) == K && scale.size(0) == N);
    TORCH_CHECK(K % 16 == 0, "K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 4 == 0, "N must be divisible by 4 for vectorized stores, got ", N);
    TORCH_CHECK(weight_layout == 0 || weight_layout == 1, "weight_layout must be 0(raw) or 1(marlin)");
    if (weight_layout == 1) {
        TORCH_CHECK(K % 128 == 0, "marlin weight layout requires K divisible by 128, got ", K);
        TORCH_CHECK(N % 16 == 0, "marlin weight layout requires N divisible by 16, got ", N);
    }

    auto out = torch::empty({M, N}, a.options());
    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    select_config(M, N, K, num_k_warps, n_per_warp, k_per_warp, num_n_warps, splitk, block_m);

    const int8_t* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    static float* splitk_work_space = nullptr;
    static int* semaphores = nullptr;
    if (splitk_work_space == nullptr) {
        hipMalloc(&splitk_work_space, 100000000);
        hipMemset(splitk_work_space, 0, 100000000);
        hipMalloc(&semaphores, 1000000);
        hipMemset(semaphores, 0, 1000000);
    }

    auto launch_for_dtype = [&](auto* out_ptr, const auto* a_ptr, const auto* s_ptr) {
        using Element = std::remove_pointer_t<decltype(out_ptr)>;
        using Scale = std::remove_pointer_t<decltype(s_ptr)>;
        if (weight_layout == 1) {
            launch_w8a16_linear<Element, Scale, true>(out_ptr, a_ptr, b_ptr, s_ptr, splitk_work_space, semaphores,
                                                     stream, M, N, K, num_k_warps, n_per_warp, k_per_warp,
                                                     num_n_warps, splitk, block_m);
        } else {
            launch_w8a16_linear<Element, Scale, false>(out_ptr, a_ptr, b_ptr, s_ptr, splitk_work_space, semaphores,
                                                      stream, M, N, K, num_k_warps, n_per_warp, k_per_warp,
                                                      num_n_warps, splitk, block_m);
        }
    };

    if (a.scalar_type() == torch::kBFloat16 && scale.scalar_type() == torch::kBFloat16) {
        launch_for_dtype(reinterpret_cast<bf16_t*>(out.data_ptr()),
                         reinterpret_cast<const bf16_t*>(a.data_ptr()),
                         reinterpret_cast<const bf16_t*>(scale.data_ptr()));
    } else if (a.scalar_type() == torch::kBFloat16) {
        launch_for_dtype(reinterpret_cast<bf16_t*>(out.data_ptr()),
                         reinterpret_cast<const bf16_t*>(a.data_ptr()),
                         reinterpret_cast<const half_t*>(scale.data_ptr()));
    } else if (scale.scalar_type() == torch::kBFloat16) {
        launch_for_dtype(reinterpret_cast<half_t*>(out.data_ptr()),
                         reinterpret_cast<const half_t*>(a.data_ptr()),
                         reinterpret_cast<const bf16_t*>(scale.data_ptr()));
    } else {
        launch_for_dtype(reinterpret_cast<half_t*>(out.data_ptr()),
                         reinterpret_cast<const half_t*>(a.data_ptr()),
                         reinterpret_cast<const half_t*>(scale.data_ptr()));
    }

    DEEPGEMM_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor w8a16_linear_bias(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale,
    const torch::Tensor& bias,
    int num_k_warps,
    int n_per_warp,
    int k_per_warp,
    int num_n_warps,
    int splitk,
    int block_m,
    int weight_layout)
{
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be CUDA contiguous");
    TORCH_CHECK(bias.scalar_type() == a.scalar_type(), "bias must match a dtype");
    TORCH_CHECK(bias.numel() == b.size(0), "bias.numel must equal N");
    auto out = w8a16_linear(a, b, scale, num_k_warps, n_per_warp, k_per_warp, num_n_warps, splitk, block_m, weight_layout);
    out.add_(bias);
    return out;
}

torch::Tensor w8a16_dequant_weight_to_dtype(
    const torch::Tensor& b,
    const torch::Tensor& scale,
    c10::ScalarType out_dtype)
{
    const c10::OptionalDeviceGuard device_guard(device_of(b));
    TORCH_CHECK(b.is_cuda() && scale.is_cuda(), "b and scale must be CUDA tensors");
    TORCH_CHECK(b.scalar_type() == torch::kInt8, "b must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat16 || scale.scalar_type() == torch::kBFloat16,
                "scale must be fp16 or bf16");
    TORCH_CHECK(b.is_contiguous() && scale.is_contiguous(), "b and scale must be contiguous");
    TORCH_CHECK(b.dim() == 2, "b must be 2-D [N, K]");

    const int N = b.size(0);
    const int K = b.size(1);
    TORCH_CHECK(scale.numel() == N, "scale.numel must equal N");
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for vectorized dequant, got ", K);

    TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kBFloat16,
                "out_dtype must be fp16 or bf16");

    auto out = torch::empty({N, K}, scale.options().dtype(out_dtype));
    const int total_i32 = N * (K / 4);
    const int threads = 256;
    const int blocks = ceil_div(total_i32, threads);
    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (out_dtype == torch::kBFloat16) {
        if (scale.scalar_type() == torch::kBFloat16) {
            hipLaunchKernelGGL((w8a16_dequant_weight_kernel<bf16_t, bf16_t>),
                dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<bf16_t*>(out.data_ptr()),
                reinterpret_cast<const int32_t*>(b.data_ptr()),
                reinterpret_cast<const bf16_t*>(scale.data_ptr()),
                total_i32,
                K / 4);
        } else {
            hipLaunchKernelGGL((w8a16_dequant_weight_kernel<bf16_t, half_t>),
                dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<bf16_t*>(out.data_ptr()),
                reinterpret_cast<const int32_t*>(b.data_ptr()),
                reinterpret_cast<const half_t*>(scale.data_ptr()),
                total_i32,
                K / 4);
        }
    } else {
        if (scale.scalar_type() == torch::kBFloat16) {
            hipLaunchKernelGGL((w8a16_dequant_weight_kernel<half_t, bf16_t>),
                dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<half_t*>(out.data_ptr()),
                reinterpret_cast<const int32_t*>(b.data_ptr()),
                reinterpret_cast<const bf16_t*>(scale.data_ptr()),
                total_i32,
                K / 4);
        } else {
            hipLaunchKernelGGL((w8a16_dequant_weight_kernel<half_t, half_t>),
                dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<half_t*>(out.data_ptr()),
                reinterpret_cast<const int32_t*>(b.data_ptr()),
                reinterpret_cast<const half_t*>(scale.data_ptr()),
                total_i32,
                K / 4);
        }
    }
    DEEPGEMM_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor w8a16_dequant_weight(
    const torch::Tensor& b,
    const torch::Tensor& scale)
{
    return w8a16_dequant_weight_to_dtype(b, scale, torch::kFloat16);
}

torch::Tensor w8a16_dequant_weight_bf16(
    const torch::Tensor& b,
    const torch::Tensor& scale)
{
    return w8a16_dequant_weight_to_dtype(b, scale, torch::kBFloat16);
}

} // namespace deepgemm
