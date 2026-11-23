# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import torch
import triton
import os
import sys
from triton import language as tl
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
import deep_ep
# from moe_wna16.ops import moe_ll
# from moe_wna16 import forward_moe_masked
# from deepgemm import forward_moe_masked
from deepgemm.m_group_gemm import grouped_gemm_w4a16_nt_masked_entry


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "NUM_SM": 64,
                "GROUP_WISE_K": 32,
            }
        ),
    ],
    key=["group_size"],
)

@triton.jit
def moe_w4a16_groupwise_triton(
    matrix_a_ptr,      # [E, M, K] bf16/fp16 激活张量（自动适配）
    matrix_b_ptr,      # [E, K//128, N//16, 4, 16, 4] uint32, 与 HIP 输入一致
    matrix_c_ptr,      # [E, M, N] bf16/fp16 输出张量（与输入类型一致）
    matrix_bs_ptr,     # [E, N, K//GROUP_WISE_K] bf16/fp16/f32 权重 scale, 与 HIP 输入一致
    actual_M,          # [E] int32 每个专家实际需要处理的 token 数
    experts,           # 专家总数
    M: tl.constexpr,   # 每个专家最大 token 数
    N: tl.constexpr,   # 输出维度
    K: tl.constexpr,   # 输入维度
    NUM_SM: tl.constexpr,        # GPU 的 SM 数量（动态调度核心参数）
    BLOCK_SIZE_M: tl.constexpr,  # 每个 block 处理的 M 维度大小
    BLOCK_SIZE_N: tl.constexpr,  # 每个 block 处理的 N 维度大小
    BLOCK_SIZE_K: tl.constexpr,  # 每个 block 处理的 K 维度大小（必须是 GROUP_WISE_K 的约数）
    GROUP_WISE_K: tl.constexpr = 32,  # 量化 group 大小（固定为 AWQ 标准 32）
):
    compute_dtype = matrix_a_ptr.type.element_ty

    tile_idx = tl.program_id(0)

    stride_a_k = 1
    stride_a_m = K
    stride_a_e = M * K

    # matrix_b_ptr is w4a16_marlin_weight(weight):
    # [E, K//128, N//16, 4, 16, 4] uint32.
    stride_b_e = (K // 128) * (N // 16) * 4 * 16 * 4
    stride_b_k128 = (N // 16) * 4 * 16 * 4
    stride_b_n16 = 4 * 16 * 4
    stride_b_k32 = 16 * 4
    stride_b_n = 4
    stride_b_pack = 1

    stride_c_n = 1
    stride_c_m = N
    stride_c_e = M * N
    stride_tokens = 1

    # HIP 输入 scale 布局: [E, N, K//32]
    stride_bs_e = N * (K // GROUP_WISE_K)
    stride_bs_n = K // GROUP_WISE_K
    stride_bs_g = 1

    last_problem_end = 0
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)

    for e in range(experts):
        m = tl.load(actual_M + e * stride_tokens)
        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        num_tiles = num_m_tiles * num_n_tiles

        a_e_ptr = matrix_a_ptr + e * stride_a_e
        b_e_ptr = matrix_b_ptr + e * stride_b_e
        c_e_ptr = matrix_c_ptr + e * stride_c_e
        bs_e_ptr = matrix_bs_ptr + e * stride_bs_e

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            mask_am = offs_am[:, None] < m
            mask_bn = offs_bn[None, :] < N

            accumulator_final = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k_start in range(0, K, BLOCK_SIZE_K):
                k_abs = k_start + offs_k
                mask_k = k_abs < K

                a_ptr = a_e_ptr + offs_am[:, None] * stride_a_m + k_abs[None, :] * stride_a_k
                a = tl.load(a_ptr, mask_am & mask_k[None, :], 0.0)

                # Unpack from [K//128, N//16, 4, 16, 4] uint32 layout.
                # Each uint32 contains 8 consecutive int4 values on K.
                k128 = k_abs // 128
                k32_in_128 = (k_abs % 128) // 32
                pack_in_32 = (k_abs % 32) // 8
                nibble_in_pack = k_abs % 8
                n16 = offs_bn // 16
                n_in_16 = offs_bn % 16

                b_u32_ptr = (
                    b_e_ptr
                    + k128[:, None] * stride_b_k128
                    + n16[None, :] * stride_b_n16
                    + k32_in_128[:, None] * stride_b_k32
                    + n_in_16[None, :] * stride_b_n
                    + pack_in_32[:, None] * stride_b_pack
                )
                b_u32 = tl.load(b_u32_ptr, mask_k[:, None] & mask_bn, 0)
                b_q = ((b_u32 >> (nibble_in_pack[:, None] * 4)) & 0xF).to(tl.float32)

                scale_group = k_abs // GROUP_WISE_K
                b_s_ptr = bs_e_ptr + offs_bn[None, :] * stride_bs_n + scale_group[:, None] * stride_bs_g
                b_s = tl.load(b_s_ptr, mask_k[:, None] & mask_bn, 0.0).to(tl.float32)

                # AWQ: zero point = 8, so dequant = (q - 8) * scale.
                b = ((b_q - 8.0) * b_s).to(compute_dtype)
                accumulator_final += tl.dot(a, b)

            c = accumulator_final.to(compute_dtype)
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptr = c_e_ptr + offs_cm[:, None] * stride_c_m + offs_cn[None, :] * stride_c_n
            tl.store(c_ptr, c, mask_am & mask_bn)

            # 动态调度到下一个 SM（低延迟 MOE 核心优化）
            tile_idx += NUM_SM

        last_problem_end += num_tiles

def w4a16_marlin_weight(weight_input # [size_n, size_k// 2 ]
                        ):
    w1_qweight = weight_input
    e,n,k=w1_qweight.shape
    k = k * 2
    w1_qweight_uint32 = w1_qweight.view(-1).view(torch.uint32)
    new_shape = (e, n // 16, 16, k // 32, 4)  # uint32张量的形状
    w1_qweight_uint32_reshaped = w1_qweight_uint32.view(new_shape)
    w1_qweight_uint32_transposed = w1_qweight_uint32_reshaped.transpose(2, 3).contiguous()
    new_shape = (e, n // 16, k // 128, 4, 16, 4)
    w1_new_trans = w1_qweight_uint32_transposed.view(new_shape)
    w1_qweight_shuffle = w1_new_trans.transpose(1, 2).contiguous()

    return w1_qweight_shuffle

def benchmark_cuda_kernel(fn, warmup=20, iters=200):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def gemm_tflops(elapsed_ms, actual_tokens, n, k):
    flops = 2.0 * float(actual_tokens) * float(n) * float(k)
    return flops / (elapsed_ms * 1.0e-3) / 1.0e12

def print_benchmark_result(name, elapsed_ms, actual_tokens, n, k):
    print(
        f"{name}: {elapsed_ms:.4f} ms, "
        f"{gemm_tflops(elapsed_ms, actual_tokens, n, k):.4f} TFLOPS"
    )

def test_fp8_block_wise():

    num_experts = 48
    num_tokens = 16
    N = 256
    K = 7168
    groupwise = 32
    pack_factor = 2

    print(f"workload M-{num_tokens} N-{N} K-{K}")

    tokens = torch.rand(num_experts, num_tokens, K, device="cuda", dtype=torch.bfloat16)/10000
    weight = torch.randint(0, 256, (num_experts, N, K // pack_factor), device="cuda", dtype=torch.uint8)

    # tokens = torch.ones(num_experts, num_tokens, K, device="cuda", dtype=torch.bfloat16)
    # weight = torch.full((num_experts, N, K // pack_factor), 153, device="cuda", dtype=torch.uint8)
    
    weight_shuffle = w4a16_marlin_weight(weight)
    masked_m = torch.ones(num_experts, device="cuda", dtype=torch.int32) * num_tokens
    seq = torch.arange(0, num_tokens, device="cuda")
    mask = seq.unsqueeze(0) >= masked_m.unsqueeze(1)

    # tokens.masked_fill_(mask.unsqueeze(-1), 0.0)

    weight_scale = torch.rand(num_experts, N, K // groupwise,device="cuda",dtype=torch.bfloat16)
    # weight_scale = torch.ones(num_experts, N, K // groupwise, device="cuda", dtype=torch.bfloat16)

    triton_out = torch.zeros(
        num_experts, num_tokens, N, dtype=torch.bfloat16, device="cuda"
    )
    hip_out = torch.zeros(
        num_experts, num_tokens, N, dtype=torch.bfloat16, device="cuda"
    )

    hip_out = grouped_gemm_w4a16_nt_masked_entry(tokens, weight_shuffle, weight_scale, masked_m, hip_out, num_tokens, num_experts)


    grid = lambda META: (META["NUM_SM"],)
    input_tokens = tokens
    input_weight = weight_shuffle

    moe_w4a16_groupwise_triton[grid](
        input_tokens,
        input_weight,
        triton_out,
        weight_scale,
        masked_m,
        num_experts,
        M=num_tokens,
        N=N,
        K=K,
    )

    torch.cuda.synchronize()
    correctness_pass = torch.allclose(
        hip_out,
        triton_out,
        atol=1e-2,
        rtol=1e-2,
    )
    max_abs_diff = (hip_out.float() - triton_out.float()).abs().max().item()

    bench_warmup = 20
    bench_iters = 200
    actual_tokens_count = int(masked_m.sum().item())

    def run_hip():
        grouped_gemm_w4a16_nt_masked_entry(
            tokens,
            weight_shuffle,
            weight_scale,
            masked_m,
            hip_out,
            num_tokens,
            num_experts,
        )

    def run_triton():
        moe_w4a16_groupwise_triton[grid](
            input_tokens,
            input_weight,
            triton_out,
            weight_scale,
            masked_m,
            num_experts,
            M=num_tokens,
            N=N,
            K=K,
        )

    hip_ms = benchmark_cuda_kernel(run_hip, warmup=bench_warmup, iters=bench_iters)
    triton_ms = benchmark_cuda_kernel(run_triton, warmup=bench_warmup, iters=bench_iters)

    print("========== W4A16 Benchmark ==========")
    print(f"warmup={bench_warmup}, iters={bench_iters}, actual_tokens={actual_tokens_count}, N={N}, K={K}")
    print_benchmark_result("HIP w4a16", hip_ms, actual_tokens_count, N, K)
    print_benchmark_result("Triton w4a16 ref", triton_ms, actual_tokens_count, N, K)
    print(f"speedup(HIP/Triton): {triton_ms / hip_ms:.4f}x")
    print("========== W4A16 Summary ==========")
    print(f"workload: experts={num_experts}, M={num_tokens}, actual_tokens={actual_tokens_count}, N={N}, K={K}")
    print(f"dtype: input={tokens.dtype}, scale={weight_scale.dtype}, output={hip_out.dtype}")
    print(f"correctness: {'PASS' if correctness_pass else 'FAIL'} (atol=1e-2, rtol=1e-2, max_abs_diff={max_abs_diff:.6f})")
    print(f"HIP: {hip_ms:.4f} ms, {gemm_tflops(hip_ms, actual_tokens_count, N, K):.4f} TFLOPS")
    print(f"Triton: {triton_ms:.4f} ms, {gemm_tflops(triton_ms, actual_tokens_count, N, K):.4f} TFLOPS")


def torch_moe(a, w1, w2, score, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    # dummy_weight = torch.ones_like(topk_weight)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # 1. 计算 w1 的线性变换
            intermediate = a[mask] @ w1[i].transpose(0, 1)

            # 2. 兼容低版本 PyTorch 的 SiLU 实现
            d = intermediate.shape[-1] // 2
            # 方法1：用 F.silu（PyTorch 1.6+ 支持，比 torch.silu 兼容性更广）
            silu_part = F.silu(intermediate[..., :d])  # 优先用 functional 中的 silu
            # 方法2：如果 F.silu 也不存在，手动实现 SiLU: x * sigmoid(x)
            # silu_part = intermediate[..., :d] * torch.sigmoid(intermediate[..., :d])

            # 3. 与后半部分相乘
            mul_part = intermediate[..., d:]
            silu_and_mul_result = silu_part * mul_part

            # 4. 与 w2 相乘
            out[mask] = silu_and_mul_result @ w2[i].transpose(0, 1)

    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)

def test_low_latency_ep(world_size: int):
    debug_mode = False
    dp = world_size
    # step 1: calc target 256 expert moe out on master rank
    local_tokens = 16
    tokens = dp * local_tokens
    hidden_size = 7168
    intermediate_size = 2048
    num_experts = 32 * dp
    k_scale_range = 128
    n_scale_range = 128
    real_tokens = 128
    top_k = 8
    input_bf16 = torch.rand(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    if debug_mode:
        input_bf16 = torch.ones(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    # weigt 13
    w13_fp8 = torch.rand(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    if debug_mode:
        w13_fp8 = torch.ones_like(w13_fp8,device="cuda",dtype=torch.float8_e4m3fn)
    N_in = 16
    N_out = (2 * intermediate_size) // N_in
    K_in = 16
    K_mid = 4
    K_out = hidden_size // (K_in * K_mid)
    w13_pmt = w13_fp8.reshape(num_experts,N_out,N_in,K_out,K_mid,K_in).permute(0,3,1,4,2,5).contiguous()
    w13_scale = torch.rand(
        num_experts,
        2 * intermediate_size // n_scale_range,
        hidden_size // k_scale_range,
        device="cuda",
        dtype=torch.float32,
    )
    # if debug_mode:
    #     w13_scale = torch.ones_like(w13_scale,device="cuda",dtype=torch.float32)
    w13_bf16 = w13_fp8.to(torch.float32) * w13_scale.repeat_interleave(
        k_scale_range, dim=-1
    ).repeat_interleave(n_scale_range, dim=-2)
    w13_bf16 = w13_bf16.to(dtype=torch.bfloat16)
    # weight 2
    w2_fp8 = torch.rand(
        num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    N_out = hidden_size // N_in
    K_out = intermediate_size // (K_in * K_mid)
    w2_pmt = w2_fp8.reshape(num_experts,N_out,N_in,K_out,K_mid,K_in).permute(0,3,1,4,2,5).contiguous()
    w2_scale = torch.rand(
        num_experts,
        hidden_size // n_scale_range,
        intermediate_size // k_scale_range,
        device="cuda",
        dtype=torch.float32,
    )
    w2_bf16 = w2_fp8.to(torch.float32) * w2_scale.repeat_interleave(
        k_scale_range, dim=-1
    ).repeat_interleave(n_scale_range, dim=-2)
    w2_bf16 = w2_bf16.to(dtype=torch.bfloat16)
    # topk
    score = torch.rand(tokens, num_experts, device="cuda", dtype=torch.float32)
    # torch moe
    torch_output = torch_moe(input_bf16, w13_bf16, w2_bf16, score, top_k, None)

    test_async = True



    if test_async:
        mp.spawn(
            ep_per_rank,
            args=(
                world_size,
                num_experts,
                top_k,
                input_bf16,
                score,
                w13_pmt,
                w13_scale,
                w2_pmt,
                w2_scale,
                torch_output,
                test_async,
            ),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    # test_low_latency_ep(4)
    test_fp8_block_wise()
    # debug_nan()


