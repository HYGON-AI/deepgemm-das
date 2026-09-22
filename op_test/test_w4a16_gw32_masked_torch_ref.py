import torch
import triton
import os
import sys
from triton import language as tl
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
# from moe_wna16.ops import moe_ll
# from moe_wna16 import forward_moe_masked
# from deepgemm import forward_moe_masked
from deepgemm.m_group_gemm import (
    grouped_gemm_w4a16_nt_masked_entry,
    w4a16_marlin_weight,
)


num_experts = 48
num_tokens_lists = [16, 32]
N_list = [4096, 4096]
K_list = [7168, 4096]

def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2
) -> None:
    """
    比较两个任意维度的PyTorch张量的差异，支持绝对误差和相对误差阈值。
    直接输出详细比较结果，包括每个元素在原始张量中的多维坐标。
    无返回值。

    参数:
        tensor1: 第一个张量（如Triton模型输出）
        tensor2: 第二个张量（如PyTorch模型输出）
        atol: 绝对误差阈值，默认1e-5
        rtol: 相对误差阈值，默认1e-8
    """
    # -------------------------- 1. 输入合法性校验 --------------------------
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量（torch.Tensor）")

    if tensor1.shape != tensor2.shape:
        raise ValueError(f"张量形状不匹配！ tensor1形状: {tensor1.shape}, tensor2形状: {tensor2.shape}")

    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
        print(f"警告：张量设备不一致，已将tensor2转移到{tensor1.device}")

    # -------------------------- 2. 核心差异计算 --------------------------
    abs_diff = torch.abs(tensor1 - tensor2)
    denom = torch.maximum(torch.abs(tensor1), torch.abs(tensor2))
    rel_diff = abs_diff / (denom + 1e-12)
    match_mask = (abs_diff <= atol) | (rel_diff <= rtol)

    # -------------------------- 3. 展平张量 --------------------------
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    abs_diff_flat = abs_diff.flatten()
    match_mask_flat = match_mask.flatten()

    # -------------------------- 4. 收集一维索引 --------------------------
    def get_indices_1d(mask: torch.Tensor) -> list:
        indices = torch.nonzero(mask).squeeze(dim=1)
        return indices.tolist() if indices.numel() > 0 else []

    match_indices_1d = get_indices_1d(match_mask_flat)
    mismatch_indices_1d = get_indices_1d(~match_mask_flat)

    # -------------------------- 5. 总体统计信息 --------------------------
    total = tensor1_flat.numel()
    matched = len(match_indices_1d)
    mismatched = len(mismatch_indices_1d)
    match_rate = matched / total if total > 0 else 0.0
    max_abs_diff = abs_diff.max().item() if total > 0 else 0.0
    avg_abs_diff = abs_diff.mean().item() if total > 0 else 0.0

    # -------------------------- 6. 格式化输出 --------------------------
    print("=" * 60)
    print("张量比较结果汇总")
    print("=" * 60)
    print(f"张量形状: {tensor1.shape} | 总元素数: {total}")
    print(f"阈值设置: 绝对误差(atol)={atol:.2e}, 相对误差(rtol)={rtol:.2e}")
    print("-" * 60)
    print(f"匹配元素数: {matched} ({match_rate:.2%})")
    print(f"不匹配元素数: {mismatched} ({1 - match_rate:.2%})")
    print(f"最大绝对差异: {max_abs_diff:.6f}")
    print(f"平均绝对差异: {avg_abs_diff:.6f}")
    print("=" * 60)

    # -------------------------- 7. 输出匹配/不匹配示例 --------------------------
    def print_sample(name: str, indices_1d: list, max_samples: int = 3, elem_per_sample: int = 10) -> None:
        if not indices_1d:
            print(f"\n【{name}样本】无数据")
            return

        print(f"\n【{name}样本】（最多展示{max_samples}组，每组{elem_per_sample}个元素）")
        print("-" * 50)

        num_samples = min(max_samples, (len(indices_1d) + elem_per_sample - 1) // elem_per_sample)

        for i in range(num_samples):
            start = i * elem_per_sample
            end = start + elem_per_sample
            sample_indices_1d = indices_1d[start:end]

            # <<< 关键改动: 使用 torch.unravel_index 转换为多维坐标 >>>
            # 此函数能处理任意维度
            sample_coords = torch.unravel_index(torch.tensor(sample_indices_1d), tensor1.shape)
            # 将结果从张量元组转换为坐标元组列表
            sample_coords_list = list(zip(*[coord.tolist() for coord in sample_coords]))

            print(f"\n第{i+1}组:")
            print(f"  原始多维坐标: {sample_coords_list}")
            print(f"  tensor1: {[round(tensor1_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  tensor2: {[round(tensor2_flat[idx].item(), 6) for idx in sample_indices_1d]}")
            print(f"  绝对差异: {[round(abs_diff_flat[idx].item(), 6) for idx in sample_indices_1d]}")

    print_sample("匹配", match_indices_1d, max_samples=2)
    print_sample("不匹配", mismatch_indices_1d, max_samples=3)
    print("\n" + "=" * 60)

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
    matrix_b_ptr,      # [E, N, K//2] uint8 raw packed W4 权重
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

    # Raw packed W4 layout: [E, N, K//2] uint8.
    stride_b_e = N * (K // 2)
    stride_b_n = K // 2
    stride_b_k2 = 1

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

                # Unpack from raw [E, N, K//2] uint8 layout.
                # Each uint8 contains two consecutive int4 values on K.
                b_u8_ptr = (
                    b_e_ptr
                    + offs_bn[None, :] * stride_b_n
                    + (k_abs[:, None] // 2) * stride_b_k2
                )
                b_u8 = tl.load(b_u8_ptr, mask_k[:, None] & mask_bn, 0)
                b_q = ((b_u8 >> ((k_abs[:, None] & 1) * 4)) & 0xF).to(tl.float32)

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


def torch_w4a16_groupwise_ref(
    tokens,
    packed_weight,
    weight_scale,
    actual_m,
    out=None,
    groupwise=32,
    chunk_n=1024,
):
    """Torch reference for grouped W4A16 NT masked GEMM."""
    num_experts, size_m, k = tokens.shape
    weight_experts, n, packed_k = packed_weight.shape
    if weight_experts != num_experts:
        raise ValueError(
            f"weight experts {weight_experts} must match tokens experts {num_experts}"
        )
    if packed_k * 2 < k:
        raise ValueError(f"packed K {packed_k} is too small for activation K {k}")
    if weight_scale.shape[:2] != (num_experts, n):
        raise ValueError(
            f"weight_scale shape {weight_scale.shape} must start with {(num_experts, n)}"
        )

    if out is None:
        out = torch.zeros(
            num_experts, size_m, n, dtype=tokens.dtype, device=tokens.device
        )
    else:
        out.zero_()

    if isinstance(actual_m, torch.Tensor):
        actual_m_list = [int(x) for x in actual_m.detach().cpu().tolist()]
    else:
        actual_m_list = [int(x) for x in actual_m]
    if len(actual_m_list) != num_experts:
        raise ValueError(
            f"actual_m length {len(actual_m_list)} must match num_experts {num_experts}"
        )

    chunk_n = max(1, int(chunk_n))
    for expert_id, m in enumerate(actual_m_list):
        if m < 0 or m > size_m:
            raise ValueError(f"actual_m[{expert_id}]={m} must be in [0, {size_m}]")
        if m == 0:
            continue

        a = tokens[expert_id, :m, :]
        for n_start in range(0, n, chunk_n):
            n_end = min(n_start + chunk_n, n)
            w_u8 = packed_weight[expert_id, n_start:n_end, :]
            low = w_u8 & 0x0F
            high = (w_u8 >> 4) & 0x0F
            unpacked = torch.empty(
                n_end - n_start,
                packed_k * 2,
                dtype=torch.uint8,
                device=packed_weight.device,
            )
            unpacked[:, 0::2] = low
            unpacked[:, 1::2] = high
            unpacked = unpacked[:, :k].to(torch.float32)

            scales = weight_scale[expert_id, n_start:n_end, :].to(torch.float32)
            scales = scales.repeat_interleave(groupwise, dim=1)[:, :k]
            dequant_weight = ((unpacked - 8.0) * scales).to(tokens.dtype)
            out[expert_id, :m, n_start:n_end] = torch.matmul(
                a, dequant_weight.transpose(0, 1)
            )

    return out


def env_flag(name, default="0"):
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")

def parse_actual_m_list(value, num_experts, env_name):
    actual_m = [int(x) for x in value.split(",") if x.strip()]
    if len(actual_m) == 1:
        actual_m = actual_m * num_experts
    if len(actual_m) != num_experts:
        raise ValueError(
            f"{env_name} length {len(actual_m)} must be 1 or num_experts={num_experts}"
        )
    return actual_m

def make_framework_actual_m(num_experts):
    explicit = os.getenv("W4A16_FRAMEWORK_MIXED_M", "")
    if explicit:
        return parse_actual_m_list(explicit, num_experts, "W4A16_FRAMEWORK_MIXED_M")

    seed = int(os.getenv("W4A16_FRAMEWORK_SEED", "2026"))
    nonzero_env = os.getenv("W4A16_FRAMEWORK_NONZERO_EXPERTS", "")
    fixed_m_env = os.getenv("W4A16_FRAMEWORK_FIXED_M", "")
    min_m = int(os.getenv("W4A16_FRAMEWORK_MIN_M", "1"))
    max_m = int(os.getenv("W4A16_FRAMEWORK_MAX_M", "16"))
    if max_m < min_m:
        raise ValueError(
            f"W4A16_FRAMEWORK_MAX_M ({max_m}) must be >= W4A16_FRAMEWORK_MIN_M ({min_m})"
        )

    if nonzero_env:
        nonzero = int(nonzero_env)
    else:
        zero_ratio = float(os.getenv("W4A16_FRAMEWORK_ZERO_RATIO", "0.75"))
        zero_ratio = min(max(zero_ratio, 0.0), 1.0)
        nonzero = int(round(num_experts * (1.0 - zero_ratio)))
    nonzero = min(num_experts, max(1, nonzero))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    expert_ids = torch.randperm(num_experts, generator=generator)[:nonzero].tolist()
    if fixed_m_env:
        fixed_m = int(fixed_m_env)
        if fixed_m < 0:
            raise ValueError("W4A16_FRAMEWORK_FIXED_M must be non-negative")
        values = [fixed_m] * nonzero
    else:
        values = torch.randint(min_m, max_m + 1, (nonzero,), generator=generator).tolist()

    actual_m = [0] * num_experts
    for expert_id, value in zip(expert_ids, values):
        actual_m[expert_id] = int(value)
    return actual_m

def expected_m_from_actual_m(actual_m):
    total = sum(int(x) for x in actual_m)
    return max(1, (total + len(actual_m) - 1) // len(actual_m))

def test_fp8_block_wise(num_experts, num_tokens, N, K):

    # num_experts = int(os.getenv("W4A16_TEST_EXPERTS", "48"))
    # num_tokens = int(os.getenv("W4A16_TEST_M", "16"))
    # N = int(os.getenv("W4A16_TEST_N", "4096"))
    # K = int(os.getenv("W4A16_TEST_K", "7168"))
    groupwise = 32
    pack_factor = 2
    size_m = int(os.getenv("W4A16_TEST_SIZE_M", str(num_tokens)))
    impl = "auto"

    mixed_m_env = os.getenv("W4A16_TEST_MIXED_M", "")
    mixed_m = None
    if mixed_m_env:
        mixed_m = parse_actual_m_list(mixed_m_env, num_experts, "W4A16_TEST_MIXED_M")
        size_m = max(size_m, max(mixed_m))
        num_tokens = max(mixed_m)

    framework_m = None
    if env_flag("W4A16_FRAMEWORK_BENCH"):
        framework_m = make_framework_actual_m(num_experts)
        size_m = max(size_m, max(framework_m))

    print(f"workload M-{num_tokens} N-{N} K-{K}")

    tokens = torch.rand(num_experts, size_m, K, device="cuda", dtype=torch.bfloat16)/10000
    weight = torch.randint(0, 256, (num_experts, N, K // pack_factor), device="cuda", dtype=torch.uint8)

    # tokens = torch.ones(num_experts, num_tokens, K, device="cuda", dtype=torch.bfloat16)
    # weight = torch.full((num_experts, N, K // pack_factor), 153, device="cuda", dtype=torch.uint8)

    weight_shuffle = w4a16_marlin_weight(weight)
    if framework_m is not None:
        active_actual_m = framework_m
        masked_m = torch.tensor(framework_m, device="cuda", dtype=torch.int32)
        expected_m = expected_m_from_actual_m(framework_m)
    elif mixed_m is not None:
        active_actual_m = mixed_m
        masked_m = torch.tensor(mixed_m, device="cuda", dtype=torch.int32)
        expected_m = expected_m_from_actual_m(mixed_m)
    else:
        active_actual_m = [num_tokens] * num_experts
        masked_m = torch.ones(num_experts, device="cuda", dtype=torch.int32) * num_tokens
        expected_m = num_tokens
    # seq = torch.arange(0, num_tokens, device="cuda")
    # mask = seq.unsqueeze(0) >= masked_m.unsqueeze(1)

    # tokens.masked_fill_(mask.unsqueeze(-1), 0.0)

    weight_scale = torch.rand(num_experts, N, K // groupwise,device="cuda",dtype=torch.bfloat16)
    # weight_scale = torch.ones(num_experts, N, K // groupwise, device="cuda", dtype=torch.bfloat16)
    scale_mode = "plain"
    hip_weight_scale = weight_scale

    torch_out = torch.zeros(
        num_experts, size_m, N, dtype=torch.bfloat16, device="cuda"
    )
    hip_out = torch.zeros(
        num_experts, size_m, N, dtype=torch.bfloat16, device="cuda"
    )

    def run_hip_once(masked_m_arg, expected_m_arg, out_arg):
        return grouped_gemm_w4a16_nt_masked_entry(
            tokens,
            weight_shuffle,
            hip_weight_scale,
            out_arg,
            masked_m_arg,
            expected_m_arg,
        )

    if env_flag("W4A16_PROFILE_ONLY"):
        if framework_m is not None:
            profile_m = framework_m
        elif mixed_m is not None:
            profile_m = mixed_m
        else:
            profile_m = [num_tokens] * num_experts

        profile_masked_m = torch.tensor(profile_m, device="cuda", dtype=torch.int32)
        profile_expected_m = expected_m_from_actual_m(profile_m)
        profile_actual_tokens = int(profile_masked_m.sum().item())
        profile_zero_experts = int((profile_masked_m == 0).sum().item())

        run_hip_once(
            profile_masked_m,
            profile_expected_m,
            hip_out,
        )
        torch.cuda.synchronize()
        print("========== W4A16 Profile Only ==========")
        print(
            f"impl={impl}, scale_mode={scale_mode}, actual_tokens={profile_actual_tokens}, "
            f"zero_experts={profile_zero_experts}/{num_experts}, "
            f"max_actual_m={max(1, max(profile_m))}, expected_m={profile_expected_m}, "
            f"size_m={size_m}, N={N}, K={K}"
        )
        return

    # m_grouped_gemm_fp8_fp8_bf16_nt_masked(lhs,rhs,masked_m,)
    # print("print test_w4a16 date type")
    # print(f"tokens dtype: {tokens.dtype}, device: {tokens.device}")
    # print(f"weight_scale dtype: {weight_scale.dtype}, device: {weight_scale.device}")
    # print(f"out dtype: {hip_out.dtype}, device: {hip_out.device}")


    hip_out = run_hip_once(
        masked_m,
        expected_m,
        hip_out,
    )


    grid = lambda META: (META["NUM_SM"],)
    input_tokens = tokens
    input_weight = weight

    torch_w4a16_groupwise_ref(
        tokens,
        weight,
        weight_scale,
        masked_m,
        torch_out,
        groupwise=groupwise,
    )

    torch.cuda.synchronize()

    # compare_tensors(hip_out, torch_out)

    correctness_pass = torch.allclose(
        hip_out,
        torch_out,
        atol=1e-2,
        rtol=1e-2,
    )
    max_abs_diff = (hip_out.float() - torch_out.float()).abs().max().item()

    if env_flag("W4A16_SKIP_BENCH"):
        actual_tokens_count = int(masked_m.sum().item())
        print("========== W4A16 Single Run ==========")
        print(
            f"impl={impl}, scale_mode={scale_mode}, actual_tokens={actual_tokens_count}, "
            f"expected_m={expected_m}, size_m={size_m}, N={N}, K={K}"
        )
        print(
            f"correctness: {'PASS' if correctness_pass else 'FAIL'} "
            f"(atol=1e-2, rtol=1e-2, max_abs_diff={max_abs_diff:.6f})"
        )
        return

    bench_warmup = int(os.getenv("W4A16_BENCH_WARMUP", "20"))
    bench_iters = int(os.getenv("W4A16_BENCH_ITERS", "200"))
    actual_tokens_count = int(masked_m.sum().item())

    def run_hip():
        run_hip_once(
            masked_m,
            expected_m,
            hip_out,
        )

    def run_torch():
        torch_w4a16_groupwise_ref(
            tokens,
            weight,
            weight_scale,
            masked_m,
            torch_out,
            groupwise=groupwise,
        )

    hip_ms = benchmark_cuda_kernel(run_hip, warmup=bench_warmup, iters=bench_iters)
    torch_ms = benchmark_cuda_kernel(run_torch, warmup=bench_warmup, iters=bench_iters)

    print("========== W4A16 Benchmark ==========")
    print(
        f"warmup={bench_warmup}, iters={bench_iters}, impl={impl}, "
        f"scale_mode={scale_mode}, actual_tokens={actual_tokens_count}, "
        f"expected_m={expected_m}, size_m={size_m}, N={N}, K={K}"
    )
    print_benchmark_result("HIP w4a16", hip_ms, actual_tokens_count, N, K)
    print_benchmark_result("Torch w4a16 ref", torch_ms, actual_tokens_count, N, K)
    print(f"speedup(HIP/Torch): {torch_ms / hip_ms:.4f}x")
    print("========== W4A16 Summary ==========")
    print(
        f"workload: experts={num_experts}, actual_M={num_tokens}, size_m={size_m}, "
        f"impl={impl}, scale_mode={scale_mode}, actual_tokens={actual_tokens_count}, "
        f"expected_m={expected_m}, N={N}, K={K}"
    )
    print(
        f"dtype: input={tokens.dtype}, scale={weight_scale.dtype}, "
        f"hip_scale={hip_weight_scale.dtype}, output={hip_out.dtype}"
    )
    print(
        f"correctness: {'PASS' if correctness_pass else 'FAIL'} "
        f"(atol=1e-2, rtol=1e-2, max_abs_diff={max_abs_diff:.6f})"
    )
    print(f"HIP: {hip_ms:.4f} ms, {gemm_tflops(hip_ms, actual_tokens_count, N, K):.4f} TFLOPS")
    print(f"Torch: {torch_ms:.4f} ms, {gemm_tflops(torch_ms, actual_tokens_count, N, K):.4f} TFLOPS")

    if framework_m is not None:
        framework_masked_m = torch.tensor(framework_m, device="cuda", dtype=torch.int32)
        framework_max_tokens = max(1, max(framework_m))
        framework_expected_m = expected_m_from_actual_m(framework_m)
        framework_actual_tokens = int(framework_masked_m.sum().item())
        framework_zero_experts = int((framework_masked_m == 0).sum().item())

        torch_out.zero_()
        hip_out.zero_()
        run_hip_once(
            framework_masked_m,
            framework_expected_m,
            hip_out,
        )

        torch_w4a16_groupwise_ref(
            tokens,
            weight,
            weight_scale,
            framework_masked_m,
            torch_out,
            groupwise=groupwise,
        )
        torch.cuda.synchronize()
        framework_correctness_pass = torch.allclose(
            hip_out,
            torch_out,
            atol=1e-2,
            rtol=1e-2,
        )
        framework_max_abs_diff = (hip_out.float() - torch_out.float()).abs().max().item()

        def run_hip_framework():
            run_hip_once(
                framework_masked_m,
                framework_expected_m,
                hip_out,
            )

        def run_torch_framework():
            torch_w4a16_groupwise_ref(
                tokens,
                weight,
                weight_scale,
                framework_masked_m,
                torch_out,
                groupwise=groupwise,
            )

        hip_fw_ms = benchmark_cuda_kernel(
            run_hip_framework, warmup=bench_warmup, iters=bench_iters
        )
        torch_fw_ms = benchmark_cuda_kernel(
            run_torch_framework, warmup=bench_warmup, iters=bench_iters
        )

        print("========== W4A16 Framework-like Benchmark ==========")
        print(
            f"warmup={bench_warmup}, iters={bench_iters}, impl={impl}, "
            f"scale_mode={scale_mode}, actual_tokens={framework_actual_tokens}, "
            f"zero_experts={framework_zero_experts}/{num_experts}, "
            f"max_actual_m={framework_max_tokens}, expected_m={framework_expected_m}, "
            f"size_m={size_m}, N={N}, K={K}"
        )
        print_benchmark_result(
            "HIP w4a16 framework-like", hip_fw_ms, framework_actual_tokens, N, K
        )
        print_benchmark_result(
            "Torch w4a16 framework-like ref", torch_fw_ms, framework_actual_tokens, N, K
        )
        print(f"speedup(HIP/Torch): {torch_fw_ms / hip_fw_ms:.4f}x")
        print(
            f"correctness: {'PASS' if framework_correctness_pass else 'FAIL'} "
            f"(atol=1e-2, rtol=1e-2, max_abs_diff={framework_max_abs_diff:.6f})"
        )


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

    num_experts = 48
    num_tokens_lists = [16, 32]
    N_list = [4096, 4096]
    K_list = [7168, 4096]

    for i in range(len(num_tokens_lists)):

        # test_low_latency_ep(4)
        test_fp8_block_wise(num_experts, num_tokens_lists[i], N_list[i], K_list[i])
        # debug_nan()
