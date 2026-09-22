import os
from typing import Any, Dict, Optional, Tuple
import torch
from .print_size import PRINT_SIZE_ENABLED, print_low_latency_masked_size

from . import op

try:
    import triton
    from triton import language as tl
except Exception:  # Triton is only required by optional reference kernels.
    triton = None
    tl = None

def load_deep_gemm_library():
    lib_path = os.environ.get("DEEP_GEMM_LIB_PATH")

    if not lib_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checked = [os.path.join(current_dir, "libdeep_gemm.so")]
        for p in checked:
            if os.path.exists(p):
                lib_path = p
                break
    else:
        checked = [lib_path]

    if not lib_path or not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Could not find libdeep_gemm.so. Checked paths: {checked}"
        )

    torch.ops.load_library(lib_path)


def _ensure_torch_ops_loaded() -> None:
    try:
        import importlib

        importlib.import_module("deepgemm.op")
    except Exception:
        load_deep_gemm_library()


try:
    _ensure_torch_ops_loaded()
except Exception as e:
    print(f"Warning: Failed to load DeepGEMM native library: {e}")

_MASKED_LL_DEFAULT_CU_BY_DEVICE: Dict[int, int] = {}


def _get_default_masked_ll_cu() -> int:
    device = torch.cuda.current_device()
    cached = _MASKED_LL_DEFAULT_CU_BY_DEVICE.get(device)
    if cached is not None:
        return cached

    physical_cu = torch.cuda.get_device_properties(device).multi_processor_count
    if physical_cu == 64:
        cu = 256
    elif physical_cu == 72:
        cu = 288
    else:
        cu = 128

    _MASKED_LL_DEFAULT_CU_BY_DEVICE[device] = cu
    return cu


def _resolve_masked_ll_cu(cu: Optional[int]) -> int:
    return _get_default_masked_ll_cu() if cu is None else int(cu)



def m_grouped_gemm_fp8_fp8_bf16_nt_masked(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    experts: int,
    *,
    block_wise: bool = True,
    cu: Optional[int] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
) -> Tuple[int, int]:
    cu = _resolve_masked_ll_cu(cu)
    matrix_a, matrix_a_scale = lhs
    matrix_b, matrix_b_scale = rhs
    op.low_latency_grouped_gemm(
        matrix_a,
        matrix_b,
        matrix_a_scale,
        matrix_b_scale,
        masked_m,
        out,
        expected_m,
        experts,
        cu,
        block_wise,
        enable_overlap,
        signal,
    )
    block_m = 16
    bn = 256
    return block_m, bn


def m_grouped_w8a8_gemm_nt_masked_ll(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = None,
    *,
    block_wise: bool = False,
    cu: Optional[int] = None,
) -> torch.Tensor:
    """W8A8 entry point for low-latency masked GEMM.

    b[0] must be 6-D packed weights from pack_int8_weight_enk_to_w6_low_latency.
    The default scale layout is 2-D per-token / per-channel; set
    block_wise=True for block-wise scales. config is reserved.
    """
    cu = _resolve_masked_ll_cu(cu)

    experts = int(a[0].shape[0])
    if int(masked_m.numel()) != experts:
        raise ValueError(
            f"masked_m length {masked_m.numel()} != expert count from a[0] ({experts})"
        )
    if int(b[0].shape[0]) != experts:
        raise ValueError(
            f"b[0].shape[0] ({b[0].shape[0]}) != expert count ({experts})"
        )

    matrix_a, a_scale = a
    matrix_b, b_scale = b
    if not block_wise:
        if a_scale.dim() == 3:
            if a_scale.size(-1) != 1:
                raise ValueError(
                    "channel-wise a_scale must have shape [E,M] or [E,M,1], "
                    f"got {tuple(a_scale.shape)}"
                )
            a_scale = a_scale.squeeze(-1)
        if b_scale.dim() == 3:
            if b_scale.size(-1) != 1:
                raise ValueError(
                    "channel-wise b_scale must have shape [E,N] or [E,N,1], "
                    f"got {tuple(b_scale.shape)}"
                )
            b_scale = b_scale.squeeze(-1)
    if not a_scale.is_contiguous():
        raise ValueError("a_scale must be contiguous")
    if not b_scale.is_contiguous():
        raise ValueError("b_scale must be contiguous")
    if masked_m.dtype != torch.int32:
        raise ValueError(f"masked_m must be torch.int32, got {masked_m.dtype}")
    masked_m_i32 = masked_m

    expected_m_per_group = int(expected_m_per_group)
    if PRINT_SIZE_ENABLED:
        print_low_latency_masked_size(
            "low_latency_grouped_gemm",
            matrix_a,
            matrix_b,
            a_scale,
            b_scale,
            masked_m_i32,
            d,
            expected_m_per_group,
            experts,
            cu=cu,
            block_wise=block_wise,
            over_lap=False,
            signal=None,
        )

    op.low_latency_grouped_gemm(
        matrix_a,
        matrix_b,
        a_scale,
        b_scale,
        masked_m_i32,
        d,
        expected_m_per_group,
        experts,
        cu,
        block_wise,
        False,
        None,
    )
    return d


def m_grouped_fp8_gemm_nt_masked_ll(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    block_wise: bool = False,
    cu: Optional[int] = None,
) -> torch.Tensor:
    
    del config
    if block_wise:
        raise RuntimeError("m_grouped_fp8_gemm_nt_masked only supports channel-wise non-block-wise scales")
    cu = _resolve_masked_ll_cu(cu)

    matrix_a, a_scale = a
    matrix_b, b_scale = b

    if a_scale.dim() == 3:
        a_scale = a_scale.squeeze(-1)
    if b_scale.dim() == 3:
        b_scale = b_scale.squeeze(-1)

    experts = int(matrix_a.shape[0])
    expected_m_per_group = int(expected_m_per_group)
    if PRINT_SIZE_ENABLED:
        print_low_latency_masked_size(
            "low_latency_grouped_gemm",
            matrix_a,
            matrix_b,
            a_scale,
            b_scale,
            masked_m,
            d,
            expected_m_per_group,
            experts,
            cu=cu,
            block_wise=False,
            over_lap=enable_overlap,
            signal=signal,
        )

    op.low_latency_grouped_gemm(
        matrix_a,
        matrix_b,
        a_scale,
        b_scale,
        masked_m,
        d,
        expected_m_per_group,
        experts,
        cu,
        False,
        enable_overlap,
        signal,
    )
    return d



def m_grouped_fp8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    block_wise: bool = False,
    cu: Optional[int] = None,
) -> torch.Tensor:
    return m_grouped_fp8_gemm_nt_masked_ll(
        a,
        b,
        d,
        masked_m,
        expected_m_per_group,
        enable_overlap=enable_overlap,
        signal=signal,
        config=config,
        block_wise=block_wise,
        cu=cu,
    )


def m_grouped_fp8_gemm_nt_masked_ll_tensor_wise(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    cu: Optional[int] = None,
) -> torch.Tensor:
    del config
    cu = _resolve_masked_ll_cu(cu)

    experts = int(a[0].shape[0])
    if int(masked_m.numel()) != experts:
        raise ValueError(
            f"masked_m length {masked_m.numel()} != expert count from a[0] ({experts})"
        )
    if int(b[0].shape[0]) != experts:
        raise ValueError(
            f"b[0].shape[0] ({b[0].shape[0]}) != expert count ({experts})"
        )

    matrix_a, a_scale = a
    matrix_b, b_scale = b

    if a_scale.dim() == 3 and a_scale.size(-1) == 1:
        a_scale = a_scale.squeeze(-1)
    if b_scale.dim() == 2 and b_scale.size(1) == 1:
        pass
    elif b_scale.dim() == 1:
        pass
    else:
        raise ValueError(
            "tensor-wise B scale must have shape [E] or [E, 1], "
            f"got {tuple(b_scale.shape)}"
        )

    if not a_scale.is_contiguous():
        raise ValueError("a_scale must be contiguous")
    if not b_scale.is_contiguous():
        raise ValueError("b_scale must be contiguous")
    if masked_m.dtype != torch.int32:
        raise ValueError(f"masked_m must be torch.int32, got {masked_m.dtype}")
    masked_m_i32 = masked_m

    expected_m_per_group = int(expected_m_per_group)
    if PRINT_SIZE_ENABLED:
        print_low_latency_masked_size(
            "low_latency_grouped_gemm_tensor_wise",
            matrix_a,
            matrix_b,
            a_scale,
            b_scale,
            masked_m_i32,
            d,
            expected_m_per_group,
            experts,
            cu=cu,
            block_wise="tensor_wise",
            over_lap=enable_overlap,
            signal=signal,
        )

    op.low_latency_grouped_gemm_tensor_wise(
        matrix_a,
        matrix_b,
        a_scale,
        b_scale,
        masked_m_i32,
        d,
        expected_m_per_group,
        experts,
        cu,
        enable_overlap,
        signal,
    )
    return d


def grouped_gemm_fp8_fp8_bf16_nt_masked_tensor_wise_entry(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_a_scale: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    actual_tokens: torch.Tensor,
    matrix_c: torch.Tensor,
    max_tokens: int,
    experts: int,
    cu: int = 128,
    over_lap: bool = False,
    signal: Optional[torch.Tensor] = None,
):
    assert matrix_a_scale.is_contiguous()
    if PRINT_SIZE_ENABLED:
        print_low_latency_masked_size(
            "low_latency_grouped_gemm_tensor_wise",
            matrix_a,
            matrix_b,
            matrix_a_scale,
            matrix_b_scale,
            actual_tokens,
            matrix_c,
            max_tokens,
            experts,
            cu=cu,
            block_wise="tensor_wise",
            over_lap=over_lap,
            signal=signal,
        )
    # return torch.ops.deep_gemm.low_latency_grouped_gemm_tensor_wise(
    #     matrix_a,
    #     matrix_b,
    #     matrix_a_scale,
    #     matrix_b_scale,
    #     actual_tokens,
    #     matrix_c,
    #     max_tokens,
    #     experts,
    #     cu,
    #     over_lap,
    #     signal,
    # )
    return op.low_latency_grouped_gemm_tensor_wise(
        matrix_a,
        matrix_b,
        matrix_a_scale,
        matrix_b_scale,
        actual_tokens,
        matrix_c,
        max_tokens,
        experts,
        cu,
        over_lap,
        signal,
    )


def grouped_gemm_w4a16_nt_masked_entry(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    matrix_c: torch.Tensor,
    actual_tokens: torch.Tensor,
    max_tokens: int,
):
    experts = int(actual_tokens.numel())
    if int(matrix_b.shape[0]) != experts:
        raise ValueError(
            f"matrix_b.shape[0] ({matrix_b.shape[0]}) != actual_tokens.numel() ({experts})"
        )
    if int(matrix_b_scale.shape[0]) != experts:
        raise ValueError(
            f"matrix_b_scale.shape[0] ({matrix_b_scale.shape[0]}) != actual_tokens.numel() ({experts})"
        )
    if int(matrix_c.shape[0]) != experts:
        raise ValueError(
            f"matrix_c.shape[0] ({matrix_c.shape[0]}) != actual_tokens.numel() ({experts})"
        )
    if int(matrix_a.shape[0]) < experts:
        raise ValueError(
            f"matrix_a.shape[0] ({matrix_a.shape[0]}) must be >= actual_tokens.numel() ({experts})"
        )
    if PRINT_SIZE_ENABLED:
        print_low_latency_masked_size(
            "low_latency_w4a16_gemm",
            matrix_a,
            matrix_b,
            None,
            matrix_b_scale,
            actual_tokens,
            matrix_c,
            max_tokens,
            experts,
        )
    return torch.ops.deep_gemm.low_latency_w4a16_gemm(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens, experts
    )


def grouped_gemm_wfp4a16_nt_masked_entry(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    matrix_c: torch.Tensor,
    actual_tokens: torch.Tensor,
    max_tokens: int,
):
    """Grouped masked GEMM for packed FP4 E2M1 weights and fp16/bf16 activations.

    ``matrix_b`` uses the plain packed layout ``[E, N, K//2]`` with two FP4
    values per byte. ``matrix_b_scale`` is uint8 E8M0 scale data with shape
    ``[E, N, K//32]``.
    """
    experts = int(actual_tokens.numel())
    if int(matrix_b.shape[0]) != experts:
        raise ValueError(
            f"matrix_b.shape[0] ({matrix_b.shape[0]}) != actual_tokens.numel() ({experts})"
        )
    if int(matrix_b_scale.shape[0]) != experts:
        raise ValueError(
            f"matrix_b_scale.shape[0] ({matrix_b_scale.shape[0]}) != actual_tokens.numel() ({experts})"
        )
    if int(matrix_c.shape[0]) != experts:
        raise ValueError(
            f"matrix_c.shape[0] ({matrix_c.shape[0]}) != actual_tokens.numel() ({experts})"
        )
    if int(matrix_a.shape[0]) < experts:
        raise ValueError(
            f"matrix_a.shape[0] ({matrix_a.shape[0]}) must be >= actual_tokens.numel() ({experts})"
        )
    if PRINT_SIZE_ENABLED:
        print_low_latency_masked_size(
            "low_latency_wfp4a16_gemm",
            matrix_a,
            matrix_b,
            None,
            matrix_b_scale,
            actual_tokens,
            matrix_c,
            max_tokens,
            experts,
        )
    return torch.ops.deep_gemm.low_latency_wfp4a16_gemm(
        matrix_a, matrix_b, matrix_b_scale, actual_tokens, matrix_c, max_tokens, experts
    )


if triton is not None:
    @triton.jit
    def _moe_w4a16_groupwise_triton_kernel(
        matrix_a_ptr,
        matrix_b_ptr,
        matrix_c_ptr,
        matrix_bs_ptr,
        actual_m_ptr,
        experts: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        NUM_SM: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_WISE_K: tl.constexpr,
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

        # Plain scale layout: [E, N, K//GROUP_WISE_K].
        stride_bs_e = N * (K // GROUP_WISE_K)
        stride_bs_n = K // GROUP_WISE_K
        stride_bs_g = 1

        last_problem_end = 0
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)

        for e in range(experts):
            m = tl.load(actual_m_ptr + e)
            num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
            num_tiles = num_m_tiles * num_n_tiles

            a_e_ptr = matrix_a_ptr + e * stride_a_e
            b_e_ptr = matrix_b_ptr + e * stride_b_e
            c_e_ptr = matrix_c_ptr + e * stride_c_e
            bs_e_ptr = matrix_bs_ptr + e * stride_bs_e

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                local_tile = tile_idx - last_problem_end
                tile_m_idx = local_tile // num_n_tiles
                tile_n_idx = local_tile % num_n_tiles

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                mask_am = offs_am[:, None] < m
                mask_bn = offs_bn[None, :] < N
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for k_start in range(0, K, BLOCK_SIZE_K):
                    k_abs = k_start + offs_k
                    mask_k = k_abs < K

                    a_ptr = a_e_ptr + offs_am[:, None] * stride_a_m + k_abs[None, :] * stride_a_k
                    a = tl.load(a_ptr, mask_am & mask_k[None, :], 0.0)

                    b_u8_ptr = (
                        b_e_ptr
                        + offs_bn[None, :] * stride_b_n
                        + (k_abs[:, None] // 2) * stride_b_k2
                    )
                    b_u8 = tl.load(b_u8_ptr, mask_k[:, None] & mask_bn, 0)
                    b_q = ((b_u8 >> ((k_abs[:, None] & 1) * 4)) & 0xF).to(tl.float32)

                    scale_group = k_abs // GROUP_WISE_K
                    b_s_ptr = (
                        bs_e_ptr
                        + offs_bn[None, :] * stride_bs_n
                        + scale_group[:, None] * stride_bs_g
                    )
                    b_s = tl.load(b_s_ptr, mask_k[:, None] & mask_bn, 0.0).to(tl.float32)

                    # AWQ zero point is 8.
                    b = ((b_q - 8.0) * b_s).to(compute_dtype)
                    accumulator += tl.dot(a, b)

                c = accumulator.to(compute_dtype)
                offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptr = c_e_ptr + offs_cm[:, None] * stride_c_m + offs_cn[None, :] * stride_c_n
                tl.store(c_ptr, c, mask_am & mask_bn)

                tile_idx += NUM_SM

            last_problem_end += num_tiles


def _check_w4a16_triton_ref_inputs(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    matrix_c: torch.Tensor,
    actual_tokens: torch.Tensor,
    group_wise_k: int,
) -> tuple[int, int, int, int]:
    if triton is None:
        raise ImportError("Triton is required for moe_w4a16_groupwise_triton")
    if matrix_a.dim() != 3 or matrix_c.dim() != 3:
        raise ValueError(
            f"matrix_a and matrix_c must be 3D, got {matrix_a.dim()} and {matrix_c.dim()}"
        )
    if matrix_b.dim() != 3:
        raise ValueError(f"matrix_b must be raw packed [E,N,K//2], got {tuple(matrix_b.shape)}")
    if matrix_b_scale.dim() != 3:
        raise ValueError(f"matrix_b_scale must be plain [E,N,K//32], got {tuple(matrix_b_scale.shape)}")
    if actual_tokens.dim() != 1:
        raise ValueError(f"actual_tokens must be 1D, got {tuple(actual_tokens.shape)}")
    if actual_tokens.dtype != torch.int32:
        raise ValueError(f"actual_tokens must be torch.int32, got {actual_tokens.dtype}")
    if matrix_a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"matrix_a must be float16 or bfloat16, got {matrix_a.dtype}")
    if matrix_c.dtype != matrix_a.dtype:
        raise ValueError(f"matrix_c dtype ({matrix_c.dtype}) must match matrix_a dtype ({matrix_a.dtype})")
    if matrix_b_scale.dtype != matrix_a.dtype:
        raise ValueError(
            f"matrix_b_scale dtype ({matrix_b_scale.dtype}) must match matrix_a dtype ({matrix_a.dtype})"
        )
    if matrix_b.dtype != torch.uint8:
        raise ValueError(f"matrix_b must be raw packed torch.uint8 weights, got {matrix_b.dtype}")
    if not (
        matrix_a.is_contiguous()
        and matrix_b.is_contiguous()
        and matrix_b_scale.is_contiguous()
        and matrix_c.is_contiguous()
        and actual_tokens.is_contiguous()
    ):
        raise ValueError("all W4A16 Triton reference inputs must be contiguous")
    device = matrix_a.device
    if not (
        matrix_b.device == device
        and matrix_b_scale.device == device
        and matrix_c.device == device
        and actual_tokens.device == device
    ):
        raise ValueError("all W4A16 Triton reference tensors must be on the same device")

    experts = int(actual_tokens.numel())
    M = int(matrix_a.shape[1])
    K = int(matrix_a.shape[2])
    N = int(matrix_c.shape[2])
    if int(matrix_a.shape[0]) < experts:
        raise ValueError(f"matrix_a.shape[0] ({matrix_a.shape[0]}) must be >= experts ({experts})")
    if int(matrix_b.shape[0]) != experts:
        raise ValueError(f"matrix_b.shape[0] ({matrix_b.shape[0]}) must equal experts ({experts})")
    if int(matrix_b_scale.shape[0]) != experts:
        raise ValueError(
            f"matrix_b_scale.shape[0] ({matrix_b_scale.shape[0]}) must equal experts ({experts})"
        )
    if int(matrix_c.shape[0]) != experts:
        raise ValueError(f"matrix_c.shape[0] ({matrix_c.shape[0]}) must equal experts ({experts})")
    if int(matrix_c.shape[1]) != M:
        raise ValueError(f"matrix_c.shape[1] ({matrix_c.shape[1]}) must equal matrix_a.shape[1] ({M})")
    if K % 128 != 0 or N % 16 != 0 or K % group_wise_k != 0:
        raise ValueError(f"K={K}, N={N}, group_wise_k={group_wise_k} must satisfy K%128==0, N%16==0")
    expected_b_shape = (experts, N, K // 2)
    if tuple(matrix_b.shape) != expected_b_shape:
        raise ValueError(f"matrix_b shape must be {expected_b_shape}, got {tuple(matrix_b.shape)}")
    expected_scale_shape = (experts, N, K // group_wise_k)
    if tuple(matrix_b_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"matrix_b_scale shape must be {expected_scale_shape}, got {tuple(matrix_b_scale.shape)}"
        )
    return experts, M, N, K


def grouped_gemm_w4a16_nt_masked_triton_ref_entry(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    matrix_c: torch.Tensor,
    actual_tokens: torch.Tensor,
    *,
    num_sm: int = 64,
    block_size_m: int = 64,
    block_size_n: int = 128,
    block_size_k: int = 128,
    group_wise_k: int = 32,
) -> torch.Tensor:
    """Fill matrix_c with the Triton W4A16 grouped masked reference result."""
    experts, M, N, K = _check_w4a16_triton_ref_inputs(
        matrix_a, matrix_b, matrix_b_scale, matrix_c, actual_tokens, group_wise_k
    )
    _moe_w4a16_groupwise_triton_kernel[(num_sm,)](
        matrix_a,
        matrix_b,
        matrix_c,
        matrix_b_scale,
        actual_tokens,
        experts,
        M,
        N,
        K,
        num_sm,
        block_size_m,
        block_size_n,
        block_size_k,
        group_wise_k,
    )
    return matrix_c


def moe_w4a16_groupwise_triton(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    actual_tokens: torch.Tensor,
    matrix_c: Optional[torch.Tensor] = None,
    *,
    num_sm: int = 64,
    block_size_m: int = 64,
    block_size_n: int = 128,
    block_size_k: int = 128,
    group_wise_k: int = 32,
) -> torch.Tensor:
    """Triton golden/reference path for low-latency grouped W4A16 masked GEMM.

    Inputs use the raw packed W4 weight layout:
    matrix_a [E,M,K], matrix_b [E,N,K//2],
    matrix_b_scale [E,N,K//32], actual_tokens [E].
    """
    if matrix_c is None:
        experts = int(actual_tokens.numel())
        if matrix_b_scale.dim() != 3:
            raise ValueError(f"matrix_b_scale must be [E,N,K//32], got {tuple(matrix_b_scale.shape)}")
        matrix_c = torch.empty(
            (experts, int(matrix_a.shape[1]), int(matrix_b_scale.shape[1])),
            device=matrix_a.device,
            dtype=matrix_a.dtype,
        )
    return grouped_gemm_w4a16_nt_masked_triton_ref_entry(
        matrix_a,
        matrix_b,
        matrix_b_scale,
        matrix_c,
        actual_tokens,
        num_sm=num_sm,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        group_wise_k=group_wise_k,
    )


def w4a16_marlin_weight(weight_input: torch.Tensor) -> torch.Tensor:
    """Pack [E,N,K//2] uint8 W4 weights into the low-latency W4A16 layout."""
    if weight_input.dim() != 3:
        raise ValueError(
            f"weight_input must be [E,N,K//2], got dim {weight_input.dim()} shape {tuple(weight_input.shape)}"
        )
    if weight_input.dtype != torch.uint8:
        raise ValueError(f"weight_input must be torch.uint8, got {weight_input.dtype}")
    if not weight_input.is_contiguous():
        weight_input = weight_input.contiguous()

    e, n, packed_k = (int(v) for v in weight_input.shape)
    k = packed_k * 2
    if n % 16 != 0 or k % 128 != 0:
        raise ValueError(f"N={n}, K={k} must satisfy N%16==0 and K%128==0")

    qweight_u32 = weight_input.view(-1).view(torch.uint32)
    reshaped = qweight_u32.view(e, n // 16, 16, k // 32, 4)
    transposed = reshaped.transpose(2, 3).contiguous()
    marlin = transposed.view(e, n // 16, k // 128, 4, 16, 4)
    return marlin.transpose(1, 2).contiguous()

_CUSTOMER_N_TILE = 16
_CUSTOMER_K_TILE = 16
_CUSTOMER_K_PACK = _CUSTOMER_K_TILE * 4


def _validate_int8_weight_enk_for_w6(weight_enk: torch.Tensor) -> tuple[int, int, int]:
    if weight_enk.dim() != 3:
        raise ValueError(
            f"weight_enk must be [E,N,K], got dim {weight_enk.dim()} shape {tuple(weight_enk.shape)}"
        )
    num_experts = int(weight_enk.size(0))
    n = int(weight_enk.size(1))
    k = int(weight_enk.size(2))
    if n % _CUSTOMER_N_TILE != 0 or k % _CUSTOMER_K_PACK != 0:
        raise ValueError(
            f"N={n}, K={k} must satisfy N%{_CUSTOMER_N_TILE}==0 and K%{_CUSTOMER_K_PACK}==0"
        )
    return num_experts, n, k


def pack_int8_weight_enk_to_w6_low_latency(weight_enk: torch.Tensor) -> torch.Tensor:
    return pack_int8_weight_enk_to_w6_low_latency_inplace(weight_enk)


def pack_int8_weight_enk_to_w6_low_latency_inplace(weight_enk: torch.Tensor) -> torch.Tensor:
    num_experts, n, k = _validate_int8_weight_enk_for_w6(weight_enk)
    if not weight_enk.is_contiguous():
        raise ValueError("low-memory W6 pack requires contiguous [E,N,K] weight input")

    stride_n, stride_k = weight_enk.stride()[1:]
    w7_shape = (
        num_experts,
        k // _CUSTOMER_K_PACK,
        n // _CUSTOMER_N_TILE,
        1,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_N_TILE,
        _CUSTOMER_K_TILE,
    )
    for expert_idx in range(num_experts):
        weight = torch.as_strided(
            weight_enk[expert_idx],
            size=w7_shape[1:],
            stride=(
                _CUSTOMER_K_PACK * stride_k,
                _CUSTOMER_N_TILE * stride_n,
                0,
                _CUSTOMER_K_TILE * stride_k,
                stride_n,
                stride_k,
            ),
        ).contiguous()
        weight_enk[expert_idx].view(-1).copy_(weight.view(-1))

    return weight_enk.view(
        num_experts,
        k // _CUSTOMER_K_PACK,
        n // _CUSTOMER_N_TILE,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_N_TILE,
        _CUSTOMER_K_TILE,
    )



def m_grouped_fp8_gemm_nt_masked_tensor_wise(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    cu: Optional[int] = None,
) -> torch.Tensor:
    return m_grouped_fp8_gemm_nt_masked_ll_tensor_wise(
        a, b, d, masked_m, expected_m_per_group,
        enable_overlap=enable_overlap, signal=signal, config=config, cu=cu
    )
