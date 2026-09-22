import torch
from typing import Optional, Dict, Any
from . import op
from .print_size import PRINT_SIZE_ENABLED, print_contiguous_size
from .group_gemm_weight_pack import (
    I8_LAYOUT_MODE_MASKED_LAYOUT_ASM,
    I8_LAYOUT_MODE_W6_HIPC,
    _get_current_gfx_arch,
    _is_gfx936_contiguous_arch,
    _resolve_i8_layout_mode_from_config,
    _resolve_shuffle_unique,
    _strip_i8_layout_mode_from_config,
)
from .configs.contiguous import (
    DEFAULT_FP8_CONTIG_CONFIG,
    get_m_grouped_fp8_contiguous_config,
    resolve_contiguous_hipc_config,
    resolve_fp8_contiguous_block_m,
    resolve_fp8_contiguous_n_loop_num,
    select_fp8_contig_hipc_config,
)


# ---------------------------------------------------------------------------
# INT8 contiguous grouped GEMM
# ---------------------------------------------------------------------------

def _normalize_contiguous_w4a8_a_scale(a_scale: torch.Tensor) -> torch.Tensor:
    if a_scale.dim() == 2:
        if a_scale.size(1) != 1:
            raise ValueError(
                "contiguous W4A8 a_scale must be [M] or [M,1], "
                f"got {tuple(a_scale.shape)}"
            )
        return a_scale
    if a_scale.dim() != 1:
        raise ValueError(
            "contiguous W4A8 a_scale must be [M] or [M,1], "
            f"got {tuple(a_scale.shape)}"
        )
    return a_scale


def _normalize_contiguous_w4a8_b_scale(b_scale: torch.Tensor) -> torch.Tensor:
    if b_scale.dim() == 3:
        if b_scale.size(-1) != 1:
            raise ValueError(
                "contiguous W4A8 b_scale must be [E,N] or [E,N,1], "
                f"got {tuple(b_scale.shape)}"
            )
        return b_scale
    if b_scale.dim() != 2:
        raise ValueError(
            "contiguous W4A8 b_scale must be [E,N] or [E,N,1], "
            f"got {tuple(b_scale.shape)}"
        )
    return b_scale


def m_grouped_w4a8_gemm_nt_contiguous_hipc(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    mode = int(config.get("MODE", 1002)) if config is not None else 1002
    a_scale = _normalize_contiguous_w4a8_a_scale(a_scale)
    b_scale = _normalize_contiguous_w4a8_b_scale(b_scale)
    if input.dim() != 2:
        raise ValueError(f"contiguous W4A8 input must be [M,K], got {tuple(input.shape)}")
    if output.dim() != 2:
        raise ValueError(f"contiguous W4A8 output must be [M,N], got {tuple(output.shape)}")
    if b_qweight.dim() != 3:
        raise ValueError(
            "contiguous W4A8 packed weight must be [E,N/32,K*16] "
            "from pack_w4a8_moe_hipc_weight"
        )
    if output.size(0) != input.size(0):
        raise ValueError("contiguous W4A8 output M must match input M")
    if m_indices.numel() != input.size(0):
        raise ValueError("contiguous W4A8 m_indices length must match input M")
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    if PRINT_SIZE_ENABLED:
        print_contiguous_size(
            "m_grouped_w4a8_gemm_nt_contiguous_hipc",
            input,
            b_qweight,
            output,
            a_scale,
            b_scale,
            m_indices,
            mode,
            block_m=64,
            block_n=128,
            n_loop_num=4,
        )
    return op.m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
        input,
        b_qweight,
        output,
        a_scale.contiguous(),
        b_scale.contiguous(),
        m_indices.contiguous(),
        mode,
    )

def _normalize_contiguous_i8_b_scale(b_scale: torch.Tensor) -> torch.Tensor:
    if b_scale.dim() == 3:
        if b_scale.size(-1) != 1:
            raise ValueError(
                "contiguous int8 b_scale must be [E,N] or [E,N,1], "
                f"got {tuple(b_scale.shape)}"
            )
        b_scale = b_scale.squeeze(-1)
    if b_scale.dim() != 2:
        raise ValueError(
            "contiguous int8 b_scale must be [E,N] or [E,N,1], "
            f"got {tuple(b_scale.shape)}"
        )
    return b_scale

def _m_grouped_i8_gemm_nt_contiguous_gfx938_impl(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    b_scale = _normalize_contiguous_i8_b_scale(b_scale)
    block_m, block_n, n_loop_num = resolve_contiguous_hipc_config(config)
    if b_qweight.dim() != 6:
        raise ValueError(
            "m_grouped_i8_gemm_nt_contiguous expects W6 packed weight "
            "[E,K/64,N/16,4,16,16] from marlin_i8_contiguous_weight"
        )
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    if PRINT_SIZE_ENABLED:
        print_contiguous_size(
            "m_grouped_i8_gemm_nt_contiguous",
            input,
            b_qweight,
            output,
            a_scale,
            b_scale,
            m_indices,
            block_m,
            block_m=block_m,
            block_n=block_n,
            n_loop_num=n_loop_num,
        )
    torch.ops.deep_gemm.m_grouped_i8_gemm_nt_contiguous(
        input,
        b_qweight,
        output,
        a_scale.contiguous(),
        b_scale.contiguous(),
        m_indices.contiguous(),
        block_m,
        block_n,
        n_loop_num,
    )
    return output


def m_grouped_i8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
    *,
    shuffle_unique: int = 1,
) -> torch.Tensor:
    arch = _get_current_gfx_arch()
    shuffle_unique = _resolve_shuffle_unique(shuffle_unique)
    if _is_gfx936_contiguous_arch(arch):
        b_qweight = b[0]
        if shuffle_unique == 1:
            mode = _resolve_i8_layout_mode_from_config(config)
            if mode == I8_LAYOUT_MODE_W6_HIPC:
                if b_qweight.dim() != 6:
                    raise ValueError(
                        "gfx936 mode=1 expects W6 packed weight "
                        "[E,K/64,N/16,4,16,16]"
                    )
                hipc_config = _strip_i8_layout_mode_from_config(config)
                return _m_grouped_i8_gemm_nt_contiguous_gfx936_hipc_impl(
                    a, b, output, m_indices, hipc_config
                )
            if mode == I8_LAYOUT_MODE_MASKED_LAYOUT_ASM:
                if arch != "gfx936":
                    raise AssertionError(
                        "m_grouped_i8_gemm_nt_contiguous mode=2 is only implemented for gfx936"
                    )
                return _m_grouped_i8_gemm_nt_contiguous_gfx936_masked_layout_impl(
                    a, b, output, m_indices
                )
        if b_qweight.dim() == 3:
            return _m_grouped_i8_gemm_nt_contiguous_gfx936_asm_impl(a, b, output, m_indices, config)
        if b_qweight.dim() == 6:
            if arch == "gfx92a":
                raise AssertionError(
                    f"m_grouped_i8_gemm_nt_contiguous does not support shuffle_unique={shuffle_unique} on gfx92a"
                )
            return _m_grouped_i8_gemm_nt_contiguous_gfx936_asm_impl(a, b, output, m_indices, config)
        raise ValueError(
            "m_grouped_i8_gemm_nt_contiguous expects packed weight "
            "[E,N/16,K*16] or [E,K/64,N/16,4,16,16]"
        )
    if arch == "gfx938":
        return _m_grouped_i8_gemm_nt_contiguous_gfx938_impl(a, b, output, m_indices, config)
    raise RuntimeError(
        f"m_grouped_i8_gemm_nt_contiguous does not support GPU arch {arch}; "
        "expected gfx936/gfx92a/gfx928 or gfx938"
    )


def _m_grouped_i8_gemm_nt_contiguous_gfx936_asm_impl(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    b_scale = _normalize_contiguous_i8_b_scale(b_scale)
    mode = int(config.get("MODE", 1000)) if config is not None else 1000
    # Temporary gemm2 fast path: keep default behavior except K=2048,N=4096.
    if mode == 1000 and input.size(1) == 2048 and b_scale.size(1) == 4096:
        mode = 2000
    if b_qweight.dim() != 3:
        raise ValueError("ASM path expects packed weight [E,N/16,K*16]")
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    return op.m_grouped_w8a8_gemm_nt_contiguous(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale,
        m_indices,
        mode,
    )


def _m_grouped_i8_gemm_nt_contiguous_gfx936_masked_layout_impl(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    b_scale = _normalize_contiguous_i8_b_scale(b_scale)
    if b_qweight.dim() != 3:
        raise ValueError(
            "gfx936 mode=2 expects masked/marlin2 packed weight [E,N/16,K*16]"
        )
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    return op.m_grouped_w8a8_gemm_nt_contiguous(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale,
        m_indices,
        I8_LAYOUT_MODE_MASKED_LAYOUT_ASM,
    )


def _m_grouped_i8_gemm_nt_contiguous_gfx936_hipc_impl(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    b_scale = _normalize_contiguous_i8_b_scale(b_scale)
    block_m, block_n, n_loop_num = resolve_contiguous_hipc_config(config)
    if b_qweight.dim() != 6:
        raise ValueError(
            "HIPC path expects packed weight [E,K/64,N/16,4,16,16]"
        )
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    torch.ops.deep_gemm.m_grouped_w8a8_gemm_nt_contiguous_gfx936(
        input,
        b_qweight,
        output,
        a_scale.contiguous(),
        b_scale.contiguous(),
        m_indices.contiguous(),
        block_m,
        block_n,
        n_loop_num,
    )
    return output


# Legacy contiguous INT8 no-padding entry point.
def m_grouped_i8_gemm_nt_contiguous_nopad(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    token_per_expert: torch.Tensor,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    return op.m_grouped_w8a8_gemm_nt_nopad_contiguous(
        input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert
    )


# ---------------------------------------------------------------------------
# FP8 contiguous grouped GEMM
# ---------------------------------------------------------------------------

def m_grouped_fp8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    if config is None:
        block_m, block_n, n_loop_num = select_fp8_contig_hipc_config(
            int(input.size(0)),
            size_n=int(output.size(-1)),
            size_k=int(input.size(-1)),
            num_groups=int(b_qweight.size(0)),
        )
    else:
        block_m = int(config.get("BLOCK_M", config.get("MODE", 128)))
        block_n = int(config.get("BLOCK_N", 256))
        n_loop_num = int(config.get("N_LOOP_NUM", 4))
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    if PRINT_SIZE_ENABLED:
        print_contiguous_size(
            "m_grouped_fp8_gemm_nt_contiguous",
            input,
            b_qweight,
            output,
            a_scale,
            b_scale,
            m_indices,
            block_m,
            block_m=block_m,
            block_n=block_n,
            n_loop_num=n_loop_num,
        )
    torch.ops.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        input,
        b_qweight,
        output,
        a_scale.contiguous(),
        b_scale.contiguous(),
        m_indices.contiguous(),
        block_m,
        block_n,
        n_loop_num,
    )
    return output


def m_grouped_fp8_gemm_nt_contiguous_tensor_wise(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    size_m = int(input.size(0))
    size_n = int(output.size(-1))
    if config is None:
        config = get_m_grouped_fp8_contiguous_config(size_m, size_n)
    mode = resolve_fp8_contiguous_block_m(size_m, config.get("MODE", DEFAULT_FP8_CONTIG_CONFIG["MODE"]))
    requested_n_loop = int(config.get("N_LOOP_NUM", DEFAULT_FP8_CONTIG_CONFIG["N_LOOP_NUM"]))
    n_loop_num = resolve_fp8_contiguous_n_loop_num(size_n, requested_n_loop)
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    if not (b_scale.dim() == 1 or (b_scale.dim() == 2 and b_scale.size(1) == 1)):
        raise ValueError(
            "tensor-wise contiguous B scale must have shape [E] or [E, 1], "
            f"got {tuple(b_scale.shape)}"
        )
    if PRINT_SIZE_ENABLED:
        print_contiguous_size(
            "m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise",
            input,
            b_qweight,
            output,
            a_scale,
            b_scale,
            m_indices,
            mode,
            block_m=mode,
            n_loop_num=n_loop_num,
        )
    torch.ops.deep_gemm.m_grouped_fp8_gemm_nt_contiguous_marlin_tensor_wise(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale.contiguous(),
        m_indices,
        mode,
        n_loop_num,
    )
    return output


# ---------------------------------------------------------------------------
# WFP4A16 contiguous grouped GEMM
# ---------------------------------------------------------------------------

def m_grouped_wfp4a16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:
    """Grouped contiguous GEMM for packed FP4 E2M1 weights.

    ``a`` is contiguous activation data shaped ``[M, K]``. ``b`` is
    ``(matrix_b, matrix_b_scale)`` where ``matrix_b`` is plain packed
    ``[E, N, K//2]`` and ``matrix_b_scale`` is uint8 E8M0 scale data shaped
    ``[E, N, K//32]``. ``m_indices[row]`` selects the expert for each A row;
    ``-1`` marks padding rows.
    """
    matrix_b, matrix_b_scale = b
    if matrix_b.dim() != 3:
        raise ValueError(
            f"matrix_b must be [E,N,K//2], got shape {tuple(matrix_b.shape)}"
        )
    if matrix_b_scale.dim() != 3:
        raise ValueError(
            f"matrix_b_scale must be [E,N,K//32], got shape {tuple(matrix_b_scale.shape)}"
        )
    if output.dim() != 2:
        raise ValueError(f"output must be [M,N], got shape {tuple(output.shape)}")
    if a.dim() != 2:
        raise ValueError(f"a must be [M,K], got shape {tuple(a.shape)}")
    if output.shape[0] != a.shape[0]:
        raise ValueError(f"output.shape[0] ({output.shape[0]}) must match a.shape[0] ({a.shape[0]})")
    if output.shape[1] != matrix_b.shape[1]:
        raise ValueError(f"output.shape[1] ({output.shape[1]}) must match matrix_b.shape[1] ({matrix_b.shape[1]})")
    if matrix_b_scale.shape[0] != matrix_b.shape[0]:
        raise ValueError("matrix_b_scale.shape[0] must match matrix_b.shape[0]")
    if matrix_b_scale.shape[1] != matrix_b.shape[1]:
        raise ValueError("matrix_b_scale.shape[1] must match matrix_b.shape[1]")
    if matrix_b.shape[2] != a.shape[1] // 2:
        raise ValueError("matrix_b.shape[2] must be K//2")
    if matrix_b_scale.shape[2] != a.shape[1] // 32:
        raise ValueError("matrix_b_scale.shape[2] must be K//32")
    if m_indices.numel() != a.shape[0]:
        raise ValueError(f"m_indices length ({m_indices.numel()}) must match M ({a.shape[0]})")
    if m_indices.dtype != torch.int32:
        m_indices = m_indices.to(torch.int32)
    if PRINT_SIZE_ENABLED:
        print_contiguous_size(
            "m_grouped_wfp4a16_gemm_nt_contiguous",
            a,
            matrix_b,
            output,
            None,
            matrix_b_scale,
            m_indices,
            514,
            block_m=128,
            block_n=128,
            n_loop_num=1,
        )
    torch.ops.deep_gemm.m_grouped_wfp4a16_gemm_nt_contiguous(
        a,
        matrix_b,
        output,
        matrix_b_scale,
        m_indices.contiguous(),
    )
    return output


# ---------------------------------------------------------------------------
# BF16 contiguous grouped GEMM
# ---------------------------------------------------------------------------

def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = {"MODE": 1000},
) -> torch.Tensor:
    mode = config["MODE"]
    M = a.shape[0]
    E, N = b.shape[0], b.shape[1]
    a_scale = torch.ones(M, dtype=torch.float32, device=a.device)
    b_scale = torch.ones((E, N), dtype=torch.float32, device=b.device)
    return op.m_grouped_w16a16_gemm_nt_contiguous(
        a, b, output, a_scale, b_scale, m_indices, mode
    )
