from __future__ import annotations

import torch

from .print_size import PRINT_SIZE_ENABLED, print_moe_prefill_size

__all__ = [
    "moe_w8a8_marlin_prefill_down_fp8",
    "assert_moe_prefill_down_fp8_shapes",
]

_MOE_N_ALIGN = 512  # BLOCK_SIZE_N * n_loop_num == 128 * 4


def moe_w8a8_marlin_prefill_down_fp8(
    input: torch.Tensor,
    b_qweight: torch.Tensor,
    output: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    top_k: int,
    real_topk: int,
) -> torch.Tensor:
    """MoE Marlin GEMM2 prefill-down FP8 (fp8 activations × fp8 Marlin weights → bf16)."""
    if PRINT_SIZE_ENABLED:
        print_moe_prefill_size(
            "moe_w8a8_marlin_prefill_down_fp8",
            input,
            b_qweight,
            output,
            a_scale,
            b_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            top_k,
            real_topk,
        )
    return torch.ops.deep_gemm.moe_w8a8_marlin_prefill_down_fp8(
        input,
        b_qweight,
        output,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        top_k,
        real_topk,
    )


def assert_moe_prefill_down_fp8_shapes(*, k: int, n: int) -> None:
    if k % 64 != 0:
        raise ValueError(f"K={k} must be a multiple of 64 for prefill-down FP8")
    if n % _MOE_N_ALIGN != 0:
        raise ValueError(
            f"N={n} must be divisible by {_MOE_N_ALIGN} for tile <32,128,64,32,32,64,2>"
        )

