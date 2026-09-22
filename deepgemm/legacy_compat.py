import torch

from .group_gemm_weight_pack import (
    marlin_fp8_contiguous_weight,
    marlin_fp8_masked_weight,
    marlin_i8_contiguous_weight,
    marlin_i8_masked_weight,
    pack_i8_contiguous_weight_gfx936_asm_inplace,
    pack_i8_masked_weight_gfx936_asm_inplace,
    pack_fp8_contiguous_weight,
    pack_fp8_masked_weight,
    pack_i8_contiguous_weight,
    pack_i8_masked_weight,
)
from .m_group_gemm import (
    m_grouped_fp8_gemm_nt_masked_ll,
    m_grouped_fp8_gemm_nt_masked_ll_tensor_wise,
    m_grouped_w8a8_gemm_nt_masked_ll,
    pack_int8_weight_enk_to_w6_low_latency,
)
from .m_group_gemm_nt_masked import (
    m_grouped_w8a8_gemm_nt_masked,
    m_grouped_w8a8_gemm_nt_masked_gfx936,
    m_grouped_w8a8_gemm_nt_masked_gfx936_pack_permute,
    m_grouped_w8a8_gemm_nt_masked_impl,
)


def weight8bit_nt_kpack2_marlin(
    weight: torch.Tensor,
    k_tile: int = 16,
    n_tile: int = 16,
) -> torch.Tensor:
    return pack_i8_contiguous_weight_gfx936_asm_inplace(
        weight, k_tile=k_tile, n_tile=n_tile
    )


def weight8bit_nt_kpack2_marlin2(
    weight: torch.Tensor,
    k_tile: int = 16,
    k_tile1: int = 4,
    n_tile: int = 16,
) -> torch.Tensor:
    return pack_i8_masked_weight_gfx936_asm_inplace(
        weight, k_tile=k_tile, k_tile1=k_tile1, n_tile=n_tile
    )


__all__ = [
    "marlin_fp8_contiguous_weight",
    "marlin_fp8_masked_weight",
    "marlin_i8_contiguous_weight",
    "marlin_i8_masked_weight",
    "pack_fp8_contiguous_weight",
    "pack_fp8_masked_weight",
    "pack_i8_contiguous_weight",
    "pack_i8_masked_weight",
    "pack_int8_weight_enk_to_w6_low_latency",
    "m_grouped_fp8_gemm_nt_masked_ll",
    "m_grouped_fp8_gemm_nt_masked_ll_tensor_wise",
    "m_grouped_w8a8_gemm_nt_masked",
    "m_grouped_w8a8_gemm_nt_masked_gfx936",
    "m_grouped_w8a8_gemm_nt_masked_gfx936_pack_permute",
    "m_grouped_w8a8_gemm_nt_masked_impl",
    "m_grouped_w8a8_gemm_nt_masked_ll",
    "weight8bit_nt_kpack2_marlin",
    "weight8bit_nt_kpack2_marlin2",
]
