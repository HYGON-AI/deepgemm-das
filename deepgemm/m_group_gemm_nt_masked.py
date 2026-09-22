import torch
from typing import Optional, Dict, Any, List, NamedTuple, Tuple
import functools
import os
from . import op
from .config_deepgemm import gemm_deepgemm_masked_config
from .config import get_deepgemm_groupgemm_w8a8_config
from .group_gemm_weight_pack import (
    BF16_MASKED_BM64_BN256_BK128_AUTO_MODE,
    BF16_MASKED_BM64_BN256_BK128_LAYOUT,
    BF16_MASKED_BM64_BN256_BK128_MODES,
    I8_LAYOUT_MODE_MASKED_LAYOUT_ASM,
    I8_LAYOUT_MODE_W6_HIPC,
    _get_current_gfx_arch,
    _is_gfx936_contiguous_arch,
    _resolve_i8_layout_mode_from_config,
    _resolve_shuffle_unique,
)
from .configs.w4a8_masked_hipc import select_w4a8_hipc_masked_mode


_MASKED_MARLIN2_INTERNAL_MODE = 1002

_BF16_MASKED_BM64_BN256_BK128_MODE = BF16_MASKED_BM64_BN256_BK128_AUTO_MODE
_BF16_MASKED_BM64_BN256_BK128_MODES = set(BF16_MASKED_BM64_BN256_BK128_MODES)


def _select_w4a8_hipc_masked_mode(expected_m: int, n: int, k: int) -> int:
    return select_w4a8_hipc_masked_mode(expected_m, n, k)


def round_up_to_next_power_of_2(x):
    if x == 0:
        return 1  # 特殊情况：0 的向上取整是 1
    x -= 1  # 处理 x 已经是 2 的幂的情况
    # 填充所有低位为 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1



@functools.lru_cache(maxsize=128)
def deepgemm_masked_config(m, size_k=32):

    # # cuda选择 目前只支持7168和2048和gfx938
    if size_k == 7168 and m <= 12:
        return 41
    elif size_k == 2048 and m <= 12:
        return 33

    if m < 16:
        adapt_m = 16  # 向上取整到下一个16的倍数
    else:
        adapt_m = round_up_to_next_power_of_2(m)
    key = f"EXPECTED_M{adapt_m}"
    if key in gemm_deepgemm_masked_config.get_deepgemm_dict():
        MODE = gemm_deepgemm_masked_config.get_deepgemm_dict()[key]
    else:
        MODE = 1000
    return MODE

@functools.lru_cache(maxsize=128)
def deepgemm_masked_config_i8fp8(m, size_k=32):

    # # cuda选择 目前只支持7168和2048和gfx938
    # marlin layout 改变了
    # if size_k == 7168 and m <= 12:
    #     return 184 #248 41 
    # elif size_k == 2048 and m <= 17:
    #     return 170

    if m < 16:
        adapt_m = 16  # 向上取整到下一个16的倍数
    else:
        adapt_m = round_up_to_next_power_of_2(m)
    key = f"EXPECTED_M{adapt_m}"
    if key in gemm_deepgemm_masked_config.get_deepgemm_dict():
        MODE = gemm_deepgemm_masked_config.get_deepgemm_dict()[key]
    else:
        MODE = 1002
    return MODE

# 读取json文件的方式 待完善
@functools.lru_cache
def get_groupgemm_w8a8_mode(expected_m: int, n: int, k: int) -> int:
    """
    获取 groupgemm w8a8 kernel 的 mode
    
    优先从 JSON 配置文件读取，如果没有则使用默认配置
    """
    device_props = torch.cuda.get_device_properties(0)
    device_name = device_props.gcnArchName.split(':')[0] if hasattr(device_props, 'gcnArchName') else device_props.name
    num_cus = device_props.multi_processor_count
    E = 1  # E 不影响 config 选择
    
    config, status = get_deepgemm_groupgemm_w8a8_config(
        E=E, M=expected_m, N=n, K=k,
        device_name=device_name, num_cus=num_cus)
    
    if status and 'MODE' in config:
        return config['MODE']
    
    # 回退到旧的 config 方式
    return deepgemm_masked_config(expected_m)


def m_grouped_w4a8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1002,}) -> torch.Tensor:

    mode = config['MODE']
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w4a8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output


def m_grouped_w4a8_gemm_nt_masked_hipc(a: tuple[torch.Tensor, torch.Tensor],
                                       b: tuple[torch.Tensor, torch.Tensor],
                                       d: torch.Tensor,
                                       masked_m: torch.Tensor,
                                       expected_m_per_group: int,
                                       config: Optional[Dict[str, Any]] = None) -> torch.Tensor:

    if config is not None and 'MODE' in config:
        mode = config['MODE']
    else:
        mode = _select_w4a8_hipc_masked_mode(
            expected_m_per_group, d.shape[2], a[0].shape[2]
        )
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w4a8_gemm_nt_masked_hipc(input, b_qweight, output, a_scale, b_scale,
                                             masked_m, expected_m_per_group, mode)
    return output



def m_grouped_w8a8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1002,},
                                  *,
                                  enable_overlap: bool = False,
                                  signal: Optional[torch.Tensor] = None) -> torch.Tensor:
    if enable_overlap:
        raise RuntimeError("m_grouped_w8a8_gemm_nt_masked overlap mode has been removed")
    if signal is not None:
        raise RuntimeError("signal is only valid for the removed m_grouped_w8a8_gemm_nt_masked overlap mode")
    if config is not None and "MODE" in config:
        mode = config["MODE"]
    else:
        size_k = a[0].shape[2]
        mode = deepgemm_masked_config_i8fp8(expected_m_per_group, size_k)  # topk*(bs*ep16) / 256  -> bs / 2
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w8a8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output

def m_grouped_w8a8_gemm_nt_masked_impl(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  mode) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w8a8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output



def m_grouped_w8a8_gemm_nt_compact_masked(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expert_offsets: torch.Tensor,
    expected_m_per_group: int,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    return op.m_grouped_marlin_w8a8_gemm_nt_compact_masked(
        input,
        b_qweight,
        d,
        a_scale,
        b_scale,
        masked_m,
        expert_offsets,
        expected_m_per_group,
    )


def m_grouped_i8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = None,
    *,
    shuffle_unique: int = 1,
    block_wise: bool = False,
    cu: Optional[int] = None,
) -> torch.Tensor:
    from .m_group_gemm import m_grouped_w8a8_gemm_nt_masked_ll

    arch = _get_current_gfx_arch()
    if masked_m.dtype != torch.int32:
        raise ValueError(f"masked_m must be torch.int32, got {masked_m.dtype}")

    if arch == "gfx938":
        return m_grouped_w8a8_gemm_nt_masked_ll(
            a,
            b,
            d,
            masked_m,
            expected_m_per_group,
            config,
            block_wise=block_wise,
            cu=cu,
        )

    if not _is_gfx936_contiguous_arch(arch):
        raise RuntimeError(
            f"m_grouped_i8_gemm_nt_masked does not support GPU arch {arch}; "
            "expected gfx936/gfx92a/gfx928 or gfx938"
        )

    shuffle_unique = _resolve_shuffle_unique(shuffle_unique)
    if shuffle_unique != 1:
        if arch == "gfx92a":
            raise AssertionError(
                f"m_grouped_i8_gemm_nt_masked does not support shuffle_unique={shuffle_unique} on gfx92a"
            )
        return _i8_masked_layout_asm_impl(
            a, b, d, masked_m, expected_m_per_group, config
        )

    mode = _resolve_i8_layout_mode_from_config(config)
    if mode == I8_LAYOUT_MODE_W6_HIPC:
        return _i8_w6_hipc_layout_impl(
            a,
            b,
            d,
            masked_m,
            expected_m_per_group,
            block_wise=block_wise,
            cu=128 if cu is None else cu,
        )

    if mode == I8_LAYOUT_MODE_MASKED_LAYOUT_ASM:
        if arch != "gfx936":
            raise AssertionError(
                "m_grouped_i8_gemm_nt_masked mode=2 is only implemented for gfx936"
            )
        return _i8_masked_layout_asm_impl(
            a,
            b,
            d,
            masked_m,
            expected_m_per_group,
            {"MODE": _MASKED_MARLIN2_INTERNAL_MODE},
        )

    return _i8_masked_layout_asm_impl(
        a, b, d, masked_m, expected_m_per_group, config
    )



def _i8_masked_layout_asm_impl(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    b_qweight, _ = b
    if b_qweight.dim() != 3:
        raise ValueError("non-unified masked int8 weight expects packed weight [E,N/16,K*16]")
    return m_grouped_w8a8_gemm_nt_masked(
        a,
        b,
        d,
        masked_m,
        expected_m_per_group,
        {"MODE": _MASKED_MARLIN2_INTERNAL_MODE} if config is None else config,
    )


def _i8_w6_hipc_layout_impl(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    *,
    block_wise: bool = False,
    cu: int = 128,
) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    if masked_m.dtype != torch.int32:
        raise ValueError(f"masked_m must be torch.int32, got {masked_m.dtype}")
    if b_qweight.dim() != 6:
        raise ValueError("unified masked int8 weight expects packed weight [E,K/64,N/16,4,16,16]")

    if not block_wise:
        if a_scale.dim() == 3:
            if a_scale.size(-1) != 1:
                raise ValueError(f"a_scale must be [E,M] or [E,M,1], got {tuple(a_scale.shape)}")
            a_scale = a_scale.squeeze(-1)
        if b_scale.dim() == 3:
            if b_scale.size(-1) != 1:
                raise ValueError(f"b_scale must be [E,N] or [E,N,1], got {tuple(b_scale.shape)}")
            b_scale = b_scale.squeeze(-1)
    torch.ops.deep_gemm.m_grouped_w8a8_gemm_nt_masked_gfx936(
        input,
        b_qweight,
        a_scale.contiguous(),
        b_scale.contiguous(),
        masked_m.contiguous(),
        d,
        int(expected_m_per_group),
        int(input.size(0)),
        int(cu),
        bool(block_wise),
        False,
        None,
    )
    return d

def m_grouped_w8a8_gemm_nt_masked_gfx936(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    *,
    ifb: Optional[bool] = None,
    block_wise: bool = False,
    cu: int = 128,
) -> torch.Tensor:
    return m_grouped_i8_gemm_nt_masked(
        a,
        b,
        d,
        masked_m,
        expected_m_per_group,
        shuffle_unique=1 if ifb is True else 0,
        block_wise=block_wise,
        cu=cu,
    )


# Backward-compatible alias for earlier gfx936/gfx92a experiments.
m_grouped_w8a8_gemm_nt_masked_gfx936_pack_permute = m_grouped_w8a8_gemm_nt_masked_gfx936

def m_grouped_bf16_gemm_nt_masked(a: torch.Tensor,
                                  b: torch.Tensor,
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = None) -> torch.Tensor:

    config = config or {}
    mode = int(config.get("MODE", _BF16_MASKED_BM64_BN256_BK128_MODE))
    if mode in _BF16_MASKED_BM64_BN256_BK128_MODES:
        if config.get("WEIGHT_LAYOUT") != BF16_MASKED_BM64_BN256_BK128_LAYOUT or not config.get("PACKED_WEIGHT", False):
            raise ValueError(
                "BF16 masked BM64/BN256/BK128 kernels require pre-packed weights. "
                "Use pack_bf16_masked_weight(weight, mode) and pass the returned config."
            )
    op.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, mode)
    return d   

def m_grouped_bf16_gemm_nt_masked_hipc(a: torch.Tensor,
                                  b: torch.Tensor,
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:

    mode = config['MODE']
    op.m_grouped_bf16_gemm_nt_masked_hipc(a, b, d, masked_m, expected_m_per_group, mode)
    return d  
