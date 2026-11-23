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
    _resolve_shuffle_unique,
)
from .configs.w4a8_masked_hipc import (
    DEFAULT_W4A8_HIPC_MASKED_MODE,
    select_w4a8_hipc_masked_config,
    select_w4a8_hipc_masked_mode,
)
from .configs.bf16_masked import select_bf16_masked_mode

from .configs.w8a8_masked import select_w8a8_masked_mode


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
                                       config: Optional[Dict[str, Any]] = None,
                                       *,
                                       enable_overlap: bool = False,
                                       signal: Optional[torch.Tensor] = None) -> torch.Tensor:

    if config is not None and 'MODE' in config:
        selected_config = config
    else:
        selected_config = select_w4a8_hipc_masked_config(
            expected_m_per_group, d.shape[2], a[0].shape[2], a[0].shape[0]
        )
    mode = int(selected_config.get('MODE', DEFAULT_W4A8_HIPC_MASKED_MODE))
    persistent_multiplier = int(selected_config.get('PERSISTENT_MULTIPLIER', 0))
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    # 返回 (block_m, ceil_div(N, block_n))：signal 形状为 E * ceil_div(M / 64)，
    # 槽位按 64 行粒度组织。block_m<64 时一个槽位聚合 64/block_m 个 m tile，
    # 槽位计数值达到 (64/block_m) * ceil_div(N, block_n) 即表示专家 e 第 m64 个
    # 64 行带全部写回完成（valid_m 不足 64 行的末带由内核补齐差额）。
    return op.m_grouped_marlin_w4a8_gemm_nt_masked_hipc(input, b_qweight, output, a_scale, b_scale,
                                             masked_m, expected_m_per_group, mode,
                                             persistent_multiplier,
                                             enable_overlap, signal)



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

    if config is not None and 'MODE' in config:
        mode = config['MODE']
    else:
        mode = select_w8a8_masked_mode(
            expected_m_per_group, d.shape[2], a[0].shape[2], a[0].shape[0], "asm"
        )

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

def m_grouped_int8_gemm_nt_masked_pair32(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = None
) -> torch.Tensor:

    experts = int(a[0].shape[0])
    matrix_a, a_scale = a
    matrix_b, b_scale = b

    if config is not None and 'MODE' in config:
        mode = config['MODE']
    else:
        mode = select_w8a8_masked_mode(
            expected_m_per_group, d.shape[2], a[0].shape[2], a[0].shape[0], "hipc_pair32"
        )

    expected_m_per_group = int(expected_m_per_group)

    op.m_grouped_int8_gemm_nt_masked_pair32(
        matrix_a,
        matrix_b,
        a_scale,
        b_scale,
        masked_m,
        d,
        mode,
    )
    return d
    


def m_grouped_fp8_gemm_nt_masked_pair32_v2(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:

    matrix_a, a_scale = a
    matrix_b, b_scale = b

    expected_m_per_group = int(expected_m_per_group)
    if config is not None and 'MODE' in config:
        mode = config['MODE']
    else:
        mode = select_w8a8_masked_mode(
            expected_m_per_group, d.shape[2], matrix_a.shape[2], matrix_a.shape[0], "hipc_pair32_v2"
        )

    return op.m_grouped_fp8_gemm_nt_masked_pair32_v2(
        matrix_a,
        matrix_b,
        a_scale,
        b_scale,
        masked_m,
        d,
        enable_overlap,
        signal,
        mode
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

    return m_grouped_int8_gemm_nt_masked_pair32(
                            a, b, d, masked_m, expected_m_per_group, config)
                
    raise RuntimeError(
        f"m_grouped_i8_gemm_nt_masked does not support GPU arch {arch}; "
        "expected gfx936/gfx92a/gfx928 or gfx938"
    )

def m_grouped_fp8_gemm_nt_masked_pair32(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:

    matrix_a, a_scale = a
    matrix_b, b_scale = b

    expected_m_per_group = int(expected_m_per_group)
    if config is not None and 'MODE' in config:
        mode = config['MODE']
    else:
        mode = select_w8a8_masked_mode(
            expected_m_per_group, d.shape[2], matrix_a.shape[2], matrix_a.shape[0], "hipc_pair32"
        )

    return op.m_grouped_fp8_gemm_nt_masked_pair32(
        matrix_a,
        matrix_b,
        a_scale,
        b_scale,
        masked_m,
        d,
        enable_overlap,
        signal,
        mode
    )
    

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
    if _get_current_gfx_arch() == "gfx938":
        return m_grouped_fp8_gemm_nt_masked_pair32_v2(
            a, b, d, masked_m, expected_m_per_group,
            enable_overlap, signal, config=config,
        )
    return m_grouped_fp8_gemm_nt_masked_pair32(
        a, b, d, masked_m, expected_m_per_group,
        enable_overlap, signal, config=config,
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
        config,
    )

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

    if config is not None and 'MODE' in config:
        mode = config['MODE']
    else:
        mode = select_bf16_masked_mode(
                expected_m_per_group, d.shape[2], a.shape[2], a.shape[0]
            )
    op.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, mode)
    return d    

