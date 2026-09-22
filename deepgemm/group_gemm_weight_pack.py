import torch
from functools import lru_cache
from typing import Any, Optional

from .m_group_gemm import pack_int8_weight_enk_to_w6_low_latency


_CUSTOMER_N_TILE = 16
_CUSTOMER_K_TILE = 16
_CUSTOMER_K_PACK = _CUSTOMER_K_TILE * 4

I8_LAYOUT_MODE_W6_HIPC = 1
I8_LAYOUT_MODE_MASKED_LAYOUT_ASM = 2

# Backward-compatible aliases for older callers/configs.
I8_LAYOUT_MODE_W6 = I8_LAYOUT_MODE_W6_HIPC
I8_LAYOUT_MODE_MASKED_LAYOUT = I8_LAYOUT_MODE_MASKED_LAYOUT_ASM

# To default gfx936 shuffle_unique=1 to mode 2, change only this line.
I8_LAYOUT_MODE_DEFAULT = I8_LAYOUT_MODE_W6_HIPC
_I8_LAYOUT_MODES = (I8_LAYOUT_MODE_W6_HIPC, I8_LAYOUT_MODE_MASKED_LAYOUT_ASM)

BF16_MASKED_BM64_BN256_BK128_AUTO_MODE = 2000
BF16_MASKED_BM64_BN256_BK128_SMALL_M_MODE = 2001
BF16_MASKED_BM64_BN256_BK128_PREFETCH_MODE = 2002
BF16_MASKED_BM64_BN256_BK128_MODES = (
    BF16_MASKED_BM64_BN256_BK128_AUTO_MODE,
    BF16_MASKED_BM64_BN256_BK128_SMALL_M_MODE,
    BF16_MASKED_BM64_BN256_BK128_PREFETCH_MODE,
)
BF16_MASKED_BM64_BN256_BK128_LAYOUT = "bm64_bn256_bk128"
BF16_MASKED_ASM_MODES = (1000, 1001)
BF16_MASKED_ASM_LAYOUT = "native_nt"


def _normalize_gfx_arch(arch: Any) -> str:
    arch = str(arch).strip().lower()
    if ";" in arch:
        arch = arch.split(";")[-1]
    return arch.split(":")[0]


@lru_cache(maxsize=1)
def _get_current_gfx_arch() -> str:
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return _normalize_gfx_arch(getattr(props, "gcnArchName", props.name))
        except Exception:
            pass
    from .utils import gfx
    return _normalize_gfx_arch(gfx)


def _resolve_arch(arch: Optional[str]) -> str:
    return _get_current_gfx_arch() if arch is None else _normalize_gfx_arch(arch)


@lru_cache(maxsize=8)
def _bf16_masked_bm64_bn256_bk128_n32_order(device: str) -> torch.Tensor:
    return torch.tensor(
        [
            0, 8, 16, 24, 2, 10, 18, 26,
            4, 12, 20, 28, 6, 14, 22, 30,
            1, 9, 17, 25, 3, 11, 19, 27,
            5, 13, 21, 29, 7, 15, 23, 31,
        ],
        dtype=torch.long,
        device=torch.device(device),
    )


def _pack_bf16_masked_bm64_bn256_bk128_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack [E, N, K] BF16 weights for BM64/BN256/BK128 masked kernels."""
    if weight.dim() != 3:
        raise ValueError(f"weight must be [E, N, K], got dim={weight.dim()}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"weight must be torch.bfloat16, got {weight.dtype}")
    if not weight.is_cuda:
        raise ValueError("weight must be a CUDA/HIP tensor")
    experts, size_n, size_k = weight.shape
    if size_n % 256 != 0:
        raise ValueError(f"N must be divisible by 256, got {size_n}")
    if size_k % 128 != 0:
        raise ValueError(f"K must be divisible by 128, got {size_k}")
    weight = weight.contiguous()
    order = _bf16_masked_bm64_bn256_bk128_n32_order(str(weight.device))
    packed = weight.reshape(experts, size_n // 32, 32, size_k // 32, 4, 2, 4)
    packed = packed.index_select(2, order)
    packed = packed.reshape(experts, size_n // 16, 16, size_k // 32, 4, 2, 4)
    packed = packed.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
    return packed.view_as(weight)


def _get_bf16_masked_bm64_bn256_bk128_config(
    mode: int = BF16_MASKED_BM64_BN256_BK128_AUTO_MODE,
    *,
    packed_weight: bool = False,
) -> dict[str, Any]:
    mode = int(mode)
    if mode not in BF16_MASKED_BM64_BN256_BK128_MODES:
        raise ValueError(
            "BF16 masked BM64/BN256/BK128 mode must be 2000 (auto), "
            f"2001 (small_m), or 2002 (prefetch), got {mode}"
        )
    return {
        "MODE": mode,
        "WEIGHT_LAYOUT": BF16_MASKED_BM64_BN256_BK128_LAYOUT,
        "PACKED_WEIGHT": bool(packed_weight),
    }


def _prepare_bf16_masked_bm64_bn256_bk128_weight(
    weight: torch.Tensor,
    mode: int = BF16_MASKED_BM64_BN256_BK128_AUTO_MODE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    packed_weight = _pack_bf16_masked_bm64_bn256_bk128_weight(weight)
    config = _get_bf16_masked_bm64_bn256_bk128_config(mode, packed_weight=True)
    return packed_weight, config


def _resolve_shuffle_unique(shuffle_unique: int) -> int:
    value = int(shuffle_unique)
    if value not in (0, 1):
        raise ValueError(f"shuffle_unique must be 0 or 1, got {shuffle_unique}")
    return value


def _resolve_i8_layout_mode(mode: Any) -> int:
    value = int(mode)
    if value not in _I8_LAYOUT_MODES:
        raise ValueError(
            "INT8 layout mode must be 1 (W6 HIPC layout) or "
            f"2 (masked-layout ASM), got {mode}"
        )
    return value


def _resolve_i8_layout_mode_from_config(config: Optional[dict[str, Any]]) -> int:
    if config is None or "MODE" not in config:
        return I8_LAYOUT_MODE_DEFAULT
    value = int(config["MODE"])
    if value in _I8_LAYOUT_MODES:
        return value
    return I8_LAYOUT_MODE_DEFAULT


def _strip_i8_layout_mode_from_config(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if config is None or "MODE" not in config:
        return config
    value = int(config["MODE"])
    if value not in _I8_LAYOUT_MODES:
        return config
    stripped = dict(config)
    stripped.pop("MODE", None)
    return stripped or None


def _is_gfx936_contiguous_arch(arch: str) -> bool:
    return arch in ("gfx936", "gfx92a", "gfx928")


def _is_fp8_low_latency_arch(arch: str) -> bool:
    return arch == "gfx938"


def pack_i8_contiguous_weight_gfx936_w6(weight_enk: torch.Tensor) -> torch.Tensor:
    if weight_enk.dim() != 3:
        raise ValueError(
            f"weight_enk must be [E,N,K], got dim {weight_enk.dim()} shape {tuple(weight_enk.shape)}"
        )
    if weight_enk.element_size() != 1:
        raise ValueError("weight_enk must be an 8-bit tensor")
    if not weight_enk.is_contiguous():
        raise ValueError("low-memory gfx936 W6 pack requires contiguous [E,N,K] weight input")
    experts = int(weight_enk.size(0))
    n = int(weight_enk.size(1))
    k = int(weight_enk.size(2))
    if n % _CUSTOMER_N_TILE != 0 or k % _CUSTOMER_K_PACK != 0:
        raise ValueError(
            f"N={n}, K={k} must satisfy N%{_CUSTOMER_N_TILE}==0 and K%{_CUSTOMER_K_PACK}==0"
        )
    n_out_band = n // _CUSTOMER_N_TILE
    k_out_band = k // _CUSTOMER_K_PACK
    stride_n, stride_k = weight_enk.stride()[1:]
    packed_shape = (
        n_out_band,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        k_out_band,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_K_TILE,
    )
    packed_stride = (
        _CUSTOMER_N_TILE * stride_n,
        4 * stride_n,
        stride_n,
        _CUSTOMER_K_PACK * stride_k,
        _CUSTOMER_K_TILE * stride_k,
        stride_k,
    )
    for expert_idx in range(experts):
        packed = torch.as_strided(
            weight_enk[expert_idx],
            size=packed_shape,
            stride=packed_stride,
        ).permute(3, 0, 4, 2, 1, 5).contiguous()
        weight_enk[expert_idx].view(-1).copy_(packed.view(-1))
    return weight_enk.view(
        experts,
        k_out_band,
        n_out_band,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_N_TILE,
        _CUSTOMER_K_TILE,
    )


def pack_i8_contiguous_weight_gfx936_asm(
    weight: torch.Tensor,
    k_tile: int = 16,
    n_tile: int = 16,
) -> torch.Tensor:
    return pack_i8_contiguous_weight_gfx936_asm_inplace(weight, k_tile=k_tile, n_tile=n_tile)


def pack_i8_contiguous_weight_gfx936_asm_inplace(
    weight: torch.Tensor,
    k_tile: int = 16,
    n_tile: int = 16,
) -> torch.Tensor:
    if weight.element_size() != 1:
        raise ValueError("weight must be an 8-bit tensor")
    if not weight.is_contiguous():
        raise ValueError("low-memory contiguous ASM pack requires contiguous weight input")

    if weight.dim() == 2:
        size_n, size_k = weight.shape
        if size_n % n_tile != 0 or size_k % k_tile != 0:
            raise ValueError(f"N={size_n}, K={size_k} must satisfy N%{n_tile}==0 and K%{k_tile}==0")
        stride_n, stride_k = weight.stride()
        packed = torch.as_strided(
            weight,
            size=(size_n // n_tile, size_k // k_tile, n_tile, k_tile),
            stride=(n_tile * stride_n, k_tile * stride_k, stride_n, stride_k),
        ).contiguous()
        weight.view(-1).copy_(packed.view(-1))
        return weight.view(size_n // n_tile, size_k * n_tile)

    if weight.dim() == 3:
        experts, size_n, size_k = weight.shape
        if size_n % n_tile != 0 or size_k % k_tile != 0:
            raise ValueError(f"N={size_n}, K={size_k} must satisfy N%{n_tile}==0 and K%{k_tile}==0")
        stride_n, stride_k = weight.stride()[1:]
        packed_shape = (size_n // n_tile, size_k // k_tile, n_tile, k_tile)
        packed_stride = (n_tile * stride_n, k_tile * stride_k, stride_n, stride_k)
        for expert_idx in range(experts):
            packed = torch.as_strided(
                weight[expert_idx],
                size=packed_shape,
                stride=packed_stride,
            ).contiguous()
            weight[expert_idx].view(-1).copy_(packed.view(-1))
        return weight.view(experts, size_n // n_tile, size_k * n_tile)

    raise ValueError(f"weight must be [N,K] or [E,N,K], got dim {weight.dim()}")


def pack_i8_masked_weight_gfx936_asm(
    weight: torch.Tensor,
    k_tile: int = 16,
    k_tile1: int = 4,
    n_tile: int = 16,
) -> torch.Tensor:
    return pack_i8_masked_weight_gfx936_asm_inplace(
        weight, k_tile=k_tile, k_tile1=k_tile1, n_tile=n_tile
    )


def pack_i8_masked_weight_gfx936_asm_inplace(
    weight: torch.Tensor,
    k_tile: int = 16,
    k_tile1: int = 4,
    n_tile: int = 16,
) -> torch.Tensor:
    if weight.element_size() != 1:
        raise ValueError("weight must be an 8-bit tensor")
    if not weight.is_contiguous():
        raise ValueError("low-memory masked ASM pack requires contiguous weight input")

    k_pack = k_tile * k_tile1
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        if size_n % n_tile != 0 or size_k % k_pack != 0:
            raise ValueError(
                f"N={size_n}, K={size_k} must satisfy N%{n_tile}==0 and K%{k_pack}==0"
            )
        stride_n, stride_k = weight.stride()
        packed = torch.as_strided(
            weight,
            size=(size_k // k_pack, size_n // n_tile, k_tile1, n_tile, k_tile),
            stride=(k_pack * stride_k, n_tile * stride_n, k_tile * stride_k, stride_n, stride_k),
        ).contiguous()
        weight.view(-1).copy_(packed.view(-1))
        return weight.view(size_n // n_tile, size_k * n_tile)

    if weight.dim() == 3:
        experts, size_n, size_k = weight.shape
        if size_n % n_tile != 0 or size_k % k_pack != 0:
            raise ValueError(
                f"N={size_n}, K={size_k} must satisfy N%{n_tile}==0 and K%{k_pack}==0"
            )
        stride_n, stride_k = weight.stride()[1:]
        packed_shape = (size_k // k_pack, size_n // n_tile, k_tile1, n_tile, k_tile)
        packed_stride = (
            k_pack * stride_k,
            n_tile * stride_n,
            k_tile * stride_k,
            stride_n,
            stride_k,
        )
        for expert_idx in range(experts):
            packed = torch.as_strided(
                weight[expert_idx],
                size=packed_shape,
                stride=packed_stride,
            ).contiguous()
            weight[expert_idx].view(-1).copy_(packed.view(-1))
        return weight.view(experts, size_n // n_tile, size_k * n_tile)

    raise ValueError(f"weight must be [N,K] or [E,N,K], got dim {weight.dim()}")

def build_k_perm_index(K: int) -> torch.Tensor:
    """
    生成K维置换索引数组，实现：
    K维以128为块，每块8组每组16，group顺序 [0,1,2,3,4,5,6,7] → [0,4,1,5,2,6,3,7]
    返回 k_idx: shape [K], int64
    """
    block_size = 64
    group_size = 8
    perm_group_idx = [0, 4, 1, 5, 2, 6, 3, 7]
    assert K % block_size == 0
    num_blocks = K // block_size

    # 单块内的置换索引
    single_block_idx = []
    for g_out in perm_group_idx:
        # g_out 是输出位置对应的原始group号
        start = g_out * group_size
        single_block_idx.extend(range(start, start + group_size))
    single_block_idx = torch.tensor(single_block_idx, dtype=torch.int64)

    # 拼接所有块
    k_idx_list = []
    for b in range(num_blocks):
        base = b * block_size
        k_idx_list.append(single_block_idx + base)
    k_idx = torch.cat(k_idx_list)
    return k_idx

def pack_int8_weight_enk_to_w6_low_latency_gfx92a(weight_enk: torch.Tensor) -> torch.Tensor:
    if weight_enk.dim() != 3:
        raise ValueError(
            f"weight_enk must be [E,N,K], got dim {weight_enk.dim()} shape {tuple(weight_enk.shape)}"
        )
    if weight_enk.element_size() != 1:
        raise ValueError("weight_enk must be an 8-bit tensor")
    if not weight_enk.is_contiguous():
        raise ValueError("low-memory gfx92a W6 pack requires contiguous [E,N,K] weight input")

    experts = int(weight_enk.size(0))
    n = int(weight_enk.size(1))
    k = int(weight_enk.size(2))

    if n % _CUSTOMER_N_TILE != 0 or k % _CUSTOMER_K_PACK != 0:
        raise ValueError(
            f"N={n}, K={k} must satisfy N%{_CUSTOMER_N_TILE}==0 and K%{_CUSTOMER_K_PACK}==0"
        )

    # # 一次性预计算全局K置换索引，所有expert共用
    # k_perm_idx = build_k_perm_index(k)

    n_out_band = n // _CUSTOMER_N_TILE
    k_out_band = k // _CUSTOMER_K_PACK
    stride_n, stride_k = weight_enk.stride()[1:]

    packed_shape = (
        n_out_band,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        k_out_band,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_K_TILE,
    )
    packed_stride = (
        _CUSTOMER_N_TILE * stride_n,
        4 * stride_n,
        stride_n,
        _CUSTOMER_K_PACK * stride_k,
        _CUSTOMER_K_TILE * stride_k,
        stride_k,
    )

    for expert_idx in range(experts):
        # expert_weight = weight_enk[expert_idx] # [N,K]
        # expert_weight_rearranged = expert_weight[:, k_perm_idx]
        # --------------------------

        packed = torch.as_strided(
            weight_enk[expert_idx],
            size=packed_shape,
            stride=packed_stride,
        ).permute(3, 0, 4, 2, 1, 5).contiguous()
        weight_enk[expert_idx].view(-1).copy_(packed.view(-1))

        # packed = torch.as_strided(
        #     expert_weight_rearranged,
        #     size=packed_shape,
        #     stride=packed_stride,
        # ).permute(3, 0, 4, 2, 1, 5).contiguous()
        # weight_enk[expert_idx].view(-1).copy_(packed.view(-1))

    return weight_enk.view(
        experts,
        k_out_band,
        n_out_band,
        _CUSTOMER_K_PACK // _CUSTOMER_K_TILE,
        _CUSTOMER_N_TILE,
        _CUSTOMER_K_TILE,
    )


def _pack_bf16_masked_asm_weight_2d(
    weight: torch.Tensor
) -> torch.Tensor:
    def _w16a16_marlin_weight_perm(device: torch.device) -> torch.Tensor:
        perm = []
        for i in range(64):
            for col in range(2):
                cur_col = (i % 16) * 2 + col
                for row in range(4):
                    cur_row = (i // 16) * 4 + row
                    perm.append(cur_row * 32 + cur_col)
        return torch.tensor(perm, dtype=torch.long, device=device)

    def _w16a16_marlin_weights_npack2(
        weight: torch.Tensor,
        weight_perm: torch.Tensor,
        k_tile: int = 16,
        n_tile: int = 32,
    ) -> torch.Tensor:
        size_k, size_n = weight.shape
        weight = weight.reshape((size_k // k_tile, k_tile, size_n // n_tile, n_tile))
        weight = weight.permute((0, 2, 1, 3))
        weight = weight.reshape((size_k // k_tile, size_n * k_tile))
        weight = weight.reshape((-1, weight_perm.numel()))[:, weight_perm].reshape(weight.shape)
        return weight.contiguous()

    weight_perm = _w16a16_marlin_weight_perm(weight.device)
    return _w16a16_marlin_weights_npack2(weight.T.contiguous(), weight_perm)


def pack_bf16_masked_weight(
    weight: torch.Tensor,
    mode: int = BF16_MASKED_BM64_BN256_BK128_AUTO_MODE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prepare BF16 masked GEMM weight and matching config for the front-end."""
    mode = int(mode)
    if mode in BF16_MASKED_BM64_BN256_BK128_MODES:
        return _prepare_bf16_masked_bm64_bn256_bk128_weight(weight, mode)
    if mode in BF16_MASKED_ASM_MODES:
        return weight.contiguous(), {
            "MODE": mode,
            "WEIGHT_LAYOUT": BF16_MASKED_ASM_LAYOUT,
            "PACKED_WEIGHT": False,
        }
    raise ValueError(
        "BF16 masked mode must be 1000/1001 for asm layout or "
        f"2000/2001/2002 for BM64/BN256/BK128 layout, got {mode}"
    )

    
def marlin_i8_contiguous_weight(
    weight: torch.Tensor,
    *,
    shuffle_unique: int = 1,
    mode: int = I8_LAYOUT_MODE_DEFAULT,
    arch: Optional[str] = None,
    low_memory: bool = True,
) -> torch.Tensor:
    arch = _resolve_arch(arch)
    shuffle_unique = _resolve_shuffle_unique(shuffle_unique)
    mode = _resolve_i8_layout_mode(mode)
    if mode == I8_LAYOUT_MODE_MASKED_LAYOUT_ASM and shuffle_unique != 1:
        raise ValueError("INT8 layout mode=2 is only valid with shuffle_unique=1")
    if arch == "gfx938":
        if mode != I8_LAYOUT_MODE_W6_HIPC:
            raise RuntimeError("INT8 layout mode=2 is only implemented for gfx936")
        return pack_int8_weight_enk_to_w6_low_latency(weight)
    if _is_gfx936_contiguous_arch(arch):
        if mode == I8_LAYOUT_MODE_MASKED_LAYOUT_ASM:
            if arch != "gfx936":
                raise RuntimeError("INT8 layout mode=2 is only implemented for gfx936")
            return pack_i8_masked_weight_gfx936_asm_inplace(weight)
        if shuffle_unique == 1:
            return pack_i8_contiguous_weight_gfx936_w6(weight)
        return pack_i8_contiguous_weight_gfx936_asm_inplace(weight)
    raise RuntimeError(
        f"marlin_i8_contiguous_weight does not support GPU arch {arch}; "
        "expected gfx936/gfx92a/gfx928 or gfx938"
    )


def marlin_i8_masked_weight(
    weight: torch.Tensor,
    *,
    shuffle_unique: int = 1,
    mode: int = I8_LAYOUT_MODE_DEFAULT,
    arch: Optional[str] = None,
    low_memory: bool = True,
) -> torch.Tensor:
    arch = _resolve_arch(arch)
    shuffle_unique = _resolve_shuffle_unique(shuffle_unique)
    mode = _resolve_i8_layout_mode(mode)
    if mode == I8_LAYOUT_MODE_MASKED_LAYOUT_ASM and shuffle_unique != 1:
        raise ValueError("INT8 layout mode=2 is only valid with shuffle_unique=1")
    if arch == "gfx938":
        if mode != I8_LAYOUT_MODE_W6_HIPC:
            raise RuntimeError("INT8 layout mode=2 is only implemented for gfx936")
        return pack_int8_weight_enk_to_w6_low_latency(weight)
    if _is_gfx936_contiguous_arch(arch):
        if mode == I8_LAYOUT_MODE_MASKED_LAYOUT_ASM:
            if arch != "gfx936":
                raise RuntimeError("INT8 layout mode=2 is only implemented for gfx936")
            return pack_i8_masked_weight_gfx936_asm_inplace(weight)
        if shuffle_unique == 1:
            return pack_i8_contiguous_weight_gfx936_w6(weight)
        return pack_i8_masked_weight_gfx936_asm_inplace(weight)
    raise RuntimeError(
        f"marlin_i8_masked_weight does not support GPU arch {arch}; "
        "expected gfx936/gfx92a/gfx928 or gfx938"
    )


def marlin_bf16_masked_weight(
    weight: torch.Tensor,
    *,
    arch: Optional[str] = None,
) -> torch.Tensor:
    return torch.stack(
        [
            _pack_bf16_masked_asm_weight_2d(weight[i])
            for i in range(weight.shape[0])
        ]
    )


def marlin_fp8_masked_weight(
    weight: torch.Tensor,
    *,
    arch: Optional[str] = None,
) -> torch.Tensor:
    arch = _resolve_arch(arch)
    if _is_fp8_low_latency_arch(arch):
        return pack_int8_weight_enk_to_w6_low_latency(weight)
    if arch == "gfx92a":
        return pack_int8_weight_enk_to_w6_low_latency_gfx92a(weight)
    raise RuntimeError(f"marlin_fp8_masked_weight does not support GPU arch {arch}; expected gfx938")


def marlin_fp8_contiguous_weight(
    weight: torch.Tensor,
    *,
    arch: Optional[str] = None,
) -> torch.Tensor:
    arch = _resolve_arch(arch)
    if _is_fp8_low_latency_arch(arch):
        return pack_int8_weight_enk_to_w6_low_latency(weight)
    if arch == "gfx92a":
        return pack_int8_weight_enk_to_w6_low_latency_gfx92a(weight)
    raise RuntimeError(f"marlin_fp8_contiguous_weight does not support GPU arch {arch}; expected gfx92a/gfx938/gfx946")


pack_i8_contiguous_weight = marlin_i8_contiguous_weight
pack_i8_masked_weight = marlin_i8_masked_weight
pack_fp8_contiguous_weight = marlin_fp8_contiguous_weight
pack_fp8_masked_weight = marlin_fp8_masked_weight


pack_w8a8_gfx936_weight_to_w6 = pack_i8_contiguous_weight_gfx936_w6
pack_int8_weight_enk_to_w6_gfx936_pack_permute = pack_i8_contiguous_weight_gfx936_w6


def pack_w4a8_weight_k_contiguous(weight_u4: torch.Tensor) -> torch.Tensor:
    if weight_u4.dim() != 3:
        raise ValueError(f"weight_u4 must be [E,N,K], got shape {tuple(weight_u4.shape)}")
    experts, n, k = weight_u4.shape
    if k % 2 != 0:
        raise ValueError(f"K must be divisible by 2 for W4 packing, got {k}")
    weight_u4 = weight_u4.to(torch.uint8) & 0x0F
    weight_u4 = weight_u4.view(experts, n, k // 2, 2)
    return ((weight_u4[..., 0] << 4) | weight_u4[..., 1]).to(torch.int8)


def repack_w4a8_weight_contiguous_to_blocked(packed_k_contiguous: torch.Tensor) -> torch.Tensor:
    if packed_k_contiguous.dim() != 3:
        raise ValueError(
            f"packed_k_contiguous must be [E,N,K/2], got shape {tuple(packed_k_contiguous.shape)}"
        )
    experts, n, k_half = packed_k_contiguous.shape
    if k_half % 4 != 0:
        raise ValueError(f"K/2 must be divisible by 4 for W4A8 blocked packing, got {k_half}")
    weight_u8 = packed_k_contiguous.to(torch.uint8)
    weight_0 = (weight_u8 >> 4) & 0x0F
    weight_1 = weight_u8 & 0x0F
    weight_unpacked = torch.stack([weight_0, weight_1], dim=-1).view(experts, n, -1)
    tile_view = weight_unpacked.view(*weight_unpacked.shape[:-1], -1, 8)
    weight_low = tile_view[..., :4].reshape(*weight_unpacked.shape[:-1], -1)
    weight_high = tile_view[..., 4:].reshape(*weight_unpacked.shape[:-1], -1)
    return ((weight_low << 4) | weight_high).to(torch.int8)



def _w4a8_moe_layout_shuffle_n256(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"weight must be [N,K/2] blocked W4 bytes, got shape {tuple(weight.shape)}")
    full_weight = weight.T
    k_tile = 32
    n_tile = 256
    size_k, size_n = full_weight.shape
    if size_k % k_tile != 0 or size_n % n_tile != 0:
        raise ValueError(
            f"transposed weight shape [K/2,N]=[{size_k},{size_n}] must satisfy "
            f"K/2%{k_tile}==0 and N%{n_tile}==0"
        )
    full_weight = full_weight.reshape(size_k // k_tile, k_tile, size_n // n_tile, n_tile)
    full_weight = full_weight.permute(0, 2, 3, 1).contiguous()
    full_weight = full_weight.reshape(
        size_k // k_tile,
        size_n // n_tile,
        n_tile // 32,
        32,
        k_tile // 8,
        8,
    )
    return full_weight.permute(0, 1, 2, 4, 3, 5).contiguous()


def w4a8_moe_layout_shuffle_gemm1(weight: torch.Tensor) -> torch.Tensor:
    return _w4a8_moe_layout_shuffle_n256(weight)


def _w4a8_evenodd_n_reorder(weight: torch.Tensor) -> torch.Tensor:
    n_dim = 0 if weight.dim() == 2 else 1
    n = weight.size(n_dim)
    if n % 32 != 0:
        raise ValueError(f"N must be divisible by 32 for W4A8 even/odd layout, got {n}")
    arange_n = torch.arange(n, device=weight.device)
    local = arange_n % 32
    reg_id = local >> 3
    col_id = (local >> 1) & 3
    it = local & 1
    physical_local = col_id * 8 + reg_id * 2 + it
    physical_n = (arange_n // 32) * 32 + physical_local
    return weight.index_select(n_dim, physical_n).contiguous()


def w4a8_moe_layout_shuffle_gemm1_evenodd(weight: torch.Tensor) -> torch.Tensor:
    return _w4a8_moe_layout_shuffle_n256(_w4a8_evenodd_n_reorder(weight))


def w4a8_moe_layout_shuffle_gemm1_evenodd_n32(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"weight must be [N,K/2] blocked W4 bytes, got shape {tuple(weight.shape)}")
    full_weight = _w4a8_evenodd_n_reorder(weight).T
    k_tile = 32
    n_tile = 32
    size_k, size_n = full_weight.shape
    if size_k % k_tile != 0 or size_n % n_tile != 0:
        raise ValueError(
            f"transposed weight shape [K/2,N]=[{size_k},{size_n}] must satisfy "
            f"K/2%{k_tile}==0 and N%{n_tile}==0"
        )
    full_weight = full_weight.reshape(size_k // k_tile, k_tile, size_n // n_tile, n_tile)
    full_weight = full_weight.permute(0, 2, 3, 1).contiguous()
    full_weight = full_weight.reshape(size_k // k_tile, size_n // n_tile, 32, k_tile // 8, 8)
    return full_weight.permute(0, 1, 3, 2, 4).contiguous()



def pack_w4a8_moe_hipc_weight_inplace(packed_k_contiguous: torch.Tensor) -> torch.Tensor:
    if packed_k_contiguous.element_size() != 1:
        raise ValueError("packed_k_contiguous must be an 8-bit tensor")
    if packed_k_contiguous.dim() != 3:
        raise ValueError(
            f"packed_k_contiguous must be [E,N,K/2], got shape {tuple(packed_k_contiguous.shape)}"
        )
    if not packed_k_contiguous.is_contiguous():
        packed_k_contiguous = packed_k_contiguous.contiguous()

    original_shape = packed_k_contiguous.shape
    experts, n, k_half = original_shape
    if k_half % 32 != 0:
        raise ValueError(f"K/2 must be divisible by 32 for W4A8 HIPC N32 layout, got {k_half}")
    if n % 32 != 0:
        raise ValueError(f"N must be divisible by 32 for W4A8 HIPC N32 layout, got {n}")

    for expert_idx in range(experts):
        blocked = repack_w4a8_weight_contiguous_to_blocked(
            packed_k_contiguous[expert_idx : expert_idx + 1]
        )[0]
        shuffled = w4a8_moe_layout_shuffle_gemm1_evenodd_n32(blocked)
        packed_k_contiguous[expert_idx].view(-1).copy_(shuffled.view(-1))
    return packed_k_contiguous.view(original_shape)


def view_w4a8_moe_hipc_weight_n32_layout(packed_hipc_weight: torch.Tensor) -> torch.Tensor:
    if packed_hipc_weight.dim() != 3:
        raise ValueError(
            f"packed_hipc_weight must be [E,N,K/2], got shape {tuple(packed_hipc_weight.shape)}"
        )
    experts, n, k_half = packed_hipc_weight.shape
    if k_half % 32 != 0 or n % 32 != 0:
        raise ValueError(
            f"packed_hipc_weight shape [E,N,K/2]={tuple(packed_hipc_weight.shape)} must satisfy N%32==0 and K/2%32==0"
        )
    return packed_hipc_weight.view(experts, k_half // 32, n // 32, 4, 32, 8)


def pack_w4a8_moe_hipc_weight_evenodd(packed_k_contiguous: torch.Tensor) -> torch.Tensor:
    if packed_k_contiguous.element_size() != 1:
        raise ValueError("packed_k_contiguous must be an 8-bit tensor")
    if packed_k_contiguous.dim() != 3:
        raise ValueError(
            f"packed_k_contiguous must be [E,N,K/2], got shape {tuple(packed_k_contiguous.shape)}"
        )
    experts, n, k_half = packed_k_contiguous.shape
    if k_half % 32 != 0:
        raise ValueError(f"K/2 must be divisible by 32 for W4A8 HIPC layout, got {k_half}")
    if n % 256 != 0:
        raise ValueError(f"N must be divisible by 256 for W4A8 HIPC legacy layout, got {n}")

    output = torch.empty(
        (experts, k_half // 32, n // 256, 8, 4, 32, 8),
        device=packed_k_contiguous.device,
        dtype=packed_k_contiguous.dtype,
    )
    for expert_idx in range(experts):
        blocked = repack_w4a8_weight_contiguous_to_blocked(
            packed_k_contiguous[expert_idx : expert_idx + 1]
        )[0]
        output[expert_idx].copy_(w4a8_moe_layout_shuffle_gemm1_evenodd(blocked))
    return output.contiguous()


def pack_w4a8_moe_hipc_weight(packed_k_contiguous: torch.Tensor) -> torch.Tensor:
    """Pack framework W4A8 weights for the HIPC masked GEMM path in-place.

    Args:
        packed_k_contiguous: int8 tensor shaped [E, N, K/2]. Each byte stores
            two unsigned int4 values in normal K order: byte0=(w0,w1),
            byte1=(w2,w3), ...

    This API rewrites ``packed_k_contiguous`` storage to reduce peak memory.
    Callers that still need the original [E,N,K/2] packed values should pass a
    clone. The returned tensor keeps the original [E,N,K/2] shape; its storage
    is rewritten to the HIPC N32 layout consumed by the kernel.
    """
    return pack_w4a8_moe_hipc_weight_inplace(packed_k_contiguous)
