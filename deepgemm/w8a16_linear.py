from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

from . import op


_DEFAULT_CONFIG: Dict[str, int] = {
    "NUM_K_WARPS": 0,
    "N_PER_WARP": 0,
    "K_PER_WARP": 0,
    "NUM_N_WARPS": 0,
    "SPLITK": 0,
    "BLOCK_M": 0,
    "KERNEL_TYPE": 0,
    "WEIGHT_LAYOUT": 0,
}

_W8A16_DEQUANT_CACHE: "OrderedDict[tuple[int, int, str], Dict[str, Any]]" = OrderedDict()
_W8A16_DEQUANT_CACHE_BYTES = 0


def _w8a16_cache_enabled() -> bool:
    return os.environ.get("DEEPGEMM_W8A16_DEQUANT_CACHE", "1") != "0"


def _w8a16_cache_limit_bytes() -> int:
    value = os.environ.get("DEEPGEMM_W8A16_DEQUANT_CACHE_MAX_BYTES", str(1024 * 1024 * 1024))
    try:
        return max(0, int(value))
    except ValueError:
        return 1024 * 1024 * 1024


def _w8a16_cache_active() -> bool:
    return _w8a16_cache_enabled() and _w8a16_cache_limit_bytes() > 0


def clear_w8a16_dequant_cache() -> None:
    """Clear cached fp16 W8A16 weights used by the GEMM backend."""
    global _W8A16_DEQUANT_CACHE_BYTES
    _W8A16_DEQUANT_CACHE.clear()
    _W8A16_DEQUANT_CACHE_BYTES = 0


def _drop_w8a16_cache_key(key: tuple[int, int, str]) -> None:
    global _W8A16_DEQUANT_CACHE_BYTES
    entry = _W8A16_DEQUANT_CACHE.pop(key, None)
    if entry is not None:
        _W8A16_DEQUANT_CACHE_BYTES -= int(entry["bytes"])


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tensor.data_ptr(),
        int(tensor._version),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _w8a16_dequant_cache_bytes(size_n: int, size_k: int, dtype: torch.dtype = torch.float16) -> int:
    return int(size_n) * int(size_k) * torch.tensor([], dtype=dtype).element_size()


def _w8a16_cache_can_hold_dequant(size_n: int, size_k: int) -> bool:
    return _w8a16_cache_active() and _w8a16_cache_limit_bytes() >= _w8a16_dequant_cache_bytes(size_n, size_k)


def _cached_w8a16_dequant_weight_impl(
    weight: torch.Tensor,
    scale_owner: torch.Tensor,
    scale_1d: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return cached dequantized weight when tensors are unchanged."""
    global _W8A16_DEQUANT_CACHE_BYTES
    if not _w8a16_cache_can_hold_dequant(weight.shape[0], weight.shape[1]):
        return op.w8a16_dequant_weight_bf16(weight, scale_1d) if dtype == torch.bfloat16 else op.w8a16_dequant_weight(weight, scale_1d)

    layout = "nt:bf16" if dtype == torch.bfloat16 else "nt:fp16"
    key = (id(weight), id(scale_owner), layout)
    weight_sig = _tensor_signature(weight)
    scale_sig = _tensor_signature(scale_owner)
    entry = _W8A16_DEQUANT_CACHE.get(key)
    if (
        entry is not None
        and entry["weight_ref"]() is weight
        and entry["scale_ref"]() is scale_owner
        and entry["weight_sig"] == weight_sig
        and entry["scale_sig"] == scale_sig
    ):
        _W8A16_DEQUANT_CACHE.move_to_end(key)
        return entry["value"]

    if entry is not None:
        _drop_w8a16_cache_key(key)

    value = op.w8a16_dequant_weight_bf16(weight, scale_1d) if dtype == torch.bfloat16 else op.w8a16_dequant_weight(weight, scale_1d)
    value_bytes = value.numel() * value.element_size()
    limit = _w8a16_cache_limit_bytes()
    if value_bytes > limit:
        return value

    def drop(_unused: weakref.ReferenceType[torch.Tensor], cache_key: tuple[int, int, str] = key) -> None:
        _drop_w8a16_cache_key(cache_key)

    _W8A16_DEQUANT_CACHE[key] = {
        "weight_ref": weakref.ref(weight, drop),
        "scale_ref": weakref.ref(scale_owner, drop),
        "weight_sig": weight_sig,
        "scale_sig": scale_sig,
        "value": value,
        "bytes": value_bytes,
    }
    _W8A16_DEQUANT_CACHE_BYTES += value_bytes
    while _W8A16_DEQUANT_CACHE_BYTES > limit and _W8A16_DEQUANT_CACHE:
        old_key, old_entry = _W8A16_DEQUANT_CACHE.popitem(last=False)
        _W8A16_DEQUANT_CACHE_BYTES -= int(old_entry["bytes"])
        if old_key == key:
            break
    return value


def _cached_w8a16_dequant_weight(
    weight: torch.Tensor,
    scale_owner: torch.Tensor,
    scale_1d: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return cached dequantized weight in row-major [N, K] layout."""
    return _cached_w8a16_dequant_weight_impl(weight, scale_owner, scale_1d, dtype)


def _clamp_w8a16_splitk(config: Dict[str, int], size_k: int) -> Dict[str, int]:
    k_per_warp = int(config.get("K_PER_WARP", 0))
    splitk = int(config.get("SPLITK", 0))
    if k_per_warp > 0 and splitk > 1:
        k_tiles = max(1, int(size_k) // k_per_warp)
        if splitk > k_tiles:
            config = dict(config)
            config["SPLITK"] = k_tiles
    return config


def get_w8a16_linear_config(size_m: int, size_n: int, size_k: int) -> Dict[str, int]:
    """Return the default HIP C W8A16 linear config.

    Decode and prefill use separate HIP kernels, so the defaults intentionally
    choose both the path and its launch shape.
    """
    size_m = int(size_m)
    size_n = int(size_n)
    size_k = int(size_k)
    config = dict(_DEFAULT_CONFIG)
    config.update(
        {
            "NUM_K_WARPS": 1,
            "N_PER_WARP": 32,
            "K_PER_WARP": 128,
            "NUM_N_WARPS": 4,
            "SPLITK": 1,
            "BLOCK_M": 16,
            "KERNEL_TYPE": 0,
        }
    )
    if size_n == 2048 and size_k == 4096 and size_m <= 16:
        # Keep the small-M hot path on the native W8A16 kernel.  The previous
        # splitK=8 default over-parallelized K and paid extra reduction cost.
        if size_m == 1:
            config.update(
                {
                    "NUM_K_WARPS": 4,
                    "N_PER_WARP": 32,
                    "K_PER_WARP": 64,
                    "NUM_N_WARPS": 2,
                    "SPLITK": 4,
                    "BLOCK_M": 16,
                    "KERNEL_TYPE": 0,
                }
            )
        elif size_m == 2:
            config.update(
                {
                    "NUM_K_WARPS": 2,
                    "N_PER_WARP": 32,
                    "K_PER_WARP": 256,
                    "NUM_N_WARPS": 2,
                    "SPLITK": 4,
                    "BLOCK_M": 16,
                    "KERNEL_TYPE": 0,
                }
            )
        elif size_m <= 8:
            config.update(
                {
                    "NUM_K_WARPS": 2,
                    "N_PER_WARP": 32,
                    "K_PER_WARP": 64,
                    "NUM_N_WARPS": 2,
                    "SPLITK": 4,
                    "BLOCK_M": 16,
                    "KERNEL_TYPE": 0,
                }
            )
        else:
            config.update(
                {
                    "NUM_K_WARPS": 1,
                    "N_PER_WARP": 32,
                    "K_PER_WARP": 128,
                    "NUM_N_WARPS": 4,
                    "SPLITK": 8,
                    "BLOCK_M": 16,
                    "KERNEL_TYPE": 0,
                }
            )
    elif size_n == 12288 and size_k == 2048 and size_m <= 4:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        ) 
    elif size_n == 12288 and size_k == 2048 and 5 <= size_m <= 16:
        config.update(
            {
                "NUM_K_WARPS": 1,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 12288 and size_k == 2560 and size_m <= 16:
            config.update(
                {
                    "NUM_K_WARPS": 1,
                    "N_PER_WARP": 32,
                    "K_PER_WARP": 256,
                    "NUM_N_WARPS": 2,
                    "SPLITK": 1,
                    "BLOCK_M": 16,
                    "KERNEL_TYPE": 0,
                }
            )
    elif size_n == 12288 and size_k == 2560 and size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 32,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n == 2560 and size_k == 4096 and size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 32,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 18432 and size_k == 2560 and size_m <= 16:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 64,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 18432 and size_k == 2560 and size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 64,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n == 2560 and size_k == 9216 and size_m <= 7:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 4,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 2560 and size_k == 9216 and size_m <= 16:
        config.update(
            {
                "NUM_K_WARPS": 1,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 4,
                "SPLITK": 8,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 2560 and size_k == 9216 and size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 32,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 10240 and size_k == 2560 and size_m <= 9:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 10240 and size_k == 2560 and size_m <= 16:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 10240 and size_k == 2560 and size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n == 9216 and size_k == 2048 and 2 <= size_m <= 8:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 9216 and size_k == 2048 and size_m == 9:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 9216 and size_k == 2048 and 10 <= size_m <= 16:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n <= 2048 and size_k >= 4096 and 10 <= size_m <= 16:
        config["KERNEL_TYPE"] = 2
    elif size_n > 2048 and size_k <= 2048 and size_m == 1:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 12288 and size_k == 2048 and size_m == 4:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n > 2048 and size_k <= 2048 and 2 <= size_m <= 11:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 32,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n > 2048 and size_k <= 2048 and 12 <= size_m <= 16:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n == 9216 and size_k == 2048 and 17 <= size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 64,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n == 12288 and size_k == 2048 and 17 <= size_m <= 18:
        config.update(
            {
                "NUM_K_WARPS": 1,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n == 12288 and size_k == 2048 and 19 <= size_m <= 28:
        config.update(
            {
                "NUM_K_WARPS": 1,
                "N_PER_WARP": 32,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n == 12288 and size_k == 2048 and 29 <= size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 1,
                "N_PER_WARP": 64,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 4,
                "SPLITK": 1,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n == 2048 and size_k == 4096 and 17 <= size_m <= 32:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 32,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_m <= 16:
        config["SPLITK"] = 8
    elif size_n > 2048 and size_k <= 2048 and size_m == 32:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 64,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "BLOCK_M": 16,
                "KERNEL_TYPE": 0,
            }
        )
    elif size_n > 2048 and size_k <= 2048 and size_m == 64:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 64,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n <= 2048 and size_k >= 4096 and size_m == 32:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 32,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
            }
        )
    elif size_n <= 2048 and size_k >= 4096 and 64 <= size_m < 128:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 64,
                "K_PER_WARP": 64,
                "NUM_N_WARPS": 2,
                "SPLITK": 1,
            }
        )
    elif size_n <= 2048 and size_k >= 4096 and size_m == 128:
        config.update(
            {
                "NUM_K_WARPS": 4,
                "N_PER_WARP": 64,
                "K_PER_WARP": 128,
                "NUM_N_WARPS": 2,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n <= 2048 and size_k >= 4096 and 128 < size_m <= 256:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 64,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_n > 2048 and size_k <= 2048 and size_m in (32, 96, 128):
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 64,
                "K_PER_WARP": 256,
                "NUM_N_WARPS": 2,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    elif size_m >= 256:
        config.update(
            {
                "NUM_K_WARPS": 2,
                "N_PER_WARP": 64,
                "K_PER_WARP": 128 if size_n <= 2048 else 256,
                "NUM_N_WARPS": 2,
                "BLOCK_M": 32,
                "KERNEL_TYPE": 1,
            }
        )
    if (
        size_m >= 512
        or (size_m >= 256 and size_n <= 2048 and size_k >= 4096)
        or (size_m >= 256 and size_n > 2048 and size_k <= 2048)
        or (
            _w8a16_cache_can_hold_dequant(size_n, size_k)
            and size_m > 16
            and not (
                17 <= size_m <= 32
                and (
                    (size_n == 12288 and size_k == 2048)
                    or (size_n == 2048 and size_k == 4096)
                    or (size_n == 9216 and size_k == 2048)
                )
            )
            and (
                (size_n <= 2048 and size_k >= 4096)
                or (size_n > 2048 and size_k <= 2048)
            )
        )
    ):
        config["KERNEL_TYPE"] = 2
    return _clamp_w8a16_splitk(config, size_k)


def _cfg(config: Optional[Dict[str, Any]], key: str) -> int:
    if config is None:
        return int(_DEFAULT_CONFIG[key])
    return int(config.get(key, _DEFAULT_CONFIG[key]))


def w8a16_marlin_weight(weight: torch.Tensor) -> torch.Tensor:
    """Shuffle a 2-D int8 W8A16 weight into the marlin load layout.

    The returned tensor keeps the same ``[N, K]`` shape and int8 storage size.
    Only the byte order changes, matching aiter.ops.shuffle.w8a16_marlin_weight_1.
    """
    if weight.dtype != torch.int8:
        raise TypeError(f"weight must be int8, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [N, K], got {tuple(weight.shape)}")
    n, k = weight.shape
    if n % 16 != 0:
        raise ValueError(f"N must be divisible by 16 for marlin shuffle, got {n}")
    if k % 128 != 0:
        raise ValueError(f"K must be divisible by 128 for marlin shuffle, got {k}")

    w = weight.contiguous().unsqueeze(0)
    w_u32 = w.view(-1).view(torch.uint32)
    w_u32 = w_u32.view(1, n // 16, 16, k // 32, 8)
    w_u32 = w_u32.transpose(2, 3).contiguous()
    w_u32 = w_u32.view(1, n // 16, k // 128, 4, 16, 8)
    w_u32 = w_u32.transpose(1, 2).contiguous()
    return w_u32.view(-1).view(torch.uint8).view(1, n, k).squeeze(0).view(torch.int8)


def w8a16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """W8A16 linear: ``x @ (weight.float() * scale[:, None]).T + bias``.

    ``x`` may be 2-D or batched with the last dimension as K. ``weight`` is
    contiguous int8 with shape ``[N, K]``. ``scale`` is fp16 or bf16 and shape
    ``[N]`` or ``[N, 1]``. fp16 and bf16 input/scale are supported directly by
    the selected backend.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"x must be float16 or bfloat16, got {x.dtype}")
    if weight.dtype != torch.int8:
        raise TypeError(f"weight must be int8, got {weight.dtype}")
    if scale.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"scale must be float16 or bfloat16, got {scale.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    scale_owner = scale
    if not scale.is_contiguous():
        scale = scale.contiguous()

    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    size_m, size_k = x_2d.shape
    size_n = weight.shape[0]
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [N, K], got {tuple(weight.shape)}")
    if weight.shape[1] != size_k:
        raise ValueError(f"K mismatch: x.K={size_k}, weight.K={weight.shape[1]}")

    scale_1d = scale.view(size_n)
    if config is None:
        config = get_w8a16_linear_config(size_m, size_n, size_k)
    else:
        config = _clamp_w8a16_splitk(dict(config), size_k)
    kernel_type = int(config.get("KERNEL_TYPE", 0))
    weight_layout = _cfg(config, "WEIGHT_LAYOUT")
    if kernel_type == 0 and weight_layout != 0:
        raise ValueError("decode KERNEL_TYPE=0 uses raw/non-shuffled W8A16 weights; set WEIGHT_LAYOUT=0")
    if kernel_type == 0:
        config = {**config, "BLOCK_M": 16}
    elif kernel_type == 1 and int(config.get("BLOCK_M", 0)) <= 0:
        config = {**config, "BLOCK_M": 32}

    x_compute = x_2d

    if kernel_type == 2 and weight_layout == 0:
        # Large-M path: materialize W16 weights once, then delegate GEMM to
        # PyTorch/rocBLAS. The dequantized cache keeps repeated calls cheap.
        weight_w16 = _cached_w8a16_dequant_weight(weight, scale_owner, scale_1d, x_compute.dtype)
        out = torch.mm(x_compute, weight_w16.t())
        if bias is not None:
            out = out.add_(bias.view(size_n).to(out.dtype))
        return out.view(*orig_shape[:-1], size_n)

    args = (
        x_compute,
        weight,
        scale_1d,
        _cfg(config, "NUM_K_WARPS"),
        _cfg(config, "N_PER_WARP"),
        _cfg(config, "K_PER_WARP"),
        _cfg(config, "NUM_N_WARPS"),
        _cfg(config, "SPLITK"),
        _cfg(config, "BLOCK_M"),
        weight_layout,
    )
    if bias is None:
        out = op.w8a16_linear(*args)
    else:
        if bias.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"bias must be float16 or bfloat16, got {bias.dtype}")
        bias_fp16 = bias.view(size_n).to(x_compute.dtype)
        out = op.w8a16_linear_bias(
            x_compute,
            weight,
            scale_1d,
            bias_fp16,
            _cfg(config, "NUM_K_WARPS"),
            _cfg(config, "N_PER_WARP"),
            _cfg(config, "K_PER_WARP"),
            _cfg(config, "NUM_N_WARPS"),
            _cfg(config, "SPLITK"),
            _cfg(config, "BLOCK_M"),
            _cfg(config, "WEIGHT_LAYOUT"),
        )

    return out.view(*orig_shape[:-1], size_n)
