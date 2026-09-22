from typing import Any, Dict, Optional, Tuple

from .w8a8_fp8_contiguous import (
    DEFAULT_FP8_CONTIG_CONFIG,
    FP8_CONTIG_NLOOP4_M_THRESHOLD,
    LARGE_M_FP8_CONTIG_CONFIG,
    MARLIN_CONTIG_BLOCK_N,
    VALID_FP8_CONTIG_BLOCK_M,
    select_fp8_contig_hipc_config,
)


def resolve_contiguous_hipc_config(
    config: Optional[Dict[str, Any]],
) -> Tuple[int, int, int]:
    if config is None:
        return 128, 256, 4
    block_m = int(config.get("BLOCK_M", config.get("MODE", 128)))
    block_n = int(config.get("BLOCK_N", 256))
    n_loop_num = int(config.get("N_LOOP_NUM", 4))
    return block_m, block_n, n_loop_num


def resolve_fp8_contiguous_n_loop_num(size_n: int, requested: int) -> int:
    if requested not in (1, 2, 4):
        if requested != 0:
            raise ValueError(f"N_LOOP_NUM must be 0 (auto), 1, 2, or 4; got {requested}")
    block_n = MARLIN_CONTIG_BLOCK_N
    if requested == 0:
        for candidate in (4, 2, 1):
            if size_n % (block_n * candidate) == 0:
                return candidate
    elif size_n % (block_n * requested) == 0:
        return requested
    for candidate in (4, 2, 1):
        if candidate <= requested and size_n % (block_n * candidate) == 0:
            return candidate
    for candidate in (1, 2, 4):
        if size_n % (block_n * candidate) == 0:
            return candidate
    raise ValueError(
        f"size_n={size_n} is not divisible by any of "
        f"{block_n * 4}, {block_n * 2}, or {block_n}"
    )


def normalize_fp8_contiguous_mode(mode: Any) -> int:
    if isinstance(mode, str):
        if mode.lower() != "auto":
            raise ValueError(f"MODE must be 0/auto, 1000/32, 64, or 128; got {mode}")
        return 0
    return int(mode)


def resolve_fp8_contiguous_block_m(size_m: int, requested: Any) -> int:
    mode = normalize_fp8_contiguous_mode(requested)
    if mode == 0:
        if size_m % 64 == 0:
            return 64
        if size_m % 32 == 0:
            return 32
        raise ValueError(f"size_m={size_m} is not divisible by 32/64/128")
    if mode == 1000:
        return 32
    if mode in VALID_FP8_CONTIG_BLOCK_M:
        return mode
    raise ValueError(f"MODE must be 0/auto, 1000/32, 64, or 128; got {requested}")


def get_m_grouped_fp8_contiguous_config(size_m: int, size_n: int) -> Dict[str, int]:
    size_m = int(size_m)
    size_n = int(size_n)
    config = (
        LARGE_M_FP8_CONTIG_CONFIG
        if size_m > FP8_CONTIG_NLOOP4_M_THRESHOLD
        else DEFAULT_FP8_CONTIG_CONFIG
    )
    block_m = resolve_fp8_contiguous_block_m(
        size_m, config.get("MODE", DEFAULT_FP8_CONTIG_CONFIG["MODE"])
    )
    n_loop_num = resolve_fp8_contiguous_n_loop_num(
        size_n, int(config.get("N_LOOP_NUM", DEFAULT_FP8_CONTIG_CONFIG["N_LOOP_NUM"]))
    )
    return {"MODE": block_m, "N_LOOP_NUM": n_loop_num}
