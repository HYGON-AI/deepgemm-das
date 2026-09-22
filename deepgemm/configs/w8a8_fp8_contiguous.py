from typing import Optional, Tuple


MARLIN_PREFILL_DOWN_BLOCK_M = 32
MARLIN_CONTIG_BLOCK_N = 128
DEFAULT_FP8_CONTIG_CONFIG = {"MODE": 64, "N_LOOP_NUM": 2}
LARGE_M_FP8_CONTIG_CONFIG = {"MODE": 64, "N_LOOP_NUM": 4}
FP8_CONTIG_NLOOP4_M_THRESHOLD = 8192 + 4096
VALID_FP8_CONTIG_BLOCK_M = (32, 64, 128)

FP8_CONTIG_HIPC_OPTIMIZED = (64, 256, 2)
FP8_CONTIG_HIPC_V75 = (128, 256, 4)
FP8_CONTIG_HIPC_DEFAULT_SHAPE = (16, 2048, 4096)
FP8_CONTIG_HIPC_M_THRESHOLDS = {
    # Tuned on gfx938 for 10 random token layouts.
    # key: (E, N, K), value: padded-M threshold for bm64/nloop2 -> bm128/nloop4.
    FP8_CONTIG_HIPC_DEFAULT_SHAPE: 6400,
    # Tuned on BW1000B for 3 repeat rounds over random token layouts.
    (24, 1536, 4096): 12288,
}
FP8_CONTIG_HIPC_SMALL_M_THRESHOLD = FP8_CONTIG_HIPC_M_THRESHOLDS[
    FP8_CONTIG_HIPC_DEFAULT_SHAPE
]


def select_fp8_contig_hipc_config(
    size_m: int,
    size_n: Optional[int] = None,
    size_k: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> Tuple[int, int, int]:
    # Keep runtime overhead tiny: one shape-table lookup and one padded-M comparison.
    threshold = FP8_CONTIG_HIPC_SMALL_M_THRESHOLD
    if num_groups is not None and size_n is not None and size_k is not None:
        shape_key = (int(num_groups), int(size_n), int(size_k))
        threshold = FP8_CONTIG_HIPC_M_THRESHOLDS.get(shape_key, threshold)

    if int(size_m) <= threshold:
        return FP8_CONTIG_HIPC_OPTIMIZED
    return FP8_CONTIG_HIPC_V75
