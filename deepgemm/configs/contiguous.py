from typing import Any, Dict, Optional, Tuple

DEFAULT_W4A8_CONTIG_CONFIG = {"MODE": 1006}


def resolve_contiguous_hipc_config(
    config: Optional[Dict[str, Any]],
) -> Tuple[int, int, int]:
    if config is None:
        return 128, 256, 4
    block_m = int(config.get("BLOCK_M", config.get("MODE", 128)))
    block_n = int(config.get("BLOCK_N", 256))
    n_loop_num = int(config.get("N_LOOP_NUM", 4))
    return block_m, block_n, n_loop_num
