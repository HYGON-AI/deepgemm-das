# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""JSON-based selection for gfx938 contiguous pair32_v2 FP8 kernels."""
import json
import os
from functools import lru_cache
from pathlib import Path


CONFIG_PREFIX = "DEEPGEMM_W8A8_FP8_CONTIGUOUS"
CONFIG_LAYOUT = Path("w8a8_contiguous/hipc_pair32_v2")
MODE_TILES = {1000: (256, 256), 1001: (128, 256), 1002: (128, 384)}


def _configs_dir():
    return Path(__file__).resolve().parent


def _config_file_candidates(arch, num_cus, n, k, experts):
    override = os.environ.get("DEEPGEMM_W8A8_FP8_CONTIGUOUS_CONFIG")
    if override:
        return (Path(override),)
    directory = _configs_dir() / arch / CONFIG_LAYOUT
    stems = []
    if experts is not None:
        stems.append(f"{CONFIG_PREFIX}_E{experts}_N{n}_K{k}")
    stems.append(f"{CONFIG_PREFIX}_N{n}_K{k}")
    # BN384-eligible shapes may vote among all three modes. Other shapes
    # use a separate default that never includes the BN384 candidate.
    group = "BN384" if n % 384 == 0 else "BN256"
    stems.append(f"{CONFIG_PREFIX}_DEFAULT_{group}")
    return tuple(directory / f"{stem}{suffix}.json"
                 for stem in stems
                 for suffix in (f"_CU{num_cus}", ""))


@lru_cache(maxsize=256)
def _load_config(path):
    try:
        with open(path) as file:
            data = json.load(file)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    configs = data.get("configs", data)
    return configs if isinstance(configs, dict) else {}


def _lookup_m_mode(configs, size_m):
    keys = []
    for key in configs:
        try:
            m = int(key)
        except (ValueError, TypeError):
            continue
        if m >= size_m:
            keys.append((m, key))
    if not keys:
        return None
    item = configs[min(keys, key=lambda pair: pair[0])[1]]
    try:
        return int(item["MODE"])
    except (KeyError, TypeError, ValueError):
        return None


def mode_supports_shape(mode, m, n, k):
    if mode not in MODE_TILES:
        return False
    bm, bn = MODE_TILES[mode]
    return m > 0 and n > 0 and k > 0 and m % bm == n % bn == k % 128 == 0


def select_w8a8_contiguous_mode(size_m, n, k, experts=None, device=None):
    """Look up total M: E/N/K -> N/K -> N-alignment DEFAULT -> safe fallback.

    Each level tries the current CU-specific table before the unsuffixed
    table. Missing M coverage or an incompatible mode falls through to the
    next table. Expert segments must align to the selected kernel's BM.
    """
    import torch

    m, n, k = int(size_m), int(n), int(k)
    if not mode_supports_shape(1001, m, n, k):
        raise ValueError("pair32_v2 requires positive M%128=0, N%256=0, K%128=0")
    properties = torch.cuda.get_device_properties(device)
    arch = properties.gcnArchName.split(":", 1)[0]
    if arch != "gfx938":
        raise ValueError(f"contiguous pair32_v2 requires gfx938, got {arch}")
    for path in _config_file_candidates(arch, properties.multi_processor_count,
                                        n, k, None if experts is None else int(experts)):
        mode = _lookup_m_mode(_load_config(path), m)
        if mode_supports_shape(mode, m, n, k):
            return mode
    return 1000 if m % 256 == 0 else 1001
