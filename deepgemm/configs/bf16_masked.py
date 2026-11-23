# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import json
import os
from functools import lru_cache
from typing import Any


DEFAULT_BF16_MASKED_MODE = 2000
DEFAULT_BF16_MASKED_MODE_BY_LAYOUT: dict[str, int] = {
    "asm": 1000,
    "hipc": 2000,
}
BF16_MASKED_CONFIG_DIR = "bf16_masked"


def _configs_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def _current_gfx_arch() -> str:
    override = os.environ.get("DEEPGEMM_W16A16_BF16_MASKED_ARCH")
    if override:
        return override.lower()

    try:
        import torch

        if torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
            arch = arch.split(":", 1)[0].lower()
            if arch:
                return arch
    except Exception:
        pass
    return "unknown"


def _config_file_candidates(
    arch: str, layout_dir: str, n: int, k: int, experts: int | None
) -> tuple[str, ...]:
    override = os.environ.get("DEEPGEMM_W16A16_BF16_MASKED_CONFIG")
    if override:
        return (override,)
    arch_config_dir = os.path.join(_configs_dir(), arch, BF16_MASKED_CONFIG_DIR)
    if layout_dir and os.path.isdir(os.path.join(arch_config_dir, layout_dir)):
        arch_config_dir = os.path.join(arch_config_dir, layout_dir)
    e_candidates = ()
    if experts is not None:
        e_candidates = (
            os.path.join(arch_config_dir, f"DEEPGEMM_W16A16_BF16_MASKED_E{experts}_N{n}_K{k}.json"),
            os.path.join(arch_config_dir, f"DEEPGEMM_W16A16_BF16_MASKED_E{experts}.json"),
            os.path.join(_configs_dir(), f"DEEPGEMM_W16A16_BF16_MASKED_E{experts}_N{n}_K{k}.json"),
            os.path.join(_configs_dir(), f"DEEPGEMM_W16A16_BF16_MASKED_E{experts}.json"),
        )
    return (
        *e_candidates,
        os.path.join(arch_config_dir, f"DEEPGEMM_W16A16_BF16_MASKED_N{n}_K{k}.json"),
        os.path.join(arch_config_dir, "DEEPGEMM_W16A16_BF16_MASKED_DEFAULT.json"),
        os.path.join(_configs_dir(), f"DEEPGEMM_W16A16_BF16_MASKED_N{n}_K{k}.json"),
        os.path.join(_configs_dir(), "DEEPGEMM_W16A16_BF16_MASKED_DEFAULT.json"),
    )


@lru_cache(maxsize=256)
def _load_bf16_masked_config(
    arch: str, layout_dir: str, n: int, k: int, experts: int | None
) -> dict[str, Any]:
    for path in _config_file_candidates(arch, layout_dir, n, k, experts):
        try:
            with open(path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            continue
    return {}


def _lookup_expected_m_config(
    config_infos: dict[str, Any],
    expected_m: int,
) -> dict[str, Any]:
    raw_configs = config_infos.get("configs", config_infos)
    int_keys = []
    for raw_key in raw_configs.keys():
        try:
            int_keys.append(int(raw_key))
        except (TypeError, ValueError):
            continue
    if not int_keys:
        return {}
    selected_m = min((m for m in int_keys if m >= expected_m), default=None)
    if selected_m is None:
        return {}
    config = raw_configs.get(str(selected_m), {})
    return config if isinstance(config, dict) else {}


def select_bf16_masked_mode(
    expected_m: int, n: int, k: int, experts: int = None, layout_dir: str = ""
) -> int:
    config_infos = _load_bf16_masked_config(
        _current_gfx_arch(),
        layout_dir or "",
        int(n),
        int(k),
        None if experts is None else int(experts),
    )
    config = _lookup_expected_m_config(config_infos, int(expected_m))
    default_mode = DEFAULT_BF16_MASKED_MODE_BY_LAYOUT.get(layout_dir, DEFAULT_BF16_MASKED_MODE)
    return int(config.get("MODE", default_mode))
