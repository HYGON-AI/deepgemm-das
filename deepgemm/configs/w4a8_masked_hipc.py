import json
import os
from functools import lru_cache
from typing import Any


DEFAULT_W4A8_HIPC_MASKED_MODE = 1644
W4A8_HIPC_MASKED_CONFIG_DIR = "w4a8_masked_hipc"


def _configs_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def _current_gfx_arch() -> str:
    override = os.environ.get("DEEPGEMM_W4A8_MASKED_HIPC_ARCH")
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


def _config_file_candidates(arch: str, n: int, k: int) -> tuple[str, ...]:
    override = os.environ.get("DEEPGEMM_W4A8_MASKED_HIPC_CONFIG")
    if override:
        return (override,)
    arch_config_dir = os.path.join(_configs_dir(), arch, W4A8_HIPC_MASKED_CONFIG_DIR)
    return (
        os.path.join(arch_config_dir, f"DEEPGEMM_W4A8_MASKED_HIPC_N{n}_K{k}.json"),
        os.path.join(arch_config_dir, "DEEPGEMM_W4A8_MASKED_HIPC_DEFAULT.json"),
        os.path.join(_configs_dir(), f"DEEPGEMM_W4A8_MASKED_HIPC_N{n}_K{k}.json"),
        os.path.join(_configs_dir(), "DEEPGEMM_W4A8_MASKED_HIPC_DEFAULT.json"),
    )


@lru_cache(maxsize=256)
def _load_w4a8_masked_hipc_config(arch: str, n: int, k: int) -> dict[str, Any]:
    for path in _config_file_candidates(arch, n, k):
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


def select_w4a8_hipc_masked_mode(expected_m: int, n: int, k: int) -> int:
    config_infos = _load_w4a8_masked_hipc_config(_current_gfx_arch(), int(n), int(k))
    config = _lookup_expected_m_config(config_infos, int(expected_m))
    return int(config.get("MODE", DEFAULT_W4A8_HIPC_MASKED_MODE))
