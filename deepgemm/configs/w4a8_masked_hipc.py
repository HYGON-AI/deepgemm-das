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


def _current_num_cus() -> int | None:
    try:
        override = int(os.environ.get("DEEPGEMM_GPU_CUS", "0"))
        if override > 0:
            return override
    except ValueError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    except Exception:
        pass
    return None


def _config_file_candidates(
    arch: str, n: int, k: int, experts: int | None = None, num_cus: int | None = None
) -> tuple[str, ...]:
    override = os.environ.get("DEEPGEMM_W4A8_MASKED_HIPC_CONFIG")
    if override:
        return (override,)
    arch_config_dir = os.path.join(_configs_dir(), arch, W4A8_HIPC_MASKED_CONFIG_DIR)
    expert_candidates = ()
    if experts is not None:
        expert_candidates = (
            os.path.join(arch_config_dir, f"DEEPGEMM_W4A8_MASKED_HIPC_E{experts}_N{n}_K{k}.json"),
        )
    arch_candidates = (
        *expert_candidates,
        os.path.join(arch_config_dir, f"DEEPGEMM_W4A8_MASKED_HIPC_N{n}_K{k}.json"),
        os.path.join(arch_config_dir, "DEEPGEMM_W4A8_MASKED_HIPC_DEFAULT.json"),
    )
    cu_candidates = ()
    if num_cus is not None and num_cus > 0:
        cu_candidates = tuple(f"{os.path.splitext(path)[0]}_CU{num_cus}.json" for path in arch_candidates)
    return (
        *cu_candidates,
        *arch_candidates,
        os.path.join(_configs_dir(), f"DEEPGEMM_W4A8_MASKED_HIPC_N{n}_K{k}.json"),
        os.path.join(_configs_dir(), "DEEPGEMM_W4A8_MASKED_HIPC_DEFAULT.json"),
    )


@lru_cache(maxsize=256)
def _load_w4a8_masked_hipc_config(
    arch: str, n: int, k: int, experts: int | None = None, num_cus: int | None = None
) -> dict[str, Any]:
    for path in _config_file_candidates(arch, n, k, experts, num_cus):
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
    selected_m = min(int_keys, key=lambda m: (abs(m - expected_m), -m))
    config = raw_configs.get(str(selected_m), {})
    return config if isinstance(config, dict) else {}


def select_w4a8_hipc_masked_config(
    expected_m: int, n: int, k: int, experts: int | None = None
) -> dict[str, Any]:
    arch = _current_gfx_arch()
    n, k = int(n), int(k)
    experts = None if experts is None else int(experts)
    num_cus = _current_num_cus() if arch == "gfx938" else None
    config_infos = _load_w4a8_masked_hipc_config(
        arch, n, k, experts, num_cus
    )
    return _lookup_expected_m_config(config_infos, int(expected_m))


def select_w4a8_hipc_masked_mode(expected_m: int, n: int, k: int, experts: int | None = None) -> int:
    config = select_w4a8_hipc_masked_config(expected_m, n, k, experts)
    return int(config.get("MODE", DEFAULT_W4A8_HIPC_MASKED_MODE))
