"""
DeepGEMM Grouped W8A8 GEMM Configuration Module

用于获取 m_grouped_w8a8_gemm_nt_masked kernel 的最优配置
参考 lightop/lightop/config.py 设计
"""

import functools
import os
import json
import torch
from typing import Any, Optional


@functools.lru_cache
def get_deepgemm_groupgemm_w8a8_config(
    E: int,
    M: int,
    N: int,
    K: int,
    device_name: str,
    num_cus: int,
    dtype: Optional[str] = "bfloat16",
) -> tuple[dict[str, Any], bool]:
    """
    获取 m_grouped_w8a8_gemm_nt_masked kernel 的优化配置
    
    Args:
        E: 专家数量 (num_experts)
        M: 输入 token 数量 (expected_m_per_group)
        N: 输出维度
        K: 输入维度
        device_name: GPU 设备名称 (如 gfx942)
        num_cus: GPU CU 数量
        dtype: 数据类型
        
    Returns:
        config: 配置字典, 包含 MODE 字段
        status: 是否找到有效配置
    """
    
    # 配置文件路径
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    config_file_name = f"DEEPGEMM_GROUPGEMM_W8A8_{N}_{K}_{gfx_version.upper()}_CU{num_cus}.json"
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", config_file_name)
    
    config = {}
    status = False
    
    try:
        with open(config_file_path, 'r') as file:
            config_infos = json.load(file)
            key = f"{N}_{K}"
            if key in config_infos:
                config_info = config_infos[key]
                int_keys = [int(k) for k in config_info.keys()]
                
                # 找到 >= M 的最小 key
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                config = config_info.get(f"{num_token}", {})
                
                # 如果没有找到，使用最接近的 key
                if not config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    config = config_info.get(f"{closest_token}", {})
                    
                if config:
                    status = True
    except FileNotFoundError:
        pass
    
    # 默认配置
    if not config:
        config = {"MODE": 0}  # 使用默认 mode
        
    return config, status


def _lookup_size_config(config_infos: dict[str, Any], key: str, m: int) -> tuple[dict[str, Any], bool]:
    if key not in config_infos:
        return {}, False

    config_info = config_infos[key]
    int_keys = []
    for raw_key in config_info.keys():
        try:
            int_keys.append(int(raw_key))
        except (TypeError, ValueError):
            continue
    if not int_keys:
        return {}, False

    # Prefer the smallest tuned M that covers the requested size; otherwise use
    # the closest tuned point so large shapes can still reuse the nearest result.
    greater_numbers = [num for num in int_keys if num >= m]
    selected_m = min(greater_numbers) if greater_numbers else min(
        int_keys, key=lambda token: abs(token - m)
    )
    config = config_info.get(str(selected_m), {})
    return config, bool(config)


@functools.lru_cache
def get_deepgemm_contiguous_fp8_config(
    M: int,
    N: int,
    K: int,
    device_name: str,
    num_cus: int,
    dtype: Optional[str] = "bfloat16",
) -> tuple[dict[str, Any], bool]:
    """
    获取 contiguous FP8 grouped GEMM 的 size-aware 配置。

    JSON 格式与 masked tune 配置保持一致:
    {
        "N_K": {
            "M": {"MODE": 64, "N_LOOP_NUM": 2}
        }
    }
    可以通过 DEEPGEMM_CONTIGUOUS_FP8_CONFIG 指向自定义配置文件。
    """
    del dtype

    config_file_path = os.environ.get("DEEPGEMM_CONTIGUOUS_FP8_CONFIG")
    if not config_file_path:
        gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        config_file_name = (
            f"DEEPGEMM_CONTIGUOUS_FP8_{N}_{K}_{gfx_version.upper()}_CU{num_cus}.json"
        )
        config_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", config_file_name
        )

    try:
        with open(config_file_path, "r") as file:
            config_infos = json.load(file)
    except FileNotFoundError:
        return {}, False

    key = f"{N}_{K}"
    config, status = _lookup_size_config(config_infos, key, M)
    if status:
        return config, True

    # Also allow a flatter file layout keyed only by M for quick experiments.
    config, status = _lookup_size_config(config_infos, "default", M)
    return config, status


def save_deepgemm_contiguous_fp8_config(
    config: dict[str, Any],
    M: int,
    N: int,
    K: int,
    num_cus: int,
) -> None:
    """保存 contiguous FP8 grouped GEMM tune 结果。"""
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    config_file_name = f"DEEPGEMM_CONTIGUOUS_FP8_{N}_{K}_{gfx_version.upper()}_CU{num_cus}.json"
    configs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    os.makedirs(configs_dir, exist_ok=True)
    config_file_path = os.path.join(configs_dir, config_file_name)

    try:
        with open(config_file_path, "r") as file:
            config_infos = json.load(file)
    except FileNotFoundError:
        config_infos = {}

    key = f"{N}_{K}"
    config_infos.setdefault(key, {})
    config_infos[key][str(M)] = config

    with open(config_file_path, "w") as file:
        json.dump(config_infos, file, indent=4)

    print(f"Config saved to {config_file_path}")


def get_all_groupgemm_modes() -> list[int]:
    """
    获取所有可用的 groupgemm mode 列表
    
    对应 m_grouped_gemm_config.h 中 kernel_maps_groupgemm_decode 的所有 key
    """
    # CUDA 版本 modes (< 1000)
    cuda_modes = [
        # BLOCK_SIZE_K = 64
        0, 1, 2, 3, 4, 5,
        # BLOCK_SIZE_K = 128
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        # BLOCK_SIZE_K = 256
        20, 21, 22, 23, 24, 25, 26, 27, 28,
        # 4 STAGES
        30, 31, 32, 33, 34, 35,
        # 更大的 BLOCK_N (256)
        40, 41, 42, 43, 44, 45,
        # BLOCK_K = 512
        50, 51, 52, 53, 54, 55,
    ]
    
    # ASM 版本 modes (>= 1000)
    asm_modes = [1000, 1001, 1002]
    
    return cuda_modes + asm_modes


def get_mode_block_m(mode: int) -> int:
    """
    根据 mode 获取对应的 BLOCK_SIZE_M
    
    当前所有 CUDA kernel 都使用 BLOCK_SIZE_M = 16
    ASM kernel 使用不同的 block 配置
    """
    if mode >= 1000:
        # ASM modes
        if mode == 1000:
            return 256
        elif mode == 1001:
            return 128
        elif mode == 1002:
            return 64
    # CUDA modes 都是 BLOCK_SIZE_M = 16
    return 16


def save_groupgemm_config(
    config: dict[str, Any],
    M: int,
    N: int,
    K: int,
    num_cus: int,
) -> None:
    """
    保存 groupgemm 配置到 JSON 文件
    
    Args:
        config: 配置字典
        M: expected_m_per_group
        N: 输出维度
        K: 输入维度
        num_cus: GPU CU 数量
    """
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    config_file_name = f"DEEPGEMM_GROUPGEMM_W8A8_{N}_{K}_{gfx_version.upper()}_CU{num_cus}.json"
    
    # 确保 configs 目录存在
    configs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    os.makedirs(configs_dir, exist_ok=True)
    
    config_file_path = os.path.join(configs_dir, config_file_name)
    
    # 读取现有配置或创建新配置
    try:
        with open(config_file_path, 'r') as file:
            config_infos = json.load(file)
    except FileNotFoundError:
        config_infos = {}
    
    key = f"{N}_{K}"
    if key not in config_infos:
        config_infos[key] = {}
    
    config_infos[key][f"{M}"] = config
    
    # 保存配置
    with open(config_file_path, 'w') as file:
        json.dump(config_infos, file, indent=4)
    
    print(f"Config saved to {config_file_path}")
