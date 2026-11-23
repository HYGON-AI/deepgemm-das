# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""DeepGEMM ``mega_mhc`` 算子的 Python 包装。

保留官方 DeepGEMM 的参数、可选输出以及返回 ``None`` 的行为；另提供显式 Fn 预打包扩展
（``csrc/apis/mega_mhc.hpp``，commit ``5081f8513af57e61696de4f4f04b0e707ffb503e``），
Split-K、workspace 与 scratch 由内部实现选择。
"""

from __future__ import annotations

import torch

from . import op


def mega_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    shifted_prev_mix,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    new_residual: torch.Tensor,
    new_prev_mix,
    new_post_mix: torch.Tensor,
    new_comb_res_mix: torch.Tensor,
    y_bf16=None,
    y_fp8=None,
    y_gemm_sf=None,
    y_routed_sf=None,
    y_shared_sf=None,
    shared_sf_block_m: int = 0,
    fn_packed=None,
):
    """执行 shifted 或 normal 的 mHC，输出下一组 mix 与 RMSNorm 结果。

    参数：
        x：``[T, H]`` BF16，子层输出。
        residual：``[T, hc_mult, H]`` BF16，多路残差流。
        shifted_prev_mix：``[T, hc_mult, 1]`` FP32，本接缝的 pre 门控；
            normal 模式传 ``None``（normal 用的是本次调用**生成**的 pre 门控）。
        post_mix：``[T, hc_mult, 1]`` FP32。
        comb_res_mix：``[T, hc_mult, hc_mult]`` FP32，输入路 i -> 输出路 j。
        fn：``[hc_mult*(hc_mult+2), hc_mult*H]`` FP32，门控投影。
        mix_scales：``[3]`` FP32；``mix_bases``：``[O]`` FP32。
        rmsnorm_weight：``[H]`` BF16。
        new_residual / new_post_mix / new_comb_res_mix / new_prev_mix：输出，
            由调用方预先分配。
        y_bf16：``[T, H]`` BF16 输出，或 ``None``。
        y_fp8：``[T, H]`` E4M3FN 输出或 ``None``；必须同时给对应的 SF。
        y_gemm_sf：``[T, H/128]`` int32，列主序，或 ``None``。
        y_routed_sf / y_shared_sf：``[T, H/128]`` int32，分别为行主序 / 共享布局；
            ``y_shared_sf`` 要求 ``shared_sf_block_m > 0``。
        fn_packed：可选，来自 :func:`mega_mhc_prepare_fn` 的缓冲。给了就跳过
            每次调用的权重打包。该缓冲与那批权重值绑定；``fn`` 变了要重新 prepare。

    返回：
        ``None``。所有结果都写进调用方传入的张量。
    """
    return op.mega_mhc(
        x,
        residual,
        shifted_prev_mix,
        post_mix,
        comb_res_mix,
        fn,
        mix_scales,
        mix_bases,
        hc_mult,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        num_sinkhorn_iters,
        rmsnorm_weight,
        rmsnorm_eps,
        rmsnorm_scale,
        new_residual,
        new_prev_mix,
        new_post_mix,
        new_comb_res_mix,
        y_bf16=y_bf16,
        y_fp8=y_fp8,
        y_gemm_sf=y_gemm_sf,
        y_routed_sf=y_routed_sf,
        y_shared_sf=y_shared_sf,
        shared_sf_block_m=shared_sf_block_m,
        fn_packed=fn_packed,
    )


def mega_mhc_prepare_fn(fn: torch.Tensor) -> torch.Tensor:
    """把 ``fn`` 打包一次，供 :func:`mega_mhc` 的 ``fn_packed=`` 复用。

    失效是显式的：返回的缓冲只匹配这批权重值。``fn`` 一旦变化就要再调用一次；
    算子自身从不做缓存。
    """
    return op.mega_mhc_prepare_fn(fn)
