# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import random
from typing import Type

import torch
import triton
from deepgemm import m_grouped_w8a8_gemm_nt_masked
from deepgemm_marlin_quant import weight8bit_nt_kpack2_marlin2  # pyright: ignore[reportMissingImports]
from lmslim.layers.gemm.int8_utils import per_token_quant_int8


seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def _get_gfx_arch() -> str:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(props, "gcnArchName", props.name)
    return arch.split(":")[0].lower()


def _require_gfx936_or_gfx92a() -> None:
    arch = _get_gfx_arch()
    if arch not in ("gfx936", "gfx92a"):
        raise RuntimeError(f"This test is for gfx936/gfx92a masked W8A8 only, got {arch}")


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def generate_m_grouped_masked_w8a8(
    m: int,
    n: int,
    k: int,
    num_groups: int,
    expected_m_per_group: int,
    out_dtype: Type[torch.dtype] = torch.bfloat16,
    device: str = "cuda",
):
    hidden_states = (torch.randn((num_groups, m, k), device=device) * 5).to(
        dtype=out_dtype
    )
    weight = to_int8(torch.randn((num_groups, n, k), device=device)).contiguous()
    w_scale = torch.randn((num_groups, n, 1), device=device, dtype=torch.float32)
    d = torch.empty((num_groups, m, n), device=device, dtype=out_dtype)

    masked_m = torch.empty((num_groups,), device=device, dtype=torch.int32)
    for expert_id in range(num_groups):
        masked_m[expert_id] = min(
            m,
            int(expected_m_per_group * random.uniform(0.7, 3.3)),
        )

    return hidden_states, weight, w_scale, masked_m, d


def torch_gemm(
    q_a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: Type[torch.dtype],
) -> torch.Tensor:
    a = q_a.to(torch.float32)
    b = weight.to(torch.float32)

    assert a.shape[-1] == b.shape[-1], "Dimension mismatch"

    c = torch.bmm(a, b.transpose(1, 2))
    c = a_scale * c * w_scale.transpose(1, 2)
    return c.to(output_dtype)


def check_w8a8_precision(
    ref_d: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    valid = True
    for expert_id in range(masked_m.shape[0]):
        cur_valid_m = int(masked_m[expert_id].item())
        if cur_valid_m == 0:
            continue

        ref_valid = ref_d[expert_id, :cur_valid_m, :]
        d_valid = d[expert_id, :cur_valid_m, :]
        abs_err = (ref_valid.float() - d_valid.float()).abs()
        max_abs_err = abs_err.max().item()
        mean_abs_err = abs_err.mean().item()
        ref_abs = ref_valid.float().abs().clamp(min=1e-7)
        rel_err = abs_err / ref_abs
        max_rel_err = rel_err.max().item()
        mean_rel_err = rel_err.mean().item()
        print(
            f"  expert {expert_id}: valid_m={cur_valid_m}, "
            f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}, "
            f"max_rel_err={max_rel_err:.6f}, mean_rel_err={mean_rel_err:.6f}"
        )

        if torch.allclose(ref_valid, d_valid, rtol=rtol, atol=atol):
            continue

        max_idx = torch.argmax(abs_err)
        row = int((max_idx // abs_err.shape[1]).item())
        col = int((max_idx % abs_err.shape[1]).item())
        print(
            "  precision error:",
            f"expert={expert_id}",
            f"at=({row}, {col})",
            f"ref={ref_valid[row, col].item():.6f}",
            f"asm={d_valid[row, col].item():.6f}",
        )
        valid = False
        break

    if valid:
        print("W8A8 精度检查合格！！！")
    else:
        print("W8A8 精度检查不合格！！！")
    return valid


def test_m_grouped_gemm_masked_w8a8_gfx936(
    warmup_iters: int = 5,
    acc_check: bool = True,
) -> None:
    _require_gfx936_or_gfx92a()
    print(
        "[gfx936/gfx92a masked w8a8]",
        f"arch={_get_gfx_arch()}",
        "gemm=m_grouped_w8a8_gemm_nt_masked",
        "weight_reorder=weight8bit_nt_kpack2_marlin2_flat",
    )

    num_groups = 16
    m = 2048
    expected_m_per_group_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    n_list = [4096, 6144]
    k_list = [6144, 2048]
    out_dtype = torch.bfloat16

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(len(k_list)):
            n = n_list[i]
            k = k_list[i]
            print(
                "m:{} n:{} k:{} expected_m_per_group:{} (w8a8 masked gfx936/gfx92a)".format(
                    m, n, k, expected_m_per_group
                )
            )

            hidden_states, weight, w_scale, masked_m, d = (
                generate_m_grouped_masked_w8a8(
                    m=m,
                    n=n,
                    k=k,
                    num_groups=num_groups,
                    expected_m_per_group=expected_m_per_group,
                    out_dtype=out_dtype,
                )
            )
            print("masked_m:", masked_m)

            q_a, a_scale = per_token_quant_int8(hidden_states)
            q_a = q_a.contiguous()
            a_scale = a_scale.contiguous()
            w_scale = w_scale.contiguous()
            weight_marlin = (
                weight8bit_nt_kpack2_marlin2(weight)
                .contiguous()
                .view(num_groups, n // 16, k * 16)
                .contiguous()
            )

            a = [q_a, a_scale]
            b = [weight_marlin, w_scale]
            launch_expected_m_per_group = int(masked_m.max().item())

            for _ in range(warmup_iters):
                m_grouped_w8a8_gemm_nt_masked(
                    a,
                    b,
                    d,
                    masked_m,
                    launch_expected_m_per_group,
                )
            torch.cuda.synchronize()

            if acc_check:
                ref_d = torch_gemm(
                    q_a,
                    weight.to(torch.int32),
                    a_scale,
                    w_scale,
                    output_dtype=out_dtype,
                )
                ok = check_w8a8_precision(ref_d, d, masked_m)
                if not ok:
                    raise AssertionError("gfx936/gfx92a masked W8A8 precision check failed")

            elapsed_time = triton.testing.do_bench(
                lambda: m_grouped_w8a8_gemm_nt_masked(
                    a,
                    b,
                    d,
                    masked_m,
                    launch_expected_m_per_group,
                ),
                quantiles=None,
                return_mode="mean",
            ) * 1000

            valid_m = int(masked_m.sum().item())
            computes = 2 * valid_m * n * k
            tflops = computes / (elapsed_time / 1e6) / 1e12
            data_bytes = valid_m * k + k * n * num_groups + valid_m * n * 2
            bandwidth = data_bytes / (elapsed_time / 1e6) / 1e9
            print(
                "deepGemm masked W8A8 算子 - "
                f"valid_m{valid_m}xN{n}xK{k}, "
                f"平均耗时: {elapsed_time:.2f} us, "
                f"{tflops:.2f} TFlops, {bandwidth:.2f} GB/s\n"
            )


test_m_grouped_gemm_masked_w8a8_gfx936(acc_check=True)
