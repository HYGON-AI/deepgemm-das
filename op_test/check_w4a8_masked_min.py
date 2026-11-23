# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OP_TEST = os.path.join(ROOT, "op_test")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if OP_TEST not in sys.path:
    sys.path.insert(0, OP_TEST)

from deepgemm import m_grouped_w4a8_gemm_nt_masked
from deepgemm_marlin_quant import weight4bit_nt_kpack2_marlin2_qqq_from_packed


def add_aiter_to_path() -> None:
    aiter_root = os.path.abspath(os.path.join(ROOT, "..", "..", "7_6", "aiter"))
    if aiter_root not in sys.path:
        sys.path.insert(0, aiter_root)


def unpack_int8_to_int4_high_nibble(tensor_int8: torch.Tensor) -> torch.Tensor:
    high4 = tensor_int8 & 0xF0
    low4 = (tensor_int8 << 4) & 0xF0
    unpacked = torch.empty(
        (*tensor_int8.shape[:-1], tensor_int8.shape[-1] * 2),
        dtype=torch.int8,
        device=tensor_int8.device,
    )
    unpacked[..., 0::2] = high4.to(torch.int8)
    unpacked[..., 1::2] = low4.to(torch.int8)
    return unpacked


def pack_weight_k_contiguous(weight: torch.Tensor) -> torch.Tensor:
    e, n, k = weight.shape
    assert k % 2 == 0
    w_unsigned = weight.to(torch.uint8) & 0x0F
    w_reshaped = w_unsigned.view(e, n, k // 2, 2)
    packed_weight = (w_reshaped[..., 0] << 4) | w_reshaped[..., 1]
    return packed_weight.to(torch.int8)


def repack_contiguous_to_blocked(packed_k_contiguous: torch.Tensor) -> torch.Tensor:
    e, n, k_half = packed_k_contiguous.shape
    assert k_half % 4 == 0
    w_u8 = packed_k_contiguous.to(torch.uint8)
    w0 = (w_u8 >> 4) & 0x0F
    w1 = w_u8 & 0x0F
    w_unpacked = torch.stack([w0, w1], dim=-1).view(e, n, -1)
    temp_view = w_unpacked.view(*w_unpacked.shape[:-1], -1, 8)
    w_low = temp_view[..., :4].reshape(*w_unpacked.shape[:-1], -1)
    w_high = temp_view[..., 4:].reshape(*w_unpacked.shape[:-1], -1)
    return ((w_low << 4) | w_high).to(torch.int8)


def pack_aiter_weight(weight_u4: torch.Tensor, gemm: str) -> torch.Tensor:
    add_aiter_to_path()
    from aiter.ops.shuffle import w4a8_moe_layout_shuffle_gemm1, w4a8_moe_layout_shuffle_gemm2

    shuffle = w4a8_moe_layout_shuffle_gemm1 if gemm == "gemm1" else w4a8_moe_layout_shuffle_gemm2
    packed = pack_weight_k_contiguous(weight_u4)
    blocked = repack_contiguous_to_blocked(packed)
    return torch.stack([shuffle(blocked[i]) for i in range(weight_u4.size(0))], dim=0).contiguous()


def run_case(
    e: int,
    m: int,
    n: int,
    k: int,
    masked: int,
    dtype: torch.dtype,
    layout: str,
    gemm: str,
) -> None:
    torch.manual_seed(1)
    device = "cuda"

    q_a = torch.randint(-8, 8, (e, m, k), device=device, dtype=torch.int8).contiguous()
    a_scale = torch.rand((e, m, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.001)

    w_scale = torch.rand((e, n, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.001)

    if layout == "marlin2":
        packed_w_source = torch.randint(
            -128, 127, (e, n, k // 2), device=device, dtype=torch.int8
        ).contiguous()
        packed_w = torch.stack(
            [weight4bit_nt_kpack2_marlin2_qqq_from_packed(packed_w_source[i]) for i in range(e)],
            dim=0,
        ).contiguous()
        w_ref = unpack_int8_to_int4_high_nibble(packed_w_source)
    else:
        weight_u4 = torch.randint(0, 15, (e, n, k), device=device, dtype=torch.int8)
        packed_w = pack_aiter_weight(weight_u4, gemm)
        w_ref = torch.where(weight_u4 > 7, weight_u4 - 16, weight_u4)

    ref = torch.bmm(q_a.float(), w_ref.float().transpose(1, 2))
    ref = (ref * a_scale * w_scale.transpose(1, 2)).to(dtype)

    out = torch.empty((e, m, n), device=device, dtype=dtype)
    masked_m = torch.full((e,), masked, device=device, dtype=torch.int32)

    m_grouped_w4a8_gemm_nt_masked(
        (q_a, a_scale),
        (packed_w, w_scale),
        out,
        masked_m,
        masked,
    )
    torch.cuda.synchronize()

    valid = out[:, :masked, :]
    ref_valid = ref[:, :masked, :]
    torch.testing.assert_close(valid, ref_valid, rtol=1e-2, atol=1e-2)

    if masked < m:
        zero_tail = out[:, masked:, :]
        torch.testing.assert_close(zero_tail, torch.zeros_like(zero_tail), rtol=0, atol=0)

    max_abs = (valid.float() - ref_valid.float()).abs().max().item()
    print(
        f"PASS layout={layout} E={e} M={m} N={n} K={k} "
        f"masked={masked} dtype={dtype} max_abs={max_abs:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e", type=int, default=2)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--masked", type=int, default=24)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--layout", choices=["marlin2", "aiter"], default="marlin2")
    parser.add_argument("--gemm", choices=["gemm1", "gemm2"], default="gemm1")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    run_case(args.e, args.m, args.n, args.k, args.masked, dtype, args.layout, args.gemm)


if __name__ == "__main__":
    main()
