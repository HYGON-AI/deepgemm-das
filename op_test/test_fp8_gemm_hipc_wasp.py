# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Small gfx92a WASP regressions, including both consumer schedules."""
import pytest
import torch

from deepgemm import op


@pytest.mark.parametrize("k", [0, 16, 112, 128, 144, 256, 384, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("batch,scaled", [(1, False), (2, True)])
def test_wasp_tails_and_epilogue(k, dtype, batch, scaled):
    if not torch.cuda.is_available():
        pytest.skip("requires gfx92a")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith("gfx92a"):
        pytest.skip("requires gfx92a")
    torch.manual_seed(123)
    m, n = 137, 271
    prefix = () if batch == 1 else (batch,)
    a = (torch.randn(*prefix, m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(*prefix, n, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    sa = torch.rand(*prefix, m, device="cuda") if scaled else torch.empty(0, device="cuda")
    sb = torch.rand(n, device="cuda") if scaled else torch.empty(0, device="cuda")
    out = torch.randn(*prefix, m, n, device="cuda", dtype=dtype)
    old = out.float().clone()
    ref = a.float() @ b.float().transpose(-1, -2)
    if scaled:
        ref = ref * sa.unsqueeze(-1) * sb
    ref = ref * 0.75 + old * 0.25
    op.fp8_gemm_hipc(
        b, a, sb, sa, out, n, m, k, batch, "TN",
        torch.tensor(0.75), torch.tensor(0.25), None,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.to(dtype).float(), rtol=0.02, atol=0.02)
