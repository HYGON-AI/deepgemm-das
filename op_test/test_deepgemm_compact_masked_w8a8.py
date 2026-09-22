import sys
from pathlib import Path

import torch
from quant_utils import per_token_quant_int8

from deepgemm import (
    m_grouped_w8a8_gemm_nt_compact_masked,
    m_grouped_w8a8_gemm_nt_masked_impl,
)

sys.path.insert(0, str(Path(__file__).parent))
from deepgemm_marlin_quant import weight8bit_nt_kpack2_marlin2


def _reference(
    q_x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros(
        (q_x.shape[0], weight.shape[1]),
        dtype=torch.bfloat16,
        device=q_x.device,
    )
    for expert in range(weight.shape[0]):
        count = int(counts[expert])
        if count == 0:
            continue
        begin = int(offsets[expert])
        end = begin + count
        value = torch.matmul(q_x[begin:end].float(), weight[expert].float().t())
        value *= x_scale[begin:end].float()
        value *= weight_scale[expert].view(1, -1).float()
        output[begin:end] = value.bfloat16()
    return output


@torch.inference_mode()
def test_compact_and_dense_masked_match_reference() -> None:
    torch.manual_seed(7)
    device = "cuda"
    num_experts = 4
    capacity = 1024
    hidden = 2048
    output_size = 256

    counts = torch.tensor([300, 0, 511, 101], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 300, 300, 811], dtype=torch.int32, device=device)

    x = torch.randn((capacity, hidden), dtype=torch.bfloat16, device=device)
    q_x, x_scale = per_token_quant_int8(x)
    weight = torch.randint(
        -8,
        8,
        (num_experts, output_size, hidden),
        dtype=torch.int8,
        device=device,
    )
    weight_scale = torch.rand(
        (num_experts, output_size, 1), dtype=torch.float32, device=device
    )
    packed_weight = weight8bit_nt_kpack2_marlin2(weight)

    compact_output = torch.full(
        (capacity, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    m_grouped_w8a8_gemm_nt_compact_masked(
        (q_x, x_scale),
        (packed_weight, weight_scale),
        compact_output,
        counts,
        offsets,
        expected_m_per_group=512,
    )

    reference = _reference(q_x, x_scale, weight, weight_scale, counts, offsets)
    for expert in range(num_experts):
        count = int(counts[expert])
        begin = int(offsets[expert])
        end = begin + count
        torch.testing.assert_close(
            compact_output[begin:end], reference[begin:end], rtol=1e-2, atol=1e-2
        )

    dense_x = torch.zeros(
        (num_experts, capacity, hidden), dtype=torch.int8, device=device
    )
    dense_scale = torch.ones(
        (num_experts, capacity, 1), dtype=torch.float32, device=device
    )
    dense_output = torch.full(
        (num_experts, capacity, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    for expert in range(num_experts):
        count = int(counts[expert])
        begin = int(offsets[expert])
        end = begin + count
        dense_x[expert, :count] = q_x[begin:end]
        dense_scale[expert, :count] = x_scale[begin:end]
    m_grouped_w8a8_gemm_nt_masked_impl(
        (dense_x, dense_scale),
        (packed_weight, weight_scale),
        dense_output,
        counts,
        expected_m_per_group=512,
        mode=1000,
    )
    for expert in range(num_experts):
        count = int(counts[expert])
        begin = int(offsets[expert])
        end = begin + count
        torch.testing.assert_close(
            dense_output[expert, :count],
            reference[begin:end],
            rtol=1e-2,
            atol=1e-2,
        )


@torch.inference_mode()
def test_compact_masked_empty_page() -> None:
    device = "cuda"
    num_experts = 4
    capacity = 256
    hidden = 2048
    output_size = 256
    q_x = torch.zeros((capacity, hidden), dtype=torch.int8, device=device)
    x_scale = torch.ones((capacity, 1), dtype=torch.float32, device=device)
    weight = torch.zeros(
        (num_experts, output_size, hidden), dtype=torch.int8, device=device
    )
    packed_weight = weight8bit_nt_kpack2_marlin2(weight)
    weight_scale = torch.ones(
        (num_experts, output_size, 1), dtype=torch.float32, device=device
    )
    output = torch.full(
        (capacity, output_size),
        17,
        dtype=torch.bfloat16,
        device=device,
    )
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    m_grouped_w8a8_gemm_nt_compact_masked(
        (q_x, x_scale),
        (packed_weight, weight_scale),
        output,
        counts,
        offsets,
        expected_m_per_group=1,
    )
    torch.cuda.synchronize()
    assert torch.all(output == 17)


@torch.inference_mode()
def test_compact_large_page_preserves_persistent_scheduler_state() -> None:
    """覆盖物理滑块需要循环处理多个逻辑滑块的 compact 大页场景。"""
    torch.manual_seed(17)
    device = "cuda"
    num_experts = 8
    rows_per_expert = 1024
    hidden = 2048
    output_size = 4096
    capacity = num_experts * rows_per_expert

    dense_x = torch.randint(
        -127,
        128,
        (num_experts, rows_per_expert, hidden),
        dtype=torch.int8,
        device=device,
    )
    dense_scale = torch.rand(
        (num_experts, rows_per_expert, 1),
        dtype=torch.float32,
        device=device,
    )
    weight = torch.randint(
        -8,
        8,
        (num_experts, output_size, hidden),
        dtype=torch.int8,
        device=device,
    )
    packed_weight = weight8bit_nt_kpack2_marlin2(weight)
    weight_scale = torch.rand(
        (num_experts, output_size, 1),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.full(
        (num_experts,), rows_per_expert, dtype=torch.int32, device=device
    )
    offsets = torch.arange(
        num_experts, dtype=torch.int32, device=device
    ) * rows_per_expert

    compact_output = torch.empty(
        (capacity, output_size), dtype=torch.bfloat16, device=device
    )
    dense_output = torch.empty(
        (num_experts, rows_per_expert, output_size),
        dtype=torch.bfloat16,
        device=device,
    )

    # 先调用 compact，再调用原 masked，覆盖两个独立静态内核对象的初始化顺序。
    m_grouped_w8a8_gemm_nt_compact_masked(
        (dense_x.view(capacity, hidden), dense_scale.view(capacity, 1)),
        (packed_weight, weight_scale),
        compact_output,
        counts,
        offsets,
        expected_m_per_group=rows_per_expert,
    )
    m_grouped_w8a8_gemm_nt_masked_impl(
        (dense_x, dense_scale),
        (packed_weight, weight_scale),
        dense_output,
        counts,
        expected_m_per_group=rows_per_expert,
        mode=1000,
    )

    torch.testing.assert_close(
        compact_output,
        dense_output.view(capacity, output_size),
        rtol=1e-2,
        atol=1e-2,
    )
