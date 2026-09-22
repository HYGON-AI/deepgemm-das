import random
from typing import Type

import torch
from quant_utils import per_token_quant_int8

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

NUM_GROUPS = 16
M = 2048
EXPECTED_M_PER_GROUP_LIST = [8, 16, 32, 64, 128, 256, 512, 1024]
N_K_LIST = [(4096, 6144), (6144, 2048)]


def iter_masked_w8a8_int8_cases():
    for expected_m_per_group in EXPECTED_M_PER_GROUP_LIST:
        for n, k in N_K_LIST:
            yield NUM_GROUPS, M, expected_m_per_group, n, k


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
    hidden_states = (torch.randn((num_groups, m, k), device=device) * 5).to(dtype=out_dtype)
    weight = to_int8(torch.randn((num_groups, n, k), device=device)).contiguous()
    w_scale = torch.randn((num_groups, n, 1), device=device, dtype=torch.float32)
    d = torch.empty((num_groups, m, n), device=device, dtype=out_dtype)

    masked_m = torch.empty((num_groups,), device=device, dtype=torch.int32)
    for expert_id in range(num_groups):
        masked_m[expert_id] = min(m, int(expected_m_per_group * random.uniform(0.7, 3.3)))

    return hidden_states, weight, w_scale, masked_m, d


def quantize_activation(hidden_states: torch.Tensor):
    q_a, a_scale = per_token_quant_int8(hidden_states)
    return q_a.contiguous(), a_scale.contiguous()


def torch_gemm(
    q_a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: Type[torch.dtype],
) -> torch.Tensor:
    a = q_a.detach().cpu().to(torch.float32)
    b = weight.detach().cpu().to(torch.float32)
    a_scale_cpu = a_scale.detach().cpu().to(torch.float32)
    w_scale_cpu = w_scale.detach().cpu().to(torch.float32)

    c = torch.bmm(a, b.transpose(1, 2))
    c = a_scale_cpu * c * w_scale_cpu.transpose(1, 2)
    return c.to(output_dtype)


def check_w8a8_precision(
    ref_d: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    ref_d_cpu = ref_d.detach().cpu()
    d_cpu = d.detach().cpu()
    masked_m_cpu = masked_m.detach().cpu()

    valid = True
    for expert_id in range(masked_m_cpu.shape[0]):
        cur_valid_m = int(masked_m_cpu[expert_id].item())
        if cur_valid_m == 0:
            continue

        ref_valid = ref_d_cpu[expert_id, :cur_valid_m, :]
        d_valid = d_cpu[expert_id, :cur_valid_m, :]
        abs_err = (ref_valid.float() - d_valid.float()).abs()
        ref_abs = ref_valid.float().abs().clamp(min=1e-7)
        rel_err = abs_err / ref_abs
        print(
            f"  CPU expert {expert_id}: valid_m={cur_valid_m}, "
            f"max_abs_err={abs_err.max().item():.6f}, mean_abs_err={abs_err.mean().item():.6f}, "
            f"max_rel_err={rel_err.max().item():.6f}, mean_rel_err={rel_err.mean().item():.6f}"
        )

        if not torch.allclose(ref_valid, d_valid, rtol=rtol, atol=atol):
            valid = False
            break

    print("W8A8 CPU 精度检查合格！！！" if valid else "W8A8 CPU 精度检查不合格！！！")
    return valid
