import torch

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

NUM_GROUPS = 16
EXPECTED_M_PER_GROUP_LIST = [128, 256, 512]
N_K_LIST = [(2048, 2048)]


def iter_contiguous_w8a8_int8_cases():
    for expected_m_per_group in EXPECTED_M_PER_GROUP_LIST:
        for n, k in N_K_LIST:
            yield NUM_GROUPS, expected_m_per_group, n, k


def _get_gfx_arch() -> str:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(props, "gcnArchName", props.name)
    return arch.split(":")[0].lower()


def _contiguous_i8_path_name(shuffle_unique: int) -> str:
    arch = _get_gfx_arch()
    if arch == "gfx938":
        return "hipc"
    return "hipc" if int(shuffle_unique) == 1 else "asm"


def align(x, y):
    return ((x + y - 1) // y) * y

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int):
    #每个group实际的M
    actual_ms = [int(expected_m_per_group) for _ in range(num_groups)]
    # contiguous接口要求每个group的M按256对齐，padding行用m_indices=-1标记
    aligned_ms = [align(actual_m, 256) for actual_m in actual_ms]
    #group的M的总和
    m = sum(aligned_ms)
    print("align_m after m : ", m)
    # a 是整体input
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)  
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    ref_d = torch.zeros((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        #表示M属于哪个group id
        m_indices[start:actual_end] = i        #[000...-1-1-1, 111...-1-1]           
        m_indices[actual_end:aligned_end] = -1

        start = aligned_end

    a_int8 = to_int8(a)
    b_int8 = to_int8(b)
    a_scale = torch.randn(m, device='cuda', dtype=torch.float)
    b_scale = torch.randn((num_groups, n, 1), device='cuda', dtype=torch.float)

    group_a = [a_int8, a_scale]
    group_b = [b_int8, b_scale]
    return m, group_a, group_b, m_indices, d, ref_d

def torch_gemm(
    A: torch.Tensor,          # [M, K]
    B: torch.Tensor,          # [E, N, K]
    C: torch.Tensor,          # [M, N]
    scaleA: torch.Tensor,     # [M] or [M, 1]
    scaleB: torch.Tensor,     # [E, N] or [E, N, 1]
    alpha: float,
    beta: float,
    m_index_h: torch.Tensor = None  # [M]，元素为 e_id，-1 表示跳过
) -> torch.Tensor:

    if m_index_h is None:
        raise ValueError('m_index_h is required for contiguous grouped GEMM')

    A_cpu = A.detach().cpu().to(torch.float32)
    B_cpu = B.detach().cpu().to(torch.float32)
    C_cpu = C.detach().cpu().to(torch.float32)
    scaleA_cpu = scaleA.detach().cpu().to(torch.float32)
    scaleB_cpu = scaleB.detach().cpu().to(torch.float32)
    m_index_cpu = m_index_h.detach().cpu().to(torch.long)

    D_cpu = torch.zeros(C_cpu.shape, device='cpu', dtype=C.dtype)
    B_ext = B_cpu.shape[0]
    for e_id in range(B_ext):
        rows = m_index_cpu == e_id
        if not torch.any(rows):
            continue
        out = A_cpu[rows] @ B_cpu[e_id].t()
        out = out * scaleA_cpu[rows].reshape(-1, 1)
        out = out * scaleB_cpu[e_id].reshape(1, -1)
        if beta != 0:
            out = alpha * out + beta * C_cpu[rows]
        elif alpha != 1:
            out = alpha * out
        D_cpu[rows] = out.to(D_cpu.dtype)

    return D_cpu


def check_w8a8_precision(
    ref_d: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 5e-2,
) -> bool:
    ref_d_cpu = ref_d.detach().cpu()
    d_cpu = d.detach().cpu()
    m_indices_cpu = m_indices.detach().cpu()

    valid_mask = (m_indices_cpu >= 0).unsqueeze(1).expand_as(ref_d_cpu)
    ref_valid = ref_d_cpu.masked_select(valid_mask)
    d_valid = d_cpu.masked_select(valid_mask)
    if ref_valid.numel() == 0:
        print('W8A8 CPU 精度检查: 无有效行，跳过')
        return True

    abs_err = (ref_valid.float() - d_valid.float()).abs()
    ref_abs = ref_valid.float().abs().clamp(min=1e-7)
    rel_err = abs_err / ref_abs
    print(
        f'  W8A8 CPU 精度统计: max_abs_err={abs_err.max().item():.6f}, '
        f'mean_abs_err={abs_err.mean().item():.6f}, '
        f'max_rel_err={rel_err.max().item():.6f}, '
        f'mean_rel_err={rel_err.mean().item():.6f}'
    )
    return torch.allclose(ref_valid, d_valid, rtol=rtol, atol=atol)
