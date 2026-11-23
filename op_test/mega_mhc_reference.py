"""DeepGEMM mega_mhc 算子的独立 Torch 参考。

数学与输入输出契约来源：deepseek-ai/DeepGEMM。

- tests/test_mega_mhc.py：输入输出构造、参考计算及误差检查。
- deep_gemm/utils/math.py：align、UE8M0 和 per_token_cast_to_fp8。
- deep_gemm/testing/numeric.py：calc_diff、assert_bitwise_equal、count_bytes。

License: MIT, Copyright (c) 2024 DeepSeek.
保留参考数学、舍入位置、epsilon、Sinkhorn 顺序和误差门限；
不依赖 SM100 原生扩展，可独立构造参考输出。"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# deep_gemm/utils/math.py：参考数学，MIT, Copyright (c) 2024 DeepSeek。
# ---------------------------------------------------------------------------


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    x_int = x.view(torch.int)
    assert (x_int >= 0).all() and (x_int & 0x7fffff == 0).all()
    return (x_int >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert not use_packed_ue8m0 or use_ue8m0
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    if use_packed_ue8m0:
        # pack_ue8m0_to_int 将4个UE8M0字节打包为一个int32，缩放因子数量须为4的倍数。
        # 较小的 hidden 维度可能不足4组，需要补齐缩放因子。
        # 补充值为1.0，即尾数为零的2^0。
        # 补齐的缩放因子不对应有效元素，kernel 不会读取它们。
        num_sf = sf.size(-1)
        if num_sf % 4 != 0:
            pad = align(num_sf, 4) - num_sf
            sf = torch.nn.functional.pad(sf, (0, pad), value=1.0)
        return x_fp8, pack_ue8m0_to_int(sf)
    return x_fp8, sf


def unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
    return (packed_sf.view(torch.uint8).to(torch.int) << 23).view(torch.float)


# ---------------------------------------------------------------------------
# deep_gemm/testing/numeric.py：误差与字节比较，MIT, Copyright (c) 2024 DeepSeek。
# ---------------------------------------------------------------------------


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # 分母为零表示两侧元素全部为零。
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def assert_bitwise_equal(x: torch.Tensor, y: torch.Tensor, label: str = ''):
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    x_bytes = x.contiguous().view(torch.uint8)
    y_bytes = y.contiguous().view(torch.uint8)
    if torch.equal(x_bytes, y_bytes):
        return

    mismatch = x_bytes != y_bytes
    mismatch_idx = mismatch.flatten().nonzero()[0].item()
    elem_idx = mismatch_idx // x.element_size()
    byte_in_elem = mismatch_idx % x.element_size()
    coord = tuple(torch.unravel_index(torch.tensor(elem_idx, device=x.device), x.shape))
    coord = tuple(v.item() for v in coord)
    raise AssertionError(
        f'bitwise mismatch{f" ({label})" if label else ""}: '
        f'num_bytes={mismatch.numel()}, num_mismatch={mismatch.sum().item()}, '
        f'first_byte={mismatch_idx}, elem={elem_idx}, coord={coord}, byte_in_elem={byte_in_elem}, '
        f'x_byte={x_bytes.flatten()[mismatch_idx].item()}, y_byte={y_bytes.flatten()[mismatch_idx].item()}, '
        f'x_val={x.flatten()[elem_idx].item()}, y_val={y.flatten()[elem_idx].item()}'
    )


# ---------------------------------------------------------------------------
# 测试矩阵
# ---------------------------------------------------------------------------

#: 完整数值矩阵：2种模式×2种SF布局×2种hidden×7种token数。
OFFICIAL_SF_LAYOUTS = ('col', 'extra')
OFFICIAL_HIDDENS = (4096, 7168)
OFFICIAL_TOKEN_COUNTS = (1, 64, 65, 200, 1025, 4096, 32768)

#: H=5120的token矩阵，包含分块边界形状。
GFX936_PR_HIDDEN = 5120
GFX936_PR_TOKEN_COUNTS = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                          4096, 8192, 16384)
GFX936_EXTRA_TOKEN_COUNTS = (4, 65, 200, 1025, 4160)

#: 官方kernel的非空输入要求H能被1024整除。
SUPPORTED_HIDDEN_MULTIPLE = 1024


def enumerate_cases():
    for sf_layout in OFFICIAL_SF_LAYOUTS:
        for hidden in OFFICIAL_HIDDENS:
            for num_tokens in OFFICIAL_TOKEN_COUNTS:
                yield sf_layout, num_tokens, hidden


def enumerate_gfx936_cases(hidden: int = GFX936_PR_HIDDEN):
    """H=5120 的 token 数矩阵，包含 tile 边界形状。"""
    for num_tokens in GFX936_PR_TOKEN_COUNTS + GFX936_EXTRA_TOKEN_COUNTS:
        yield 'extra', num_tokens, hidden


def check_hidden(hidden: int) -> None:
    assert hidden % SUPPORTED_HIDDEN_MULTIPLE == 0, (
        f'official mega_mhc requires H % {SUPPORTED_HIDDEN_MULTIPLE} == 0, got H={hidden}')


# ---------------------------------------------------------------------------
# 输入输出构造，默认参数取自 tests/test_mega_mhc.py
# ---------------------------------------------------------------------------


def make_inputs(num_tokens: int, hidden: int, seed: int, is_shifted: bool,
                device: str = 'cuda', num_sinkhorn_iters: int = 10,
                rmsnorm_scale: float = 1.25):
    hc_mult = 4
    num_hc_outputs = hc_mult * (hc_mult + 2)
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    inputs = dict(
        x=randn((num_tokens, hidden), torch.bfloat16),
        residual=randn((num_tokens, hc_mult, hidden), torch.bfloat16),
        post_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
        comb_res_mix=torch.rand((num_tokens, hc_mult, hc_mult), device=device, generator=generator),
        shifted_prev_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid() if is_shifted else None,
        fn=randn((num_hc_outputs, hc_mult * hidden), torch.float).mul_(1e-2),
        mix_scales=randn((3,), torch.float).mul_(0.1),
        mix_bases=randn((num_hc_outputs,), torch.float).mul_(0.1),
        rmsnorm_weight=randn((hidden,), torch.float).mul_(0.1).add_(1).bfloat16(),
        hc_mult=hc_mult,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        num_sinkhorn_iters=num_sinkhorn_iters,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=rmsnorm_scale,
    )
    for _ in range(3):
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-1, keepdim=True)
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-2, keepdim=True)
    duplicates = tuple(i for i in (63, 64, 511, 512, num_tokens - 1) if 0 < i < num_tokens)
    for name in ('x', 'residual', 'post_mix', 'comb_res_mix', 'shifted_prev_mix'):
        if inputs[name] is not None:
            inputs[name][list(set(duplicates))] = inputs[name][0].clone()
    return inputs, duplicates


def make_outputs(inputs, sf_layout: str, shared_sf_block_m: int = 224, device: str = 'cuda'):
    num_tokens, hidden = inputs['x'].shape
    is_shifted = inputs['shifted_prev_mix'] is not None
    outputs = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=torch.empty_like(inputs['shifted_prev_mix']) if is_shifted else None,
        new_post_mix=torch.empty_like(inputs['post_mix']),
        new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
        y_bf16=torch.empty_like(inputs['x']),
    )
    if sf_layout == 'bf16':
        return outputs
    sf_shape = (num_tokens, hidden // 128)
    outputs['y_fp8'] = torch.empty((num_tokens, hidden), dtype=torch.float8_e4m3fn, device=device)
    if sf_layout == 'col':
        outputs['y_gemm_sf'] = torch.empty_strided(
            sf_shape, (1, align(num_tokens, 4)), dtype=torch.int32, device=device)
    else:
        outputs['y_routed_sf'] = torch.empty(sf_shape, dtype=torch.int32, device=device)
        num_rows = (num_tokens + shared_sf_block_m - 1) // shared_sf_block_m * align(shared_sf_block_m, 128)
        storage = torch.empty_strided(
            (num_rows, hidden // 128), (1, num_rows), dtype=torch.int32, device=device)
        outputs.update(
            y_shared_sf=storage[:num_tokens],
            y_shared_sf_storage=storage,
            shared_sf_block_m=shared_sf_block_m,
        )
    return outputs


def extra_sf_rows(x: torch.Tensor, block_m: int) -> torch.Tensor:
    token_idx = torch.arange(x.size(0), device=x.device)
    index_in_block = token_idx % block_m
    return (
        token_idx // block_m * align(block_m, 128)
        + index_in_block // 128 * 128
        + index_in_block % 32 * 4
        + index_in_block % 128 // 32
    )


def _allocate_extra_sf(x: torch.Tensor, block_m: int) -> torch.Tensor:
    num_tokens, hidden = x.shape
    num_rows = (num_tokens + block_m - 1) // block_m * align(block_m, 128)
    return torch.empty_strided(
        (num_rows, hidden // 128), (1, num_rows), dtype=torch.int32, device=x.device)


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


@torch.no_grad()
def _exact_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """投影采用FP32矩阵乘法，临时关闭TF32并恢复调用前的配置。"""
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return a @ b
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


def matmul_precision_report() -> Dict[str, object]:
    """只读获取影响参考计算的矩阵乘法精度配置。"""
    report: Dict[str, object] = {}
    try:
        report['allow_tf32'] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:                                    # pragma: no cover
        report['allow_tf32'] = None
    try:
        report['cudnn_allow_tf32'] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:                                    # pragma: no cover
        report['cudnn_allow_tf32'] = None
    try:
        report['float32_matmul_precision'] = torch.get_float32_matmul_precision()
    except Exception:                                    # pragma: no cover
        report['float32_matmul_precision'] = None
    return report


# ---------------------------------------------------------------------------
# 参考数学与可选的中间值诊断
# ------------------------------------------------------------------------------


@torch.no_grad()
def mhc_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    sf_layout: str = 'bf16',
    shared_sf_block_m: int = 0,
    shifted_prev_mix: torch.Tensor | None = None,
    with_diagnostics: bool = False,
):
    """计算一次 mega_mhc 的 Torch 参考结果。

    返回按官方接口命名的逻辑输出字典。with_diagnostics=True 时返回
    (result, diagnostics)，后者包含 Z、投影、HC 平方和、U、Y 和量化中间值。"""
    diag: Dict[str, torch.Tensor] = {}

    new_residual_f32 = x.float().unsqueeze(1) * post_mix
    new_residual_f32 += torch.einsum(
        'tij,tih->tjh', comb_res_mix, residual.float())
    new_residual = new_residual_f32.bfloat16()
    norm_input = None if shifted_prev_mix is None else (
        new_residual.float() * shifted_prev_mix).sum(dim=1).bfloat16()
    diag['z_fp32'] = new_residual_f32
    diag['z_bf16'] = new_residual

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = _exact_matmul(new_residual.flatten(1).float(), fn.mT)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    hc_sqr_sum = new_residual.float().square().sum((1, 2))
    diag['proj'] = mixes
    diag['hc_sqr_sum'] = hc_sqr_sum

    mixes = mixes * torch.rsqrt(
        hc_sqr_sum / (hc_mult * x.size(1)) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((
        mix_scales[0].expand(hc_mult),
        mix_scales[1].expand(hc_mult),
        mix_scales[2].expand(hc_mult * hc_mult),
    ))
    mixes = mixes * scales + mix_bases
    diag['mix_logits'] = mixes

    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (
        mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale
    ).unsqueeze(2)
    new_comb_res_mix = _sinkhorn_reference(
        mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult),
        num_sinkhorn_iters,
        sinkhorn_eps,
    )
    diag['new_prev_mix'] = new_prev_mix
    diag['new_post_mix'] = new_post_mix
    diag['new_comb_res_mix'] = new_comb_res_mix
    if norm_input is None:
        norm_input = (new_residual.float() * new_prev_mix).sum(dim=1).bfloat16()
    diag['u_bf16'] = norm_input

    norm_input_float = norm_input.float()
    y_fp32 = (
        norm_input_float
        * torch.rsqrt(norm_input_float.square().mean(1) + rmsnorm_eps).unsqueeze(1)
        * (rmsnorm_weight.float() * rmsnorm_scale)
    )
    y_bf16 = y_fp32.bfloat16()
    diag['y_fp32'] = y_fp32
    diag['y_bf16'] = y_bf16

    result = {
        'new_residual': new_residual,
        'new_post_mix': new_post_mix,
        'new_comb_res_mix': new_comb_res_mix,
        'y_bf16': y_bf16,
    }
    if shifted_prev_mix is not None:
        result['new_prev_mix'] = new_prev_mix
    if sf_layout == 'bf16':
        return (result, diag) if with_diagnostics else result

    y_fp8, y_fp8_sf = per_token_cast_to_fp8(
        y_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    diag['y_fp8'] = y_fp8
    diag['sf_packed'] = y_fp8_sf
    diag['sf_unpacked'] = unpack_ue8m0_from_int(y_fp8_sf)
    if sf_layout == 'col':
        col_major_sf = torch.empty_strided(
            y_fp8_sf.shape, (1, align(x.size(0), 4)),
            dtype=torch.int32, device=x.device)
        col_major_sf.copy_(y_fp8_sf)
        y_fp8_sf = col_major_sf
    sf_name = 'y_gemm_sf' if sf_layout == 'col' else 'y_routed_sf'
    result.update(y_fp8=y_fp8, **{sf_name: y_fp8_sf})
    if sf_layout == 'extra':
        storage = _allocate_extra_sf(x, shared_sf_block_m)
        storage[extra_sf_rows(x, shared_sf_block_m)] = y_fp8_sf
        result['y_shared_sf_storage'] = storage
    return (result, diag) if with_diagnostics else result


# ---------------------------------------------------------------------------
# 输出比较辅助函数
# ---------------------------------------------------------------------------


def logical_outputs(result, shared_sf_block_m=None):
    names = ('new_residual', 'new_post_mix', 'new_comb_res_mix', 'y_bf16')
    names += ('new_prev_mix',) if result.get('new_prev_mix') is not None else ()
    names += tuple(name for name in ('y_fp8', 'y_gemm_sf', 'y_routed_sf') if name in result)
    values = {name: result[name] for name in names}
    if 'y_shared_sf_storage' in result:
        y_bf16 = result['y_bf16']
        rows = extra_sf_rows(y_bf16, result.get('shared_sf_block_m', shared_sf_block_m))
        values['y_shared_sf'] = result['y_shared_sf_storage'][rows]
    return values


def comparable(tensor: torch.Tensor):
    return tensor.contiguous().view(torch.uint8) if tensor.dtype == torch.int32 else tensor


#: 按各逻辑输出分别应用 calc_diff 门限。
CORRECTNESS_LIMITS = {'y_fp8': 2e-4}
DEFAULT_CORRECTNESS_LIMIT = 5e-5


def check_correctness(actual, reference, case):
    """按各输出的 calc_diff 门限检查结果，并返回逐输出误差。"""
    max_diffs = []
    for name in actual:
        actual_tensor = comparable(actual[name])
        reference_tensor = comparable(reference[name])
        diff = float(calc_diff(actual_tensor, reference_tensor))
        limit = CORRECTNESS_LIMITS.get(name, DEFAULT_CORRECTNESS_LIMIT)
        assert diff < limit, f'{case}, {name=}, {diff=}'
        max_diffs.append(diff)
    return max_diffs


def diagnose_outputs(actual, reference, case) -> Dict[str, Dict[str, float]]:
    """报告每个逻辑输出的 calc_diff、最大绝对误差、非有限值数量与最大误差位置。

    同时保留数值误差和逐元素诊断，便于识别稀疏异常。"""
    report: Dict[str, Dict[str, float]] = {}
    for name in actual:
        a = comparable(actual[name]).double()
        b = comparable(reference[name]).double()
        diff = float(calc_diff(a, b))
        absdiff = (a - b).abs()
        worst = int(absdiff.argmax().item())
        report[name] = {
            'calc_diff': diff,
            'limit': CORRECTNESS_LIMITS.get(name, DEFAULT_CORRECTNESS_LIMIT),
            'max_abs': float(absdiff.flatten()[worst].item()),
            'worst_index': worst,
            'nonfinite_actual': int((~torch.isfinite(a)).sum().item()),
            'nonfinite_reference': int((~torch.isfinite(b)).sum().item()),
            'pass': diff < CORRECTNESS_LIMITS.get(name, DEFAULT_CORRECTNESS_LIMIT),
        }
    return report


def compare_diagnostics(port_diag: Dict[str, torch.Tensor],
                        ref_diag: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """按参考数据流顺序比较中间张量的 calc_diff 与 max_abs。

    第一个出现差异的中间值用于定位偏差起点。"""
    order = ('z_fp32', 'z_bf16', 'proj', 'hc_sqr_sum', 'mix_logits',
             'new_prev_mix', 'new_post_mix', 'new_comb_res_mix',
             'u_bf16', 'y_fp32', 'y_bf16', 'y_fp8', 'sf_packed', 'sf_unpacked')
    report: Dict[str, Dict[str, float]] = {}
    for name in order:
        if name not in ref_diag or name not in port_diag:
            continue
        a = comparable(port_diag[name]).double()
        b = comparable(ref_diag[name]).double()
        if a.shape != b.shape:
            report[name] = {'shape_mismatch': 1.0,
                            'port': float(a.numel()), 'reference': float(b.numel())}
            continue
        absdiff = (a - b).abs()
        report[name] = {
            'calc_diff': float(calc_diff(a, b)),
            'max_abs': float(absdiff.max().item()) if absdiff.numel() else 0.0,
            'nonfinite': int((~torch.isfinite(a)).sum().item()),
        }
    return report


# ---------------------------------------------------------------------------
# 延迟接缝参考：collapse 消费输入的 pre_mix
# ---------------------------------------------------------------------------


@torch.no_grad()
def delayed_seam_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: Optional[torch.Tensor],
    sublayer_out: Optional[torch.Tensor] = None,
    post_layer_mix: Optional[torch.Tensor] = None,
    comb_res_mix: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    with_diagnostics: bool = False,
):
    """延迟接缝的 Torch 参考，消费输入的 pre_mix。

    与 mhc_reference 的差异：
    - collapse 使用携带的 pre_mix；为 None 时选择第0路，不消费本次新生成的 pre。
    - 不含 rmsnorm_scale 乘法，需要时由调用方预先合入 norm_weight。
    - norm_eps 与 HC 归一化使用的 epsilon 相互独立。"""
    assert residual.dim() == 3 and residual.dtype == torch.bfloat16
    T, n, H = residual.shape
    assert n == 4, f'delayed seam is specialised for hc_mult=4, got {n}'
    device = residual.device
    diag: Dict[str, torch.Tensor] = {}

    if pre_mix is None:
        pre_2d = torch.zeros(T, 4, dtype=torch.float32, device=device)
        pre_2d[:, 0] = 1.0
    else:
        pre_2d = pre_mix.reshape(T, 4).float()

    if sublayer_out is not None:
        post_2d = post_layer_mix.reshape(T, 4).float()
        comb = comb_res_mix.reshape(T, 4, 4).float()
        z_f32 = sublayer_out.float().unsqueeze(1) * post_2d[:, :, None]
        z_f32 = z_f32 + torch.einsum('tij,tih->tjh', comb, residual.float())
        z = z_f32.bfloat16()
        residual_out = z.clone()
    else:
        z = residual
        residual_out = residual
    diag['z_bf16'] = z
    zf = z.float()

    hc_sqr_sum = zf.square().sum((1, 2))
    mixes = _exact_matmul(zf.flatten(1), fn.mT)
    diag['proj'] = mixes
    diag['hc_sqr_sum'] = hc_sqr_sum
    rstd = torch.rsqrt(hc_sqr_sum / (4 * H) + rms_eps).unsqueeze(1)
    logits = mixes * rstd * torch.cat((
        hc_scale[0].expand(4), hc_scale[1].expand(4), hc_scale[2].expand(16)))
    logits = logits + hc_base
    diag['mix_logits'] = logits

    next_pre = logits[:, :4].sigmoid() + hc_pre_eps
    post_out = logits[:, 4:8].sigmoid() * hc_post_mult_value
    comb_out = _sinkhorn_reference(
        logits[:, 8:24].view(-1, 4, 4), sinkhorn_repeat, hc_sinkhorn_eps)
    diag['new_prev_mix'] = next_pre.unsqueeze(2)
    diag['new_post_mix'] = post_out.unsqueeze(2)
    diag['new_comb_res_mix'] = comb_out

    x1 = (zf * pre_2d[:, :, None]).sum(dim=1).bfloat16()
    diag['u_bf16'] = x1
    x1f = x1.float()
    y_fp32 = (x1f * torch.rsqrt(x1f.square().mean(1) + norm_eps).unsqueeze(1)
              * norm_weight.float())
    y = y_fp32.bfloat16()
    diag['y_fp32'] = y_fp32
    diag['y_bf16'] = y
    result = (residual_out, post_out.unsqueeze(2), comb_out, y, next_pre)
    return (result, diag) if with_diagnostics else result


# ---------------------------------------------------------------------------
# 可写入JSON的环境与设备标识
# ---------------------------------------------------------------------------


def describe_environment() -> Dict[str, object]:
    """只读获取适合写入结果清单的环境和设备标识，不改变状态。"""
    import os
    import subprocess

    info: Dict[str, object] = {
        'torch_version': torch.__version__,
        'torch_hip': torch.version.hip,
        'python': None,
        'cuda_available': torch.cuda.is_available(),
    }
    try:
        import sys
        info['python'] = sys.version.replace('\n', ' ')
    except Exception:                                    # pragma: no cover
        pass
    try:
        import triton
        info['triton_version'] = triton.__version__
    except Exception:                                    # pragma: no cover
        info['triton_version'] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            'device_name': props.name,
            'gcn_arch_name': getattr(props, 'gcnArchName', None),
            'multi_processor_count': props.multi_processor_count,
            'total_memory_bytes': props.total_memory,
            'uuid': getattr(props, 'uuid', None),
        })
    try:
        info['dtk_version'] = open('/opt/dtk/.info/rocm_version').read().strip()
    except Exception:                                    # pragma: no cover
        info['dtk_version'] = os.environ.get('DTK_VERSION')
    for key, cmd in (('git_head', ['git', 'rev-parse', 'HEAD']),
                     ('git_branch', ['git', 'rev-parse', '--abbrev-ref', 'HEAD'])):
        try:
            info[key] = subprocess.check_output(
                cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:                                # pragma: no cover
            info[key] = None
    return info
