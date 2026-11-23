# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""mega_mhc 的独立参考、数值与行为测试。"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mega_mhc_reference as ref  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    return torch.cuda.is_available()


needs_cuda = pytest.mark.skipif(not _cuda_available(), reason='no CUDA/HIP device')


def _fast() -> bool:
    """选择缩小的测试矩阵；默认使用完整矩阵。"""
    return os.environ.get('MEGA_MHC_TEST_FAST', '0') == '1'


def _empty_cache() -> None:
    if _cuda_available():
        torch.cuda.empty_cache()


def _now() -> float:
    if _cuda_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _native_available() -> bool:
    try:
        import deepgemm
        return hasattr(deepgemm, 'mega_mhc')
    except Exception:
        return False


requires_native = pytest.mark.skipif(
    not _native_available(),
    reason='deepgemm.mega_mhc is unavailable in the loaded extension; '
           'reference/primitives gates do not depend on it')


def _call_native(deepgemm, inputs, outputs, **overrides):
    """调用 native 算子，并剔除仅供测试读取布局的存储元数据。

    make_outputs 中的 y_shared_sf_storage 用于保存底层存储，
    算子只接收它的 y_shared_sf 视图，因此不将该元数据传入接口。"""
    kwargs = {**inputs, **outputs}
    kwargs.pop('y_shared_sf_storage', None)
    kwargs.update(overrides)
    return deepgemm.mega_mhc(**kwargs)


# ---------------------------------------------------------------------------
# 参考计算：环境、接口和布局
# ---------------------------------------------------------------------------


@needs_cuda
def test_reference_environment_identity():
    """记录运行环境与设备标识，不改变设备状态。"""
    info = ref.describe_environment()
    print('\n' + '\n'.join(f'  {k}: {v}' for k, v in info.items()))
    assert info['torch_version']
    assert info['cuda_available'], 'reference requires a visible device'
    assert info['multi_processor_count'] > 0
    gcn = str(info['gcn_arch_name'])
    assert gcn.startswith('gfx'), gcn
    # 只读记录设备频率，不修改设备状态。
    print(f"  device={info['device_name']} arch={gcn} "
          f"CU={info['multi_processor_count']}")
    precision = ref.matmul_precision_report()
    print('  matmul precision knobs: '
          + ', '.join(f'{k}={v}' for k, v in precision.items()))
    assert 'allow_tf32' in precision


@needs_cuda
@pytest.mark.parametrize('hidden', [4096, 5120, 7168])
def test_reference_input_output_contract(hidden):
    """检查 BF16 与两组 SF 布局的形状、类型和步长契约。"""
    num_tokens = 200
    for is_shifted in (False, True):
        inputs, duplicates = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
        assert inputs['x'].shape == (num_tokens, hidden)
        assert inputs['residual'].shape == (num_tokens, 4, hidden)
        assert inputs['fn'].shape == (24, 4 * hidden)
        assert inputs['post_mix'].shape == (num_tokens, 4, 1)
        assert inputs['comb_res_mix'].shape == (num_tokens, 4, 4)
        assert (inputs['shifted_prev_mix'] is not None) == is_shifted
        # make_inputs 执行三轮行、列归一化，最后一步为列归一化。
        # 因此应检查 sum_i comb[i,j]=1，即每个输出流是输入流的凸组合。
        # 行和仅近似为1，不能用严格的双随机条件替代列和检查。

        col = inputs['comb_res_mix'].sum(-2)
        roww = inputs['comb_res_mix'].sum(-1)
        assert torch.allclose(col, torch.ones_like(col), atol=1e-5)
        print(f'  comb row-sum max deviation: '
              f'{float((roww - 1).abs().max()):.3e} (diagnostic, not a gate)')

        for sf_layout in ('bf16', 'col', 'extra'):
            outputs = ref.make_outputs(inputs, sf_layout)
            assert outputs['new_residual'].dtype == torch.bfloat16
            if is_shifted:
                assert outputs['new_prev_mix'] is not None
            else:
                assert outputs['new_prev_mix'] is None
            assert outputs['new_post_mix'].shape == (num_tokens, 4, 1)
            assert outputs['y_bf16'].shape == (num_tokens, hidden)
            if sf_layout == 'bf16':
                assert 'y_fp8' not in outputs
                continue
            assert outputs['y_fp8'].dtype == torch.float8_e4m3fn
            assert outputs['y_fp8'].shape == (num_tokens, hidden)
            sf_shape = (num_tokens, hidden // 128)
            if sf_layout == 'col':
                sf = outputs['y_gemm_sf']
                assert sf.shape == sf_shape and sf.dtype == torch.int32
                # 列主序存储中，每列的 token 维按4对齐。
                assert sf.stride() == (1, ref.align(num_tokens, 4)), sf.stride()
            else:
                sf = outputs['y_routed_sf']
                assert sf.shape == sf_shape and sf.dtype == torch.int32
                assert sf.is_contiguous(), sf.stride()
                storage = outputs['y_shared_sf_storage']
                block_m = outputs['shared_sf_block_m']
                assert block_m == 224
                num_rows = ((num_tokens + block_m - 1) // block_m
                            * ref.align(block_m, 128))
                assert storage.shape == (num_rows, hidden // 128)
                assert storage.stride() == (1, num_rows)
                assert outputs['y_shared_sf'].shape == sf_shape
                assert outputs['y_shared_sf'].data_ptr() == storage.data_ptr()
                # 行置换必须单射到预留存储行，且不越界。
                rows = ref.extra_sf_rows(outputs['y_bf16'], block_m)
                assert rows.min() >= 0 and rows.max() < num_rows
                assert len(torch.unique(rows)) == num_tokens
        del inputs, outputs
    _empty_cache()


@needs_cuda
def test_reference_shared_sf_row_permutation_matches_storage_view():
    """通过 extra_sf_rows 读取 y_shared_sf 的逻辑行置换。"""
    num_tokens, hidden, block_m = 700, 4096, 224
    inputs, _ = ref.make_inputs(num_tokens, hidden, 7, False)
    outputs = ref.make_outputs(inputs, 'extra', shared_sf_block_m=block_m)
    storage = outputs['y_shared_sf_storage']
    storage.zero_()                  # empty_strided 返回未初始化的存储。
    sf = torch.arange(num_tokens * (hidden // 128), device='cuda',
                      dtype=torch.int32).view(num_tokens, hidden // 128)
    outputs['y_routed_sf'].copy_(sf)
    rows = ref.extra_sf_rows(outputs['y_bf16'], block_m)
    storage.index_copy_(0, rows, sf)
    assert torch.equal(storage[rows], sf)
    # 逻辑读取使用 storage[extra_sf_rows(...)]；y_shared_sf 只是 storage[:T] 的别名。
    # 该视图的原始行序不等于逻辑置换行序。
    assert outputs['y_shared_sf'].data_ptr() == storage.data_ptr()
    # 置换范围之外的存储不得被修改。
    touched = torch.zeros_like(storage, dtype=torch.bool)
    touched[rows] = True
    assert storage[~touched].eq(0).all()
    _empty_cache()


@needs_cuda
def test_reference_fp8_scale_contract():
    """检查 UE8M0 打包、amax 下限及 E4M3FN 编码契约。"""
    torch.manual_seed(0)
    m, n = 5, 128
    x = torch.randn((m, n), device='cuda', dtype=torch.float32).bfloat16()
    x[0] = 0.0                                  # 全零分组的 amax 使用1e-4下限。
    x[1, :32] = 1e-6                            # 低于 amax 下限的输入。
    y_fp8, sf_packed = ref.per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    assert y_fp8.dtype == torch.float8_e4m3fn
    assert sf_packed.dtype == torch.int32
    assert sf_packed.shape == (m, n // 128)
    # 每个 int32 打包4个指数，UE8M0 指数范围为[1,254]。
    exponents = sf_packed.view(torch.uint8).view(m, -1)
    assert exponents.shape == (m, n // 32)
    assert int(exponents.min()) >= 1 and int(exponents.max()) <= 254
    unpacked = ref.unpack_ue8m0_from_int(sf_packed)
    assert torch.equal(
        unpacked.view(torch.int) >> 23, exponents.to(torch.int))
    # 全零组的缩放因子为 ceil_ue8m0(1e-4/448)=2^-22。
    floor_sf = ref.ceil_to_ue8m0(torch.tensor([1e-4 / 448.0], device='cuda'))
    expected_exp = int((floor_sf.view(torch.int) >> 23) & 0xFF)
    assert int(exponents[0, 0]) == expected_exp, (int(exponents[0, 0]), expected_exp)
    assert float(floor_sf.item()) == 2.0 ** (-22)
    # 未饱和区间的 E4M3 往返转换有3位尾数，相对误差不超过2^-4。
    scale = unpacked.repeat_interleave(32, dim=1)
    deq = y_fp8.float() * scale
    err = (deq - x.float()).abs()
    bound = (2.0 ** -4) * x.float().abs() + 1e-30
    assert bool((err <= bound).all()), (err.max().item(), bound.max().item())
    _empty_cache()


# ---------------------------------------------------------------------------
# 参考数值：确定性、重复 token 与中间值一致性
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize('is_shifted', [False, True])
def test_reference_repeatability(is_shifted):
    """相同输入配合新分配的输出，逻辑输出应逐字节一致。"""
    num_tokens, hidden = 200, 4096
    inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
    kwargs = dict(inputs, sf_layout='extra', shared_sf_block_m=224)
    first = ref.mhc_reference(**kwargs)
    second = ref.mhc_reference(**kwargs)
    first_logical = ref.logical_outputs(first, 224)
    second_logical = ref.logical_outputs(second, 224)
    for name, tensor in first_logical.items():
        ref.assert_bitwise_equal(tensor, second_logical[name], name)


@needs_cuda
@pytest.mark.parametrize('is_shifted', [False, True])
def test_reference_duplicate_tokens(is_shifted):
    """重复的输入行应产生逐字节一致的输出行。"""
    num_tokens, hidden = 513, 4096
    inputs, duplicates = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
    assert duplicates, 'make_inputs must produce duplicate rows for T=513'
    out = ref.mhc_reference(**dict(inputs, sf_layout='bf16'))
    for name, tensor in ref.logical_outputs(out).items():
        for token_idx in duplicates:
            ref.assert_bitwise_equal(tensor[token_idx], tensor[0], f'{name}, token={token_idx}')


@needs_cuda
def test_reference_diagnostics_are_consistent():
    """分段中间值应准确重现最终参考输出。"""
    num_tokens, hidden = 65, 4096
    for is_shifted in (False, True):
        inputs, _ = ref.make_inputs(num_tokens, hidden, 11, is_shifted)
        out, diag = ref.mhc_reference(
            **dict(inputs, sf_layout='extra', shared_sf_block_m=224),
            with_diagnostics=True)
        ref.assert_bitwise_equal(diag['z_bf16'], out['new_residual'], 'diag z')
        ref.assert_bitwise_equal(diag['y_bf16'], out['y_bf16'], 'diag y')
        ref.assert_bitwise_equal(diag['new_post_mix'], out['new_post_mix'], 'diag post')
        ref.assert_bitwise_equal(diag['new_comb_res_mix'], out['new_comb_res_mix'], 'diag comb')
        if is_shifted:
            ref.assert_bitwise_equal(diag['new_prev_mix'], out['new_prev_mix'], 'diag pre')
        # 投影消费的是按 BF16 舍入后的 Z，中间张量必须保持这一数据依赖。
        # 不得把舍入前的 FP32 值直接用于投影。

        z16 = diag['z_bf16'].float()
        assert torch.allclose(diag['proj'],
                              ref._exact_matmul(z16.flatten(1), inputs['fn'].mT),
                              rtol=2e-3, atol=2e-3)
        exact = ref._exact_matmul(z16.flatten(1).double(), inputs['fn'].mT.double())
        assert torch.allclose(diag['proj'].double(), exact, rtol=5e-6, atol=5e-6), \
            'reference projection must be IEEE fp32 (allow_tf32 disabled), not TF32'
        assert torch.allclose(diag['hc_sqr_sum'], z16.square().sum((1, 2)), rtol=1e-4)
        assert torch.allclose(diag['y_fp32'], out['y_bf16'].float(),
                              rtol=1e-2, atol=1e-2)
        report = ref.compare_diagnostics(diag, diag)
        assert all(v['calc_diff'] < 1e-12 for v in report.values())


@needs_cuda
@pytest.mark.parametrize('num_tokens', [1, 64, 65, 200, 1025])
@pytest.mark.parametrize('is_shifted', [False, True])
def test_reference_numeric_matrix_small(num_tokens, is_shifted):
    """独立运行两种模式、两种 H 和三组输出的参考计算。"""
    for hidden in (4096, 7168):
        for sf_layout in ('bf16', 'col', 'extra'):
            inputs, _ = ref.make_inputs(num_tokens, hidden, 2026 + num_tokens, is_shifted)
            outputs = ref.make_outputs(inputs, sf_layout)
            result = ref.mhc_reference(
                **dict(inputs, sf_layout=sf_layout,
                       shared_sf_block_m=outputs.get('shared_sf_block_m', 0)))
            logical = ref.logical_outputs(result, outputs.get('shared_sf_block_m'))
            for name, tensor in logical.items():
                assert torch.isfinite(tensor.float()).all(), f'{name} non-finite'
            assert logical['y_bf16'].shape == (num_tokens, hidden)
            if sf_layout != 'bf16':
                assert 'y_fp8' in logical
            del inputs, outputs, result, logical
        _empty_cache()


@needs_cuda
@pytest.mark.parametrize('hidden', ref.OFFICIAL_HIDDENS)
def test_reference_numeric_matrix_large(hidden):
    """运行完整矩阵中的大 token 形状，检查参考计算。"""
    tokens = (4096,) if _fast() else (4096, 32768)
    for num_tokens in tokens:
        for is_shifted in (False, True):
            start = _now()
            inputs, _ = ref.make_inputs(num_tokens, hidden, 99, is_shifted)
            outputs = ref.make_outputs(inputs, 'col')
            result = ref.mhc_reference(
                **dict(inputs, sf_layout='col'))
            logical = ref.logical_outputs(result)
            for name, tensor in logical.items():
                assert torch.isfinite(tensor.float()).all(), f'{name} non-finite'
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f'  reference T={num_tokens:6} H={hidden} '
                  f'{"shifted" if is_shifted else "normal "}: '
                  f'{_now() - start:.2f}s peak_alloc={peak:.2f}GiB')
            del inputs, outputs, result, logical
            _empty_cache()
            torch.cuda.reset_peak_memory_stats()




# ---------------------------------------------------------------------------
# 矩阵片段布局与数值转换契约
# ---------------------------------------------------------------------------


def _mmac_column_owner(tile: int, lane: int, element: int) -> int:
    """计算 HCU 映射下由 tile、lane 和元素下标共同确定的逻辑输出列。"""
    return tile * 16 + element * 4 + (lane // 16)


def test_primitives_mmac_c_mapping_differs_from_amd_contiguous_n():
    """HCU 的 C 分片为 n=e*4+lane/16；AMD 形式为 n=4*(lane/16)+e。"""
    lanes = range(64)
    differing = 0
    for lane in lanes:
        for element in range(4):
            hcu = (lane % 16, element * 4 + (lane // 16))
            amd = (lane % 16, (lane // 16) * 4 + element)
            assert 0 <= hcu[0] < 16 and 0 <= hcu[1] < 16
            if hcu != amd:
                differing += 1
    # 256个 lane/元素组合中有192个落在不同列；错误映射会静默置换输出列。
    # 仅 e=lane/16 时两种解释一致。

    assert differing == 192, differing
    # 两种解释覆盖相同的16×16位置集合，但对应关系不同。
    hcu_cells = sorted((lane % 16, e * 4 + lane // 16) for lane in lanes for e in range(4))
    amd_cells = sorted((lane % 16, (lane // 16) * 4 + e) for lane in lanes for e in range(4))
    assert hcu_cells == amd_cells == sorted((m, n) for m in range(16) for n in range(16))
    # 24个有效输出由第0个 tile 的16列与第1个 tile 的前8列构成。
    tile1_cols = sorted({_mmac_column_owner(1, lane, e) for lane in lanes
                         for e in range(4)})
    assert tile1_cols == list(range(16, 32))
    used_elements = sorted({e for lane in lanes for e in range(4)
                            if _mmac_column_owner(1, lane, e) < 24})
    # 第1个 tile 的有效列 n=0..7 对应各 lane 的累加器元素 e=0/1。
    # 四组 lane 都参与，但只使用第1个 tile 的一半累加器元素。
    assert used_elements == [0, 1], used_elements


@needs_cuda
def test_primitives_fp32_fn_precision_routes():
    """比较FP32、BF16 hi/lo、BF16与TF32投影权重表示的端到端输出误差。"""
    hidden, num_tokens = 4096, 200
    inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, True)
    fn = inputs['fn']

    hi = fn.bfloat16()
    lo = (fn - hi.float()).bfloat16()
    tf32_bits = fn.view(torch.int32) & -8192          # 保留10位尾数。
    routes = {
        'fp32': fn,
        'bf16_hilo': hi.float() + lo.float(),
        'bf16': hi.float(),
        'tf32': tf32_bits.view(torch.float32),
    }

    baseline = ref.mhc_reference(**dict(inputs, sf_layout='bf16'))
    base_logical = ref.logical_outputs(baseline)
    hilo_worst = None
    print(f'  {"route":10} {"proj max_abs":>14} {"proj calc_diff":>15} '
          f'{"y_bf16 calc_diff":>17} {"worse_output":>14}')
    for name, route_fn in routes.items():
        diag_kwargs = dict(inputs, sf_layout='bf16', with_diagnostics=True)
        diag_kwargs['fn'] = route_fn
        result, diag = ref.mhc_reference(**diag_kwargs)
        report = ref.diagnose_outputs(ref.logical_outputs(result), base_logical,
                                      (name, 'bf16', num_tokens, hidden))
        worst_name = max(report, key=lambda k: report[k]['calc_diff'])
        worst = report[worst_name]
        proj_exact = ref._exact_matmul(
            diag['z_bf16'].float().flatten(1).double(), fn.double().mT)
        proj_max_abs = float((diag['proj'].double() - proj_exact).abs().max())
        print(f'  {name:10} {proj_max_abs:14.3e} '
              f'{float(ref.calc_diff(diag["proj"].double(), proj_exact)):15.3e} '
              f'{report["y_bf16"]["calc_diff"]:17.3e} {worst_name:>14}')
        # 各种权重表示的全部输出均应满足误差门限。
        assert worst['calc_diff'] < worst['limit'], (name, worst_name, worst)
        if name == 'bf16_hilo':
            hilo_worst = worst['calc_diff']
    # BF16 hi/lo 路径的输出误差应低于门限的一半。
    assert hilo_worst is not None and hilo_worst < 5e-5 * 0.5, hilo_worst
    del inputs, baseline
    _empty_cache()


@needs_cuda
def test_primitives_bf16_rounding_is_rne_not_truncation():
    """FP32转BF16采用RNE舍入，与截断结果区分。"""
    torch.manual_seed(0)
    x = (torch.randn(4096, device='cuda', dtype=torch.float32)
         * torch.exp2(torch.randint(-8, 8, (4096,), device='cuda').float()))
    rne = x.bfloat16()
    trunc_bits = (x.view(torch.int32) & ~torch.tensor(0xFFFF, dtype=torch.int32,
                                                      device='cuda'))
    trunc = trunc_bits.view(torch.float32).bfloat16()
    differing = int((rne.view(torch.uint8) != trunc.view(torch.uint8)).sum().item())
    print(f'  RNE vs truncation differ on {differing}/{x.numel()} elements')
    assert differing > 0
    # Torch 的 BF16 转换采用 RNE，参考计算按相同位置舍入。
    assert torch.equal(rne, x.to(torch.bfloat16))


def test_primitives_padded_columns_contribute_nothing():
    """N=32 的 tile 中，24至31列必须补零，不参与有效输出。"""
    hidden = 4096
    fn = torch.randn((24, 4 * hidden), dtype=torch.float32) * 1e-2
    z = torch.randn((8, 4, hidden), dtype=torch.bfloat16).float()
    real = ref._exact_matmul(z.flatten(1).double(), fn.double().mT)
    padded = torch.zeros((32, 4 * hidden), dtype=torch.float32)
    padded[:24] = fn
    out = ref._exact_matmul(z.flatten(1).double(), padded.double().mT)
    assert torch.equal(out[:, :24], real)
    assert torch.count_nonzero(out[:, 24:]) == 0


@requires_native
@needs_cuda
@pytest.mark.parametrize('sf_layout', ['col', 'extra'])
@pytest.mark.parametrize('is_shifted', [False, True])
def test_fp8_native_matches_reference(launch_mode, sf_layout, is_shifted):
    """检查 FP8/UE8M0 的逻辑数值、SF 字节和写入范围。

    SF 字节要求完全一致，或仅在 amax 的二次幂边界处相差一个 UE8M0 指数步长，
    差异数量不超过每百万字节两个。该边界允许 collapse 舍入相差一个 ULP。
    y_fp8 另有对应的差异预算，因为 BF16 源值相差一个 ULP 可能改变 E4M3FN 舍入。
    数值误差和字节差异分别检查。"""
    import deepgemm
    tokens = (1, 65, 200) if (_fast() or launch_mode == 'fused') \
        else ref.OFFICIAL_TOKEN_COUNTS
    for num_tokens in tokens:
        for hidden in ref.OFFICIAL_HIDDENS:
            inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
            outputs = ref.make_outputs(inputs, sf_layout)
            block_m = outputs.get('shared_sf_block_m', 0)
            if sf_layout == 'extra':
                # 用确定的初值检查未被写入的行。
                outputs['y_shared_sf_storage'].zero_()
                before = outputs['y_shared_sf_storage'].clone()
            _call_native(deepgemm, inputs, outputs)
            expected = ref.mhc_reference(**dict(inputs, sf_layout=sf_layout,
                                                shared_sf_block_m=block_m))
            actual_logical = ref.logical_outputs(outputs)
            ref_logical = ref.logical_outputs(expected, block_m)
            report = ref.diagnose_outputs(actual_logical, ref_logical,
                                          ('fp8', sf_layout, num_tokens, hidden, is_shifted))
            worst = max(report, key=lambda k: report[k]['calc_diff'])
            print(f'  fp8 {sf_layout:5} T={num_tokens:6} H={hidden} '
                  f'{"shifted" if is_shifted else "normal "}: '
                  f'worst {worst} calc_diff={report[worst]["calc_diff"]:.3e} '
                  f'(limit {report[worst]["limit"]:.0e})')
            ref.check_correctness(actual_logical, ref_logical,
                                  ('fp8', sf_layout, num_tokens, hidden, is_shifted))
            # 比较 SF 的字节、布局、指数打包与分组 amax。
            # 字节差异单独检查，不能仅依赖整体数值误差。
            sf_checked = 0
            want_sf_map = ref.logical_outputs(expected, block_m)
            for sf_name in ('y_gemm_sf', 'y_routed_sf', 'y_shared_sf'):
                # y_shared_sf 是 storage[:T] 的别名；比较时通过逻辑输出应用行置换。
                # 两侧均使用 logical_outputs 读取。
                tensor = actual_logical.get(sf_name)
                want_sf = want_sf_map.get(sf_name)
                if not isinstance(tensor, torch.Tensor) or want_sf is None:
                    continue
                got_bytes = tensor.detach().contiguous().view(torch.uint8)
                want_bytes = want_sf.detach().contiguous().view(torch.uint8)
                assert got_bytes.shape == want_bytes.shape, (sf_name, got_bytes.shape,
                                                             want_bytes.shape)
                neq = got_bytes != want_bytes
                diff = int(neq.sum())
                if diff:
                    # collapse 相差一个 ULP 时，可能跨过二次幂 amax 边界并改变 ceil_to_ue8m0。
                    # 允许的指数差异必须恰好为一个步长。
                    # 任意更大的差异均不属于该舍入边界容差。

                    deltas = torch.unique(
                        (got_bytes.to(torch.int16) - want_bytes.to(torch.int16))[neq])
                    assert int(deltas.abs().max()) == 1, (
                        f'{sf_name}: {diff} bytes differ with deltas {deltas.tolist()} '
                        f'({launch_mode}, T={num_tokens}, H={hidden})')
                    assert diff * 500000 <= got_bytes.numel(), (
                        f'{sf_name}: {diff}/{got_bytes.numel()} bytes differ, over the '
                        f'2-per-million budget ({launch_mode}, T={num_tokens}, '
                        f'H={hidden})')
                    print(f'  SF note: {sf_name} {diff}/{got_bytes.numel()} bytes differ '
                          f'by one exponent step (amax boundary)')
                sf_checked += 1
            assert sf_checked >= 1, 'no SF tensor was compared against torch'
            # y_fp8 的容差仅适用于 BF16 源值相差一个 ULP 的元素。
            got_fp8 = actual_logical['y_fp8'].detach().contiguous().view(torch.uint8)
            want_fp8 = ref_logical['y_fp8'].detach().contiguous().view(torch.uint8)
            fp8_diff = int((got_fp8 != want_fp8).sum())
            budget = max(64, got_fp8.numel() // 20000)
            assert fp8_diff <= budget, (
                f'y_fp8: {fp8_diff} bytes differ from torch, budget {budget} '
                f'({launch_mode}, T={num_tokens}, H={hidden})')
            if sf_layout == 'extra':
                storage = outputs['y_shared_sf_storage']
                rows = ref.extra_sf_rows(outputs['y_bf16'], block_m)
                touched = torch.zeros_like(storage, dtype=torch.bool)
                touched[rows] = True
                assert torch.equal(storage[~touched], before[~touched]), \
                    'the shared SF write must not touch rows outside the permutation'
            del inputs, outputs, expected
            _empty_cache()


@requires_native
@needs_cuda
def test_fp8_only_matches_dual_output():
    """仅 FP8 输出与 BF16/FP8 双输出调用应产生相同的 FP8 字节。"""
    import deepgemm
    num_tokens, hidden = 200, 4096
    for is_shifted in (False, True):
        inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
        dual = ref.make_outputs(inputs, 'col')
        _call_native(deepgemm, inputs, dual)
        fp8_only = ref.make_outputs(inputs, 'col')
        _call_native(deepgemm, inputs, fp8_only, y_bf16=None)
        ref.assert_bitwise_equal(fp8_only['y_fp8'], dual['y_fp8'], 'y_fp8')
        ref.assert_bitwise_equal(fp8_only['y_gemm_sf'], dual['y_gemm_sf'], 'y_gemm_sf')
        del inputs, dual, fp8_only
        _empty_cache()


@requires_native
@needs_cuda
def test_native_shape_contract_errors():
    """非法输入组合应在 host 入口明确报错，不交给 kernel 处理。"""
    import deepgemm
    num_tokens, hidden = 65, 4096
    inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, False)
    outputs = ref.make_outputs(inputs, 'bf16')

    def expect_error(**overrides):
        with pytest.raises(Exception):
            _call_native(deepgemm, inputs, outputs, **overrides)

    # shifted 输入和输出必须成对提供。
    expect_error(new_prev_mix=torch.zeros((num_tokens, 4, 1), dtype=torch.float,
                                          device='cuda'))
    # 至少请求一种 Y 输出。
    expect_error(y_bf16=None)
    # FP8 输出必须配套缩放因子。
    fp8_out = torch.empty((num_tokens, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    expect_error(y_fp8=fp8_out)
    # hidden 维度不满足分块整除约束。
    bad, _ = ref.make_inputs(num_tokens, 4096 + 8, 2026, False)
    with pytest.raises(Exception):
        _call_native(deepgemm, bad, outputs)
    # 路由数必须为4。
    expect_error(hc_mult=2)
    # CPU 张量应在 host 入口被拒绝，不能作为 GPU 指针使用。
    cpu = {name: (value.cpu() if isinstance(value, torch.Tensor) else value)
           for name, value in inputs.items()}
    with pytest.raises(Exception, match='HIP device'):
        _call_native(deepgemm, cpu, outputs)
    # 所有输入都必须与主输入位于同一 GPU 设备。
    mixed = dict(inputs)
    mixed['rmsnorm_weight'] = inputs['rmsnorm_weight'].cpu()
    with pytest.raises(Exception):
        _call_native(deepgemm, mixed, outputs)
    # H=0 必须被拒绝，不能仅检查整除关系。
    empty = {name: (value[:, :0] if isinstance(value, torch.Tensor) and value.dim() == 2
                    and value.shape[1] == hidden else value)
             for name, value in inputs.items()}
    with pytest.raises(Exception):
        _call_native(deepgemm, empty, outputs)
    del inputs, outputs, bad, cpu, mixed, empty
    _empty_cache()


@requires_native
@needs_cuda
def test_fused_single_launch():
    """fused 路径每次调用只发射一次主计算 kernel。

    本用例临时设置 MEGA_MHC_LAUNCH=fused，结束后恢复原环境变量。
    权重打包与主计算分别计数。"""
    import deepgemm

    previous = os.environ.get('MEGA_MHC_LAUNCH')
    os.environ['MEGA_MHC_LAUNCH'] = 'fused'
    try:
        num_tokens, hidden = 200, 4096
        inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, True)
        outputs = ref.make_outputs(inputs, 'bf16')
        kwargs = {**inputs, **outputs}
        kwargs.pop('y_shared_sf_storage', None)
        deepgemm.mega_mhc(**kwargs)                     # 预热并完成延迟初始化。
        expected = ref.mhc_reference(**dict(inputs, sf_layout='bf16'))
        ref.check_correctness(ref.logical_outputs(outputs),
                              ref.logical_outputs(expected),
                              ('fused-single-launch', num_tokens))
        torch.cuda.synchronize()

        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                deepgemm.mega_mhc(**kwargs)
            torch.cuda.synchronize()
        launches = {}
        for evt in prof.key_averages():
            if evt.self_device_time_total and 'mega_mhc' in evt.key:
                launches[evt.key] = evt.count
        print('  device launches per 5 calls:', launches)
        assert launches, 'the profiler saw no mega_mhc kernel'
        main = {k: v for k, v in launches.items() if 'mega_mhc_main_kernel' in k}
        # 每次调用只发射一次 main，且没有独立 reduce；逐次权重打包另计为 pack kernel。
        assert len(main) == 1, launches
        assert list(main.values())[0] == 5, launches
        others = [k for k in launches if 'mega_mhc_main_kernel' not in k]
        for key in others:
            assert 'pack_fn_kernel' in key, key
            assert launches[key] == 5, launches
        del inputs, outputs, expected
    finally:
        if previous is None:
            os.environ.pop('MEGA_MHC_LAUNCH', None)
        else:
            os.environ['MEGA_MHC_LAUNCH'] = previous
    _empty_cache()





@requires_native
@needs_cuda
def test_native_empty_input():
    """T=0 时不发射计算，直接返回且不修改输出。"""
    import deepgemm
    hidden = 4096
    inputs, _ = ref.make_inputs(1, hidden, 2026, False)
    zero = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            zero[name] = value[:0]
        else:
            zero[name] = value
    outputs = dict(
        new_residual=torch.empty((0, 4, hidden), dtype=torch.bfloat16, device='cuda'),
        new_prev_mix=None,
        new_post_mix=torch.empty((0, 4, 1), dtype=torch.float, device='cuda'),
        new_comb_res_mix=torch.empty((0, 4, 4), dtype=torch.float, device='cuda'),
        y_bf16=torch.empty((0, hidden), dtype=torch.bfloat16, device='cuda'),
    )
    before = {n: t.clone() for n, t in outputs.items() if isinstance(t, torch.Tensor)}
    deepgemm.mega_mhc(**{**zero, **outputs})
    torch.cuda.synchronize()
    for name, tensor in outputs.items():
        if isinstance(tensor, torch.Tensor):
            assert tensor.numel() == 0
            ref.assert_bitwise_equal(tensor, before[name], name)
    _empty_cache()


@requires_native
@needs_cuda
def test_native_multi_stream_isolation(launch_mode):
    """不同 stream 的并发调用不得共享 workspace 或输出状态。"""
    import deepgemm
    num_tokens, hidden = 200, 4096
    inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, True)
    reference = ref.mhc_reference(**dict(inputs, sf_layout='bf16'))
    expected = ref.logical_outputs(reference)

    # 默认 stream 上的结果先通过独立参考的数值门限。
    # 跨 stream 的逐字节比较以同实现结果为基准。
    # 不同累加顺序或 FMA 可能使 native 与 Torch 的 Z 在个别位置相差一个 BF16 ULP。
    # native 与参考采用数值误差门，stream 隔离则采用同实现的字节一致性门。

    base = ref.make_outputs(inputs, 'bf16')
    _call_native(deepgemm, inputs, base)
    ref.check_correctness(ref.logical_outputs(base), expected, ('multi_stream', num_tokens))
    base_logical = {n: t.clone() for n, t in ref.logical_outputs(base).items()}

    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            o = ref.make_outputs(inputs, 'bf16')
            _call_native(deepgemm, inputs, o)
            outs.append(o)
    for stream in streams:
        stream.synchronize()
    for idx, o in enumerate(outs):
        for name, tensor in ref.logical_outputs(o).items():
            ref.assert_bitwise_equal(tensor, base_logical[name], f'stream {idx}: {name}')
    del inputs, base, outs, reference, base_logical
    _empty_cache()


@requires_native
@needs_cuda
def test_native_graph_capture_and_replay(launch_mode):
    """完成预热后捕获 Graph，重放结果应逐字节一致。"""
    import deepgemm
    num_tokens, hidden = 200, 4096
    inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, True)
    capture_stream = torch.cuda.Stream()

    # 捕获前先在目标 stream 上预热。
    with torch.cuda.stream(capture_stream):
        warm = ref.make_outputs(inputs, 'bf16')
        _call_native(deepgemm, inputs, warm)
    capture_stream.synchronize()
    expected = ref.logical_outputs(warm)
    expected = {k: v.clone() for k, v in expected.items()}

    graph = torch.cuda.CUDAGraph()
    graph_out = ref.make_outputs(inputs, 'bf16')
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            _call_native(deepgemm, inputs, graph_out)
    graph.replay()
    capture_stream.synchronize()
    for name, tensor in ref.logical_outputs(graph_out).items():
        ref.assert_bitwise_equal(tensor, expected[name], f'graph replay: {name}')

    # 再次重放应产生相同字节，不得消费残留的计数器或 scratch。
    graph_out['new_prev_mix'].zero_()
    graph.replay()
    capture_stream.synchronize()
    for name, tensor in ref.logical_outputs(graph_out).items():
        ref.assert_bitwise_equal(tensor, expected[name], f'graph replay 2: {name}')
    del inputs, warm, graph_out
    _empty_cache()


@requires_native
@needs_cuda
def test_native_interleaved_sizes_reuse_workspace():
    """插入不同尺寸的调用不得污染下一次相同尺寸调用的结果。"""
    import deepgemm
    hidden = 4096
    big, _ = ref.make_inputs(256, hidden, 2026, True)
    small, _ = ref.make_inputs(17, hidden, 2027, True)
    big_out = ref.make_outputs(big, 'bf16')
    _call_native(deepgemm, big, big_out)
    first = {n: t.clone() for n, t in ref.logical_outputs(big_out).items()}
    small_out = ref.make_outputs(small, 'bf16')
    _call_native(deepgemm, small, small_out)
    _call_native(deepgemm, big, big_out)
    for name, tensor in ref.logical_outputs(big_out).items():
        ref.assert_bitwise_equal(tensor, first[name], name)
    del big, small, big_out, small_out, first
    _empty_cache()


# ---------------------------------------------------------------------------
# native 正确性与行为测试
# ---------------------------------------------------------------------------

@pytest.fixture(params=['split', 'fused'])
def launch_mode(request, monkeypatch):
    """分别运行默认 split 路径和 fused 路径。

    dispatch 每次调用读取 MEGA_MHC_LAUNCH；monkeypatch 在用例结束后恢复环境变量。"""
    monkeypatch.setenv('MEGA_MHC_LAUNCH', request.param)
    return request.param


@requires_native
@needs_cuda
def test_native_prepare_fn_matches_per_call_packing(monkeypatch):
    """显式预打包与逐次打包的输出应逐字节一致。

    权重变化后由调用方重新 prepare；尺寸不符的 packed 缓冲必须被拒绝。"""
    monkeypatch.setenv('MEGA_MHC_LAUNCH', 'split')
    import deepgemm
    for num_tokens, hidden, is_shifted in ((1, 4096, True), (200, 4096, True),
                                           (200, 4096, False)):
        inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
        per_call = ref.make_outputs(inputs, 'bf16')
        _call_native(deepgemm, inputs, per_call)
        per_call_logical = {n: t.clone() for n, t in ref.logical_outputs(per_call).items()}

        packed = deepgemm.mega_mhc_prepare_fn(inputs['fn'])
        assert packed.dtype == torch.bfloat16 and packed.is_contiguous()
        assert packed.numel() == 2 * 24 * 4 * hidden
        once = ref.make_outputs(inputs, 'bf16')
        kwargs = {**inputs, **once, 'fn_packed': packed}
        kwargs.pop('y_shared_sf_storage', None)
        deepgemm.mega_mhc(**kwargs)
        for name, tensor in ref.logical_outputs(once).items():
            ref.assert_bitwise_equal(tensor, per_call_logical[name], name)
        for _ in range(2):                       # 多次复用预打包缓冲的结果应保持一致。
            deepgemm.mega_mhc(**kwargs)
            for name, tensor in ref.logical_outputs(once).items():
                ref.assert_bitwise_equal(tensor, per_call_logical[name], name)
        with pytest.raises(Exception):
            deepgemm.mega_mhc(**{**inputs, **ref.make_outputs(inputs, 'bf16'),
                                 'fn_packed': packed[:100]})
        del inputs, per_call, once, packed, per_call_logical
        _empty_cache()


@requires_native
@needs_cuda
@pytest.mark.parametrize('is_shifted', [False, True])
def test_bf16_native_matches_reference(launch_mode, is_shifted):
    """运行两种模式、两种 H 和完整 token 数矩阵的 native BF16 对拍。

    分别运行 split/fused，两条路径均与独立参考比较。"""
    import deepgemm
    tokens = (1, 64, 65, 200, 1025) if _fast() else ref.OFFICIAL_TOKEN_COUNTS
    worst = 0.0
    for num_tokens in tokens:
        for hidden in ref.OFFICIAL_HIDDENS:
            inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
            outputs = ref.make_outputs(inputs, 'bf16')
            deepgemm.mega_mhc(**{**inputs, **outputs})
            expected = ref.mhc_reference(**dict(inputs, sf_layout='bf16'))
            actual = ref.logical_outputs(outputs)
            report = ref.diagnose_outputs(actual, ref.logical_outputs(expected),
                                          ('bf16', num_tokens, hidden, is_shifted))
            worst_name = max(report, key=lambda k: report[k]['calc_diff'])
            worst = max(worst, report[worst_name]['calc_diff'])
            print(f'  bf16 T={num_tokens:6} H={hidden} '
                  f'{"shifted" if is_shifted else "normal "}: '
                  f'worst {worst_name} calc_diff={report[worst_name]["calc_diff"]:.3e}')
            ref.check_correctness(actual, ref.logical_outputs(expected),
                                  ('bf16', num_tokens, hidden, is_shifted))
            del inputs, outputs, expected
            _empty_cache()
    assert worst < 5e-5


@requires_native
@needs_cuda
def test_bf16_native_repeatability(launch_mode):
    """复用同一组缓冲重复调用，输出应逐字节一致。"""
    import deepgemm
    num_tokens, hidden = 200, 4096
    for is_shifted in (False, True):
        inputs, _ = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
        outputs = ref.make_outputs(inputs, 'bf16')
        kwargs = {**inputs, **outputs}
        deepgemm.mega_mhc(**kwargs)
        first = {name: t.clone() for name, t in ref.logical_outputs(outputs).items()}
        for _ in range(3):
            deepgemm.mega_mhc(**kwargs)
            for name, tensor in ref.logical_outputs(outputs).items():
                ref.assert_bitwise_equal(tensor, first[name], name)
        del inputs, outputs, first
        _empty_cache()


@requires_native
@needs_cuda
@pytest.mark.parametrize('is_shifted', [False, True])
def test_bf16_native_duplicate_tokens(launch_mode, is_shifted):
    """重复的输入行经 native kernel 后应产生逐字节一致的输出行。"""
    import deepgemm
    num_tokens, hidden = 513, 4096
    inputs, duplicates = ref.make_inputs(num_tokens, hidden, 2026, is_shifted)
    assert duplicates
    outputs = ref.make_outputs(inputs, 'bf16')
    deepgemm.mega_mhc(**{**inputs, **outputs})
    for name, tensor in ref.logical_outputs(outputs).items():
        for token_idx in duplicates:
            ref.assert_bitwise_equal(tensor[token_idx], tensor[0],
                                     f'{name}, token={token_idx}')
    del inputs, outputs
    _empty_cache()


@requires_native
@needs_cuda
def test_bf16_native_tail_shapes(launch_mode):
    """检查 token 分块边界附近的尾块形状。"""
    import deepgemm
    hidden = 4096
    for num_tokens in (1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 79, 80):
        for is_shifted in (False, True):
            inputs, _ = ref.make_inputs(num_tokens, hidden, 7, is_shifted)
            outputs = ref.make_outputs(inputs, 'bf16')
            deepgemm.mega_mhc(**{**inputs, **outputs})
            expected = ref.mhc_reference(**dict(inputs, sf_layout='bf16'))
            ref.check_correctness(ref.logical_outputs(outputs),
                                  ref.logical_outputs(expected),
                                  ('tail', num_tokens, hidden, is_shifted))
            del inputs, outputs, expected
            _empty_cache()
