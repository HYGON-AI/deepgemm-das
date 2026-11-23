# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Non-paged MQA correctness and performance checks.

Run the regression tests with pytest, or run a correctness/timing matrix directly:
    python -m pytest -q op_test/test_mqa_logits.py
    python op_test/test_mqa_logits.py --dtype bf16 --output both
    python op_test/test_mqa_logits.py --suite lightop --dtype int8 --csv results.csv
    python op_test/test_mqa_logits.py --shape 1 4096 128 --no-benchmark

The Lightop preset uses test/mqa_logits/test_mqa_logits.py's M/N/H matrix.
INT8/FP8 use per-token scales; Q scales are folded into Weights. References
use the stored quantized values, separating kernel error from quantization error.
Eager timings use GPU events around API calls; Graph timings use replay events.
Results go to stdout, with optional CSV output. No Lightop installation is needed.
"""

import argparse
from contextlib import contextmanager
import csv
from functools import partial
import itertools
import json
from pathlib import Path
import statistics

import pytest
import torch

# Adapted from tensor_scores/tests/test_correctness.py.
@pytest.fixture(autouse=True)
def supported_device(request):
    if not torch.version.hip:
        pytest.skip("HIP PyTorch is required")
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except RuntimeError as error:
        pytest.skip("HIP device unavailable: " + str(error))
    if arch not in {"gfx936", "gfx938"}:
        pytest.skip("MQA requires gfx936 or gfx938")
    dtype = getattr(request.node, "callspec", None)
    dtype = dtype.params.get("dtype") if dtype else None
    if dtype == "fp8" and arch != "gfx938":
        pytest.skip("FP8 MQA requires gfx938")


def call_mqa_logits(*args, **kwargs):
    from deepgemm import mqa_logits
    return mqa_logits(*args, **kwargs)


DTYPES = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn, "int8": torch.int8}


def inputs(m, n, h, dtype, signed=False):
    torch.manual_seed(20261022 + m + n + h)
    if dtype == "int8":
        q = torch.randint(-8, 9, (m, h, 128), device="cuda", dtype=torch.int8)
        k = torch.randint(-8, 9, (n, 128), device="cuda", dtype=torch.int8)
    else:
        q = torch.randn(m, h, 128, device="cuda").to(DTYPES[dtype])
        k = torch.randn(n, 128, device="cuda").to(DTYPES[dtype])
    w = torch.randn(m, h, device="cuda")
    rows = torch.arange(m, dtype=torch.int32, device="cuda")
    ks = (rows % 5).clamp(max=n)
    ke = (n - rows % 7).clamp(min=0)
    if signed and m > 1:
        ks = ks - 3
        ke[::7] = -2
        ke[1::7] = n + 9
        ks[2::7] = n + 3
    elif signed:
        ks = ks - 1
        ke = ke + 1
    scale = torch.rand(n, device="cuda") * 0.1 + 0.01 if dtype in {"int8", "fp8"} else None
    return q, k, w, ks, ke, scale


def reference(q, k, w, ks, ke, scale):
    # Chunk N so the reference remains usable for long KV performance cases.
    chunks = []
    for start in range(0, k.shape[0], 512):
        score = torch.einsum("mhd,nd->mhn", q.float(), k[start:start + 512].float())
        chunks.append((score.relu() * w.unsqueeze(-1)).sum(dim=1))
    out = torch.cat(chunks, dim=1) if chunks else w.new_empty((q.shape[0], 0))
    if scale is not None:
        out = out * scale[None, :]
    columns = torch.arange(k.shape[0], device=q.device)[None, :]
    valid = (columns >= ks[:, None].clamp(0, k.shape[0])) & (columns < ke[:, None].clamp(0, k.shape[0]))
    return out.masked_fill(~valid, -float("inf")), valid


def check(actual, expected, valid, clean=True):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    if clean:
        assert torch.equal(torch.isneginf(actual), ~valid)
    if valid.any():
        a, b = actual[valid], expected[valid]
        assert torch.isfinite(a).all()
        relative_max = (a - b).abs().max() / b.abs().max().clamp_min(1)
        assert relative_max < 3e-4, float(relative_max)
        torch.testing.assert_close(a, b, atol=0.03, rtol=5e-4)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "int8"])
@pytest.mark.parametrize("shape", [(1, 33, h) for h in (1, 2, 4, 8, 16, 32, 64, 128)] +
                         [(16, 512, 8), (17, 513, 64), (64, 1024, 8),
                          (129, 1025, 8), (16, 9216, 8), (256, 1024, 32)])
def test_reference(dtype, shape):
    args = inputs(*shape, dtype, signed=True)
    expected, valid = reference(*args)
    check(call_mqa_logits(*args), expected, valid)


@pytest.mark.parametrize('h', [64, 128])
@pytest.mark.parametrize('n', [64, 65, 4096])
@pytest.mark.parametrize('output', ['dense_clean', 'dense_valid', 'compact'])
def test_bf16_smallkv_scheduling_boundaries(h, n, output):
    # H128 covers the gfx936 spill fix; H64/H128 cover the gfx938 load boundary.
    # N65 checks the tail route, which retains its original accumulation order.
    args = inputs(1, n, h, 'bf16', signed=True)
    if output == 'compact':
        result = call_mqa_logits(*args, max_seqlen_k=n)
        check_compact(result, args, n)
    else:
        clean = output == 'dense_clean'
        buffer = torch.full((2, n + 7), 1234.0, device='cuda')
        result = call_mqa_logits(*args, clean_logit=clean, D_out=buffer)
        expected, valid = reference(*args)
        check(result, expected, valid, clean)
        assert torch.all(buffer[1:, :] == 1234)
        assert torch.all(buffer[:, n:] == 1234)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "int8"])
@pytest.mark.parametrize("clean", [False, True])
def test_strides_and_out(dtype, clean):
    q, k, w, ks, ke, scale = inputs(17, 65, 8, dtype, signed=True)
    padded_weights = torch.full((17, 13), 777.0, device="cuda")
    padded_weights[:, :8] = w
    w = padded_weights[:, :8]
    output = torch.full((19, 72), 1234.0, device="cuda")
    result = call_mqa_logits(q, k, w, ks, ke, scale, clean, output)
    assert result.data_ptr() == output.data_ptr()
    expected, valid = reference(q, k, w, ks, ke, scale)
    check(result, expected, valid, clean)
    assert torch.all(output[17:, :] == 1234)
    assert torch.all(output[:, 65:] == 1234)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "int8"])
@pytest.mark.parametrize("shape", [(0, 64, 8), (16, 0, 8)])
def test_empty(dtype, shape):
    result = call_mqa_logits(*inputs(*shape, dtype))
    assert result.shape == shape[:2]
    assert result.numel() == 0


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "int8"])
def test_current_stream(dtype):
    args = inputs(17, 513, 8, dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = call_mqa_logits(*args)
        expected, valid = reference(*args)
    stream.synchronize()
    check(result, expected, valid)


def test_rejects_invalid_contract():
    q, k, w, ks, ke, _ = inputs(16, 64, 8, "bf16")
    with pytest.raises(RuntimeError, match="does not accept kv_scale"):
        call_mqa_logits(q, k, w, ks, ke, torch.ones(64, device="cuda"))
    with pytest.raises(RuntimeError, match="dtype"):
        call_mqa_logits(q, k, w, ks.long(), ke)
    with pytest.raises(RuntimeError, match="Q/K dtype must match"):
        call_mqa_logits(q, k.float(), w, ks, ke)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rejects_removed_dtype(dtype):
    q, k, w, ks, ke, _ = inputs(16, 64, 8, "bf16")
    with pytest.raises(RuntimeError, match="supports BF16, INT8 and FP8 E4M3FN"):
        call_mqa_logits(q.to(dtype), k.to(dtype), w, ks, ke)


@pytest.mark.parametrize("setting", [None, "0", "1", "invalid"])
@pytest.mark.parametrize("dtype", ["bf16", "fp8", "int8"])
def test_removed_environment_switch_is_ignored(monkeypatch, setting, dtype):
    if setting is None:
        monkeypatch.delenv("LIGHTOP_MQA_LOGITS_BETA", raising=False)
    else:
        monkeypatch.setenv("LIGHTOP_MQA_LOGITS_BETA", setting)
    args = inputs(17, 65, 8, dtype, signed=True)
    expected, valid = reference(*args)
    check(call_mqa_logits(*args), expected, valid)


def test_no_lightop_import():
    import sys
    assert "lightop" not in sys.modules


def check_compact(actual, args, width):
    q, k, w, ks, ke, scale = args
    dense, _ = reference(*args)
    assert actual.shape == (q.shape[0], width)
    assert actual.dtype == torch.float32
    assert actual.stride(0) == (width + 255) // 256 * 256
    assert not actual.is_inference()
    for row in range(q.shape[0]):
        start = int(ks[row].clamp(0, k.shape[0]))
        end = int(ke[row].clamp(0, k.shape[0]))
        count = max(0, end - start)
        if count:
            torch.testing.assert_close(actual[row, :count], dense[row, start:end], atol=0.03, rtol=5e-4)


@pytest.mark.parametrize("dtype,shape", [
    pytest.param("bf16", (1023, 138, 16), id="bf16-m1023-n138-h16"),
    pytest.param("bf16", (1010, 65, 128), id="bf16-m1010-n65-h128"),
    pytest.param("int8", (17, 1, 16), id="int8-m17-n1-h16"),
    pytest.param("int8", (64, 15, 128), id="int8-m64-n15-h128"),
])
@pytest.mark.parametrize("output", ["dense_clean", "dense_valid", "compact"])
def test_shared_dispatch_short_n_and_tails(dtype, shape, output):
    # Signed windows exercise clamping, empty rows and masked tail tiles.
    # Keep packed weights/output so these cases exercise the measured layouts.
    args = inputs(*shape, dtype, signed=True)
    if output == "compact":
        width = shape[1]
        result = call_mqa_logits(*args, max_seqlen_k=width)
        check_compact(result, args, width)
    else:
        clean = output == "dense_clean"
        result = call_mqa_logits(*args, clean_logit=clean)
        expected, valid = reference(*args)
        check(result, expected, valid, clean)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('h', [1, 8, 32, 128])
@pytest.mark.parametrize('clean', [False, True])
def test_compact_signed_windows_and_tail(dtype, h, clean):
    args = inputs(17, 257, h, dtype, signed=True)
    result = call_mqa_logits(*args, clean_logit=clean, max_seqlen_k=259)
    check_compact(result, args, 259)


@pytest.mark.parametrize('dtype', DTYPES)
def test_compact_rejects_d_out(dtype):
    args = inputs(17, 257, 8, dtype)
    with pytest.raises(ValueError, match='D_out is unsupported'):
        call_mqa_logits(*args, D_out=torch.empty(17, 257, device='cuda'), max_seqlen_k=257)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('clean', [False, True])
def test_graph_replay_reads_updated_device_windows(dtype, compact, clean):
    args = inputs(17, 257, 8, dtype, signed=True)
    options = {'clean_logit': clean}
    if compact:
        options['max_seqlen_k'] = 257
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call_mqa_logits(*args, **options)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call_mqa_logits(*args, **options)
    args[3].fill_(0)
    args[4].fill_(257)
    graph.replay()
    torch.cuda.synchronize()
    if compact:
        check_compact(result, args, 257)
    else:
        expected, valid = reference(*args)
        check(result, expected, valid, clean)


@pytest.mark.parametrize('m', [16, 17])
@pytest.mark.parametrize('n', [4096, 4097])
def test_bf16_compact_window_reload_edges(m, n):
    # Cover every M/N tail specialization of the grouped M16/BN64 path.
    args = inputs(m, n, 8, 'bf16', signed=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call_mqa_logits(*args, max_seqlen_k=n)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call_mqa_logits(*args, max_seqlen_k=n)
    graph.replay()
    torch.cuda.synchronize()
    check_compact(result, args, n)

    args[3].fill_(0)
    args[4].fill_(n)
    graph.replay()
    torch.cuda.synchronize()
    check_compact(result, args, n)


# The standalone runner follows Lightop's quantization and windowed-dot checks.
# Keep the existing Tensor-scores-derived regression cases above independently
# runnable by pytest, without making performance measurements part of pytest.
QUICK_SHAPES = [(1, 33, h) for h in (1, 2, 4, 8, 16, 32, 64, 128)] + [
    (1, 64, 128), (1, 65, 128), (1, 4096, 64), (1, 4096, 128), (1, 65536, 128),
    (16, 64, 8), (16, 4096, 8), (17, 513, 64), (64, 1024, 8),
    (127, 4096, 64), (256, 1024, 32),
]
LIGHTOP_SHAPES = list(itertools.product(
    (16, 32, 64, 127), (66, 4096, 8192, 16384, 32768, 65536, 131072), (8, 64)))
WINDOWS = ('full', 'causal', 'sliding', 'signed')


@contextmanager
def reference_precision():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def quantize_per_token(value, dtype):
    limit = 127.0 if dtype == 'int8' else 448.0
    scale = value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / limit
    quantized = value / scale
    if dtype == 'int8':
        quantized = quantized.round().clamp(-128, 127)
    return quantized.to(DTYPES[dtype]), scale.squeeze(-1).contiguous()


def matrix_inputs(m, n, h, dtype, window, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn((m, h, 128), device='cuda', generator=generator)
    k = torch.randn((n, 128), device='cuda', generator=generator)
    weights = torch.randn((m, h), device='cuda', generator=generator)
    scale = None
    if dtype == 'bf16':
        q, k = q.bfloat16(), k.bfloat16()
    else:
        q, q_scale = quantize_per_token(q, dtype)
        k, scale = quantize_per_token(k, dtype)
        weights = weights * q_scale

    # Generate bounds on CPU outside all timed regions. Compact capacity is
    # known from this test fixture, never inferred inside the production API.
    rows = torch.arange(m, dtype=torch.int64)
    ks = torch.zeros(m, dtype=torch.int64)
    ke = torch.full((m,), n, dtype=torch.int64)
    if window == 'causal':
        ke = (n - m + rows + 1).clamp(0, n)
    elif window == 'sliding':
        width = min(n, max(1, n // 4 + 1))
        ks = (rows * 103 + 3) % (n - width + 1)
        ke = ks + (width - rows % 17).clamp_min(0)
    elif window == 'signed':
        ks = rows % 9 - 3
        ke = n - rows % 13
        if m > 1:
            ke[::7] = -2
            ke[1::7] = min(n + 9, 2**31 - 1)
            ks[2::7] = min(n + 3, 2**31 - 1)
        else:
            ke.fill_(min(n + 9, 2**31 - 1))
    elif window != 'full':
        raise ValueError('unknown window: ' + window)
    lengths = (ke.clamp(0, n) - ks.clamp(0, n)).clamp_min(0)
    width = max(1, int(lengths.max()))
    return (q, k, weights, ks.to(device='cuda', dtype=torch.int32),
            ke.to(device='cuda', dtype=torch.int32), scale), width


def check_matrix_output(actual, args, output, clean, width):
    """Check every defined output, with a bounded FP32 reference workspace."""
    q, k, weights, ks, ke, scale = args
    m, h, _ = q.shape
    n = k.shape[0]
    assert actual.dtype == torch.float32 and actual.device == q.device
    assert actual.shape == (m, n if output == 'dense' else width)
    if output == 'compact':
        assert actual.stride() == ((width + 255) // 256 * 256, 1)
        assert not actual.is_inference()
    start, end = ks.long().clamp(0, n), ke.long().clamp(0, n)
    assert output != 'compact' or int((end - start).clamp_min(0).max()) <= width
    chunk = max(1, min(512, (32 * 1024 * 1024) // (m * h * 4)))
    q_float = q.float().reshape(m * h, 128)
    max_abs = max_ref = 0.0
    squared_error = energy = 0.0
    valid_count = 0
    with reference_precision():
        for begin in range(0, n, chunk):
            stop = min(n, begin + chunk)
            dot = (q_float @ k[begin:stop].float().T).view(m, h, stop - begin)
            expected = (dot.relu() * weights[:, :, None]).sum(dim=1)
            if scale is not None:
                expected = expected * scale[None, begin:stop]
            columns = torch.arange(begin, stop, device=q.device)[None, :]
            valid = (columns >= start[:, None]) & (columns < end[:, None])
            if output == 'dense':
                values = actual[:, begin:stop]
                if clean:
                    assert torch.equal(torch.isneginf(values), ~valid), 'dense mask differs'
                a = values[valid]
            else:
                # Gather only valid prefixes; compact padding is undefined.
                rr, cc = valid.nonzero(as_tuple=True)
                a = actual[rr, begin + cc - start[rr]]
            b = expected[valid]
            if not a.numel():
                continue
            torch.testing.assert_close(a, b, atol=0.03, rtol=5e-4)
            assert torch.isfinite(a).all(), 'nonfinite valid output'
            delta = a.double() - b.double()
            max_abs = max(max_abs, float(delta.abs().max()))
            max_ref = max(max_ref, float(b.abs().max()))
            squared_error += float(delta.square().sum())
            energy += float(a.double().square().sum() + b.double().square().sum())
            valid_count += a.numel()
    normalized = max_abs / max(1.0, max_ref)
    assert normalized < 3e-4, 'normalized maximum error: {}'.format(normalized)
    # This is Lightop's 1 - 2<x,y>/(||x||^2+||y||^2), computed stably at zero.
    return dict(valid_values=valid_count, max_abs_error=max_abs,
                normalized_max_error=normalized, diff=squared_error / max(energy, 1e-30))


def benchmark_call(call, actual, args, output, clean, timing, warmup, rounds, iterations):
    for _ in range(warmup):
        result = call()
    torch.cuda.synchronize()
    graph = None
    nodes = 0
    if timing in ('graph', 'both'):
        expected_output = actual.clone()
        # Bound capture allocations for long dense outputs. Compact allocations
        # remain inside capture, matching the library-owned output contract.
        output_bytes = actual.shape[0] * actual.stride(0) * 4 + 1024
        nodes = max(1, min(32, (64 * 1024 * 1024) // output_bytes))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result = call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(nodes):
                result = call()
        graph.replay()
        torch.cuda.synchronize()
        n = args[1].shape[0]
        columns = torch.arange(actual.shape[1], device=actual.device)[None, :]
        start, end = args[3].clamp(0, n), args[4].clamp(0, n)
        valid = (columns < (end - start).clamp_min(0)[:, None] if output == 'compact'
                 else (columns >= start[:, None]) & (columns < end[:, None]))
        torch.testing.assert_close(result[valid], expected_output[valid], atol=0.03, rtol=5e-4)
        if output == 'dense' and clean:
            assert torch.equal(torch.isneginf(result), ~valid), 'Graph dense mask differs'

    samples = {kind: [] for kind in ('eager', 'graph') if timing in (kind, 'both')}
    start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    try:
        for sample in range(rounds):
            order = tuple(samples) if sample % 2 == 0 else tuple(reversed(samples))
            for kind in order:
                start_event.record()
                if kind == 'eager':
                    for _ in range(iterations):
                        result = call()
                    count = iterations
                else:
                    for _ in range(iterations):
                        graph.replay()
                    count = iterations * nodes
                end_event.record()
                end_event.synchronize()
                us = start_event.elapsed_time(end_event) * 1000.0 / count
                assert us > 0, 'nonpositive event timing'
                samples[kind].append(us)
    finally:
        if graph is not None:
            graph.reset()
    values = dict(graph_nodes=nodes)
    for kind, times in samples.items():
        name = 'eager_event' if kind == 'eager' else 'graph'
        values[name + '_us'] = statistics.median(times)
        values[name + '_samples_us'] = times
    return values


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(1, 65, 128), (17, 257, 8)])
@pytest.mark.parametrize('window', ['sliding', 'signed'])
@pytest.mark.parametrize('output,clean', [('dense', True), ('dense', False), ('compact', False)])
def test_quantized_matrix_reference(dtype, shape, window, output, clean):
    args, width = matrix_inputs(*shape, dtype, window, seed=17)
    options = dict(clean_logit=clean)
    if output == 'compact':
        options['max_seqlen_k'] = width
    actual = call_mqa_logits(*args, **options)
    check_matrix_output(actual, args, output, clean, width)


@pytest.mark.parametrize('dtype', ['fp8'])
@pytest.mark.parametrize('output', ['dense', 'compact'])
def test_fp8_optional_scale(dtype, output):
    args = inputs(17, 65, 8, dtype, signed=True)
    args = args[:-1] + (None,)
    options = dict(max_seqlen_k=65) if output == 'compact' else {}
    actual = call_mqa_logits(*args, **options)
    check_matrix_output(actual, args, output, True, 65)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dtype', choices=('all', 'bf16', 'int8', 'fp8'), default='all')
    parser.add_argument('--output', choices=('both', 'dense', 'compact'), default='both')
    parser.add_argument('--suite', choices=('quick', 'lightop'), default='quick')
    parser.add_argument('--shape', nargs=3, action='append', type=positive_int, metavar=('M', 'N', 'H'),
                        help='repeatable; overrides --suite; D is always 128')
    parser.add_argument('--window', choices=('all',) + WINDOWS, default='causal')
    parser.add_argument('--clean-logit', choices=('true', 'false'), default='true', help='dense only')
    parser.add_argument('--reuse-output', action='store_true', help='reuse D_out for dense timing')
    parser.add_argument('--timing', choices=('eager', 'graph', 'both'), default='both')
    parser.add_argument('--warmup', type=positive_int, default=10)
    parser.add_argument('--rounds', type=positive_int, default=5)
    parser.add_argument('--iterations', type=positive_int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-benchmark', action='store_true', help='check the selected matrix without timing')
    parser.add_argument('--csv', type=Path, help='optional per-case errors, median timings and samples')
    options = parser.parse_args(argv)
    shapes = options.shape or (QUICK_SHAPES if options.suite == 'quick' else LIGHTOP_SHAPES)
    for m, n, h in shapes:
        if m > 2**31 - 1 or n > 2**31 - 1 or h not in (1, 2, 4, 8, 16, 32, 64, 128):
            parser.error('M/N must fit int32; H must be a power of two in [1,128]')
    if options.reuse_output and options.output == 'compact':
        parser.error('--reuse-output requires dense output; compact is library-allocated')
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error('a HIP PyTorch build and gfx936/gfx938 device are required')
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = props.gcnArchName.split(':')[0]
    if arch not in ('gfx936', 'gfx938'):
        parser.error('unsupported architecture: ' + arch)
    if options.dtype == 'fp8' and arch != 'gfx938':
        parser.error('FP8 MQA requires gfx938')
    dtypes = ('bf16', 'int8', 'fp8') if options.dtype == 'all' else (options.dtype,)
    if arch == 'gfx936' and 'fp8' in dtypes:
        print('SKIP fp8: requires gfx938', flush=True)
        dtypes = tuple(dtype for dtype in dtypes if dtype != 'fp8')
    from deepgemm import mqa_logits, op
    environment = dict(arch=arch, device=props.name, torch_version=torch.__version__,
                       hip_version=torch.version.hip, native_module=getattr(op, '__file__', 'unknown'))
    print('MQA environment: ' + json.dumps(environment), flush=True)
    print('D=128; each case validates all defined outputs before timing. '
          'Eager = GPU events around API calls; Graph = replay GPU events. '
          'Inputs remain resident; no cache flush.', flush=True)
    layouts = ('dense', 'compact') if options.output == 'both' else (options.output,)
    windows = WINDOWS if options.window == 'all' else (options.window,)
    rows = []
    torch.set_num_threads(1)
    with torch.no_grad():
        for dtype, shape, window in itertools.product(dtypes, shapes, windows):
            m, n, h = shape
            seed = (options.seed + m * 1000003 + n * 101 + h) % (2**63 - 1)
            args, width = matrix_inputs(m, n, h, dtype, window, seed)
            for output in layouts:
                clean = options.clean_logit == 'true' if output == 'dense' else False
                kwargs = dict(clean_logit=clean)
                if output == 'compact':
                    kwargs['max_seqlen_k'] = width
                elif options.reuse_output:
                    kwargs['D_out'] = torch.empty((m, n), device=args[0].device)
                call = partial(mqa_logits, *args, **kwargs)
                actual = call()
                context = '{} {} M={} N={} H={} window={}'.format(dtype, output, m, n, h, window)
                try:
                    metrics = check_matrix_output(actual, args, output, clean, width)
                    times = {} if options.no_benchmark else benchmark_call(
                        call, actual, args, output, clean, options.timing, options.warmup,
                        options.rounds, options.iterations)
                except Exception as error:
                    raise RuntimeError('MQA case failed: ' + context) from error
                row = dict(environment, dtype=dtype, output=output, M=m, N=n, H=h, D=128,
                           window=window, seed=seed, clean_logit=clean, output_width=actual.shape[1],
                           output_stride=actual.stride(0), reused_output='D_out' in kwargs,
                           warmup=options.warmup, rounds=options.rounds, iterations=options.iterations,
                           **metrics, **times)
                for metric in ('eager_event', 'graph'):
                    if metric + '_us' in times:
                        row[metric + '_useful_tflops'] = 2 * 128 * h * metrics['valid_values'] / times[metric + '_us'] / 1e6
                rows.append(row)
                print('PASS {} valid={} max_abs={:.3e} normalized={:.3e} diff={:.3e}'.format(
                    context, metrics['valid_values'], metrics['max_abs_error'],
                    metrics['normalized_max_error'], metrics['diff']), flush=True)
                for metric in ('eager_event', 'graph'):
                    if metric + '_us' in times:
                        print('  {}={:.3f} us, useful={:.3f} TFLOPS, samples(us)={}'.format(
                            metric, times[metric + '_us'], row[metric + '_useful_tflops'],
                            [round(value, 4) for value in times[metric + '_samples_us']]), flush=True)
                del call, actual, kwargs
    if options.csv:
        with options.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                                 for key, value in row.items()})
        print('CSV: ' + str(options.csv.resolve()), flush=True)
    print('MQA matrix passed: {} cases{}'.format(len(rows), ' (correctness only)' if options.no_benchmark else ''), flush=True)
    return rows


if __name__ == '__main__':
    main()
