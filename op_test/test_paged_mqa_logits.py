# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Unified paged_mqa_logits decode/prefill/sparse cases and eager/Graph/API checks.

Run all tests:
    python3 op_test/test_paged_mqa_logits.py

Run one dtype from Python:
    test_paged_mqa_logits("int8")
    test_paged_mqa_logits("fp8", benchmark=False)

Correctness and timing use the same cases, clean modes and backend/output variants.

Run all correctness checks without timing:
    python3 op_test/test_paged_mqa_logits.py --no-benchmark

Results are printed to stdout. No result files are created.
"""

import argparse
import itertools
import random
import os
import statistics
from functools import partial
from unittest.mock import patch

import torch

from deepgemm import (
    op, paged_mqa_logits, page_mqa_logits_sparse_mask,
    get_paged_sparse_mqa_logits_metadata, fp8_fp4_paged_sparse_mqa_logits,
    page_mqa_logits_sparse_mask_grouped, page_mqa_logits_sparse_mask_grouped_int8,
    fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_fp4_metadata,
)


D = 128
SEED = 20260914
STANDARD_BATCHES = (1, 4, 8, 16, 32, 64)
STANDARD_NEXT_N = (1, 2, 4)
STANDARD_HEADS = (16, 32, 64)
STANDARD_AVG_KV = (4096, 72000)
STANDARD_BLOCK_KV = 64
STANDARD_MAX_LEN = 111000


def _arch():
    if not torch.cuda.is_available():
        raise RuntimeError("paged_mqa_logits tests require a gfx936/gfx938 GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938"):
        raise RuntimeError(f"unsupported GPU architecture: {arch}")
    return arch


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _context_lengths(batch, next_n, heads, average):
    """Keep the historical deterministic length rule shared by both matrices."""
    if batch == 1:
        return (average,)
    generator = torch.Generator().manual_seed(SEED + batch + next_n + heads + average)
    return tuple(torch.randint(int(0.9 * average), int(1.1 * average),
                               (batch,), generator=generator).tolist())


def _pack_quantized_cache(kv, scale, storage_dtype):
    pages, block_kv, _, dim = kv.shape
    packed = torch.empty((pages, block_kv * (dim + 4)), device=kv.device, dtype=storage_dtype)
    packed[:, :block_kv * dim] = kv.reshape(pages, block_kv * dim).view(storage_dtype)
    packed[:, block_kv * dim:] = scale.contiguous().reshape(pages, block_kv).view(storage_dtype)
    return packed.view(pages, block_kv, 1, dim + 4)


def kv_cache_cast_to_fp8(kv):
    """BF16 randn -> E4M3FN + per-token FP32 amax/448 quantization."""
    amax = kv.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = amax / 448.0
    quantized = (kv * (1.0 / scale)).to(torch.float8_e4m3fn)
    return _pack_quantized_cache(quantized, scale, torch.uint8)


def pack_int8_fused_cache(kv, scale):
    return _pack_quantized_cache(kv, scale, torch.int8)


def _make_inputs(mode, next_n, heads, block_kv, lengths, shared_pages=False, contiguous_pages=False,
                 *, table_max_model_len=None, data_pattern="random"):
    assert data_pattern in ("random", "fp8_subnormal")
    assert data_pattern == "random" or mode == "fp8"
    batch = len(lengths)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    page_counts = [_ceil_div(length, block_kv) for length in lengths]
    pages = (max(page_counts) if shared_pages else sum(page_counts)) + 11
    permutation = (torch.arange(pages, device="cuda", dtype=torch.int32) if contiguous_pages else
                   torch.randperm(pages, generator=generator, device="cuda", dtype=torch.int32))
    # Frameworks retain a model-capacity page table across graph replays, even
    # when only a short prefix is populated. Physical KV pages remain demand-sized.
    table_pages = (max(page_counts) if table_max_model_len is None else
                   _ceil_div(table_max_model_len, block_kv))
    assert table_pages >= max(page_counts), "page table cannot hold the active contexts"
    table = torch.zeros((batch, table_pages), device="cuda", dtype=torch.int32)
    offset = 0
    for row, count in enumerate(page_counts):
        table[row, :count] = permutation[:count] if shared_pages else permutation[offset:offset + count]
        if not shared_pages:
            offset += count

    if mode == "int8":
        q = torch.randint(-127, 128, (batch, next_n, heads, D),
                          generator=generator, device="cuda", dtype=torch.int8)
        kv = torch.randint(-127, 128, (pages, block_kv, 1, D),
                           generator=generator, device="cuda", dtype=torch.int8)
        scale = torch.rand((pages, block_kv), generator=generator,
                           device="cuda", dtype=torch.float32) * 0.09 + 0.01
        cache = pack_int8_fused_cache(kv, scale)
    elif data_pattern == "fp8_subnormal":
        # Preserve scale=amax/448=1 while isolating signed subnormal contributions.
        q_float = torch.ones((batch, next_n, heads, D), device="cuda")
        q_float[:, 1::2].neg_()
        q_float[..., 0] = 0
        q = q_float.to(torch.float8_e4m3fn)
        tokens = torch.arange(block_kv, device="cuda")
        codes = ((tokens % 8) | ((tokens // 8 % 2) << 7)).to(torch.uint8)
        kv = codes[None, :, None, None].expand(pages, block_kv, 1, D).clone()
        kv[..., 0] = 0x7e  # E4M3FN 448; Q's matching coordinate is zero.
        scale = torch.ones((pages, block_kv), device="cuda", dtype=torch.float32)
        cache = _pack_quantized_cache(kv, scale, torch.uint8)
    elif mode == "fp16":
        q = torch.randn((batch, next_n, heads, D), generator=generator, device="cuda", dtype=torch.float16)
        cache = torch.randn((pages, block_kv, 1, D), generator=generator, device="cuda", dtype=torch.float16)
    else:
        q_bf16 = torch.randn((batch, next_n, heads, D), generator=generator,
                             device="cuda", dtype=torch.bfloat16)
        q = (q_bf16.to(torch.float8_e4m3fn) if mode == "fp8" else
             q_bf16.to(torch.float8_e5m2) if mode == "e5m2" else q_bf16)
        kv = torch.randn((pages, block_kv, 1, D), generator=generator,
                         device="cuda", dtype=torch.bfloat16)
        cache = kv if mode == "bf16" else kv_cache_cast_to_fp8(kv)

    weights = torch.randn((batch * next_n, heads), generator=generator, device="cuda", dtype=torch.float32)
    context = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q, cache, weights, context, table


def ref_paged_mqa_logits(q, cache, weights, context, table, max_model_len):
    """Reference uses the actual stored Q/KV bytes and scales, not pre-quantized tensors."""
    batch, next_n, _, dim = q.shape
    pages, block_kv = cache.shape[:2]
    scale = None
    if cache.dtype in (torch.uint8, torch.int8):
        raw = cache.view(pages, -1)
        kv_bytes = raw[:, :block_kv * dim]
        kv_dtype = torch.float8_e4m3fn if cache.dtype == torch.uint8 else torch.int8
        kv = kv_bytes.contiguous().view(kv_dtype).reshape(pages, block_kv, dim).float()
        scale = raw[:, block_kv * dim:].contiguous().view(torch.float32)
    else:
        kv = cache[:, :, 0, :].float()
    output = torch.full((batch * next_n, max_model_len), -float("inf"), device=q.device, dtype=torch.float32)
    valid = torch.zeros_like(output, dtype=torch.bool)
    for batch_index, length in enumerate(context.cpu().tolist()):
        ids = table[batch_index, :_ceil_div(length, block_kv)].long()
        keys = kv[ids].reshape(-1, dim)[:length]
        if scale is not None:
            keys = keys * scale[ids].reshape(-1)[:length, None]
        scores = torch.einsum("rhd,td->rht", q[batch_index].float(), keys)
        row_weights = weights[batch_index * next_n:(batch_index + 1) * next_n, :, None]
        scores = (scores.relu() * row_weights).sum(1)
        for query in range(next_n):
            end = max(0, min(max_model_len, length - next_n + query + 1))
            row = batch_index * next_n + query
            output[row, :end] = scores[query, :end]
            valid[row, :end] = True
    return output, valid


def _check_output(actual, reference, valid, *, clean_logits=True, context=None):
    assert actual.shape == reference.shape and actual.dtype == torch.float32
    if not clean_logits:
        assert context is not None
        next_n = actual.size(0) // context.numel()
        ignored = torch.arange(actual.size(1), device=actual.device)[None, :] >= context.repeat_interleave(next_n)[:, None]
        # Mask both tensors with the same value; causal positions inside context
        # remain observable and must still be -inf.
        actual = actual.masked_fill(ignored, 0)
        reference = reference.masked_fill(ignored, 0)
    assert torch.isfinite(actual[valid]).all(), "non-finite valid logits"
    assert torch.equal(actual[~valid], reference[~valid]), "causal mask or required tail differs"
    error = actual[valid] - reference[valid]
    relative_l2 = (error.double().norm() / reference[valid].double().norm().clamp_min(1e-30)).item()
    assert relative_l2 <= 2e-5, f"relative_l2={relative_l2}, limit=2e-5"
    return relative_l2, error.abs().max().item() if error.numel() else 0.0

def _print_inputs(label, inputs, max_model_len, clean_logits=True):
    q, cache, weights, context, table = inputs
    lengths = context.cpu().tolist()
    print(f"{label}: Q={tuple(q.shape)} {q.dtype}, KV={tuple(cache.shape)} {cache.dtype}, "
          f"weights={tuple(weights.shape)}, context={tuple(context.shape)}, table={tuple(table.shape)}, "
          f"max_model_len={max_model_len}, ctx(min/mean/max)="
          f"{min(lengths, default=0)}/{statistics.mean(lengths) if lengths else 0:.1f}/"
          f"{max(lengths, default=0)}, seed={SEED}, clean_logits={clean_logits}, "
          f"context_lens={lengths}",
          flush=True)
    if any(not tensor.is_contiguous() for tensor in inputs):
        print("  strides:", [tuple(tensor.stride()) for tensor in inputs], flush=True)


def _bench(calls):
    # Full API calls, including output allocation, conversions and clean_logits.
    for call in calls.values():
        for _ in range(10):
            call()
    torch.cuda.synchronize()
    samples = {name: [] for name in calls}
    names = tuple(calls)
    for sample in range(5):
        for name in (names if sample % 2 == 0 else names[::-1]):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(20):
                calls[name]()
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0 / 20)
    return samples


def _backends(case):
    if case["kernel_id"] != -1:
        return ("opus",)
    mode, r, h, block = (case[key] for key in ("mode", "next_n", "heads", "block_kv"))
    native = block in (4, 8, 16, 32, 64) and (mode not in ("fp8", "mixed") or r in (1, 2))
    opus = mode == "fp8" and block in (1, 64) and h in (32, 64) and r in (1, 2, 4) and not case["strided"]
    return ("auto",) + (("native",) if native else ()) + (("opus",) if opus else ())


def make_prefix_prefill_case(chunk, mode="int8"):
    """Four CP8 chunks; FP8 Q/K derive from the same seeded INT8 fixture.

    INT8 cache stores signed K bytes; FP8 cache stores E4M3FN bytes in uint8.
    Both layouts append per-token FP32 KV scales and fold Q scales into weights.
    """
    if chunk not in (1, 2, 3, 4):
        raise ValueError("chunk must be in [1, 4]")
    tokens, heads, block_kv, pages = 4096, 64, 64, 1161
    width = chunk * 8192
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    kv = torch.randint(-128, 128, (pages, block_kv, 1, D),
                       dtype=torch.int8, device="cuda", generator=generator)
    scales = torch.rand((pages, block_kv), dtype=torch.float32,
                        device="cuda", generator=generator) * 0.09 + 0.01
    shared_pages = torch.randperm(pages, device="cuda", generator=generator)[:512].int()
    table = shared_pages[:width // block_kv].repeat(tokens, 1)
    generator.manual_seed(SEED + chunk)
    q = torch.randint(-128, 128, (tokens, 1, heads, D), dtype=torch.int8,
                      device="cuda", generator=generator)
    weights = torch.randn((tokens, heads), dtype=torch.float32,
                          device="cuda", generator=generator)
    q_scales = torch.rand((tokens, heads), dtype=torch.float32,
                         device="cuda", generator=generator) * 0.09 + 0.01
    weights *= q_scales
    context = torch.arange(tokens, dtype=torch.int32, device="cuda") * 2 + (chunk - 1) * 8192
    cache = pack_int8_fused_cache(kv, scales)
    if mode == "fp8":
        q = q.to(torch.float8_e4m3fn)
        cache = torch.empty_like(cache, dtype=torch.uint8)
        raw = cache.view(pages, -1)
        raw[:, :block_kv * D] = kv.to(torch.float8_e4m3fn).reshape(pages, -1).view(torch.uint8)
        raw[:, block_kv * D:] = scales.view(torch.uint8)
    elif mode != "int8":
        raise ValueError("mode must be int8 or fp8")
    return (q, cache, weights, context, table), width

def check_prefill(output, inputs, width):
    """Stored-byte FP32 dots (TF32 disabled), FP64 weighting/head reduction."""
    q, cache, weights, context, table = inputs
    tokens, _, heads, dim = q.shape
    assert output.shape == (tokens, width) and output.dtype == torch.float32
    if tokens == 0 or width == 0:
        return 0.0
    pages, block_kv = cache.shape[:2]
    raw = cache.view(pages, -1)
    ids = table[0, :_ceil_div(width, block_kv)].long()
    key_dtype = torch.float8_e4m3fn if cache.dtype == torch.uint8 else torch.int8
    keys = raw[:, :block_kv * dim].contiguous().view(key_dtype).reshape(pages, block_kv, dim)[ids].reshape(-1, dim)[:width].float()
    scales = raw[:, block_kv * dim:].contiguous().view(torch.float32)[ids].reshape(-1)[:width].double()
    positions = torch.arange(width, device=q.device)
    max_error = torch.zeros((), device=q.device, dtype=torch.float64)
    max_reference = torch.zeros_like(max_error)
    square_error = torch.zeros_like(max_error)
    square_reference = torch.zeros_like(max_error)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row in range(0, tokens, 32):
            dots = q[row:row + 32, 0].reshape(-1, dim).float() @ keys.T
            reference = (dots.reshape(-1, heads, width).double().relu()
                         * weights[row:row + 32, :, None].double()).sum(1) * scales
            valid = positions[None, :] < context[row:row + 32, None]
            actual = output[row:row + 32]
            assert torch.equal(torch.isneginf(actual), ~valid), "incorrect prefill mask/tail"
            assert torch.isfinite(actual[valid]).all()
            max_reference = torch.maximum(max_reference, reference.masked_fill(~valid, 0).abs().max())
            max_error = torch.maximum(max_error, (actual.double() - reference).masked_fill(~valid, 0).abs().max())
            square_error += (actual.double() - reference).masked_fill(~valid, 0).square().sum()
            square_reference += reference.masked_fill(~valid, 0).square().sum()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    normalized_error = (max_error / max_reference.clamp_min(1)).item()
    assert normalized_error < 2e-5, f"normalized max error={normalized_error}"
    relative_l2 = (square_error / square_reference.clamp_min(1e-30)).sqrt().item()
    assert relative_l2 <= 2e-5, f"relative_l2={relative_l2}"
    return normalized_error

def make_sparse_case(mode, heads, group_size, width=4096, requests=1):
    batch = 8 if group_size == 1 and requests == 1 else group_size * requests
    inputs = list(_make_inputs("fp8" if mode == "fp8_fnuz" else mode, 1, heads, 64,
                               (width,) * batch, shared_pages=group_size != 1))
    q, cache, weights, context, table = inputs
    context.copy_(torch.tensor([max(0, width - 3 * row) for row in range(batch)],
                               device=q.device, dtype=torch.int32))
    if group_size != 1:
        # Different requests use different logical page orderings, but each
        # group's rows share one table. The cache still has only width/64+11 pages.
        for request in range(requests):
            begin = request * group_size
            table[begin:begin + group_size].copy_(table[0].roll(request))
    if mode == "int8":
        inputs[0] = q.to(torch.bfloat16)
        inputs[1] = cache.view(torch.uint8)
    elif mode == "fp8_fnuz":
        inputs[0] = q.float().to(torch.float8_e4m3fnuz)
    return tuple(inputs)

def sparse_call(inputs, width, group_size, mode, *, grouped=False, persistent_ctas=0):
    if mode == "int8":
        function = page_mqa_logits_sparse_mask_grouped_int8
    elif group_size == 1 and not grouped:
        function = page_mqa_logits_sparse_mask
    else:
        function = page_mqa_logits_sparse_mask_grouped
    options = dict(clean_logits=True, num_warps=4, persistent_ctas=persistent_ctas)
    if function != page_mqa_logits_sparse_mask:
        options["group_size"] = group_size
    return partial(function, *inputs, width, **options)

def sparse_valid_bits(result, context, width):
    values, mask = result
    assert values.shape == (context.numel(), width) and values.dtype == torch.float32
    assert mask.shape == (context.numel(), _ceil_div(width, 16)) and mask.dtype == torch.int16
    positions = torch.arange(width, device=values.device)
    valid = positions[None, :] < context[:, None].clamp(0, width)
    bits = ((mask.int()[:, positions // 16] >> (positions % 16)) & 1).bool() & valid
    return valid, bits

def ref_sparse_mqa(inputs, width, mode):
    q, cache, weights, context, table = inputs
    pages, block_kv = cache.shape[:2]
    scale = None
    if cache.dtype == torch.uint8:
        raw = cache.reshape(pages, -1)
        dtype = torch.int8 if mode == "int8" else torch.float8_e4m3fn
        keys = raw[:, :block_kv * D].contiguous().view(dtype).reshape(pages, block_kv, D).float()
        scale = raw[:, block_kv * D:].contiguous().view(torch.float32)
        if mode == "int8":
            keys = (keys * scale[:, :, None]).bfloat16().float()
            scale = None
    else:
        keys = cache[:, :, 0].float()
    output = torch.zeros((q.size(0), width), device=q.device, dtype=torch.float64)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row, length in enumerate(context.clamp(0, width).cpu().tolist()):
            ids = table[row, :_ceil_div(length, block_kv)].long()
            kv = keys[ids].reshape(-1, D)[:length]
            dots = q[row, 0].float() @ kv.T
            scores = (dots.double().relu() * weights[row, :, None].double()).sum(0)
            if scale is not None:
                scores *= scale[ids].reshape(-1)[:length].double()
            output[row, :length] = scores
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return output

def check_sparse(result, inputs, width, mode):
    valid, bits = sparse_valid_bits(result, inputs[3], width)
    # Never inspect uninitialized values, or legacy mask bits beyond context.
    dense = torch.zeros(result[0].shape, device=result[0].device, dtype=torch.float32)
    dense[bits] = result[0][bits]
    assert torch.isfinite(dense[valid]).all()
    reference = ref_sparse_mqa(inputs, width, mode)
    error = dense.double()[valid] - reference[valid]
    relative_l2 = (error.norm() / reference[valid].norm().clamp_min(1e-30)).item()
    assert relative_l2 <= 2e-5, f"sparse relative_l2={relative_l2}, limit=2e-5"
    return relative_l2

def _prefill_cases(mode, *, graph=False):
    """Prefill variants share the eager/Graph runners; only case data differs."""
    if mode not in (("int8", "fp8") if _arch() == "gfx938" else ("int8",)):
        return
    rows = []
    if graph:
        rows.append(dict(case="prefill_capacity", heads=32, block_kv=64,
                         context_lengths=_context_lengths(4, 1, 32, 4096),
                         max_model_len=STANDARD_MAX_LEN, backends=("auto", "native")))
        for tokens, width in (((33, 257), (64, 257), (33, 256), (64, 256)) if mode == "fp8" else ((33, 257),)):
            for heads, block in ((8, 4), (16, 8), (32, 16), (32, 32), (64, 64)):
                rows.append(dict(case="prefill_alignment", heads=heads, block_kv=block,
                                 context_lengths=(width,) * tokens, max_model_len=width,
                                 graph_context="alternating", clean_modes=(True,)))
    else:
        shapes = [(tokens, heads, STANDARD_BLOCK_KV, average)
                  for tokens in STANDARD_BATCHES for heads in STANDARD_HEADS for average in STANDARD_AVG_KV]
        shapes += [(4, heads, block, average)
                   for heads in (8, 16, 32, 64) for block in (4, 8, 16, 32, 64)
                   for average in STANDARD_AVG_KV if block != 64 or heads == 8]
        for tokens, heads, block, average in shapes:
            rows.append(dict(case="prefill_capacity", heads=heads, block_kv=block,
                             context_lengths=_context_lengths(tokens, 1, heads, average),
                             max_model_len=STANDARD_MAX_LEN, backends=("auto", "native"), benchmark=True))
        for heads, tokens in ((8, 1), (16, 31), (32, 32), (64, 33)):
            for block in (4, 8, 16, 32, 64):
                width = block * 3 + 1
                lengths = (width, 0, block - 1, block, block + 1)
                rows.append(dict(case="prefill_boundary", heads=heads, block_kv=block,
                                 input_lengths=(width,) * tokens,
                                 context_lengths=tuple(lengths[i % len(lengths)] for i in range(tokens)),
                                 max_model_len=width))
        rows.extend((dict(case="prefill_empty_rows", heads=8, block_kv=4, context_lengths=(8,),
                          max_model_len=8, empty_rows=True, clean_modes=(True,)),
                     dict(case="prefill_empty_columns", heads=8, block_kv=4, context_lengths=(8,),
                          max_model_len=0, clean_modes=(True,))))
        if mode == "fp8":
            for block in (4, 8, 16, 32, 64):
                width = block * 3 + 1
                rows.append(dict(case="prefill_subnormal", heads=64, block_kv=block,
                                 context_lengths=(width,) * 33, max_model_len=width,
                                 data_pattern="fp8_subnormal", clean_modes=(True,)))
            for block in (16, 32, 64):
                rows.append(dict(case="prefill_fp8_byte_pairs", heads=8, block_kv=block,
                                 context_lengths=(256,) * 256, max_model_len=256,
                                 data_pattern="fp8_byte_pairs"))
            for tokens in (65, 127, 129):
                for width in (255, 256, 257):
                    for heads in (8, 16, 32, 64):
                        for block in (16, 32, 64):
                            lengths = (width, 0, block - 1, block, block + 1)
                            rows.append(dict(case="prefill_tail_cta", heads=heads, block_kv=block,
                                             input_lengths=(width,) * tokens,
                                             context_lengths=tuple(lengths[i % len(lengths)] for i in range(tokens)),
                                             max_model_len=width))
        if mode == "int8":
            for tokens in (31, 32, 33):
                for width in (255, 256, 257):
                    for heads, block in ((8, 4), (16, 8), (32, 16), (32, 32), (64, 64)):
                        lengths = (width, 0, block - 1, block, block + 1)
                        rows.append(dict(case="prefill_tail_cta", heads=heads, block_kv=block,
                                         input_lengths=(width,) * tokens,
                                         context_lengths=tuple(lengths[i % len(lengths)] for i in range(tokens)),
                                         max_model_len=width))
        for chunk in (1, 2, 3, 4):
            rows.append(dict(case=f"prefill_chunk_{chunk}", heads=64, block_kv=64,
                             context_lengths=tuple((chunk - 1) * 8192 + 2 * t for t in range(4096)),
                             max_model_len=chunk * 8192, chunk=chunk, benchmark=True, clean_modes=(True,)))
    for row in rows:
        for clean in row.get("clean_modes", (True, False)):
            yield {"kind": "prefill", "mode": mode, "next_n": 1, "shared_pages": True,
                   "contiguous_pages": False, "strided": False, "kernel_id": -1,
                   "backends": ("auto",), "benchmark": False, **row, "clean_logits": clean}


def _sparse_cases(mode, *, graph=False):
    if mode not in ("bf16", "mixed", "fp8", "fp8_fnuz", "int8"):
        return
    rows = []
    if graph:
        for value, group, heads, q_dtype in (("bf16", 1, 64, None), ("mixed", 3, 8, None),
                ("fp8", 6, 32, None), ("fp8", 3, 8, None), ("fp8", 5, 16, None),
                ("fp8", 4, 64, None), ("fp8", 1, 16, None), ("int8", 5, 16, None),
                ("int8", 3, 8, torch.float8_e4m3fn), ("int8", 4, 64, torch.float8_e4m3fnuz)):
            if value == mode:
                rows.append(dict(case="sparse_graph", heads=heads, group_size=group,
                                 max_model_len=320, q_dtype=q_dtype, graph_context="alternating"))
    else:
        for heads in (8, 16, 32, 64):
            for group in (1, 3, 4, 5, 6):
                for q_dtype in ((None, torch.float8_e4m3fn, torch.float8_e4m3fnuz) if mode == "int8" else (None,)):
                    rows.append(dict(case="sparse_matrix", heads=heads, group_size=group,
                                     max_model_len=4096, q_dtype=q_dtype,
                                     benchmark=q_dtype is None and heads == (32 if _arch() == "gfx938" else 64)))
        if mode == "fp8":
            # Subnormal and signed-zero inputs must reach every head count's kernel.
            for heads in (8, 16, 32, 64):
                rows.append(dict(case="sparse_subnormal", heads=heads, group_size=4,
                                 max_model_len=4096, data_pattern="fp8_subnormal"))
        if mode != "fp8_fnuz":
            # Native FP8 has a separate kernel instantiation per head count.
            for heads in ((8, 16, 32, 64) if mode == "fp8" else (32,)):
                for group in (1, 3, 4, 5, 6):
                    for persistent in (0, 1, 32):
                        rows.append(dict(case="sparse_edges", heads=heads, group_size=group,
                                         max_model_len=320, requests=5, grouped=True, persistent_ctas=persistent,
                                         strided=True, context_pattern="sparse_boundary"))
        if mode == "bf16":
            rows.append(dict(case="sparse_independent_b40", heads=64, group_size=1,
                             max_model_len=320, requests=40))
    for row in rows:
        yield {"kind": "sparse", "mode": mode, "clean_logits": True, "benchmark": False,
               "num_warps": 4, "persistent_ctas": 0, "requests": 1, "data_pattern": "random", **row}


def _make_case_inputs(case):
    kind, mode, width = case.get("kind", "decode"), case["mode"], case["max_model_len"]
    if kind == "sparse":
        inputs = make_sparse_case(mode, case["heads"], case["group_size"], width, case["requests"])
        if case["data_pattern"] == "fp8_subnormal":
            inputs = _make_inputs(mode, 1, case["heads"], inputs[1].size(1),
                                  (width,) * inputs[0].size(0), shared_pages=True,
                                  data_pattern="fp8_subnormal")
        if case.get("q_dtype") is not None:
            inputs = (inputs[0].to(case["q_dtype"]), *inputs[1:])
        if case.get("context_pattern") == "sparse_boundary":
            endpoints = (-1, 0, 1, 15, 16, 63, 64, 319, 320, 321)
            inputs[3].copy_(torch.tensor([endpoints[i % len(endpoints)] for i in range(inputs[0].size(0))],
                                         device="cuda", dtype=torch.int32))
    elif "chunk" in case:
        inputs, actual_width = make_prefix_prefill_case(case["chunk"], mode)
        assert actual_width == width
    elif case.get("data_pattern") == "fp8_byte_pairs":
        tokens, heads, block = len(case["context_lengths"]), case["heads"], case["block_kv"]
        pages = width // block
        codes = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
        q = torch.empty((tokens, 1, heads, D), device="cuda", dtype=torch.float8_e4m3fn)
        q.view(torch.uint8).copy_(codes[:, None, None, None].expand_as(q))
        cache = torch.empty((pages, block, 1, D + 4), device="cuda", dtype=torch.uint8)
        raw = cache.view(pages, -1)
        raw[:, :block * D] = codes[:, None].expand(width, D).reshape(pages, block * D)
        raw[:, block * D:].view(torch.float32).fill_(1)
        inputs = (q, cache, torch.full((tokens, heads), 1 / heads, device="cuda", dtype=torch.float32),
                  torch.full((tokens,), width, device="cuda", dtype=torch.int32),
                  torch.arange(pages, device="cuda", dtype=torch.int32).repeat(tokens, 1))
    else:
        lengths = case.get("input_lengths", case["context_lengths"])
        inputs = _make_inputs(mode, case["next_n"], case["heads"], case["block_kv"], lengths,
                              case["shared_pages"], case["contiguous_pages"],
                              table_max_model_len=max(width, max(lengths)),
                              data_pattern=case.get("data_pattern", "random"))
        if kind == "prefill":
            inputs[4].copy_(inputs[4][lengths.index(max(lengths))].clone().expand_as(inputs[4]))
            inputs[3].copy_(torch.tensor(case["context_lengths"], device="cuda", dtype=torch.int32))
        if case.get("empty_rows"):
            inputs = tuple(tensor[:0] if index != 1 else tensor for index, tensor in enumerate(inputs))
    if case.get("strided"):
        q, cache, weights, context, table = inputs
        raw = cache.reshape(cache.size(0), -1)
        if kind == "sparse":
            raw[:, :16 * D] = 0
        padding = 16 if kind == "sparse" else 8
        padded_cache = torch.empty((cache.size(0), raw.size(1) + padding), device=q.device, dtype=cache.dtype)
        padded_cache[:, :raw.size(1)].copy_(raw)
        padded_table = torch.empty((table.size(0), table.size(1) + 3), device=q.device, dtype=table.dtype)
        padded_table[:, :table.size(1)].copy_(table)
        inputs = q, padded_cache[:, :raw.size(1)].view(cache.shape), weights, context, padded_table[:, :table.size(1)]
    return inputs


def _case_call(case, inputs, *, supplied=None):
    width = case["max_model_len"]
    if case.get("kind") == "sparse":
        return sparse_call(inputs, width, case["group_size"], case["mode"],
                           grouped=case.get("grouped", False), persistent_ctas=case["persistent_ctas"])
    function = paged_mqa_logits if case.get("kernel_id", -1) == -1 else partial(
        op.paged_mqa_logits, kernel_id=case["kernel_id"])
    return partial(function, *inputs, width, out=supplied, clean_logits=case["clean_logits"],
                   is_prefill=case.get("kind") == "prefill")


def _check_case_output(actual, inputs, case):
    kind, width = case.get("kind", "decode"), case["max_model_len"]
    if kind == "sparse":
        return {"rel_l2": check_sparse(actual, inputs, width, case["mode"])}
    if kind == "prefill":
        if case.get("data_pattern") == "fp8_byte_pairs":
            values = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
            reference = torch.fmax(values[:, None] * values[None, :] * D,
                                   torch.zeros((), device="cuda", dtype=torch.float32))
            assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
            return {"all_65536_fp8_byte_pairs_bitwise_exact": True}
        return {"normalized_max_error": check_prefill(actual, inputs, width)}
    rel, absolute = _check_output(actual, *ref_paged_mqa_logits(*inputs, width),
                                  clean_logits=case["clean_logits"], context=inputs[3])
    return {"rel_l2": rel, "max_abs": absolute}


def _check_eager_case(case, *, benchmark):
    width, clean = case["max_model_len"], case["clean_logits"]
    inputs = _make_case_inputs(case)
    kind = case.get("kind", "decode")
    if kind != "decode":
        _print_inputs(f"eager kind={kind} case={case['case']} data_pattern={case.get('data_pattern', 'random')} "
                      f"group_size={case.get('group_size')} num_warps={case.get('num_warps')} "
                      f"persistent_ctas={case.get('persistent_ctas')}", inputs, width, clean)
        for backend in case.get("backends", ("auto",)):
            with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": "" if backend == "auto" else backend}):
                call = _case_call(case, inputs)
                metrics = _check_case_output(call(), inputs, case)
                print(f"  PASS | backend={backend} | metrics={metrics}", flush=True)
                if benchmark and case.get("benchmark", False):
                    samples = _bench({"allocated": call})["allocated"]
                    median = statistics.median(samples)
                    if kind == "prefill":
                        useful = 2 * D * inputs[0].size(2) * inputs[3].long().clamp(0, width).sum().item()
                        dense = 2 * D * inputs[0].size(2) * inputs[0].size(0) * width
                        print(f"  prefill {backend}/allocated: {median:.3f} us | "
                              f"{useful / median / 1e6:.3f} useful TFLOPS | "
                              f"{dense / median / 1e6:.3f} dense TFLOPS | samples(us)={samples}", flush=True)
                    else:
                        print(f"  sparse: {median:.3f} us | samples(us)={samples}", flush=True)
        return
    mode, r, h, block, lengths = (case[key] for key in
        ("mode", "next_n", "heads", "block_kv", "context_lengths"))
    _print_inputs(f"matrix {case['case']} data_pattern={case.get('data_pattern', 'random')} shared_pages={case['shared_pages']} "
                  f"contiguous_pages={case['contiguous_pages']} kernel_id={case['kernel_id']}", inputs, width, clean)
    backends = _backends(case)
    if backends == ("auto",):
        _expect_error(partial(paged_mqa_logits, *inputs, width, clean_logits=clean),
                      "neither backend supports these tensor metadata")
        print("  UNSUPPORTED | API rejection verified", flush=True)
        return
    reference, valid = ref_paged_mqa_logits(*inputs, width)
    in_context = torch.arange(width, device=inputs[0].device)[None, :] < inputs[3].repeat_interleave(r)[:, None]
    def comparable(output):
        return (output if clean else output[in_context]).contiguous().view(torch.int32)
    native = "native" in backends
    opus = "opus" in backends
    small_single = len(lengths) == 1 and (r == 1 or inputs[1].size(0) * block <= width // 2)
    default = "opus" if opus and (not native or _arch() == "gfx938" or small_single) else "native"
    with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": ""}):
        expected = op.paged_mqa_logits(*inputs, width, clean_logits=clean,
                                      implementation=default, kernel_id=case["kernel_id"])
    q, cache, weights, context, table = inputs
    input_bytes = (q.numel() * q.element_size() + weights.numel() * 4 + context.numel() * 4 +
                   sum(_ceil_div(length, block) for length in lengths) * 4 +
                   sum(lengths) * (D + 4 if cache.dtype in (torch.uint8, torch.int8) else D * cache.element_size()))
    logical_bytes = input_bytes + (len(lengths) * r * width if clean else sum(lengths) * r) * 4
    effective_bytes = input_bytes + valid.sum().item() * 4
    for backend in backends:
        # Use the supported environment override through the public API.
        # Explicit calls below, with the override unset, verify the selected route.
        with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": ""}):
            route = expected if backend in ("auto", default) else op.paged_mqa_logits(
                *inputs, width, clean_logits=clean, implementation=backend, kernel_id=case["kernel_id"])
        with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": "" if backend == "auto" else backend}):
            function = paged_mqa_logits if case["kernel_id"] == -1 else partial(
                op.paged_mqa_logits, kernel_id=case["kernel_id"])
            supplied = torch.full_like(reference, float("nan"))
            calls = {"allocated": partial(function, *inputs, width, clean_logits=clean),
                     "supplied": partial(function, *inputs, width, out=supplied, clean_logits=clean)}
            for output_mode, call in calls.items():
                actual = call()
                if output_mode == "supplied":
                    assert actual.data_ptr() == supplied.data_ptr(), "supplied output was not used"
                rel, absolute = _check_output(actual, reference, valid, clean_logits=clean, context=context)
                assert torch.equal(comparable(actual), comparable(route)), "environment/default route or supplied output differs"
                if output_mode == "allocated":
                    assert actual.stride() == route.stride(), "selected route has unexpected output stride"
                    if backend == "native" or (backend == "auto" and default == "native"):
                        alignment = 256 if block < 16 else block
                        assert actual.stride(0) == _ceil_div(width, alignment) * alignment
                    else:
                        assert actual.is_contiguous(), "OPUS output must be contiguous"
                print(f"  PASS | backend={backend} out={output_mode} | rel_l2={rel:.3e} | max_abs={absolute:.3e}", flush=True)
            if benchmark:
                for output_mode, values in _bench(calls).items():
                    median = statistics.median(values)
                    assert median > 0
                    print(f"  {backend}/{output_mode}: {median:.3f} us | "
                          f"{logical_bytes / median / 1000:.1f} logical GB/s | "
                          f"{effective_bytes / median / 1000:.1f} effective GB/s | "
                          f"samples(us)={values}", flush=True)

def _extra_cases(mode):
    # Each row: NextN, heads, BLOCK_KV, context lengths, output width, optional shared-pages flag.
    cases = {
        "fp8": [
            (1, 32, 64, (64,), 257),
            (2, 64, 64, (67, 129, 191, 257), 320, True),
            (1, 32, 64, (65, 127, 193, 255, 257, 319, 385, 511), 513),
            (2, 64, 64, tuple(64 * (2 + i % 8) for i in range(16)), 1025, True),
            (1, 32, 64, tuple(1025 + i * 17 for i in range(32)), 2049),
            (2, 64, 64, tuple(1024 - i % 8 * 17 for i in range(64)), 1088, True),
            (1, 64, 64, tuple(513 + i % 16 * 31 for i in range(128)), 1025),
            (1, 8, 4, (16, 31, 64, 67), 96),
            (2, 16, 8, (17, 24, 33, 64, 79, 96, 127, 145), 160, True),
            (1, 8, 16, (16, 31, 64, 67), 96),
            (2, 16, 32, (32, 63, 96, 129), 160),
            (2, 8, 64, (64, 65, 127, 129), 160),
            (1, 32, 1, (96, 128), 129),
            (4, 64, 64, (192, 257), 320),
            (2, 32, 64, (72000,), 111000),
        ],
        "bf16": [(1, 32, 16, tuple(257 + i % 8 * 31 for i in range(16)), 544)],
        "fp16": [(2, 64, 32, tuple(513 + i % 16 * 29 for i in range(32)), 1088, True)],
        "int8": [
            (1, 8, 64, tuple(257 + i % 16 * 17 for i in range(64)), 640),
            (2, 32, 8, (129, 192, 255, 287), 320, True),
        ],
        "mixed": [
            (2, 16, 16, tuple(3584 + i * 64 for i in range(8)), 4097),
            (2, 16, 8, (17, 24, 33, 64, 79, 96, 127, 145), 160),
            (1, 64, 4, (16, 31, 64, 67), 96),
        ],
    }
    yield from cases[mode]
    if mode == "fp8":
        for batch, next_n in ((1, 1), (1, 2), (64, 1), (64, 2), (128, 1)):
            for heads in (32, 64):
                for average in (4096, 72000):
                    lengths = _context_lengths(batch, next_n, heads, average)
                    yield next_n, heads, 64, lengths, 111000

    # BLOCK_KV / NextN / heads. Keep the original dense branch coverage, including multi-next-n.
    if mode == "fp8":
        shapes = [(4, 1, 8), (4, 2, 16), (8, 1, 16), (8, 2, 8), (16, 1, 8),
                  (16, 2, 16), (32, 1, 16), (32, 2, 8), (64, 1, 8), (64, 2, 16)]
    elif mode == "mixed":
        shapes = [(4, 1, 8), (4, 2, 16), (8, 1, 16), (8, 2, 8), (16, 1, 32),
                  (16, 2, 16), (32, 1, 64), (32, 2, 8), (64, 1, 32), (64, 2, 64)]
    else:
        shapes = [(4, 1, 8), (4, 5, 64), (8, 2, 16), (8, 4, 32), (16, 3, 8),
                  (16, 4, 16), (32, 2, 64), (32, 5, 16), (64, 1, 32), (64, 3, 64),
                  (32, 2, 16), (32, 4, 32), (64, 5, 64)]
    for block_kv, next_n, heads in shapes:
        batch = 4 if next_n >= 3 else 8
        lengths = tuple(16 * block_kv + (i * 13) % block_kv for i in range(batch))
        yield next_n, heads, block_kv, lengths, 17 * block_kv + 3

    next_n, heads, block_kv = {
        "fp8": (2, 8, 16), "mixed": (2, 16, 32), "bf16": (3, 8, 32),
        "fp16": (4, 16, 64), "int8": (5, 8, 64),
    }[mode]
    yield next_n, heads, block_kv, (71903, 72011, 72079, 72007), 111000

    if mode == "fp8":
        yield 2, 64, 1, (255, 256, 257, 259), 320
        yield 4, 32, 1, (255, 256, 257, 259), 320
        yield 4, 64, 64, (4093, 4096, 4101, 4111), 8192


def _standard_modes(arch=None):
    arch = _arch() if arch is None else arch
    if arch == "gfx936":
        return ("bf16", "fp8", "int8")
    if arch == "gfx938":
        return ("fp8",)
    raise ValueError(f"unsupported architecture: {arch}")


def _standard_cases(mode):
    """One parameter source for correctness and timing, with clean as a dimension.

    The requested full-model-capacity matrix is followed by existing API branch
    and boundary cases; extras are timed too and carry explicit parameter labels.
    """
    rows = []
    if mode in _standard_modes():
        for r in STANDARD_NEXT_N:
            for batch in STANDARD_BATCHES:
                for heads in STANDARD_HEADS:
                    for average in STANDARD_AVG_KV:
                        rows.append((f"{mode}_b{batch:02}_r{r}_h{heads:02}_kv{average}",
                                     (r, heads, STANDARD_BLOCK_KV, _context_lengths(batch, r, heads, average), STANDARD_MAX_LEN)))
    rows.extend((f"{mode}_api{index}", row) for index, row in enumerate(_extra_cases(mode)))
    if mode in ("fp8", "bf16", "int8"):
        rows.append((f"{mode}_tile_boundary", (1, 32, 64, (0, 1, 63, 64, 65, 255, 256, 257), 513)))
    if mode == "fp8":
        for heads in (32, 64):
            rows.append((f"fp8_subnormal_h{heads}",
                         (2, heads, 64, (72000,), STANDARD_MAX_LEN), "fp8_subnormal"))
        ids = (6,) if _arch() == "gfx936" else (0, 1, 2, 3, 6)
        for kernel_id in ids:
            rows.append((f"fp8_s1_partial_output_id{kernel_id}",
                         (1, 32, 1, (96, 128), 129, False, False, kernel_id)))
        for r in STANDARD_NEXT_N:
            for heads in (32, 64):
                for kernel_id in ids:
                    rows.append((f"fp8_s1_r{r}_h{heads}_id{kernel_id}",
                                 (r, heads, 1, (65, 129, 257), 320, False, False, kernel_id)))
        for block, lengths, width, shared, contiguous in (
            (64, tuple(1025 + i * 17 for i in range(32)), 2049, False, False),
            (64, (257,) * 4, 320, True, False), (1, (129,) * 4, 160, True, True)):
            rows.append((f"fp8_grouped_s{block}_shared{shared}", (1, 32, block, lengths, width, shared, contiguous, 7)))
        rows.append(("fp8_strided_cache_table", (2, 64, 64, (67, 129, 191, 257), 320, True, False, -1, True)))
    seen = set()
    for entry in rows:
        name, row = entry[:2]
        data_pattern = entry[2] if len(entry) > 2 else "random"
        r, heads, block, lengths, width = row[:5]
        shared = row[5] if len(row) > 5 else False
        contiguous = row[6] if len(row) > 6 else False
        kernel_id = row[7] if len(row) > 7 else -1
        strided = row[8] if len(row) > 8 else False
        identity = (r, heads, block, tuple(lengths), width, shared, contiguous, kernel_id, strided, data_pattern)
        if identity in seen:
            continue
        seen.add(identity)
        for clean in (True, False):
            yield {"case": name, "mode": mode, "batch": len(lengths), "next_n": r, "heads": heads,
                   "D": D, "block_kv": block, "context_lengths": tuple(lengths), "max_model_len": width,
                   "shared_pages": shared, "contiguous_pages": contiguous, "kernel_id": kernel_id,
                   "strided": strided, "clean_logits": clean, "data_pattern": data_pattern}


def test_paged_mqa_logits(mode=None, benchmark=True):
    """One eager runner for decode, prefix prefill and sparse case matrices."""
    _arch()
    if os.environ.get("DEEPGEMM_PAGED_MQA_LOGITS_BACKEND"):
        raise RuntimeError("unset DEEPGEMM_PAGED_MQA_LOGITS_BACKEND; the matrix tests its own overrides")
    modes = (mode,) if mode is not None else tuple(dict.fromkeys(
        _standard_modes() + ("bf16", "fp16", "fp8", "mixed", "int8", "fp8_fnuz")))
    assert all(value in ("bf16", "fp16", "fp8", "mixed", "int8", "fp8_fnuz") for value in modes)
    for value in modes:
        if value != "fp8_fnuz":
            for case in _standard_cases(value):
                _check_eager_case(case, benchmark=benchmark)
        for case in _prefill_cases(value):
            _check_eager_case(case, benchmark=benchmark)
        for case in _sparse_cases(value):
            _check_eager_case(case, benchmark=benchmark)

def _check_graph(inputs, max_model_len, supplied=False, kernel_id=-1, clean_logits=True, *, case=None):
    case = ({"kind": "decode", "mode": "fp8" if inputs[0].dtype == torch.float8_e4m3fn else "bf16",
             "max_model_len": max_model_len, "kernel_id": kernel_id, "clean_logits": clean_logits}
            if case is None else case)
    kind = case.get("kind", "decode")
    q, cache, weights, context, table = inputs
    if case.get("graph_context") == "alternating":
        context[::2] = 0
    _print_inputs(f"graph kind={kind} case={case.get('case', 'api')} out={supplied} kernel_id={kernel_id} "
                  f"group_size={case.get('group_size')} persistent_ctas={case.get('persistent_ctas')}",
                  inputs, max_model_len, clean_logits)
    storage = (torch.empty((q.size(0) * q.size(1), max_model_len), device=q.device, dtype=torch.float32)
               if supplied else None)
    call = _case_call(case, inputs, supplied=storage)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    tensors = output if kind == "sparse" else (output,)
    pointers = [tensor.data_ptr() for tensor in tensors]

    for update in ("initial", "Q-only", "K-only", "scale/weights/context/pages", "context growth"):
        if update == "Q-only":
            if q.dtype == torch.float8_e4m3fn:
                q.view(torch.uint8).bitwise_xor_(0x80)
            elif q.dtype == torch.float8_e4m3fnuz:
                # FNUZ has no negative zero; a raw sign-bit flip would create NaN.
                q.copy_(q.float().neg().to(q.dtype))
            elif q.dtype == torch.int8:
                q.neg_()
            else:
                q.mul_(-0.5 if kind == "sparse" else -0.75)
        elif update == "K-only":
            if cache.dtype in (torch.uint8, torch.int8):
                keys = cache.view(cache.size(0), -1)[:, :cache.size(1) * D]
                if cache.dtype == torch.int8 or (kind == "sparse" and case["mode"] == "int8"):
                    keys.view(torch.int8).neg_()
                else:
                    keys.bitwise_xor_(0x80)
            else:
                cache.mul_(-0.5)
        elif update == "scale/weights/context/pages":
            if cache.dtype in (torch.uint8, torch.int8):
                cache.view(cache.size(0), -1)[:, cache.size(1) * D:].view(torch.float32).mul_(1.125)
            weights.mul_(-0.75 if kind == "sparse" else -0.5)
            if kind == "sparse":
                context.copy_(torch.arange(q.size(0), device=q.device, dtype=torch.int32) * 7 + 257)
            elif case.get("graph_context") == "alternating":
                context.copy_(torch.arange(q.size(0), device=q.device, dtype=torch.int32) * 7)
            else:
                context.sub_(1).clamp_min_(0)
            if kind == "prefill":
                # Row zero owns the shared prefix even when its own context is zero.
                table[:, :2] = table[:, :2].flip(1)
            else:
                for row, length in enumerate(context.cpu().tolist()):
                    if _ceil_div(length, cache.size(1)) >= 2:
                        table[row, :2] = table[row, :2].flip(0)
        elif update == "context growth":
            # Grow lengths without recapturing; launch metadata must remain fixed.
            context.fill_(min(max_model_len, table.size(1) * cache.size(1)))
            if kind != "sparse":
                output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert [tensor.data_ptr() for tensor in tensors] == pointers, "Graph output pointer changed"
        metrics = _check_case_output(output, inputs, case)
        if kind == "prefill":
            assert torch.equal(output.view(torch.int32), call().view(torch.int32))
        elif kind == "sparse":
            eager = call()
            _, bits = sparse_valid_bits(output, context, max_model_len)
            assert torch.equal(bits, sparse_valid_bits(eager, context, max_model_len)[1])
            assert torch.equal(output[0][bits].view(torch.int32), eager[0][bits].view(torch.int32))
        print(f"  PASS | {update} replay | metrics={metrics}", flush=True)


def test_paged_mqa_logits_graph():
    """Capture/replay, including in-place Q/KV/scale/weights/context/page-table updates."""
    _arch()
    # mode, NextN, heads, BLOCK_KV, lengths, width, shared, contiguous, supplied out, kernel ID.
    cases = [
        ("fp8", 2, 64, 64, (67, 129, 191, 257), 320, True, False, False, -1),
        ("fp8", 1, 8, 4, (16, 31, 64, 67), 96, False, False, True, -1),
        ("bf16", 1, 32, 16, tuple(257 + i % 8 * 31 for i in range(16)), 544, False, False, True, -1),
        ("fp8", 2, 32, 64, (72000,), 111000, False, False, False, -1),
        ("bf16", 3, 8, 16, tuple(256 + i * 13 % 16 for i in range(4)), 275, False, False, True, -1),
        ("int8", 5, 64, 4, (64, 65, 66, 67), 71, False, False, False, -1),
        ("fp8", 2, 8, 32, tuple(512 + i * 13 % 32 for i in range(8)), 547, False, False, True, -1),
        ("bf16", 2, 16, 32, tuple(512 + i * 13 % 32 for i in range(8)), 547, False, False, False, -1),
        ("fp16", 4, 32, 32, tuple(512 + i * 13 % 32 for i in range(4)), 547, False, False, True, -1),
        ("int8", 5, 64, 64, tuple(1024 + i * 13 % 64 for i in range(4)), 1091, False, False, False, -1),
        ("fp8", 1, 32, 64, (257,) * 4, 320, True, False, False, 7),
        ("fp8", 1, 32, 1, (129,) * 4, 160, True, True, True, 7),
    ]
    # Exercise regular R=1 native persistent capture/replay through the environment,
    # in addition to OPUS. Reuse exactly the fixed-matrix parameter construction.
    for mode in ("fp8", "bf16", "int8"):
        for r in (1, 2):
            lengths = _context_lengths(4, r, 32, 4096)
            cases.append((mode, r, 32, 64, lengths, STANDARD_MAX_LEN, False, False, True, -1))
    # Automatic ID selection must remain fixed when lengths/page contents change
    # on replay. Cover the former gfx938 ID7 shape with independent/shared pages.
    for shared in (False, True):
        cases.append(("fp8", 1, 32, 64, _context_lengths(32, 1, 32, 4096),
                      STANDARD_MAX_LEN, shared, False, False, -1))
    # ID7 reuse also requires equal group lengths. Cover both storage bands
    # using the specified KV lengths, then mutate/grow them on graph replay.
    for average in STANDARD_AVG_KV:
        cases.append(("fp8", 1, 32, 64, (average,) * 32,
                      STANDARD_MAX_LEN, True, False, False, -1))
    for heads in (32, 64):
        cases.append(("fp8", 2, heads, 64, (72000,), STANDARD_MAX_LEN,
                      False, False, False, -1, "fp8_subnormal"))
    graph_cases = []
    for row in cases:
        mode, next_n, heads, block_kv, lengths, width, shared, contiguous, supplied, kernel_id = row[:10]
        data_pattern = row[10] if len(row) > 10 else "random"
        case = {"kind": "decode", "case": "decode_graph", "mode": mode, "next_n": next_n,
                "heads": heads, "block_kv": block_kv, "kernel_id": kernel_id, "strided": False,
                "context_lengths": lengths, "max_model_len": width, "shared_pages": shared,
                "contiguous_pages": contiguous, "data_pattern": data_pattern, "supplied": supplied}
        for clean in (True, False):
            for backend in _backends(case):
                graph_cases.append({**case, "clean_logits": clean, "backends": (backend,)})
    for mode in ("int8", "fp8"):
        graph_cases.extend(_prefill_cases(mode, graph=True))
    for mode in ("bf16", "mixed", "fp8", "int8"):
        graph_cases.extend(_sparse_cases(mode, graph=True))
    for case in graph_cases:
        for backend in case.get("backends", ("auto",)):
            with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": "" if backend == "auto" else backend}):
                inputs = _make_case_inputs(case)
                print(f"Graph backend={backend} case={case['case']} mode={case['mode']}", flush=True)
                _check_graph(inputs, case["max_model_len"], case.get("supplied", False),
                             case.get("kernel_id", -1), case["clean_logits"], case=case)


def _expect_error(call, message):
    try:
        call()
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected RuntimeError containing {message!r}")


def test_paged_mqa_logits_api(clean_logits=True):
    """API rejection and override priority need different call/check flows."""
    inputs = _make_inputs("e5m2", 1, 32, 16, (129, 192, 255, 287))
    _print_inputs("unsupported E5M2", inputs, 320, clean_logits)
    _expect_error(partial(paged_mqa_logits, *inputs, 320, clean_logits=clean_logits), "neither backend supports")
    inputs = _make_inputs("fp8", 1, 32, 64, (65, 127, 193, 255))
    width = 257
    _print_inputs("environment backend priority", inputs, width, clean_logits)
    reference, valid = ref_paged_mqa_logits(*inputs, width)
    expected = {
        name: op.paged_mqa_logits(*inputs, width, clean_logits=clean_logits, implementation=name)
        for name in ("native", "opus")
    }
    assert expected["native"].stride() != expected["opus"].stride()
    for name in ("native", "opus"):
        with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": name}):
            for implementation in ("auto", "native", "opus"):
                for kernel_id in (-1, 6):
                    actual = op.paged_mqa_logits(*inputs, width, clean_logits=clean_logits,
                                                implementation=implementation, kernel_id=kernel_id)
                    _check_output(actual, reference, valid, clean_logits=clean_logits, context=inputs[3])
                    assert actual.stride() == expected[name].stride()
                    assert torch.equal(actual[valid].view(torch.int32), expected[name][valid].view(torch.int32))
                    print(f"  PASS | env={name}, implementation={implementation}, kernel_id={kernel_id}",
                          flush=True)
            public_output = paged_mqa_logits(*inputs, width, clean_logits=clean_logits)
            assert torch.equal(public_output[valid].view(torch.int32), expected[name][valid].view(torch.int32))
            _check_graph(tuple(tensor.clone() for tensor in inputs), width, supplied=True, kernel_id=6, clean_logits=clean_logits)

    with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": "native"}):
        actual = op.paged_mqa_logits(*inputs, width, clean_logits=clean_logits, implementation="opus", kernel_id=0)
        assert actual.stride() == expected["native"].stride()
        assert torch.equal(actual[valid].view(torch.int32), expected["native"][valid].view(torch.int32))
        print("  PASS | native override ignores opus-only kernel_id=0", flush=True)
        unsupported = _make_inputs("fp8", 1, 32, 1, (96, 128))
        _print_inputs("native override unsupported BLOCK_KV=1", unsupported, 129, clean_logits)
        _expect_error(partial(paged_mqa_logits, *unsupported, 129, clean_logits=clean_logits),
                      "native backend does not support")
    with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": "opus"}):
        unsupported = _make_inputs("bf16", 1, 32, 16, (257, 288, 319, 350))
        _print_inputs("opus override unsupported BF16", unsupported, 384, clean_logits)
        _expect_error(partial(paged_mqa_logits, *unsupported, 384, clean_logits=clean_logits),
                      "OPUS backend does not support")
    with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": "invalid"}):
        _expect_error(partial(op.paged_mqa_logits, *inputs, width, clean_logits=clean_logits,
                              implementation="native"), "BACKEND must be opus or native")
    with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": ""}):
        actual = op.paged_mqa_logits(*inputs, width, clean_logits=clean_logits, implementation="native")
        assert torch.equal(actual[valid].view(torch.int32), expected["native"][valid].view(torch.int32))
        _expect_error(partial(op.paged_mqa_logits, *inputs, width, clean_logits=clean_logits,
                              implementation="native", kernel_id=6), "kernel_id applies only")
    print("  PASS | unsupported/invalid overrides rejected; empty override preserves explicit arguments", flush=True)

    for mode in (("int8", "fp8") if _arch() == "gfx938" else ("int8",)):
        clean = clean_logits
        lengths = _context_lengths(4, 1, 32, 4096)
        inputs = _make_inputs(mode, 1, 32, 64, lengths, True, table_max_model_len=STANDARD_MAX_LEN)
        inputs[4].copy_(inputs[4][lengths.index(max(lengths))].clone().expand_as(inputs[4]))
        _print_inputs("prefill API", inputs, STANDARD_MAX_LEN, clean)
        call = partial(paged_mqa_logits, *inputs, STANDARD_MAX_LEN, is_prefill=True, clean_logits=clean)
        out = torch.empty((4, STANDARD_MAX_LEN), device=inputs[0].device, dtype=torch.float32)
        _expect_error(partial(call, out=out), "prefix prefill does not support supplied out")
        with patch.dict(os.environ, {"DEEPGEMM_PAGED_MQA_LOGITS_BACKEND": "opus"}):
            _expect_error(call, "OPUS backend does not support")
        wrong_dtype = (inputs[0].to(torch.bfloat16), *inputs[1:])
        _print_inputs("prefill unsupported Q dtype", wrong_dtype, STANDARD_MAX_LEN, clean)
        _expect_error(partial(paged_mqa_logits, *wrong_dtype, STANDARD_MAX_LEN,
                              is_prefill=True, clean_logits=clean), "prefix prefill Q must be INT8 or E4M3FN FP8")
        wrong_cache = (inputs[0], inputs[1].view(torch.uint8 if mode == "int8" else torch.int8), *inputs[2:])
        _print_inputs("prefill mismatched Q/cache dtypes", wrong_cache, STANDARD_MAX_LEN, clean)
        cache_error = ("INT8 prefix prefill requires packed INT8 KV cache" if mode == "int8" else
                       "FP8 prefix prefill requires packed uint8 KV cache")
        _expect_error(partial(paged_mqa_logits, *wrong_cache, STANDARD_MAX_LEN,
                              is_prefill=True, clean_logits=clean), cache_error)
        wrong_r = (inputs[0].repeat(1, 2, 1, 1), *inputs[1:])
        _expect_error(partial(paged_mqa_logits, *wrong_r, STANDARD_MAX_LEN,
                              is_prefill=True, clean_logits=clean), "prefix prefill requires next_n=1")
        _expect_error(partial(paged_mqa_logits, *inputs[:4], inputs[4][:, :-1], STANDARD_MAX_LEN,
                              is_prefill=True, clean_logits=clean), "page table does not cover")
        empty_q = (inputs[0][:0], inputs[1], inputs[2][:0], inputs[3][:0], inputs[4][:0])
        for empty_inputs, width in ((empty_q, STANDARD_MAX_LEN), (inputs, 0)):
            actual = paged_mqa_logits(*empty_inputs, width, is_prefill=True, clean_logits=clean)
            assert actual.shape == (empty_inputs[0].size(0), width) and actual.dtype == torch.float32
        print(f"  PASS | prefill API restrictions and empty outputs | clean_logits={clean}", flush=True)

    if clean_logits:
        inputs = make_sparse_case("bf16", 64, 1, 320, requests=40)
        _print_inputs("sparse API group_size=1 requests=40 num_warps=4 persistent_ctas=0", inputs, 320)
        _expect_error(partial(page_mqa_logits_sparse_mask, *inputs, 320, clean_logits=False), "clean_logits=True")
        _expect_error(partial(page_mqa_logits_sparse_mask_grouped, *inputs, 320, group_size=2), "invalid")


# MXFP4/MXFP8 paged subset of DeepSeek tests/test_attention.py (MIT, 1fc9c669).
# Keep its case order, request construction, group32 quantization and 30 repeats.
def _sparse_mx_quantize(x, fmt):
    groups = x.reshape(-1, 4, 32).float()
    scales = groups.abs().amax(-1).clamp_min(1e-4) / (6 if fmt == "mxfp4" else 448)
    bits = scales.contiguous().view(torch.int32)
    exponents = ((bits >> 23) + ((bits & 0x7fffff) != 0).int()).clamp(1, 254)
    scales = (exponents << 23).contiguous().view(torch.float32)
    values = (groups * (1.0 / scales[..., None])).flatten(1)
    if fmt == "mxfp8":
        payload = values.to(torch.float8_e4m3fn)
    else:
        codes = torch.zeros_like(values, dtype=torch.uint8)
        for boundary, inclusive in zip((.25, .75, 1.25, 1.75, 2.5, 3.5, 5),
                                       (False, True, False, True, False, True, False)):
            codes += (values.abs() >= boundary if inclusive else values.abs() > boundary).byte()
        codes |= ((values < 0) & (codes != 0)).byte() << 3
        payload = (codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    scales = sum(exponents[:, g].long() << (8 * g) for g in range(4)).int()
    return payload, scales


def _sparse_mx_dequantize(payload, scales, fmt):
    if fmt == "mxfp4":
        codes = torch.stack((payload.byte() & 15, payload.byte() >> 4), -1).flatten(-2).long()
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        values = lut[codes]
    else:
        values = payload.view(torch.float8_e4m3fn).float()
    exponents = torch.stack([(scales.long() >> (8 * g)) & 255 for g in range(4)], -1).int() - 127
    return torch.ldexp(values.unflatten(-1, (4, 32)), exponents[..., None]).flatten(-2)


def make_sparse_kv_block_indices(context_lens, request_indices, sparse_block_kv, num_max_sparse_blocks, seed):
    rng = random.Random(seed)
    rows, counts, previous, previous_request = [], [], None, None
    for length, request in zip(context_lens, request_indices):
        available = _ceil_div(length, sparse_block_kv)
        count = min(available, num_max_sparse_blocks)
        if count == available:
            blocks = list(range(available))
        elif request != previous_request:
            blocks = rng.sample(range(available), count)
        else:
            kept = [b for b in previous if b < available]
            kept = rng.sample(kept, min(round(count * .8), len(kept)))
            kept_set, replacements = set(kept), set()
            while len(kept) + len(replacements) < count:
                block = rng.randrange(available)
                if block not in kept_set:
                    replacements.add(block)
            blocks = kept + list(replacements)
        blocks.sort()
        rows.append(blocks + [blocks[-1] if blocks else 0] * (num_max_sparse_blocks - count))
        counts.append(count)
        previous, previous_request = blocks, request
    return torch.tensor(rows, device="cuda", dtype=torch.int32), counts


def test_sparse_mqa_logits(benchmark=True):
    """Upstream paged loop, followed by LightOp's nonduplicated shape/request set."""
    _arch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count

    def enumerate_sparse_mqa_logits():
        for fmt in ("mxfp4", "mxfp8"):
            for capacity in (2048, 1024, 512):
                for context in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
                    for heads in (8, 12, 20, 32):
                        yield heads, fmt, 512, context, 8, capacity, 64, None
        for rows, context, block, capacity in (
            (1, 64, 16, 8), (3, 64, 16, 8), (num_cus - 1, 16387, 16, 1024),
            (num_cus + 1, 16387, 16, 1024), (2 * num_cus + 1, 16387, 16, 1024),
            (3, 1048576, 8, 2048),
        ):
            for heads in (8, 12, 20, 32):
                yield heads, "mxfp4", rows, context, block, capacity, 64, None
        for heads in (8, 12, 20, 32):
            yield heads, "mxfp8", 3, 64, 16, 8, 64, None
        # LightOp 37c08dbd: deduplicated geometry/request parameters from all
        # 85 compute cases. Numeric/layout variants use the compact graph check below.
        # H, R, context, block, capacity, page, (requests, causal, CP world, rank, prefix).
        lightop_cases = [
            (4, 2, 79, 8, 8, 64, (1, False, 1, 0, 0)),
            (4, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (8, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (12, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (16, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (20, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (24, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (28, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (32, 1, 31, 8, 4, 64, (1, False, 1, 0, 0)),
            (32, 2, 0, 8, 8, 64, (1, False, 1, 0, 0)),
            (32, 2, 65, 8, 8, 64, (1, False, 1, 0, 0)),
            (32, 3, 1, 8, 128, 64, (1, False, 1, 0, 0)),
            (32, 3, 7, 8, 128, 64, (1, False, 1, 0, 0)),
            (32, 3, 8, 8, 128, 64, (1, False, 1, 0, 0)),
            (32, 3, 9, 8, 128, 64, (1, False, 1, 0, 0)),
            (32, 3, 79, 8, 8, 64, (1, False, 1, 0, 0)),
            (32, 3, 1027, 8, 32, 64, (1, False, 1, 0, 0)),
            (32, 3, 2051, 8, 32, 64, (1, False, 1, 0, 0)),
            (32, 3, 3073, 8, 32, 64, (1, False, 1, 0, 0)),
            (32, 3, 4099, 8, 32, 64, (1, False, 1, 0, 0)),
            (32, 3, 65535, 8, 128, 64, (1, True, 1, 0, 0)),
            (32, 3, 65536, 8, 128, 64, (1, True, 1, 0, 0)),
            (32, 3, 65537, 8, 128, 64, (1, True, 1, 0, 0)),
            (32, 3, 65539, 16, 8, 16, (1, False, 1, 0, 0)),
            (32, 3, 65539, 16, 128, 64, (1, False, 1, 0, 0)),
            (32, 3, 65539, 16, 2052, 256, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 4, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 8, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 12, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 32, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 64, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 128, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 128, 64, (1, True, 1, 0, 0)),
            (32, 3, 131072, 8, 256, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 512, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 1024, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 2044, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 2048, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 2052, 64, (1, False, 1, 0, 0)),
            (32, 3, 131072, 8, 4096, 64, (1, False, 1, 0, 0)),
            (32, 3, 1048576, 8, 128, 64, (1, True, 1, 0, 0)),
            (32, 4, 1024, 8, 4, 64, (1, False, 1, 0, 0)),
            (32, 8, 65, 8, 8, 64, (2, True, 1, 0, 0)),
            (32, 8, 2049, 8, 32, 8, (2, True, 1, 0, 0)),
            (32, 8, 2049, 8, 32, 16, (2, True, 1, 0, 0)),
            (32, 8, 2049, 8, 32, 32, (2, True, 1, 0, 0)),
            (32, 8, 2049, 8, 32, 64, (2, True, 1, 0, 0)),
            (32, 8, 2049, 8, 32, 128, (2, True, 1, 0, 0)),
            (32, 8, 2049, 8, 32, 256, (2, True, 1, 0, 0)),
            (32, 8, 4097, 8, 128, 64, (2, True, 1, 0, 0)),
            (32, 8, 65537, 8, 128, 64, (2, False, 1, 0, 0)),
            (32, 32, 8191, 8, 128, 64, (2, True, 1, 0, 0)),
            (32, 64, 4096, 8, 2048, 64, (1, False, 1, 0, 0)),
            (32, 64, 32768, 8, 2048, 64, (1, True, 1, 0, 0)),
            (32, 64, 65536, 8, 2048, 64, (1, True, 1, 0, 0)),
            (32, 64, 98304, 8, 2048, 64, (1, True, 1, 0, 0)),
            (32, 64, 131072, 8, 2048, 64, (1, True, 1, 0, 0)),
            (32, 128, 65535, 8, 32, 64, (2, True, 1, 0, 0)),
            (32, 512, 257, 8, 8, 64, (2, True, 1, 0, 0)),
            (32, 512, 4096, 8, 2048, 64, (1, True, 8, 0, 0)),
            (32, 1024, 257, 8, 8, 64, (2, True, 1, 0, 0)),
            (32, 2048, 16384, 8, 2048, 64, (1, True, 8, 0, 0)),
            (32, 4096, 257, 8, 8, 64, (2, True, 1, 0, 0)),
            (32, 4096, 32768, 8, 2048, 64, (1, True, 8, 0, 0)),
            (32, 4096, 65536, 8, 2048, 64, (1, True, 8, 0, 32768)),
            (32, 4096, 131072, 8, 2048, 64, (1, True, 8, 0, 98304)),
        ]
        for heads, rows, context, block, capacity, page, requests in lightop_cases:
            yield heads, "mxfp4", rows, context, block, capacity, page, requests

    for heads, fmt, rows, context, block, capacity, page, request_spec in enumerate_sparse_mqa_logits():
        print(f"Sparse source={'NV' if request_spec is None else 'LightOp'} fmt={fmt} R={rows} H={heads} "
              f"D=128 C={context} B={block} M={capacity} P={page} requests={request_spec}", flush=True)
        rng = random.Random(rows * 1000000 + context + block)
        sizes = []
        if request_spec is None:
            remaining = rows
            while remaining:
                size = min(rng.randint(2, 6), remaining)
                sizes.append(size)
                remaining -= size
            lengths = [rng.randint(int(.7 * context) // block, int(1.3 * context) // block) * block for _ in sizes]
            lengths = [length + j for length, size in zip(lengths, sizes) for j in range(size)]
        else:
            requests, causal, cp_world, cp_rank, prefix = request_spec
            sizes = [rows // requests] * requests
            sizes[-1] += rows - sum(sizes)
            lengths = [max(0, (prefix + cp_rank + 1 + cp_world * j if cp_world > 1 else
                              context - (size - 1 - j) if causal else context) - r * (page + 3))
                       for r, size in enumerate(sizes) for j in range(size)]
        request_indices = [r for r, size in enumerate(sizes) for _ in range(size)]
        counts = [max(1, _ceil_div(max(lengths[start:start + size]), page))
                  for start, size in zip([0] + list(itertools.accumulate(sizes))[:-1], sizes)]
        pages, elem_dim = sum(counts), 64 if fmt == "mxfp4" else 128
        q_fp, q_sf = _sparse_mx_quantize(torch.randn((rows * heads, D), device="cuda", dtype=torch.bfloat16), fmt)
        q = q_fp.view(rows, 1, heads, elem_dim), q_sf.view(rows, 1, heads)
        weights = torch.randn((rows, heads), device="cuda", dtype=torch.bfloat16).float()
        sparse_indices, num_blocks = make_sparse_kv_block_indices(lengths, request_indices, block, capacity, context)
        page_stride = _ceil_div(page * (elem_dim + 4), 128) * 128
        storage = torch.empty((pages, page_stride), device="cuda", dtype=torch.uint8)
        payload = storage[:, :page * elem_dim].view(pages, page, elem_dim)
        scales = storage[:, page * elem_dim:page * (elem_dim + 4)].view(torch.int32)
        for begin in range(0, pages, 16384):
            end = min(begin + 16384, pages)
            fp, sf = _sparse_mx_quantize(torch.randn(((end - begin) * page, D), device="cuda", dtype=torch.bfloat16), fmt)
            payload[begin:end].copy_(fp.view(torch.uint8).view(end - begin, page, elem_dim))
            scales[begin:end].copy_(sf.view(end - begin, page))
        kv_cache = storage.as_strided((pages, page, 1, elem_dim + 4), (page_stride, elem_dim + 4, elem_dim + 4, 1))
        indices = torch.tensor(request_indices, device="cuda", dtype=torch.int32)
        context_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        pool = torch.randperm(pages, device="cuda", dtype=torch.int32)
        table = torch.zeros((len(sizes), max(counts)), device="cuda", dtype=torch.int32)
        begin = 0
        for r, count in enumerate(counts):
            table[r, :count] = pool[begin:begin + count]
            begin += count
        block_table = table[indices.long()].contiguous()
        run_metadata = lambda: get_paged_sparse_mqa_logits_metadata(
            context_lens, block_table, indices, page, sparse_indices, q[0].dtype, block)
        metadata = run_metadata()
        run_sparse = lambda: fp8_fp4_paged_sparse_mqa_logits(
            q=q, kv_cache=kv_cache, weights=weights, metadata=metadata,
            num_max_sparse_blocks=capacity, sparse_block_kv=block)
        logits = run_sparse()
        token_indices = (sparse_indices[..., None].long() * block + torch.arange(block, device="cuda")).flatten(1)
        live = torch.arange(capacity * block, device="cuda")[None, :] < torch.tensor(num_blocks, device="cuda")[:, None] * block
        valid = live & (token_indices < context_lens[:, None])
        q_ref = _sparse_mx_dequantize(q[0].view(rows, heads, elem_dim), q[1].view(rows, heads), fmt)
        # The HCU dense API has a different quantization format. Gather the same
        # selected KV tokens into an independent FP32 reference in bounded chunks.
        for begin in range(0, rows, 8):
            end = min(begin + 8, rows)
            tokens = token_indices[begin:end]
            physical = block_table[begin:end].gather(1, tokens // page).long()
            k_ref = _sparse_mx_dequantize(payload[physical, tokens % page], scales[physical, tokens % page], fmt)
            reference = (torch.bmm(q_ref[begin:end], k_ref.transpose(1, 2)).relu() * weights[begin:end, :, None]).sum(1)
            mask = valid[begin:end]
            torch.testing.assert_close(logits[begin:end][mask], reference[mask], atol=1e-3, rtol=5e-6)
        assert torch.isneginf(logits[~live]).all()
        for _ in range(30):
            assert torch.equal(run_sparse()[valid], logits[valid])
        if benchmark:
            timings = _bench({"metadata": run_metadata, "sparse": run_sparse})
            latency = statistics.median(timings["sparse"])
            flops = 2 * sum(num_blocks) * block * heads * D
            physical = block_table.gather(1, sparse_indices.long() * block // page).long() * (page // block)
            physical += sparse_indices.long() % (page // block)
            union = physical[live[:, ::block]].unique().numel()
            byte_count = q[0].numel() + q[1].numel() * 4 + weights.numel() * 4 + metadata.numel() + logits.numel() * 4
            byte_count += union * block * (elem_dim + 4)
            print(f"  PASS | metadata={statistics.median(timings['metadata']):.2f} us, sparse={latency:.2f} us, "
                  f"{flops / latency / 1e6:.2f} TFLOPS, effective={byte_count / latency / 1e3:.2f} GB/s", flush=True)
        else:
            print("  PASS | reference and 30 bitwise repeats", flush=True)
        del q, kv_cache, storage, payload, scales, weights, metadata, logits, q_ref, k_ref, reference
        torch.cuda.empty_cache()


def test_sparse_mqa_logits_graph():
    """One small graph checks FP32 weights, strided operands, remapping and extreme scales."""
    rows, heads, page, block, capacity = 3, 12, 64, 16, 8
    for fmt, elem_dim, one, two in (("mxfp4", 64, 0x22, 0x44), ("mxfp8", 128, 0x38, 0x40)):
        q = torch.full((rows, heads, elem_dim + 1), one, device="cuda", dtype=torch.uint8)[..., :elem_dim]
        q = q.view(torch.int8 if fmt == "mxfp4" else torch.float8_e4m3fn)
        sf = torch.empty((rows, heads + 1), device="cuda", dtype=torch.int32)[:, :heads]
        storage = torch.empty((2, page * (elem_dim + 4) + 1), device="cuda", dtype=torch.uint8)
        storage[0, :page * elem_dim].fill_(one)
        storage[1, :page * elem_dim].fill_(two)
        kv_scales = storage[:, page * elem_dim:page * (elem_dim + 4)]
        kv = storage[:, :page * (elem_dim + 4)].view(2, page, 1, elem_dim + 4)
        weights = torch.empty((rows, heads + 2), device="cuda")[:, :heads]
        weights.copy_(.50005 + torch.arange(heads, device="cuda") * .00002)
        lens = torch.full((rows,), page, device="cuda", dtype=torch.int32)
        table = torch.zeros((rows, 1), device="cuda", dtype=torch.int32)
        ids = torch.arange(rows, device="cuda", dtype=torch.int32)
        candidates = torch.arange(capacity, device="cuda", dtype=torch.int32).clamp_max(3).expand(rows, -1).contiguous()
        output = torch.empty((rows, capacity * block + 4), device="cuda")[:, :capacity * block]
        def run():
            metadata = get_paged_sparse_mqa_logits_metadata(lens, table, ids, page, candidates, q.dtype, block)
            return fp8_fp4_paged_sparse_mqa_logits((q, sf), kv, weights, metadata, capacity, block,
                                                  context_lens=lens, sparse_kv_block_indices=candidates, output=output)
        sf.fill_(0x7f7f7f7f)
        kv_scales.fill_(127)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        for qexp, kexp in ((127, 127), (254, 0), (0, 254), (0, 127), (95, 159), (96, 158), (160, 94), (161, 93)):
            print(f"Sparse graph fmt={fmt} R3 H12 D128 C64 B16 M8 P64 qexp={qexp} kexp={kexp}", flush=True)
            sf.fill_((qexp * 0x01010101 + 2**31) % 2**32 - 2**31)
            kv_scales.fill_(kexp)
            table.fill_(1 - int(table[0, 0]))
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.ldexp(weights.sum(1) * (D * (1 + int(table[0, 0]))),
                                   torch.full((rows,), qexp + kexp - 254, device="cuda", dtype=torch.int32))
            torch.testing.assert_close(captured[:, :page], expected[:, None].expand(rows, page), atol=0, rtol=5e-6)
            assert captured.data_ptr() == output.data_ptr() and torch.isneginf(captured[:, page:]).all()
        lens.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isneginf(captured).all()
        print("  PASS | graph, reciprocal scales, strides and empty contexts", flush=True)


def _bench_fp4_paged(call):
    """NV bench_kineto protocol: 30 samples, 256 MB L2 flush per sample."""
    call()
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True,
    ) as profiler:
        for _ in range(2):
            for _ in range(30):
                torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()
                call()
            torch.cuda.synchronize()
            profiler.step()
    # The HCU scorer has a main kernel and an exceptional-scale kernel.
    # Sum their per-call times rather than averaging across kernel names.
    events = [e for e in profiler.key_averages() if "paged_mqa_fp4_" in e.key]
    assert events, "paged MXFP4 kernels missing from profiler"
    return sum(e.self_device_time_total / e.count for e in events)


def _make_fp4_inputs(rows, heads, length, block=64):
    pages = _ceil_div(length, 64) + 1
    qp = torch.randint(0, 256, (rows, 1, heads, 64), device="cuda", dtype=torch.uint8)
    qs = torch.full((rows, 1, heads), 0x7f7f7f7f, device="cuda", dtype=torch.int32)
    storage = torch.empty(pages, 4368, device="cuda", dtype=torch.uint8)
    storage[:, :4096].random_(0, 256)
    storage[:, 4096:4352].random_(118, 138)
    cache = storage[:, :4352]
    weights = torch.randn(rows, heads + 8, device="cuda")[:, :heads]
    lens = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    table = torch.arange(_ceil_div(length, block), device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
    output = torch.full((rows, length + 32), 12345., device="cuda")[:, :length]
    return (qp, qs), cache, weights, lens, table, output


def _check_fp4_case(x, block=64, n=1):
    q, cache, weights, lens, table, output = x
    rows, heads = weights.shape
    print(f"FP4 correctness R={rows} H={heads} D128 P64 TB={block} width={output.shape[1]} "
          f"rows_per_request={n} seed=17 "
          f"KV_stride={cache.stride()} weights_stride={weights.stride()} out_stride={output.stride()}", flush=True)
    meta = get_paged_mqa_logits_fp4_metadata(lens, rows_per_request=n)
    fp8_fp4_paged_mqa_logits(
        q, cache, weights, lens, table, meta, output.shape[1],
        table_block_size=block, out=output, rows_per_request=n,
    )

    def decode(p, sf):
        codes = torch.stack((p & 15, p >> 4), -1).flatten(-2).long()
        lut = torch.tensor(
            [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda", dtype=torch.float64,
        )
        scales = sf.contiguous().view(torch.uint8).reshape(*p.shape[:-1], 4)
        values = lut[codes] * torch.exp2(scales.double() - 127).repeat_interleave(32, -1)
        valid = (scales != 255).repeat_interleave(32, -1) & (values.abs() < 2.**128)
        return values.masked_fill(~valid, 0)

    qref = decode(q[0].reshape(rows, heads, 64), q[1])
    for row, length in enumerate(lens.tolist()):
        length = max(0, min(length, output.shape[1], table.shape[1] * block))
        if length:
            tokens = torch.arange(length, device="cuda")
            table_row = row - (row % n) % 2 if n > 1 and rows // n >= 8 else row
            slots = table[table_row, tokens // block].long() * block + tokens % block
            payload = cache[:, :4096].reshape(-1, 64, 64)[slots // 64, slots % 64]
            scales = cache[:, 4096:4352].reshape(-1, 64, 4)[slots // 64, slots % 64]
            key = decode(payload, scales)
            want = ((qref[row] @ key.T).relu() * weights[row].double()[:, None]).sum(0)
            mag = ((qref[row].abs() @ key.abs().T) * weights[row].double().abs()[:, None]).sum(0)
            got = output[row, :length].double()
            assert torch.isfinite(got).all()
            assert ((got - want).abs() <= 4 * (128 + heads) * 2**-24 * mag + 2**-148).all()
        assert (output[row, length:] == 12345.).all()
    print("  PASS", flush=True)


def test_fp4_paged_mqa_logits(benchmark=True):
    """NV paged MXFP4 loop in LightOp's supported subset, then LightOp shapes.

    NV 5081f851: page64, D128, H16/32/64, varlen queries, growing contexts,
    random pages and a 32M-token pool. Repeat request tables in place of NV's
    indices; weights/logits remain FP32 as required by the HCU interface.
    """
    _arch()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    def cases():
        for requests, max_queries, heads, average in itertools.product(
                (64, 256), (6, 8), (16, 32, 64), (8192, 65536, 262144)):
            if requests * average <= 32 * 1024 * 1024:
                yield requests, max_queries, heads, average, 64, True
        for heads, (block, length) in itertools.product(
                (16, 32, 64), ((64, 1), (64, 63), (64, 257), (8, 1), (8, 63), (8, 4097))):
            yield 3, 1, heads, length, block, False
        for heads, length in itertools.product((16, 32), (2049, 49152)):
            yield 1, 1, heads, length, 64, False

    def calc_diff(x, y):
        x, y = x.double(), y.double()
        denominator = (x.square() + y.square()).sum()
        return (1 - 2 * (x * y).sum() / denominator).item() if denominator else 0.

    for case_id, (requests, max_queries, heads, average, block, official) in enumerate(cases()):
        counts = (torch.randint(1, max_queries + 1, (requests,), device="cuda", dtype=torch.int32)
                  if official else torch.ones(requests, device="cuda", dtype=torch.int32))
        ids = torch.arange(requests, device="cuda").repeat_interleave(counts)
        rows = ids.numel()
        q_float = torch.randn(rows, 1, heads, D, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, heads, device="cuda")
        lengths = (torch.randint(int(.7 * average), int(1.3 * average), (requests,), device="cuda", dtype=torch.int32)
                   if official else torch.full((requests,), average, device="cuda", dtype=torch.int32))
        offsets = torch.cat([torch.arange(int(n), device="cuda", dtype=torch.int32) for n in counts])
        lens = lengths[ids] + offsets
        sizes = (lengths + counts - 1 + block - 1) // block
        width, units = int(sizes.max()) * block, int(sizes.sum())
        pages = _ceil_div(units * block, 64)
        kv_float = torch.randn(pages * 64, D, device="cuda", dtype=torch.bfloat16)
        table = torch.zeros(requests, width // block, device="cuda", dtype=torch.int32)
        order = torch.randperm(units, device="cuda", dtype=torch.int32)
        cursor = 0
        for request, size in enumerate(sizes.tolist()):
            table[request, :size] = order[cursor:cursor + size]
            cursor += size
        table = table[ids].contiguous()
        reference, valid = ref_paged_mqa_logits(
            q_float, kv_float.reshape(-1, block, 1, D), weights, lens, table, width)
        qp, qs = _sparse_mx_quantize(q_float, "mxfp4")
        kp, ks = _sparse_mx_quantize(kv_float, "mxfp4")
        q = (qp.reshape(rows, 1, heads, 64), qs.reshape(rows, 1, heads))
        cache = torch.cat((kp.byte().reshape(pages, 4096), ks.contiguous().view(torch.uint8).reshape(pages, 256)), 1)
        cache = cache.reshape(pages, 64, 1, 68)
        simulated, _ = ref_paged_mqa_logits(
            _sparse_mx_dequantize(qp, qs, "mxfp4").to(torch.bfloat16).reshape(rows, 1, heads, D),
            _sparse_mx_dequantize(kp, ks, "mxfp4").to(torch.bfloat16).reshape(-1, block, 1, D),
            weights, lens, table, width)
        del q_float, kv_float
        print(f"FP4 case={case_id} official={official} requests={requests} max_queries={max_queries} "
              f"R={rows} H={heads} D128 P64 TB={block} avgKV={average} width={width} "
              f"seed=0 rows_per_request=1 FP32 weights/logits", flush=True)
        meta = get_paged_mqa_logits_fp4_metadata(lens)
        output = torch.full((rows, width), 12345., device="cuda")
        call = lambda schedule: fp8_fp4_paged_mqa_logits(
            q, cache, weights, lens, table, schedule, width, table_block_size=block, out=output)
        call(meta)
        masked = output.masked_fill(~valid, 0)
        for _ in range(20):
            call(meta)
            assert torch.equal(
                masked.contiguous().view(torch.uint8),
                output.masked_fill(~valid, 0).contiguous().view(torch.uint8),
            ), "paged MXFP4 self-consistency"
        diff = calc_diff(masked, reference.masked_fill(~valid, 0))
        simulated_diff = calc_diff(masked, simulated.masked_fill(~valid, 0))
        # NV's original-input tolerance applies to its random matrix; tiny
        # LightOp layout cases validate the actual stored MXFP4 arithmetic.
        assert not official or diff < .02, f"case={case_id} diff={diff}"
        assert simulated_diff < 3e-5, f"case={case_id} simulated_diff={simulated_diff}"
        assert (output[~valid] == 12345.).all()
        output.fill_(12345.)
        call(None)
        assert calc_diff(output.masked_fill(~valid, 0), simulated.masked_fill(~valid, 0)) < 3e-5
        assert (output[~valid] == 12345.).all()
        message = f"  PASS diff={diff:.3e} simulated_diff={simulated_diff:.3e}"
        if benchmark:
            meta_us = _bench_fp4_paged(lambda: get_paged_mqa_logits_fp4_metadata(lens))
            us = _bench_fp4_paged(lambda: call(meta))
            valid_tokens = int(lens.sum())
            # NV's effective-byte convention: KV once per request, packed Q,
            # FP32 weights and valid FP32 logits. This is not an HBM counter.
            effective_bytes = rows * heads * 72 + int(lengths.sum()) * 68 + valid_tokens * 4
            message += (f" {us:.3f} us metadata={meta_us:.3f} us "
                        f"{2 * valid_tokens * heads * D / us / 1e6:.3f} TFLOPS "
                        f"effective={effective_bytes / us / 1000:.3f} GB/s")
        print(message, flush=True)
        del reference, simulated, valid, masked, output, cache, qp, qs, kp, ks
        torch.cuda.empty_cache()


    torch.manual_seed(17)
    for heads in (16, 32, 64):
        x = _make_fp4_inputs(2, heads, 1025)
        x[0][1].view(torch.uint8).random_(100, 110)
        x[1][:, 4096:4352].random_(0, 4)
        _check_fp4_case(x)
    for weight_exponent in (-64, -48, 48, 64):
        x = _make_fp4_inputs(2, 32, 257)
        x[0][1].view(torch.uint8).fill_(140)
        x[1][:, 4096:4352].fill_(110)
        x[2].mul_(2.**weight_exponent)
        print(f"FP4 weight scale 2**{weight_exponent}, Q scale140 K scale110", flush=True)
        _check_fp4_case(x)
    for low, high in ((110, 146), (120, 133), (100, 156)):
        x = _make_fp4_inputs(2, 32, 4097)
        x[1][:, 4096:4352].random_(low, high)
        print(f"FP4 K scale range=[{low},{high})", flush=True)
        _check_fp4_case(x)
    for span, group, sign in itertools.product((0, 1, 5, 6, 10, 11, 12), range(4), (2, 10)):
        x = _make_fp4_inputs(1, 16, 64)
        x[0][0].zero_()
        x[0][0][0, 0, 0, group * 16] = sign
        x[0][1].view(torch.uint8).fill_(127 + span)
        x[0][1].view(torch.uint8).reshape(1, 16, 4)[:, :, group] = 127
        x[1][:, :4096].zero_()
        x[1][:, :4096].reshape(-1, 64, 64)[0, :, group * 16] = torch.arange(64, device="cuda", dtype=torch.uint8) % 16
        x[1][:, 4096:4352].fill_(127 + span)
        x[1][:, 4096:4352].reshape(-1, 64, 4)[:, :, group] = 127
        x[2].zero_()
        x[2][0, 0] = 1
        _check_fp4_case(x)
    for heads in (16, 32):
        x = _make_fp4_inputs(1, heads, 4096)
        x[0][0].zero_()
        x[0][0][0, 0, 0, 0] = 2
        x[1][:, :4096].zero_()
        x[1][:, :4096].reshape(-1, 64, 64)[:64, :, 0] = torch.arange(4096, device="cuda", dtype=torch.uint8).reshape(64, 64) % 16
        x[1][:64, 4096:4352].copy_((torch.arange(4096, device="cuda") // 16).byte().repeat_interleave(4).reshape(64, 256))
        x[2].zero_()
        x[2][0, 0] = 1
        _check_fp4_case(x)
    x = list(_make_fp4_inputs(1, 16, 128))
    x[4] = x[4][:, :1]
    _check_fp4_case(x)
    x = list(_make_fp4_inputs(2, 16, 257))
    x[1] = x[1][:1].expand(x[1].shape[0], -1)
    _check_fp4_case(x)
    x = _make_fp4_inputs(1, 16, 1)
    x[0][0].zero_()
    x[0][0][0, 0, 0, 0], x[0][0][0, 0, 0, 16], x[0][0][0, 0, 0, 32] = 2, 1, 10
    x[0][1].view(torch.uint8).reshape(1, 16, 4)[0, 0] = torch.tensor([147, 120, 147, 127], device="cuda", dtype=torch.uint8)
    x[1][:, :4096].fill_(0x22)
    x[1][:, 4096:4352].fill_(127)
    x[2].zero_()
    x[2][0, 0] = 1
    _check_fp4_case(x)


def test_fp4_paged_mqa_logits_graph():
    """Capture metadata and compute; replay with live inputs and lengths."""
    torch.manual_seed(17)
    cases = [(requests * n, heads, 2049, 64, n) for requests, n, heads in
             itertools.product((4, 8), (2, 3, 4, 6, 7), (16, 32))]
    cases += [(3, 32, 4097, block, 1) for block in (8, 64)]
    for rows, heads, length, block, n in cases:
        x = _make_fp4_inputs(rows, heads, length, block)
        q, cache, weights, lens, table, output = x

        def call():
            meta = get_paged_mqa_logits_fp4_metadata(lens, rows_per_request=n)
            return fp8_fp4_paged_mqa_logits(
                q, cache, weights, lens, table, meta, length,
                table_block_size=block, out=output, rows_per_request=n,
            )

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for replay in range(2):
            lens.copy_((torch.arange(rows, device="cuda", dtype=torch.int32) * 521 + replay * 257) % length)
            q[0].bitwise_xor_(0x88)
            cache[:, :4096].bitwise_xor_(0x11)
            if replay == 0:
                q[1].view(torch.uint8).fill_(127)
                cache[:, 4096:4352].fill_(127)
            else:
                q[1].view(torch.uint8).reshape(rows, heads, 4).copy_(
                    torch.tensor([100, 112, 100, 112], device="cuda", dtype=torch.uint8),
                )
                cache[:, 4096:4352].fill_(150)
            weights.neg_()
            table.copy_(table.flip(1))
            output.fill_(12345.)
            graph.replay()
            captured = output.clone()
            _check_fp4_case(x, block, n)
            assert torch.equal(
                captured.contiguous().view(torch.uint8), output.contiguous().view(torch.uint8),
            ), "paged MXFP4 Graph/eager mismatch"
        print(f"FP4 Graph R={rows} H={heads} D128 P64 TB={block} width={length} "
              f"rows_per_request={n} seed=17 replays=2 Qsf=127/[100,112,100,112] Ksf=127/150 "
              f"lens=(row*521+replay*257)%width PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-benchmark", action="store_true", help="run the same complete matrix without timing")
    args = parser.parse_args()
    print(f"Testing paged_mqa_logits on {_arch()}", flush=True)
    test_paged_mqa_logits(benchmark=not args.no_benchmark)
    test_paged_mqa_logits_graph()
    for clean in (True, False):
        test_paged_mqa_logits_api(clean)
    test_sparse_mqa_logits(benchmark=not args.no_benchmark)
    test_sparse_mqa_logits_graph()
    test_fp4_paged_mqa_logits(benchmark=not args.no_benchmark)
    test_fp4_paged_mqa_logits_graph()
    print("\nAll paged_mqa_logits tests passed.", flush=True)


if __name__ == "__main__":
    main()
