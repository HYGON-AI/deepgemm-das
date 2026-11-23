# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dense and sparse paged MQA logits on gfx936/gfx938."""

from typing import Optional

import torch

from . import op


def paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    *,
    out: Optional[torch.Tensor] = None,
    clean_logits: bool = True,
    is_prefill: bool = False,
) -> torch.Tensor:
    """Return FP32 logits with causal masking.

    For decode, ``clean_logits=False`` leaves positions at or beyond each
    context length unspecified. Valid logits and causal masking inside the
    context are identical to the clean path. Launch geometry depends only on
    tensor metadata, so device context lengths may change during graph replay.

    Prefix prefill on gfx938 supports native E4M3FN FP8 Q and FP8 KV bytes
    in a packed uint8 cache. INT8 Q and packed INT8 KV remain available on
    gfx936/gfx938. The shared native entry dispatches by Q/cache dtype.
    Both paths take one context length per query and a shared block table;
    Q scales must be folded into weights. Launch parameters depend only on
    tensor metadata and remain fixed during Graph replay. Supplied ``out``
    is supported only for decode.
    """
    return op.paged_mqa_logits(
        q, fused_kv_cache, weights, context_lens, block_table, max_model_len,
        out, clean_logits, is_prefill=is_prefill,
    )


def page_mqa_logits_sparse_mask(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    *,
    clean_logits: bool = True,
    num_warps: int = 4,
    persistent_ctas: int = 0,
):
    """Return FP32 sparse values and a packed INT16 mask for independent rows.

    Only values with a set mask bit within the row's context are initialized.
    Each mask element covers 16 tokens; this is not a dense logits output.
    """
    return op.page_mqa_logits_sparse_mask(
        q, fused_kv_cache, weights, context_lens, block_table, max_model_len,
        clean_logits, num_warps, persistent_ctas,
    )


def page_mqa_logits_sparse_mask_grouped(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    *,
    clean_logits: bool = True,
    num_warps: int = 4,
    persistent_ctas: int = 0,
    group_size: int = 5,
):
    """Sparse MQA with independent rows (1) or groups of 3/4/5/6 rows.

    Rows in a group must have identical block-table contents. Q scales must
    be folded into weights. The sparse output contract matches the ungrouped API.
    """
    return op.page_mqa_logits_sparse_mask_grouped(
        q, fused_kv_cache, weights, context_lens, block_table, max_model_len,
        clean_logits, num_warps, persistent_ctas, group_size,
    )


def page_mqa_logits_sparse_mask_grouped_int8(
    q: torch.Tensor,
    packed_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    *,
    clean_logits: bool = True,
    num_warps: int = 4,
    persistent_ctas: int = 0,
    group_size: int = 5,
):
    """Grouped sparse MQA with packed INT8 KV bytes and FP32 scales in uint8 storage."""
    return op.page_mqa_logits_sparse_mask_grouped_int8(
        q, packed_kv_cache, weights, context_lens, block_table, max_model_len,
        clean_logits, num_warps, persistent_ctas, group_size,
    )


def get_paged_sparse_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    indices: torch.Tensor,
    page_kv: int,
    sparse_kv_block_indices: torch.Tensor,
    qk_dtype: torch.dtype,
    sparse_block_kv: int,
) -> torch.Tensor:
    """Build opaque metadata for the selected blocks of each query.

    Valid candidate indices must be unique and strictly increasing per row.
    Request IDs follow the upstream interface; scheduling is chosen internally.
    Capture this producer with the scorer when mappings change during replay.
    """
    return op.get_paged_sparse_mqa_logits_metadata(
        context_lens, block_table, indices, page_kv,
        sparse_kv_block_indices, qk_dtype, sparse_block_kv,
    )


def fp8_fp4_paged_sparse_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    metadata: Optional[torch.Tensor],
    num_max_sparse_blocks: int,
    sparse_block_kv: int,
    *,
    context_lens: Optional[torch.Tensor] = None,
    sparse_kv_block_indices: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    mask_tail: Optional[bool] = None,
) -> torch.Tensor:
    """Return FP32 candidate-slot logits for group32 MXFP4/MXFP8, D=128.

    Q contains the payload and packed UE8M0 scales. Weights are FP32.
    The architecture selects BF16 or native FP8 computation internally, with
    complete exceptional-scale arithmetic. No prefill/decode flag is needed.
    With context lengths and candidates, partial tails are masked unless
    mask_tail=False; unused candidate slots are always -inf.
    If metadata is None, supply block_table, context_lens and candidates:
    the same metadata producer runs before compute inside this call.
    """
    if not isinstance(q, (tuple, list)) or len(q) != 2:
        raise TypeError("q must be (payload, packed UE8M0 scales)")
    return op.fp8_fp4_paged_sparse_mqa_logits(
        q[0], q[1], kv_cache, weights, metadata, num_max_sparse_blocks,
        sparse_block_kv, context_lens, sparse_kv_block_indices, output,
        block_table, mask_tail,
    )


sparse_fp8_fp4_paged_mqa_logits = fp8_fp4_paged_sparse_mqa_logits


def get_paged_mqa_logits_fp4_metadata(
    context_lens: torch.Tensor,
    block_kv: int = 64,
    num_sms: Optional[int] = None,
    rows_per_request: int = 1,
) -> torch.Tensor:
    """Build the fp8_fp4_paged_mqa_logits schedule with graph-live lengths.

    num_sms, if supplied, is the physical CU count; the persistent CTA count
    is selected internally. Requests contain rows_per_request consecutive rows.
    """
    return op.get_paged_mqa_logits_fp4_metadata(context_lens, block_kv, num_sms, rows_per_request)


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: Optional[torch.Tensor] = None,
    max_context_len: Optional[int] = None,
    *,
    table_block_size: int = 64,
    out: Optional[torch.Tensor] = None,
    rows_per_request: int = 1,
) -> torch.Tensor:
    """FP32 logits for packed MXFP4 Q/K, H16/32/64, D128, page64.

    q=(payload, packed UE8M0 scales). This LightOp-compatible subset currently
    excludes NV MXFP8, other head dimensions/pages and varlen indices. Weights
    and logits are FP32. A KV page stores 4096 payload bytes followed by 256
    scale bytes, viewed as uint8 [pages,64,1,68] or [pages,4352]. A table entry
    addresses 64 physical slots (dense) or 8 slots (candidates). Grouping with
    rows_per_request>1 requires table_block_size=64 and H<=32; rows within a
    request share the table. Columns beyond the visible context/table are
    left untouched. Graph replay may update tensor values; request grouping
    and tensor shapes stay fixed. Schedule metadata must be generated for the
    same row grouping; shapes fitting one CTA per CU use a direct grid instead.
    """
    if not isinstance(q, (tuple, list)) or len(q) != 2:
        raise TypeError("q must be (packed FP4 payload, UE8M0 scales)")
    if max_context_len is None:
        max_context_len = block_table.shape[1] * table_block_size
    return op.fp8_fp4_paged_mqa_logits(
        q[0], q[1], kv_cache, weights, context_lens, block_table,
        schedule_meta, max_context_len, table_block_size, out, rows_per_request,
    )
