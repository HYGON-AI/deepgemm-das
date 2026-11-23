# SPDX-License-Identifier: Apache-2.0
"""Non-paged BF16/INT8/FP8 MQA with dense or compact output."""
from typing import Optional

import torch

from . import op

__all__ = ["mqa_logits"]


def mqa_logits(
    Q: torch.Tensor,
    K: torch.Tensor,
    Weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    kv_scale: Optional[torch.Tensor] = None,
    clean_logit: bool = True,
    D_out: Optional[torch.Tensor] = None,
    *, max_seqlen_k=None,
) -> torch.Tensor:
    """Sum_h(ReLU(Q[m,h,:] @ K[n,:]) * Weights[m,h]), masked by [ks,ke).

    Q/K dtype selects BF16, INT8 or FP8 E4M3FN. Quantized Q scales must
    already be included in Weights. INT8 requires kv_scale; FP8 accepts it
    optionally; BF16 rejects it. BF16/INT8 target gfx936/gfx938; FP8 targets gfx938.

    Dense output is FP32 [M,N]. clean_logit=True fills invalid cells with -inf.
    With clean_logit=False only window values are defined; skipped tiles
    preserve caller contents. D_out may supply a compatible output buffer.

    max_seqlen_k=W selects compact output: allocate [M,align(W,256)] and
    return [:,:W]. D_out is rejected and clean_logit ignored. Only each row's
    valid prefix is defined; tails are not cleared. The caller must guarantee
    W covers every clamped window length. Capacity is not checked on the host.
    Calls are asynchronous on the current stream and support graph capture.
    """
    if max_seqlen_k is None:
        q_seq_len, num_heads, head_dim = Q.shape
        kv_seq_len = K.shape[0]
        assert num_heads in [1, 2, 4, 8, 16, 32, 64, 128], "num_heads must be a power of two in [1,128]"
        assert head_dim == 128, "head_dim only supports 128"
        return op.mqa_logits(
            Q, K, Weights, ks, ke, q_seq_len, kv_seq_len, num_heads, head_dim,
            kv_scale, clean_logit, D_out,
        )
    return _mqa_logits_compact(Q, K, Weights, ks, ke, kv_scale, D_out, max_seqlen_k)


def _mqa_logits_compact(Q, K, Weights, ks, ke, kv_scale, D_out, max_seqlen_k):
    if isinstance(max_seqlen_k, bool) or not isinstance(max_seqlen_k, int):
        raise TypeError("max_seqlen_k must be an int, excluding bool")
    if not 1 <= max_seqlen_k <= 2**31 - 1:
        raise ValueError("max_seqlen_k must be in [1, INT32_MAX]")
    if D_out is not None:
        raise ValueError("max_seqlen_k selects compact output: output is library-allocated; D_out is unsupported")
    q_seq_len, num_heads, head_dim = Q.shape
    kv_seq_len = K.shape[0]
    return op.mqa_logits_compact(
        Q, K, Weights, ks, ke, q_seq_len, kv_seq_len, num_heads, head_dim,
        kv_scale, max_seqlen_k,
    )
