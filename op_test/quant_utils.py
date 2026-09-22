import torch


def per_token_quant_int8(x: torch.Tensor, eps: float = 1e-10):
    """Quantize each token along the last dimension to int8.

    This mirrors the lmslim test helper formula without importing lmslim,
    which can transitively load lightop in CI.
    """
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")

    x_float = x.float()
    absmax = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = absmax / 127.0
    q_x = torch.round(x_float * (127.0 / absmax)).clamp(-128, 127).to(torch.int8)
    return q_x.contiguous(), scale.contiguous()
