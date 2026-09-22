from __future__ import annotations

import functools
import os
from typing import Any, Callable


_ENV_ENABLE = "DEEPGEMM_PRINT_OP_SIZE"
_ENV_STATS = "DEEPGEMM_PRINT_OP_SIZE_STATS"
_ENV_FILTER = "DEEPGEMM_PRINT_OP_SIZE_FILTER"
_PATCH_MARKER = "__deepgemm_print_size_wrapped__"


def _enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "0") not in ("", "0", "false", "False")


PRINT_SIZE_ENABLED = _enabled()


def _stats_enabled() -> bool:
    return os.environ.get(_ENV_STATS, "0") not in ("", "0", "false", "False")


def _shape(x: Any) -> str:
    shape = getattr(x, "shape", None)
    if shape is None:
        return str(x)
    return "(" + ",".join(str(int(v)) for v in shape) + ")"


def _dtype(x: Any) -> str:
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        return ""
    return str(dtype).replace("torch.", "")


def _device(x: Any) -> str:
    device = getattr(x, "device", None)
    if device is None:
        return ""
    return str(device)


def _tensor_desc(x: Any) -> str:
    if x is None:
        return "None"
    dtype = _dtype(x)
    device = _device(x)
    suffix = ""
    if dtype:
        suffix += f":{dtype}"
    if device:
        suffix += f":{device}"
    return f"{_shape(x)}{suffix}"


def _maybe_int(value: Any) -> str:
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _dim(x: Any, index: int, default: Any = 0) -> Any:
    shape = getattr(x, "shape", None)
    if shape is None:
        return default
    try:
        return shape[index]
    except Exception:
        return default


def _maybe_mask_stats(masked_m: Any) -> list[str]:
    if not _stats_enabled():
        return []
    try:
        return [
            f"masked_m_sum={int(masked_m.sum().item())}",
            f"masked_m_min={int(masked_m.min().item())}",
            f"masked_m_max={int(masked_m.max().item())}",
        ]
    except Exception as exc:
        return [f"masked_m_stats_error={type(exc).__name__}"]


def _should_print(op_name: str) -> bool:
    value = os.environ.get(_ENV_FILTER)
    if not value:
        return True
    needles = [item.strip() for item in value.split(",") if item.strip()]
    return any(needle in op_name for needle in needles)


def _print_size(op_name: str, fields: list[str]) -> None:
    if PRINT_SIZE_ENABLED and _should_print(op_name):
        print("[DeepGEMM][size]", op_name, *fields)


def _wrap_pybind_op(
    op_name: str,
    fn: Callable[..., Any],
    formatter: Callable[..., list[str]],
) -> Callable[..., Any]:
    if getattr(fn, _PATCH_MARKER, False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            _print_size(op_name, formatter(*args, **kwargs))
        except Exception as exc:
            print("[DeepGEMM][size]", op_name, f"format_error={type(exc).__name__}")
        return fn(*args, **kwargs)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _format_contiguous(
    input: Any,
    b_qweight: Any,
    output: Any,
    a_scale: Any,
    b_scale: Any,
    m_indices: Any,
    mode: Any,
) -> list[str]:
    return [
        f"input={_tensor_desc(input)}",
        f"b_qweight={_tensor_desc(b_qweight)}",
        f"output={_tensor_desc(output)}",
        f"a_scale={_tensor_desc(a_scale)}",
        f"b_scale={_tensor_desc(b_scale)}",
        f"m_indices={_tensor_desc(m_indices)}",
        f"E={_maybe_int(_dim(b_qweight, 0))}",
        f"M={_maybe_int(_dim(input, 0))}",
        f"N={_maybe_int(_dim(output, -1))}",
        f"K={_maybe_int(_dim(input, -1))}",
        f"mode={_maybe_int(mode)}",
    ]


def _format_contiguous_nopad(
    input: Any,
    b_qweight: Any,
    output: Any,
    a_scale: Any,
    b_scale: Any,
    m_indices: Any,
    token_per_expert: Any,
) -> list[str]:
    fields = _format_contiguous(input, b_qweight, output, a_scale, b_scale, m_indices, "nopad")
    fields.append(f"token_per_expert={_tensor_desc(token_per_expert)}")
    fields.extend(_maybe_mask_stats(token_per_expert))
    return fields


def _format_masked(
    input: Any,
    b_qweight: Any,
    output: Any,
    a_scale: Any,
    b_scale: Any,
    masked_m: Any,
    expected_m: Any,
    mode: Any,
) -> list[str]:
    fields = [
        f"input={_tensor_desc(input)}",
        f"b_qweight={_tensor_desc(b_qweight)}",
        f"output={_tensor_desc(output)}",
        f"a_scale={_tensor_desc(a_scale)}",
        f"b_scale={_tensor_desc(b_scale)}",
        f"masked_m={_tensor_desc(masked_m)}",
        f"E={_maybe_int(_dim(input, 0))}",
        f"M={_maybe_int(_dim(input, 1))}",
        f"N={_maybe_int(_dim(output, -1))}",
        f"K={_maybe_int(_dim(input, -1))}",
        f"expected_m_per_group={_maybe_int(expected_m)}",
        f"mode={_maybe_int(mode)}",
    ]
    fields.extend(_maybe_mask_stats(masked_m))
    return fields


def _format_bf16_masked(
    input: Any,
    b_qweight: Any,
    output: Any,
    masked_m: Any,
    expected_m: Any,
    mode: Any,
) -> list[str]:
    return _format_masked(input, b_qweight, output, None, None, masked_m, expected_m, mode)



def _format_fp8_gemm(
    mat_a: Any,
    mat_b: Any,
    scale_a: Any,
    scale_b: Any,
    output: Any,
    m: Any,
    n: Any,
    k: Any,
    batch: Any,
    transpose_flag: Any,
    alpha: Any,
    beta: Any,
    bias: Any = None,
) -> list[str]:
    return [
        f"mat_a={_tensor_desc(mat_a)}",
        f"mat_b={_tensor_desc(mat_b)}",
        f"scale_a={_tensor_desc(scale_a)}",
        f"scale_b={_tensor_desc(scale_b)}",
        f"output={_tensor_desc(output)}",
        f"M={_maybe_int(m)}",
        f"N={_maybe_int(n)}",
        f"K={_maybe_int(k)}",
        f"batch={_maybe_int(batch)}",
        f"layout={transpose_flag}",
        f"bias={_tensor_desc(bias)}",
    ]


def _format_w8a16_linear(
    a: Any,
    b: Any,
    scale: Any,
    num_k_warps: Any = 0,
    n_per_warp: Any = 0,
    k_per_warp: Any = 0,
    num_n_warps: Any = 0,
    splitk: Any = 0,
    block_m: Any = 0,
    weight_layout: Any = 0,
) -> list[str]:
    return [
        f"a={_tensor_desc(a)}",
        f"b={_tensor_desc(b)}",
        f"scale={_tensor_desc(scale)}",
        f"M={_maybe_int(_dim(a, 0))}",
        f"N={_maybe_int(_dim(b, 0))}",
        f"K={_maybe_int(_dim(a, -1))}",
        f"num_k_warps={_maybe_int(num_k_warps)}",
        f"n_per_warp={_maybe_int(n_per_warp)}",
        f"k_per_warp={_maybe_int(k_per_warp)}",
        f"num_n_warps={_maybe_int(num_n_warps)}",
        f"splitk={_maybe_int(splitk)}",
        f"block_m={_maybe_int(block_m)}",
        f"weight_layout={_maybe_int(weight_layout)}",
    ]


def _format_w8a16_linear_bias(a: Any, b: Any, scale: Any, bias: Any, *rest: Any) -> list[str]:
    fields = _format_w8a16_linear(a, b, scale, *rest)
    fields.insert(3, f"bias={_tensor_desc(bias)}")
    return fields


def _format_tf32(A: Any, B: Any, D: Any, sqr_sum: Any, num_splits: Any = None) -> list[str]:
    return [
        f"A={_tensor_desc(A)}",
        f"B={_tensor_desc(B)}",
        f"D={_tensor_desc(D)}",
        f"sqr_sum={_tensor_desc(sqr_sum)}",
        f"M={_maybe_int(_dim(A, 0))}",
        f"N={_maybe_int(_dim(B, -1))}",
        f"K={_maybe_int(_dim(A, -1))}",
        f"num_splits={_maybe_int(num_splits)}",
    ]


def _format_einsum(expr: Any, a: Any, b: Any, d: Any, c: Any = None) -> list[str]:
    return [
        f"expr={expr}",
        f"a={_tensor_desc(a)}",
        f"b={_tensor_desc(b)}",
        f"d={_tensor_desc(d)}",
        f"c={_tensor_desc(c)}",
    ]


def _format_fp8_einsum(expr: Any, a: Any, b: Any, d: Any, c: Any = None, recipe: Any = None) -> list[str]:
    a_tensor = a[0] if isinstance(a, (tuple, list)) and a else a
    a_scale = a[1] if isinstance(a, (tuple, list)) and len(a) > 1 else None
    b_tensor = b[0] if isinstance(b, (tuple, list)) and b else b
    b_scale = b[1] if isinstance(b, (tuple, list)) and len(b) > 1 else None
    return [
        f"expr={expr}",
        f"a={_tensor_desc(a_tensor)}",
        f"a_scale={_tensor_desc(a_scale)}",
        f"b={_tensor_desc(b_tensor)}",
        f"b_scale={_tensor_desc(b_scale)}",
        f"d={_tensor_desc(d)}",
        f"c={_tensor_desc(c)}",
        f"recipe={recipe}",
    ]


def print_w8a16_linear_size(op_name: str, x: Any, weight: Any, scale: Any, bias: Any, config: Any) -> None:
    if not PRINT_SIZE_ENABLED:
        return
    fields = [
        f"x={_tensor_desc(x)}",
        f"weight={_tensor_desc(weight)}",
        f"scale={_tensor_desc(scale)}",
        f"bias={_tensor_desc(bias)}",
        f"M={_maybe_int(_dim(x, 0))}",
        f"N={_maybe_int(_dim(weight, 0))}",
        f"K={_maybe_int(_dim(x, -1))}",
    ]
    if isinstance(config, dict):
        for key in (
            "KERNEL_TYPE",
            "NUM_K_WARPS",
            "N_PER_WARP",
            "K_PER_WARP",
            "NUM_N_WARPS",
            "SPLITK",
            "BLOCK_M",
            "WEIGHT_LAYOUT",
        ):
            fields.append(f"{key.lower()}={_maybe_int(config.get(key, 0))}")
    _print_size(op_name, fields)


def print_contiguous_size(
    op_name: str,
    input: Any,
    b_qweight: Any,
    output: Any,
    a_scale: Any,
    b_scale: Any,
    m_indices: Any,
    mode: Any,
    *,
    block_m: Any = None,
    block_n: Any = None,
    n_loop_num: Any = None,
) -> None:
    if not PRINT_SIZE_ENABLED:
        return
    fields = _format_contiguous(input, b_qweight, output, a_scale, b_scale, m_indices, mode)
    if block_m is not None:
        fields.append(f"block_m={_maybe_int(block_m)}")
    if block_n is not None:
        fields.append(f"block_n={_maybe_int(block_n)}")
    if n_loop_num is not None:
        fields.append(f"n_loop_num={_maybe_int(n_loop_num)}")
    _print_size(op_name, fields)


def print_low_latency_masked_size(
    op_name: str,
    matrix_a: Any,
    matrix_b: Any,
    matrix_a_scale: Any,
    matrix_b_scale: Any,
    actual_tokens: Any,
    matrix_c: Any,
    max_tokens: Any,
    experts: Any,
    *,
    cu: Any = None,
    block_wise: Any = None,
    over_lap: Any = None,
    signal: Any = None,
) -> None:
    if not PRINT_SIZE_ENABLED:
        return
    fields = _format_masked(
        matrix_a,
        matrix_b,
        matrix_c,
        matrix_a_scale,
        matrix_b_scale,
        actual_tokens,
        max_tokens,
        "low_latency",
    )
    fields.extend([
        f"experts={_maybe_int(experts)}",
        f"cu={_maybe_int(cu)}",
        f"block_wise={block_wise}",
        f"over_lap={over_lap}",
        f"signal={_tensor_desc(signal)}",
    ])
    _print_size(op_name, fields)


def print_moe_prefill_size(
    op_name: str,
    input: Any,
    b_qweight: Any,
    output: Any,
    a_scale: Any,
    b_scale: Any,
    topk_weights: Any,
    sorted_token_ids: Any,
    expert_ids: Any,
    num_tokens_post_pad: Any,
    top_k: Any,
    real_topk: Any,
) -> None:
    if not PRINT_SIZE_ENABLED:
        return
    fields = [
        f"input={_tensor_desc(input)}",
        f"b_qweight={_tensor_desc(b_qweight)}",
        f"output={_tensor_desc(output)}",
        f"a_scale={_tensor_desc(a_scale)}",
        f"b_scale={_tensor_desc(b_scale)}",
        f"topk_weights={_tensor_desc(topk_weights)}",
        f"sorted_token_ids={_tensor_desc(sorted_token_ids)}",
        f"expert_ids={_tensor_desc(expert_ids)}",
        f"num_tokens_post_pad={_tensor_desc(num_tokens_post_pad)}",
        f"M={_maybe_int(_dim(input, 0))}",
        f"N={_maybe_int(_dim(output, -1))}",
        f"K={_maybe_int(_dim(input, -1))}",
        f"top_k={_maybe_int(top_k)}",
        f"real_topk={_maybe_int(real_topk)}",
    ]
    _print_size(op_name, fields)


def install_op_size_printer(op_module: Any) -> None:
    """Patch selected direct ``deepgemm.op`` entry points when size printing is enabled."""
    if not PRINT_SIZE_ENABLED:
        return

    specs: dict[str, Callable[..., list[str]]] = {
        "m_grouped_w8a8_gemm_nt_nopad_contiguous": _format_contiguous_nopad,
        "m_grouped_w8a8_gemm_nt_contiguous": _format_contiguous,
        "m_grouped_w16a16_gemm_nt_contiguous": _format_contiguous,
        "m_grouped_fp8_gemm_nt_contiguous": _format_contiguous,
        "m_grouped_marlin_w4a8_gemm_nt_masked": _format_masked,
        "m_grouped_marlin_w8a8_gemm_nt_masked": _format_masked,
        "m_grouped_bf16_gemm_nt_masked": _format_bf16_masked,
        "fp8_gemm": _format_fp8_gemm,
        "w8a16_linear": _format_w8a16_linear,
        "w8a16_linear_bias": _format_w8a16_linear_bias,
        "tf32_hc_pernorm_gemm": _format_tf32,
        "einsum": _format_einsum,
        "fp8_einsum": _format_fp8_einsum,
    }

    for op_name, formatter in specs.items():
        fn = getattr(op_module, op_name, None)
        if fn is None:
            continue
        setattr(op_module, op_name, _wrap_pybind_op(op_name, fn, formatter))
