# deepgemm API 接口文档

## 通用接口（gfx936/gfx938）

### `m_grouped_i8_gemm_nt_contiguous`

来源：`test_deepgemm_contiguous_w8a8_int8_shuffle_unique.py`、`test_deepgemm_contiguous_w8a8_int8_non_shuffle_unique.py`

contiguous grouped W8A8/INT8 GEMM 统一入口。

```python
def m_grouped_i8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
    *,
    shuffle_unique: int = 1,
) -> torch.Tensor:
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[M, K]` int8，`a_scale` 为 `[M]` 或 `[M, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用 `marlin_i8_contiguous_weight` 生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `output`：输出 tensor，shape `[M, N]`，通常为 bf16/fp16。
- `m_indices`：shape `[M]`，dtype int32。`m_indices[i]` 表示第 `i` 行属于哪个 expert，padding 行填 `-1`。
- `shuffle_unique`：权重重排格式选择，默认 `1`。`1` 表示统一权重重排格式，`0` 表示非统一权重重排格式。

调用顺序：先重排权重，再调用 GEMM。`marlin_i8_contiguous_weight` 和 GEMM 的 `shuffle_unique` 必须保持一致。

```python
packed_b = marlin_i8_contiguous_weight(
    b_int8,
    shuffle_unique=shuffle_unique,
)
b = [packed_b, b_scale]

m_grouped_i8_gemm_nt_contiguous(
    a,
    b,
    output,
    m_indices,
    shuffle_unique=shuffle_unique,
)
```

计算语义：

```text
output[i] = q_a[i] @ weight[e].T * a_scale[i] * b_scale[e]
e = m_indices[i], m_indices[i] >= 0
```

`m_indices[i] == -1` 的 padding 行不保证输出有效。

### `marlin_i8_contiguous_weight`

来源：`deepgemm/group_gemm_weight_pack.py`

contiguous INT8/W8A8 权重重排入口。

```python
def marlin_i8_contiguous_weight(
    weight: torch.Tensor,
    *,
    shuffle_unique: int = 1,
    arch: Optional[str] = None,
) -> torch.Tensor:
```

参数：

- `weight`：原始权重，shape `[E, N, K]`，dtype int8。
- `shuffle_unique`：权重重排格式选择，需与 `m_grouped_i8_gemm_nt_contiguous` 的同名参数保持一致。`1` 表示统一权重重排格式，`0` 表示非统一权重重排格式。
- `arch`：默认 `None`，内部使用当前 GPU 架构；一般不需要显式传入。

### `m_grouped_i8_gemm_nt_masked`

来源：`test_deepgemm_masked_w8a8_int8_shuffle_unique.py`、`test_deepgemm_masked_w8a8_int8_non_shuffle_unique.py`

masked grouped W8A8/INT8 GEMM 统一入口。

```python
def m_grouped_i8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = None,
    *,
    shuffle_unique: int = 1,
    block_wise: bool = False,
    cu: Optional[int] = None,
) -> torch.Tensor:
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[E, M_pad, K]` int8，`a_scale` 为 `[E, M_pad]` 或 `[E, M_pad, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用 `marlin_i8_masked_weight` 生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `d`：输出 tensor，shape `[E, M_pad, N]`，通常为 bf16/fp16。
- `masked_m`：shape `[E]`，dtype int32，表示每个 expert 的有效行数。
- `expected_m_per_group`：kernel launch 使用的每个 expert token 上限，通常由框架侧按固定 bucket/capacity 传入，可大于实际 `masked_m`；避免在 GEMM 调用热路径里通过 `masked_m.max().item()` 临时计算。
- `shuffle_unique`：权重重排格式选择，默认 `1`。`1` 表示统一权重重排格式，`0` 表示非统一权重重排格式。

调用顺序：先重排权重，再调用 GEMM。`marlin_i8_masked_weight` 和 GEMM 的 `shuffle_unique` 必须保持一致。

```python
packed_weight = marlin_i8_masked_weight(
    weight,
    shuffle_unique=shuffle_unique,
)
b = [packed_weight, b_scale]

m_grouped_i8_gemm_nt_masked(
    a,
    b,
    d,
    masked_m,
    expected_m_per_group,
    shuffle_unique=shuffle_unique,
)
```

计算语义：

```text
d[e, :masked_m[e], :] = q_a[e, :masked_m[e], :] @ weight[e].T * a_scale * b_scale
```

`d[e, masked_m[e]:, :]` 不保证输出有效。

### `marlin_i8_masked_weight`

来源：`deepgemm/group_gemm_weight_pack.py`

masked INT8/W8A8 权重重排入口。

```python
def marlin_i8_masked_weight(
    weight: torch.Tensor,
    *,
    shuffle_unique: int = 1,
    arch: Optional[str] = None,
) -> torch.Tensor:
```

参数：

- `weight`：原始权重，shape `[E, N, K]`，dtype int8。
- `shuffle_unique`：权重重排格式选择，需与 `m_grouped_i8_gemm_nt_masked` 的同名参数保持一致。`1` 表示统一权重重排格式，`0` 表示非统一权重重排格式。
- `arch`：默认 `None`，内部使用当前 GPU 架构；一般不需要显式传入。

## gfx936

### `m_grouped_w4a8_gemm_nt_masked`

来源：`test_deepgemm_masked_w4a8.py`

```python
def m_grouped_w4a8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],   # (q_a, a_scale)，q_a [E, M_pad, K] int8
    b: tuple[torch.Tensor, torch.Tensor],   # (weight_marlin, w_scale)，w_scale [E, N, 1]
    d: torch.Tensor,                        # bf16 输出 [E, M_pad, N]
    masked_m: torch.Tensor,                 # [E] int32
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = {"MODE": 1002},
) -> torch.Tensor:
    """
    masked grouped W4A8 GEMM。
    weight 原始形状为 [E, N, K//2]，需要 weight4bit_nt_kpack2_marlin2_qqq_from_packed 预处理。
    """
```

### `m_grouped_bf16_gemm_nt_masked`

来源：`test_deepgemm_masked_w16a16_bf16.py`

```python
def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,                       # bf16 激活 [E, M_pad, K]
    b: torch.Tensor,                       # bf16 权重 [E, N, K]
    d: torch.Tensor,                       # bf16 输出 [E, M_pad, N]
    masked_m: torch.Tensor,                # [E] int32
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = {"MODE": 1000},
) -> torch.Tensor:
    """
    masked grouped BF16 GEMM。
    语义：d[e, :masked_m[e], :] = a[e, :masked_m[e], :] @ b[e].T。
    """
```

### `m_grouped_bf16_gemm_nt_contiguous`

来源：`deepgemm/m_group_gemm_nt_contiguous.py`

```python
def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,                       # bf16 激活 [M, K]
    b: torch.Tensor,                       # bf16 权重 [E, N, K]
    output: torch.Tensor,                  # bf16 输出 [M, N]
    m_indices: torch.Tensor,               # [M] int32，行到 expert 的映射，padding 行为 -1
    config: Optional[Dict[str, Any]] = {"MODE": 1000},
) -> torch.Tensor:
    """
    contiguous grouped BF16 GEMM。
    语义：对 m_indices[i] >= 0 的行，计算 output[i] = a[i] @ b[e].T。
    b 是普通 contiguous BF16 张量，不需要 pack。
    """
```

## gfx938

### `m_grouped_fp8_gemm_nt_contiguous`

来源：`test_deepgemm_contiguous_w8a8_fp8.py`

```python
def m_grouped_fp8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],   # (a_fp8, a_scale)，a_fp8 [M, K] fp8，a_scale [M]
    b: tuple[torch.Tensor, torch.Tensor],   # (weight_w6, b_scale)，b_scale [E, N]
    output: torch.Tensor,                   # bf16 输出 [M, N]
    m_indices: torch.Tensor,                # [M] int32，padding 行为 -1
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    contiguous grouped FP8 GEMM，per-channel weight scale。
    weight_w6 由 marlin_fp8_contiguous_weight(weight_fp8) 生成。
    """
```

### `marlin_fp8_contiguous_weight`

来源：`deepgemm/group_gemm_weight_pack.py`

contiguous W8A8/FP8 权重重排入口，配套 `m_grouped_fp8_gemm_nt_contiguous` 使用。

```python
def marlin_fp8_contiguous_weight(
    weight: torch.Tensor,
    *,
    arch: Optional[str] = None,
) -> torch.Tensor:
```

参数：

- `weight`：原始 FP8 权重，shape `[E, N, K]`。
- `arch`：默认 `None`，内部使用当前 GPU 架构；一般不需要显式传入。
- 返回值：重排后的 `weight_w6`，作为 GEMM 的 `b[0]` 传入。

### `m_grouped_fp8_gemm_nt_contiguous_tensor_wise`

来源：`test_deepgemm_contiguous_w8a8_fp8_tensor_wise.py`

```python
def m_grouped_fp8_gemm_nt_contiguous_tensor_wise(
    a: tuple[torch.Tensor, torch.Tensor],   # (a_fp8, a_scale)，a_fp8 [M, K]，a_scale [M]
    b: tuple[torch.Tensor, torch.Tensor],   # (weight_w6, b_scale)，b_scale [E] 或 [E, 1]
    output: torch.Tensor,                   # bf16 输出 [M, N]
    m_indices: torch.Tensor,                # [M] int32，padding 行为 -1
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    contiguous grouped FP8 GEMM，tensor-wise weight scale。
    """
```

### `m_grouped_bf16_gemm_nt_contiguous`

来源：`deepgemm/m_group_gemm_nt_contiguous.py`

```python
def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,                       # bf16 激活 [M, K]
    b: torch.Tensor,                       # bf16 权重 [E, N, K]
    output: torch.Tensor,                  # bf16 输出 [M, N]
    m_indices: torch.Tensor,               # [M] int32，行到 expert 的映射，padding 行为 -1
    config: Optional[Dict[str, Any]] = {"MODE": 1000},
) -> torch.Tensor:
    """
    contiguous grouped BF16 GEMM。
    语义：对 m_indices[i] >= 0 的行，计算 output[i] = a[i] @ b[e].T。
    b 是普通 contiguous BF16 张量，不需要 pack。
    """
```

### `m_grouped_fp8_gemm_nt_masked`

来源：`test_deepgemm_masked_w8a8_fp8.py`

```python
def m_grouped_fp8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],   # (input_fp8, input_scale)，input_fp8 [E, M_pad, K]
    b: tuple[torch.Tensor, torch.Tensor],   # (weight_w6, weight_scale)，weight_scale [E, N, 1]
    d: torch.Tensor,                        # bf16 输出 [E, M_pad, N]
    masked_m: torch.Tensor,                 # [E] int32
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    block_wise: bool = False,
    cu: int = 128,
) -> torch.Tensor:
    """
    masked grouped FP8 GEMM，per-channel weight scale。
    """
```

### `marlin_fp8_masked_weight`

来源：`deepgemm/group_gemm_weight_pack.py`

masked W8A8/FP8 权重重排入口，配套 `m_grouped_fp8_gemm_nt_masked` 使用。

```python
def marlin_fp8_masked_weight(
    weight: torch.Tensor,
    *,
    arch: Optional[str] = None,
) -> torch.Tensor:
```

参数：

- `weight`：原始 FP8 权重，shape `[E, N, K]`。
- `arch`：默认 `None`，内部使用当前 GPU 架构；一般不需要显式传入。
- 返回值：重排后的 `weight_w6`，作为 GEMM 的 `b[0]` 传入。

### `m_grouped_fp8_gemm_nt_masked_tensor_wise`

来源：`test_deepgemm_masked_w8a8_fp8_tensor_wise.py`

```python
def m_grouped_fp8_gemm_nt_masked_tensor_wise(
    a: tuple[torch.Tensor, torch.Tensor],   # (input_fp8, input_scale)，input_fp8 [E, M_pad, K]
    b: tuple[torch.Tensor, torch.Tensor],   # (weight_w6, weight_scale)，weight_scale [E] 或 [E, 1]
    d: torch.Tensor,                        # bf16 输出 [E, M_pad, N]
    masked_m: torch.Tensor,                 # [E] int32
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    cu: int = 128,
) -> torch.Tensor:
    """
    masked grouped FP8 GEMM，tensor-wise weight scale。
    """
```

### `m_grouped_w4a8_gemm_nt_masked`

来源：`test_deepgemm_masked_w4a8.py`

```python
def m_grouped_w4a8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = {"MODE": 1002},
) -> torch.Tensor:
    """
    masked grouped W4A8 GEMM。
    a = (q_a, a_scale)，b = (weight_marlin, w_scale)。
    """
```

### `m_grouped_bf16_gemm_nt_masked`

来源：`test_deepgemm_masked_w16a16_bf16.py`

```python
def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,                       # [E, M_pad, K] bf16
    b: torch.Tensor,                       # [E, N, K] bf16
    d: torch.Tensor,                       # [E, M_pad, N] bf16
    masked_m: torch.Tensor,                # [E] int32
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = {"MODE": 1000},
) -> torch.Tensor:
    """
    masked grouped BF16 GEMM。
    语义：d[e, :masked_m[e], :] = a[e, :masked_m[e], :] @ b[e].T。
    """
```

### `fp8_gemm`

来源：`test_fp8_gemm.py`

```python
def fp8_gemm(
    a: tuple[torch.Tensor, torch.Tensor],   # (a_fp8, scale_a)，a_fp8 [M, K]，scale_a [M]
    b: tuple[torch.Tensor, torch.Tensor],   # (b_fp8, scale_b)，b_fp8 [N, K]，scale_b [N]
    d: torch.Tensor,                        # 输出 [M, N]，bf16 或 fp16
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
) -> None:
    """
    FP8 GEMM，NT 布局。
    语义：d = (a_fp8 * scale_a) @ (b_fp8 * scale_b).T。
    """
```

### `einsum`

来源：`test_einsum.py`

```python
def einsum(
    pattern: str,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BF16 einsum。op_test 覆盖模式：
    - "bmk,bnk->mn"
    - "bhr,hdr->bhd"
    - "bhd,hdr->bhr"
    """
```

### `fp8_einsum`

来源：`test_einsum.py`

```python
def fp8_einsum(
    pattern: str,
    x: tuple[torch.Tensor, torch.Tensor],
    y: tuple[torch.Tensor, torch.Tensor],
    z: torch.Tensor,
    recipe: Optional[tuple[int, int, int]] = None,
) -> torch.Tensor:
    """
    FP8 einsum。op_test 覆盖模式：
    - "bhr,hdr->bhd"
    - "bhd,hdr->bhr"
    - "bhd,bhr->hdr"
    """
```

### `tf32_hc_pernorm_gemm`

来源：`test_tf32_hc_pernorm_gemm.py`

```python
def tf32_hc_pernorm_gemm(
    a: torch.Tensor,                       # bf16 输入 [M, K]
    b: torch.Tensor,                       # float32 权重 [N, K]
    d: torch.Tensor,                       # float32 输出 [M, N] 或 [num_splits, M, N]
    sqr_sum: torch.Tensor,                 # float32 行平方和 [M] 或 [num_splits, M]
    num_splits: Optional[int] = None,
) -> None:
    """
    TF32 hyperconnection pernorm GEMM。
    输出 d 对应 a.float() @ b.T；sqr_sum 对应 a.float().square().sum(-1)。
    num_splits 不为 None 时，测试中使用分片输出后再 sum(0) 得到最终结果。
    """
```

## W4A8 详细用法

### masked GEMM

当前 W4A8 masked GEMM 保留两条 Python 对外接口：

1. `m_grouped_w4a8_gemm_nt_masked`
   - 原有 ASM 路径接口，保持历史行为不变；
   - weight 使用 ASM/marlin2 layout。

2. `m_grouped_w4a8_gemm_nt_masked_hipc`
   - HIPC 路径接口；
   - weight 使用 HIPC 专用 layout，适合做 ASM/HIPC A/B 对比。

两个接口的参数形式一致：

```python
m_grouped_w4a8_gemm_nt_masked_xxx(
    (a_q, a_scale),
    (weight, w_scale),
    out,
    masked_m,
    expected_m,
)
```

其中：

- `masked_m` 是 `[E]` 的 int32 tensor，表示每个 expert 当前有效 token 数；
- `expected_m` 是框架侧传入的预期每 expert token 数；
- HIPC 路径按 `input.size(1)` 计算容量 block，并在 kernel 内通过 `masked_m[expert]` 跳过无效 block 和 tail row；
- ASM 权重 layout 和 HIPC 权重 layout 不相同，二者不能混用。

#### ASM 接口

ASM 接口使用 marlin2 权重 layout：

```python
from deepgemm import m_grouped_w4a8_gemm_nt_masked

m_grouped_w4a8_gemm_nt_masked(
    (a_q, a_scale),
    (weight_marlin2, w_scale),
    out,
    masked_m,
    expected_m,
)
```

性能和正确性测试：

```bash
python3 op_test/test_deepgemm_masked_w4a8.py
```

#### HIPC 接口

框架通常传入普通 K 连续的 int4 打包权重：

```text
packed_weight: int8, shape [E, N, K / 2]
```

每个 int8 byte 存两个 int4 权重，顺序为正常 K 顺序：

```text
byte0 = (w0, w1)
byte1 = (w2, w3)
...
```

调用 HIPC 算子前，使用统一的权重转换接口：

```python
from deepgemm import pack_w4a8_moe_hipc_weight

weight_hipc = pack_w4a8_moe_hipc_weight(packed_weight)
```

该接口会完成：

1. 将普通 K 连续 int4 packed weight 整理为 HIPC blocked layout；
2. 按 HIPC dwordx4 写回路径进行 N 维 even/odd 列重排；
3. 对每个 expert 做 HIPC weight shuffle。

框架侧不需要自行拆分 blocked、列重排或 shuffle 步骤。

调用示例：

```python
from deepgemm import (
    m_grouped_w4a8_gemm_nt_masked_hipc,
    pack_w4a8_moe_hipc_weight,
)

# a_q:        [E, M, K], int8
# a_scale:    [E, M, 1], float32
# weight_raw: [E, N, K / 2], int8, normal K-contiguous int4 packed weight
# w_scale:    [E, N, 1], float32
# out:        [E, M, N], bf16/fp16
# masked_m:   [E], int32
# expected_m: int

weight_hipc = pack_w4a8_moe_hipc_weight(weight_raw)
m_grouped_w4a8_gemm_nt_masked_hipc(
    (a_q, a_scale),
    (weight_hipc, w_scale),
    out,
    masked_m,
    expected_m,
)
```

shape 约束：

1. `packed_weight` 必须是 3D int8 tensor，shape 为 `[E, N, K / 2]`；
2. `N` 需要是 256 的倍数；
3. `K / 2` 需要是 32 的倍数，即 `K` 需要是 64 的倍数；
4. HIPC blocked、even/odd 列重排和 shuffle 由 `pack_w4a8_moe_hipc_weight` 统一处理。

性能和正确性测试：

```bash
python3 op_test/test_deepgemm_masked_w4a8_hipc.py
```

常用单 case 性能测试：

```bash
python3 op_test/test_deepgemm_masked_w4a8_hipc.py \
  --single --e 16 --m 2048 --n 4096 --k 7168 \
  --masked 2048 --expected 2048 --bench --no-verify
```

### contiguous GEMM

W4A8 contiguous GEMM 使用 HIPC 路径接口：

```python
m_grouped_w4a8_gemm_nt_contiguous_hipc(
    (a_q, a_scale),
    (weight_hipc, w_scale),
    out,
    m_indices,
)
```

参数说明：

- `a_q`：`[M, K]` int8；
- `a_scale`：`[M]` 或 `[M, 1]` float32；
- `weight_raw`：`[E, N, K / 2]` int8，普通 K 连续 int4 打包权重；
- `w_scale`：`[E, N]` 或 `[E, N, 1]` float32；
- `out`：`[M, N]` bf16/fp16；
- `m_indices`：`[M]` int32，每一行 token 对应的 expert，padding 行使用 `-1`；
- `config`：可选，常用配置为 `{"MODE": 1002}`。

调用示例：

```python
from deepgemm import (
    m_grouped_w4a8_gemm_nt_contiguous_hipc,
    pack_w4a8_moe_hipc_weight,
)

weight_hipc = pack_w4a8_moe_hipc_weight(weight_raw)
m_grouped_w4a8_gemm_nt_contiguous_hipc(
    (a_q, a_scale),
    (weight_hipc, w_scale),
    out,
    m_indices,
)
```

计算语义：

```text
out[i] = a_q[i] @ weight[m_indices[i]].T
          * a_scale[i] * w_scale[m_indices[i]]
```

`m_indices[i] == -1` 的 padding 行不保证输出有效。

shape 约束：

1. `a_q` 必须是 2D int8 tensor，shape 为 `[M, K]`；
2. `packed_weight` 必须是 3D int8 tensor，shape 为 `[E, N, K / 2]`；
3. `out` 必须是 2D bf16/fp16 tensor，shape 为 `[M, N]`；
4. `m_indices` 必须是 `[M]` int32 tensor，padding 行填 `-1`；
5. `K / 2` 需要是 32 的倍数，即 `K` 需要是 64 的倍数；
6. `N` 需要是 16 的倍数；
7. contiguous 场景建议每个 expert 按 128 行粒度 padding。

当前常用模式：

```text
MODE=1002
BM64/BN128/N_LOOP4
```

性能和正确性测试：

```bash
python3 op_test/test_deepgemm_contiguous_w4a8_hipc.py
```

常用单 case 性能测试：

```bash
python3 op_test/test_deepgemm_contiguous_w4a8_hipc.py \
  --e 32 --m 32768 65536 131072 262144 \
  --shapes 2048x8192 --mode 1002 --no-verify
```

## 兼容说明

当前版本保留部分旧接口名称，用于兼容已有框架侧调用；兼容入口集中在 `legacy_compat.py`，不作为新代码推荐用法。新代码建议使用本文档中的统一 GEMM 接口和 `marlin_*_weight` 权重重排接口。旧接口名称仅作兼容保留，后续版本会逐步移除，新增调用不要继续依赖旧名称。

