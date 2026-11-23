# deepgemm API 接口文档

## Grouped_GEMM算子接口

### 通用接口（gfx936/gfx938/gfx92a）

#### `m_grouped_i8_gemm_nt_contiguous`

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
) :
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[M, K]` int8，`a_scale` 为 `[M]` 或 `[M, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用  `marlin_i8_contiguous_weight`/`pack_i8_contiguous_weight`生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `output`：输出 tensor，shape `[M, N]`，通常为 bf16/fp16。
- `m_indices`：shape `[M]`，dtype int32。`m_indices[i]` 表示第 `i` 行属于哪个 expert，padding 行填 `-1`。
- `shuffle_unique`：权重重排格式选择，默认 `1`。`1` 表示Prefill-Decode统一权重重排格式，`0` 表示Prefill-Decode非统一权重重排格式。

调用顺序：先重排权重，再调用 GEMM。`pack_i8_contiguous_weight` 和 GEMM 的 `shuffle_unique` 必须保持一致。

```python
packed_b = pack_i8_contiguous_weight(
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

#### `marlin_i8_contiguous_weight`

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

#### `m_grouped_i8_gemm_nt_masked`

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
) :
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[E, M_pad, K]` int8，`a_scale` 为 `[E, M_pad]` 或 `[E, M_pad, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用  `marlin_i8_masked_weight`/`pack_i8_masked_weight` 生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `d`：输出 tensor，shape `[E, M_pad, N]`，通常为 bf16/fp16。
- `masked_m`：shape `[E]`，dtype int32，表示每个 expert 的有效行数。
- `expected_m_per_group`：kernel launch 使用的每个 expert token 上限，通常由框架侧按固定 bucket/capacity 传入，可大于实际 `masked_m`；避免在 GEMM 调用热路径里通过 `masked_m.max().item()` 临时计算。
- `shuffle_unique`：权重重排格式选择，默认 `1`。`1` 表示统一权重重排格式，`0` 表示非统一权重重排格式。

调用顺序：先重排权重，再调用 GEMM。`pack_i8_masked_weight` 和 GEMM 的 `shuffle_unique` 必须保持一致。

```python
packed_weight = pack_i8_masked_weight(
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

#### `marlin_i8_masked_weight`

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

#### `m_grouped_w4a8_gemm_nt_contiguous_hipc`

来源：`test_deepgemm_masked_w4a8_hipc.py`

contiguous grouped W8A8/INT8 GEMM 统一入口。

```python
def m_grouped_w4a8_gemm_nt_contiguous_hipc(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
):
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[M, K]` int8，`a_scale` 为 `[M]` 或 `[M, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用 `pack_w4a8_moe_hipc_weight`/`pack_w4a8_contiguous_weight`生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `output`：输出 tensor，shape `[M, N]`，通常为 bf16/fp16。
- `m_indices`：shape `[M]`，dtype int32。`m_indices[i]` 表示第 `i` 行属于哪个 expert，padding 行填 `-1`。

#### `m_grouped_w4a8_gemm_nt_masked_hipc`

来源：`test_deepgemm_masked_w4a8_hipc.py`

masked grouped W4A8 GEMM 入口。

```python
def m_grouped_w4a8_gemm_nt_masked_hipc(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = None,
    *,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
) -> (int, int):
    """
	masked grouped W4A8 GEMM。
    """
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[E, M_pad, K]` int8，`a_scale` 为 `[E, M_pad]` 或 `[E, M_pad, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用 `pack_w4a8_moe_hipc_weight` /`pack_w4a8_masked_weight`生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `d`：输出 tensor，shape `[E, M_pad, N]`，为 bf16/fp16。
- `masked_m`：shape `[E]`，dtype int32，表示每个 expert 的有效行数。
- `expected_m_per_group`：kernel launch 使用的每个 expert token 上限，通常由框架侧按固定 bucket/capacity 传入，可大于实际 `masked_m`；避免在 GEMM 调用热路径里通过 `masked_m.max().item()` 临时计算。
- `enable_overlap`/`signal`：overlap--SBO(Single-Batch Overlap) 特性，详见[overlap特性](#overlap特性)
- 返回值：`(batch_m, threshold)`配合实现overlap特性，不启用overlap特性可不接收。

#### `pack_w4a8_moe_hipc_weight`

来源：`deepgemm/group_gemm_weight_pack.py`

contiguous W4A8 权重重排入口。

```python
def pack_w4a8_moe_hipc_weight(
    weight: torch.Tensor,
) -> torch.Tensor:
```

参数：

- `weight`：原始`K`方向连续 int4 打包权重权重，shape `[E, N, K/2]`，dtype int8。

#### `m_grouped_bf16_gemm_nt_contiguous`

来源：`deepgemm/m_group_gemm_nt_contiguous.py`

```python
def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,                       # bf16 激活 [M, K]
    b: torch.Tensor,                       # bf16 权重 [E, N, K]
    output: torch.Tensor,                  # bf16 输出 [M, N]
    m_indices: torch.Tensor,               # [M] int32，行到 expert 的映射，padding 行为 -1
    config: Optional[Dict[str, Any]] = {"MODE": 1000},
) :
    """
    contiguous grouped BF16 GEMM。
    语义：对 m_indices[i] >= 0 的行，计算 output[i] = a[i] @ b[e].T。
    """
```

#### `m_grouped_bf16_gemm_nt_masked`

来源：`test_deepgemm_masked_w16a16_bf16.py`

```python
def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,                       # bf16 激活 [E, M_pad, K]
    b: torch.Tensor,                       # bf16 权重 [E, N, K]
    d: torch.Tensor,                       # bf16 输出 [E, M_pad, N]
    masked_m: torch.Tensor,                # [E] int32
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = {"MODE": 1000},
) :
    """
    masked grouped BF16 GEMM。
    语义：d[e, :masked_m[e], :] = a[e, :masked_m[e], :] @ b[e].T。
    """
```

### gfx938/gfx92a

#### `m_grouped_fp8_gemm_nt_contiguous`

来源：`test_deepgemm_contiguous_w8a8_fp8.py`

```python
def m_grouped_fp8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],   # (a_fp8, a_scale)，a_fp8 [M, K] fp8，a_scale [M]
    b: tuple[torch.Tensor, torch.Tensor],   # (weight, b_scale)，b_scale [E, N]
    output: torch.Tensor,                   # bf16 输出 [M, N]
    m_indices: torch.Tensor,                # [M] int32，padding 行为 -1
    config: Optional[Dict[str, Any]] = None,
) :
    """
    contiguous grouped FP8 GEMM，per-channel weight scale。
    weight 由 pack_fp8_contiguous_weight(weight_fp8) 生成。
    """
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[M, K]` fp8，`a_scale` 为 `[M]` 或 `[M, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用  `marlin_fp8_contiguous_weight`/`pack_fp8_contiguous_weight`生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `output`：输出 tensor，shape `[M, N]`，通常为 bf16。
- `m_indices`：shape `[M]`，dtype int32。`m_indices[i]` 表示第 `i` 行属于哪个 expert，padding 行填 `-1`。
- `shuffle_unique`：权重重排格式选择，默认 `1`。`1` 表示Prefill-Decode统一权重重排格式，`0` 表示Prefill-Decode非统一权重重排格式。

#### `marlin_fp8_contiguous_weight`

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
- 返回值：重排后的 `weight`，作为 GEMM 的 `b[0]` 传入。

#### `m_grouped_fp8_gemm_nt_masked`

来源：`test_deepgemm_masked_w8a8_fp8.py`

```python
def m_grouped_fp8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],   # (input_fp8, input_scale)，input_fp8 [E, M_pad, K]
    b: tuple[torch.Tensor, torch.Tensor],   # (weight, weight_scale)，weight_scale [E, N, 1]
    d: torch.Tensor,                        # bf16 输出 [E, M_pad, N]
    masked_m: torch.Tensor,                 # [E] int32
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    block_wise: bool = False,
    cu: int = 128,
) -> (int, int):
    """
    masked grouped FP8 GEMM，per-channel weight scale。
    """
```

参数：

- `a`：`(q_a, a_scale)`，`q_a` 为 `[E, M_pad, K]` fp8，`a_scale` 为 `[E, M_pad]` 或 `[E, M_pad, 1]`。
- `b`：`(packed_weight, b_scale)`，`packed_weight` 必须使用  `marlin_fp8_masked_weight`/`pack_fp8_masked_weight`生成，`b_scale` 为 `[E, N]` 或 `[E, N, 1]`。
- `d`：输出 tensor，shape `[E, M_pad, N]`，通常为 bf16。
- `masked_m`：shape `[E]`，dtype int32，表示每个 expert 的有效行数。
- `expected_m_per_group`：kernel launch 使用的每个 expert token 上限，通常由框架侧按固定 bucket/capacity 传入，可大于实际 `masked_m`；避免在 GEMM 调用热路径里通过 `masked_m.max().item()` 临时计算。
- `enable_overlap`/`signal`：overlap--SBO(Single-Batch Overlap) 特性，详见[overlap特性](#overlap特性)
- 返回值：`(batch_m, threshold)`配合实现overlap特性，不启用overlap特性可不接收。

#### `marlin_fp8_masked_weight`

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
- 返回值：重排后的 `weight`，作为 GEMM 的 `b[0]` 传入。

### overlap特性

overlap特性是在做grouped_gemm时，向外通知已完成并写回显存的数据，供下游使用的特性。

#### 接口说明

以`m_grouped_fp8_gemm_nt_masked`为例

```python
def m_grouped_fp8_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],   
    b: tuple[torch.Tensor, torch.Tensor],   
    d: torch.Tensor,                        
    masked_m: torch.Tensor,                 
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None, 
    config: Optional[Dict[str, Any]] = None,
    *,
) -> (int, int):
    """
    masked grouped FP8 GEMM，per-channel weight scale。
    """
```

当`overlap`值为`True`时启用overlap特性，已写回的结果通过`signal`传递。
- signal 需要传入全 0 数组，shape = `size_e × ceil_div(size_m, 64)`
- `m_grouped_fp8_gemm_nt_masked`返回值为`(batch_m, threshold)`, `batch_m`等于`64`或`128`
- 当grouped_gemm完成第`e`个专家的第`m`个`batch_m`计算写回时，`signal`的第`e*ceil_div(size_m/batch_m)+m`的值会等于`threshold`，表示数据已准备完成。
- 已支持overlap特性算子：`m_grouped_fp8_gemm_nt_masked`，`m_grouped_w4a8_gemm_nt_masked_hipc`

## 其余算子

#### `fp8_gemm`

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

#### `einsum`

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

#### `fp8_einsum`

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

#### `tf32_hc_prenorm_gemm`

来源：`test_tf32_hc_prenorm_gemm.py`

```python
def tf32_hc_prenorm_gemm(
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

## 兼容说明

当前版本保留部分旧接口名称，用于兼容已有框架侧调用；兼容入口集中在 `legacy_compat.py`，不作为新代码推荐用法。新代码建议使用本文档中的统一 GEMM 接口和 `marlin_*_weight` 权重重排接口。旧接口名称仅作兼容保留，后续版本会逐步移除，新增调用不要继续依赖旧名称。

### 分页 Attention 接口（gfx936/gfx938）

#### `paged_mqa_logits`

来源：`op_test/test_paged_mqa_logits.py`

分页 MQA logits，支持 dense decode 和 INT8 prefix prefill。计算各 head 的 QK 点积，经 ReLU 后按 weights 加权求和；不包含 softmax 或 value 聚合。

```python
def paged_mqa_logits(
    q: torch.Tensor,                   # [B, R, H, 128]
    fused_kv_cache: torch.Tensor,      # [P, S, 1, 128] 或量化布局 [P, S, 1, 132]
    weights: torch.Tensor,             # [B*R, H] float32
    context_lens: torch.Tensor,        # [B] int32
    block_table: torch.Tensor,         # [B, T] int32
    max_model_len: int,
    *,
    out: Optional[torch.Tensor] = None,
    clean_logits: bool = True,
    is_prefill: bool = False,
) -> torch.Tensor:
```

参数：

- `B` 为 batch，`R` 为 NextN，`H` 为 head 数；`P` 为物理页数，`S` 为每页 token 数（`BLOCK_KV`），`T` 为每个请求的页表长度。
- `fused_kv_cache`：BF16/FP16 直接存储 KV；FP8 用 uint8 存储 E4M3FN bytes，INT8 用 int8 存储。
  量化 cache 每页先存 `S*128` 个 KV bytes，再存 `S` 个 FP32 scale，而非逐 token 交错存储。
- `context_lens` 包含本次 `R` 个 decode token。须满足 `R <= context_lens[b] <= min(max_model_len, T*S)`。
- `block_table` 将逻辑页映射到物理页，所用页号须位于 `[0, P)`。
- `max_model_len` 为正 int32 输出列数；不要求为未使用的输出 tail 分配 KV 页。
- `out` 可选，须为连续 FP32 `[B*R, max_model_len]`，且不与输入共享 storage。
- `clean_logits=True` 将 causal mask 和 tail 填为 `-inf`；`False` 时非有效区不保证清理。

Decode 支持组合（`is_prefill=False`，`D=128`，`B>0`，`B*R<=65535`）：

| Q / KV dtype | R | H | S |
| --- | --- | --- | --- |
| E4M3FN / packed E4M3FN、BF16 / packed E4M3FN | 1/2 | 8/16/32/64 | 4/8/16/32/64 |
| BF16 / BF16、FP16 / FP16、INT8 / packed INT8 | 1–5 | 8/16/32/64 | 4/8/16/32/64 |
| E4M3FN / packed E4M3FN | 1/2/4 | 32/64 | 1/64 |

输入须在同一 GPU 且不参与 autograd；Q、weights、context_lens 连续，cache 页内和页表行内连续，
`cache.stride(0)`（按元素计）须为 4 的倍数。E4M3FN Q 与 packed FP8 cache 的存储地址须 4-byte 对齐。
`S=1` 或 FP8 `R=4` 时，cache 和页表须完全连续，且 `P*S*132<=INT_MAX`。
返回 FP32 `[B*R, max_model_len]`，未传 `out` 时输出可能带行间 padding。

计算语义（`k[b,t]` 为经页表寻址并按实际存储 scale 反量化后的 KV）：

```text
logits[b*R+r, t] = sum_h(weights[b*R+r, h] * max(dot(q[b,r,h], k[b,t]), 0))
有效位置：0 <= t < context_lens[b] - R + r + 1
其余位置：-inf（clean_logits=True）
```

调用示例：

```python
logits = deepgemm.paged_mqa_logits(
    q, fused_kv_cache, weights, context_lens, block_table,
    max_model_len=111000, clean_logits=True,
)
```

Prefix prefill（`is_prefill=True`，gfx936）：

- Q 为 INT8 `[tokens, 1, H, 128]`，KV 为 packed INT8 `[P, S, 1, 132]`；`H=8/16/32/64`，`S=4/8/16/32/64`。
- `weights` 为 FP32 `[tokens, H]`，须已乘入 Q 的量化 scale。KV scale 使用上述每页尾部 FP32 布局。
- `context_lens` 为 INT32 `[tokens]`，表示各 query 可见的 KV 终点，满足 `0 <= context_lens[i] <= max_model_len`，可为零或非单调。
- `block_table` 为 INT32 `[tokens, T]`，所有行必须对应同一 KV 序列且页表内容相同；`T*S >= max_model_len`。
- Q/cache/weights/context_lens 连续，页表行内连续；Q/cache/weights 的地址须 16-byte 对齐。
- 返回 FP32 `[tokens, max_model_len]`，每行 `[0, context_lens[i])` 为有效 logits，其余为 `-inf`。
  该模式始终写入完整输出，不接受 `out`；允许 `tokens=0` 或 `max_model_len=0`。

```python
logits = deepgemm.paged_mqa_logits(
    q, fused_kv_cache, weights, context_lens, block_table,
    max_model_len=32768, clean_logits=True, is_prefill=True,
)
```

单测：`python3 op_test/test_paged_mqa_logits.py`。

#### `page_mqa_logits_sparse_mask`

来源：`op_test/test_paged_mqa_logits.py`

独立 query 行的分页 MQA，返回稀疏 logits 及每 token 一位的 mask。

```python
def page_mqa_logits_sparse_mask(
    q: torch.Tensor,                   # [B, 1, H, 128]
    fused_kv_cache: torch.Tensor,      # BF16 [P, 64, 1, 128] 或 uint8 [P, 64, 1, 132]
    weights: torch.Tensor,             # [B, H] float32
    context_lens: torch.Tensor,        # [B] int32
    block_table: torch.Tensor,         # [B, T] int32
    max_model_len: int,
    *,
    clean_logits: bool = True,
    num_warps: int = 4,
    persistent_ctas: int = 0,
) -> list[torch.Tensor]:
```

- 支持 gfx936/80CU、gfx938/64CU/72CU，`B>0`，`H=8/16/32/64`，page size 固定为 64。
- Q/KV 支持 BF16/BF16，或 BF16、E4M3FN、E4M3FNUZ Q 配合 packed E4M3FN KV。
  KV 的 uint8 cache 每页先存 64×128 个 E4M3FN bytes，再存 64 个 FP32 scale；Q scale 须提前乘入 weights。
- 所有 tensor 位于同一 GPU。Q/cache/weights/context_lens 连续，页表行内连续。
- `max_model_len` 必须等于 `T*64`，`clean_logits=True`、`num_warps=4`。
  `persistent_ctas=0` 自动选择调度规模；显式值支持 1–4096。
- 返回 `[values, mask]`：values 为 FP32 `[B, max_model_len]`，mask 为 INT16 `[B, ceil(max_model_len/16)]`。
  每个 mask 元素的低至高位对应连续 16 个 token；置位表示该位置存有结果，包括非有限值和负零。
- 消费者必须先用 context 限定有效位置，再读取置位的 values；有效范围内未置位表示 `+0`。
  未置位 values 及 context 外的 mask 不保证初始化，不能作为 dense logits 直接消费。

```python
values, mask = deepgemm.page_mqa_logits_sparse_mask(
    q, fused_kv_cache, weights, context_lens, block_table,
    max_model_len=block_table.size(1) * 64,
)
```

#### `page_mqa_logits_sparse_mask_grouped`

来源：`op_test/test_paged_mqa_logits.py`

在 sparse MQA 中复用同一请求的 KV，支持独立行或连续多 query 行分组。

```python
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
) -> list[torch.Tensor]:
```

shape、dtype、架构和输出契约与 `page_mqa_logits_sparse_mask` 相同，区别如下：

- `group_size=1` 为独立行；3/4/5/6 对应每请求连续的 3/4/5/6 行，`B` 须被 group_size 整除。
- 组内所有行的 block table 内容必须相同，Q、weights 和 context_lens 则各行独立；不同组可使用不同页表。
- cache 允许页间 padding，但页内须连续；页表允许行间 padding。

```python
values, mask = deepgemm.page_mqa_logits_sparse_mask_grouped(
    q, fused_kv_cache, weights, context_lens, block_table,
    max_model_len=block_table.size(1) * 64, group_size=6,
)
```

#### `page_mqa_logits_sparse_mask_grouped_int8`

来源：`op_test/test_paged_mqa_logits.py`

支持 packed INT8 KV 的 grouped sparse MQA，KV 在 kernel 内反量化为 BF16 后计算。

```python
def page_mqa_logits_sparse_mask_grouped_int8(
    q: torch.Tensor,                   # BF16/E4M3FN/E4M3FNUZ [B, 1, H, 128]
    packed_kv_cache: torch.Tensor,     # uint8 [P, 64, 1, 132]，包含 INT8 bytes 和 FP32 scale
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    *,
    clean_logits: bool = True,
    num_warps: int = 4,
    persistent_ctas: int = 0,
    group_size: int = 5,
) -> list[torch.Tensor]:
```

分组、shape、架构和 values/mask 契约与前述 grouped 接口一致。与 dense INT8 模式不同，
此接口的 KV 容器 dtype 为 uint8，其中前 64×128 bytes 按有符号 INT8 解释；Q 不接受 INT8。

单测：`python3 op_test/test_paged_mqa_logits.py`，直接输出正确性、Graph 和计时结果。

### 非分页 Attention 接口（gfx936/gfx938）

#### `mqa_logits`

来源：`op_test/test_mqa_logits.py`

实现迁自 LightOp 的非分页 `mqa_logits` beta 版本。Q/K 使用连续存储，支持完整输出和按窗口压缩输出。计算各 head 的 QK 点积，经 ReLU 后按 Weights 加权求和；不包含 softmax 或 value 聚合。

```python
def mqa_logits(
    Q: torch.Tensor,                   # [M, H, 128]
    K: torch.Tensor,                   # [N, 128]
    Weights: torch.Tensor,             # [M, H] float32
    ks: torch.Tensor,                  # [M] int32，每行窗口起点
    ke: torch.Tensor,                  # [M] int32，每行窗口终点，不含该位置
    kv_scale: Optional[torch.Tensor] = None,
    clean_logit: bool = True,
    D_out: Optional[torch.Tensor] = None,
    *,
    max_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
```

参数：

- `M` 为 query 数，`N` 为 KV token 数，均可为零；`H` 为 head 数，支持 `1/2/4/8/16/32/64/128`，head 维度固定为 `128`。
- `Q`、`K` 的 dtype 必须相同，且按上述形状连续存储。`Weights` 为 FP32 `[M, H]`，列连续，允许行间 padding。
- `ks`、`ke` 为连续 INT32 向量，各至少包含 `M` 个元素，表示每行的半开窗口 `[ks[m], ke[m])`。
  起点和终点分别在设备端裁剪到 `[0, N]`；裁剪后终点不大于起点时，该行窗口为空。
- `kv_scale` 为按 KV token 排列的 FP32 scale，传入时须连续且至少包含 `N` 个元素。各 dtype 的要求见下表。
  量化 Q 的 scale 须由调用方提前乘入 `Weights`，接口不单独接收 Q scale。
- `clean_logit=True` 时，完整输出中窗口外的位置填为 `-inf`；`False` 时只保证窗口内结果有效，窗口外不保证初始化或保留原值。
- `D_out` 为可选的 FP32 输出缓冲区，仅用于完整输出。须为二维，大小至少覆盖 `[M, N]`，列步长为 `1`、行步长不小于 `N`，输出使用的区域不能与任何输入重叠。
  传入更大的缓冲区时返回其前 `[M, N]` 区域的视图；未传入时由算子分配输出。
- `max_seqlen_k=None` 使用完整输出；传入正整数 `W` 时使用压缩输出，范围为 `[1, 2**31-1]`，不接受布尔值。压缩模式的约束见下文。

支持组合：

| Q / K dtype | GPU 架构 | kv_scale |
| --- | --- | --- |
| BF16 / BF16 | gfx936、gfx938 | 必须为 `None` |
| INT8 / INT8 | gfx936、gfx938 | 必须传入每个 KV token 的 FP32 scale |
| FP8 E4M3FN / FP8 E4M3FN | gfx938 | 可选；未传入时按 `1` 处理 |

所有输入及 `D_out` 须位于同一 GPU。BF16 的 Q/K 地址须 16 字节对齐，INT8/FP8 的 Q/K 地址须 4 字节对齐；其余张量地址须 4 字节对齐。
接口使用当前 PyTorch stream 异步执行，支持 Graph 捕获和回放；回放时读取设备上当前的 `ks`、`ke`。

返回 FP32 `[M, N]`，每列对应原始 KV 序列中的同一位置。

计算语义（Q/K 使用实际存储值；`scale[n]` 为 `kv_scale[n]`，未传入时为 `1`）：

```text
s[m] = min(max(ks[m], 0), N)
e[m] = min(max(ke[m], 0), N)
logits[m, n] = sum_h(Weights[m, h] * max(dot(Q[m, h, :], K[n, :]), 0)) * scale[n]
有效位置：s[m] <= n < e[m]
其余位置：-inf（clean_logit=True）
```

调用示例：

```python
# BF16 时 kv_scale=None；INT8/FP8 按上表传入 scale。
logits = deepgemm.mqa_logits(
    Q, K, Weights, ks, ke,
    kv_scale=kv_scale, clean_logit=True,
)

# 复用调用方分配的完整输出缓冲区。
logits = deepgemm.mqa_logits(
    Q, K, Weights, ks, ke,
    kv_scale=kv_scale, clean_logit=True, D_out=out,
)
```

压缩输出（`max_seqlen_k=W`）：

- 返回 FP32 `[M, W]`，行步长为 `ceil(W/256)*256` 个元素，因此返回值可能不连续。
- 令 `L[m]=max(e[m]-s[m], 0)`，第 `m` 行的第 `j` 列对应原始 KV 位置 `s[m]+j`；仅 `0 <= j < L[m]` 的前缀有效。
- 调用方必须保证 `W >= L[m]` 对所有行成立。接口不会在主机端读取窗口来检查容量，Graph 回放时也须满足该约束。
- 输出由算子分配，不接受 `D_out`；`clean_logit` 在此模式下不起作用。有效前缀后的区域不保证初始化，不能作为有效 logits 使用。
- 完整输出和压缩输出使用各自的内核选择器。FP8 的 grouped M16 仅用于完整输出，压缩输出使用普通 M16、M64 或 QH 内核。

```python
# W 由调用方提供，须覆盖所有行裁剪后的窗口长度。
logits_compact = deepgemm.mqa_logits(
    Q, K, Weights, ks, ke,
    kv_scale=kv_scale, max_seqlen_k=W,
)
```

单测：`python3 -m pytest -q op_test/test_mqa_logits.py`。
正确性与测速：`python3 op_test/test_mqa_logits.py`，直接输出逐用例的正确性及计时结果；更多选项见 `--help`。
