# deepgemm
## 简介
DeepGEMM是一个专为简洁高效的通用矩阵相乘（GEMM）而设计的库。它支持普通和混合专家（MoE）分组场景的FP8/INT8/BF16.

## 上游归属

本仓库基于 DeepGEMM 进行 HCU 平台适配和扩展。

- 上游仓库：https://github.com/deepseek-ai/DeepGEMM.git
- 上游分支：main
- 上游 Commit：559d79fb6994a58b8a15b4b93bf13ccc16edf247
- 上游许可证：MIT
- Modified by Hygon Information Technology Co., Ltd.

上述上游基线继承自现有公开版本；本次同步基于内部已授权发布的固定源码版本，
保留已有第三方及文件级许可证声明。
本版本发布依赖固定为 `torch==2.13.0`，需使用匹配的 HIP/DTK 构建环境。

## 安装
环境依赖：
    pytorch2.0以上
    dtk24.04\dtk25.04\dtk26.04 \dtk26.10

安装脚本：

```python
python3 setup.py bdist_wheel
```

## 算子介绍
### 核心算子

说明：下表列出推荐使用的 Python 对外接口。`deepgemm.op.*` 中保留了部分底层接口，主要用于兼容、ASM/HIP 路径对比和内部调试；框架侧集成建议优先调用下表接口。

#### Groupd GEMM算子

| 算子接口                                 | 权重重排函数接口               | 算子介绍                                                     | 架构支持             |
| ---------------------------------------- | ------------------------------ | ------------------------------------------------------------ | -------------------- |
| `m_grouped_w4a8_gemm_nt_contiguous_hipc` | `pack_w4a8_moe_hipc_weight`    | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_w4a8_gemm_nt_contiguous_hipc) | gfx936/gfx938/gfx92a |
| `m_grouped_w4a8_gemm_nt_masked_hipc`     | `pack_w4a8_moe_hipc_weight`    | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_w4a8_gemm_nt_masked_hipc) | gfx936/gfx938/gfx92a |
| `m_grouped_i8_gemm_nt_contiguous`        | `marlin_i8_contiguous_weight`  | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_i8_gemm_nt_contiguous) | gfx936/gfx938/gfx92a |
| `m_grouped_i8_gemm_nt_masked`            | `marlin_i8_masked_weight`      | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_i8_gemm_nt_masked) | gfx936/gfx938/gfx92a |
| `m_grouped_fp8_gemm_nt_contiguous`       | `marlin_fp8_contiguous_weight` | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_fp8_gemm_nt_contiguous) | gfx938/gfx92a        |
| `m_grouped_fp8_gemm_nt_masked`           | `marlin_fp8_masked_weight`     | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_fp8_gemm_nt_masked) | gfx938/gfx92a        |
| `m_grouped_bf16_gemm_nt_contiguous`      | `pack_bf16_contiguous_weight`  | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_bf16_gemm_nt_contiguous) | gfx936/gfx938/gfx92a |
| `m_grouped_bf16_gemm_nt_masked`          | `pack_bf16_masked_weight`      | 详见[API文档](./doc/deepgemm API 接口文档.md#m_grouped_bf16_gemm_nt_masked) | gfx936/gfx938/gfx92a |

#### 非Grouped GEMM算子

| 算子接口            | 算子描述                            | 架构支持 |
| ------------------- | ----------------------------------- | -------- |
| fp8_gemm            | 普通 FP8 GEMM，NT layout，输出 bf16 | gfx938   |
| einsum / fp8_einsum | 常用 einsum 表达式封装              | gfx938   |



历史底层接口说明：`deepgemm.op.*` 下部分旧接口仍保留，例如旧的 contiguous ASM 对比路径。这些接口可能使用不同的权重 layout，主要服务于历史兼容和测试验证；不建议框架侧直接依赖。

### W4A8 masked GEMM 状态说明

当前 DeepGEMM 中 W4A8 masked GEMM 同时保留两条 Python 对外接口：

1. `m_grouped_w4a8_gemm_nt_masked`
   - 原有 ASM 路径接口，保持历史行为不变；
   - weight 使用 ASM/marlin2 layout；
   - 适合继续兼容已经接入 ASM 路径的框架代码。

2. `m_grouped_w4a8_gemm_nt_masked_hipc`
   - 新增 HIPC 路径接口；
   - weight 必须使用 HIPC 专用 layout；
   - 适合框架侧按需增量接入和做 ASM/HIPC A/B 对比。

两个接口的算子参数形式一致：

    m_grouped_w4a8_gemm_nt_masked_xxx(
        (a_q, a_scale),
        (weight, w_scale),
        out,
        masked_m,
        expected_m,
    )

其中 `masked_m` 是 `[E]` 的 int32 tensor，表示每个 expert 当前有效 token 数；`expected_m` 是框架侧传入的预期每 expert token 数。HIPC 当前仍按 `input.size(1)` 计算容量 block，并在 kernel 内通过 `masked_m[expert]` 跳过无效 block / tail row。

#### ASM 接口调用方式

ASM 接口使用原有 marlin2 权重 layout。框架侧如果已经在使用 `m_grouped_w4a8_gemm_nt_masked`，调用方式无需改变：

    from deepgemm import m_grouped_w4a8_gemm_nt_masked
    
    m_grouped_w4a8_gemm_nt_masked(
        (a_q, a_scale),
        (weight_marlin2, w_scale),
        out,
        masked_m,
        expected_m,
    )

注意：ASM 权重 layout 和 HIPC 权重 layout 不相同，二者不能混用。

ASM 性能/正确性测试脚本：

    python3 op_test/test_deepgemm_masked_w4a8.py

#### HIPC 接口调用方式

HIPC 接口要求 weight 使用 HIPC 专用 layout。框架通常传入的是普通 K 连续 int4 打包权重：

    packed_weight: int8, shape [E, N, K / 2]

其中每个 int8 byte 存两个 int4 权重，顺序为正常 K 顺序：

    byte0 = (w0, w1), byte1 = (w2, w3), ...

调用 HIPC 算子前，需要先调用：

    from deepgemm import pack_w4a8_moe_hipc_weight
    
    hipc_weight = pack_w4a8_moe_hipc_weight(packed_weight)

`pack_w4a8_moe_hipc_weight` 内部会完成完整的 HIPC weight layout 转换：

1. 将普通 K 连续 int4 packed weight 重新整理为 HIPC blocked layout；
2. 按 HIPC dwordx4 写回路径要求做 N 维 even/odd 列重排；
3. 对每个 expert 做 HIPC weight shuffle，得到 kernel 需要的最终物理 layout。

框架侧只需要调用 `pack_w4a8_moe_hipc_weight` 这一个 API，不需要也不应该自己拆分 blocked、列重排或 shuffle 步骤。

算子调用示例：

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



shape 约束：

1. `packed_weight` 必须是 3D int8 tensor，shape 为 `[E, N, K / 2]`；
2. `N` 需要是 256 的倍数；
3. `K / 2` 需要是 32 的倍数，即 `K` 需要是 64 的倍数；
4. HIPC blocked、even/odd 列重排和 shuffle 由 `pack_w4a8_moe_hipc_weight` 统一处理，用户不需要选择或感知内部 layout。

HIPC 性能/正确性测试脚本：

    python3 op_test/test_deepgemm_masked_w4a8_hipc.py 

常用单 case 性能测试命令：

    python3 op_test/test_deepgemm_masked_w4a8_hipc.py --single --e 16 --m 2048 --n 4096 --k 7168 --masked 2048 --expected 2048 --bench --no-verify

### W4A8 contiguous GEMM 状态说明

当前 DeepGEMM 中 W4A8 contiguous GEMM 提供一条推荐 Python 对外接口：

1. `m_grouped_w4a8_gemm_nt_contiguous_hipc`
   - HIPC contiguous 路径接口；
   - input 使用 int8 activation，weight 使用 int4 packed weight，输出 bf16/fp16；
   - weight 必须使用 HIPC 专用 layout；
   - 适合框架侧 contiguous MoE prefill 场景接入。

算子参数形式：

    m_grouped_w4a8_gemm_nt_contiguous_hipc(
        (a_q, a_scale),
        (weight_hipc, w_scale),
        out,
        m_indices,
    )

其中 `m_indices` 是 `[M]` 的 int32 tensor，表示每一行 token 属于哪个 expert；padding 行使用 `-1` 标记。contiguous HIPC 路径要求框架侧将每个 expert 的 token 按连续区间排布，并建议按 128 行粒度 padding。`config` 是可选参数，通常无需传入；内部调试时可以显式传入例如 `{"MODE": 1002}`。

#### HIPC contiguous 接口调用方式

HIPC contiguous 接口要求 weight 使用 HIPC 专用 layout。框架通常传入的是普通 K 连续 int4 打包权重：

    packed_weight: int8, shape [E, N, K / 2]

其中每个 int8 byte 存两个 int4 权重，顺序为正常 K 顺序：

    byte0 = (w0, w1), byte1 = (w2, w3), ...

调用 contiguous HIPC 算子前，需要先调用：

    from deepgemm import pack_w4a8_moe_hipc_weight
    
    weight_hipc = pack_w4a8_moe_hipc_weight(packed_weight)

`pack_w4a8_moe_hipc_weight` 内部会完成完整的 HIPC weight layout 转换：

1. 将普通 K 连续 int4 packed weight 重新整理为 HIPC blocked layout；
2. 按 HIPC dwordx4 写回路径要求做 N 维 even/odd 列重排；
3. 对每个 expert 做 HIPC weight shuffle，得到 kernel 需要的最终物理 layout。

框架侧只需要调用 `pack_w4a8_moe_hipc_weight` 这一个 API，不需要也不应该自己拆分 blocked、列重排或 shuffle 步骤。

算子调用示例：

    from deepgemm import (
        m_grouped_w4a8_gemm_nt_contiguous_hipc,
        pack_w4a8_moe_hipc_weight,
    )
    
    # a_q:        [M, K], int8
    # a_scale:    [M] or [M, 1], float32
    # weight_raw: [E, N, K / 2], int8, normal K-contiguous int4 packed weight
    # w_scale:    [E, N] or [E, N, 1], float32
    # out:        [M, N], bf16/fp16
    # m_indices:  [M], int32, padding row uses -1
    # config:     optional, e.g. {"MODE": 1002}
    
    weight_hipc = pack_w4a8_moe_hipc_weight(weight_raw)
    m_grouped_w4a8_gemm_nt_contiguous_hipc(
        (a_q, a_scale),
        (weight_hipc, w_scale),
        out,
        m_indices,
    )

计算语义：

    out[i] = a_q[i] @ weight[m_indices[i]].T * a_scale[i] * w_scale[m_indices[i]]

`m_indices[i] == -1` 的 padding 行不保证输出有效。

shape 约束：

1. `a_q` 必须是 2D int8 tensor，shape 为 `[M, K]`；
2. `packed_weight` 必须是 3D int8 tensor，shape 为 `[E, N, K / 2]`；
3. `out` 必须是 2D bf16/fp16 tensor，shape 为 `[M, N]`；
4. `m_indices` 必须是 `[M]` int32 tensor，padding 行填 `-1`；
5. `K / 2` 需要是 32 的倍数，即 `K` 需要是 64 的倍数；
6. `N` 需要是 16 的倍数；
7. contiguous 场景建议每个 expert 按 128 行粒度 padding。

常用 mode：

1. `MODE=1002`
   - BM64/BN128/N_LOOP4 路径；
   - 当前分支的 W4A8 contiguous HIPC 默认路径。

HIPC contiguous 性能/正确性测试脚本：

    python3 op_test/test_deepgemm_contiguous_w4a8_hipc.py

常用单 case 性能测试命令：

    python3 op_test/test_deepgemm_contiguous_w4a8_hipc.py --e 32 --m 32768 65536 131072 262144 --shapes 2048x8192 --mode 1002 --no-verify

## 差异
与官方DeepGemm仓库中的相比：

1.增加int8类型支持

2.input与weight均使用perChannel量化

3.INT8/FP8的group类gemm权重layout按路径区分：masked历史路径使用marlin/marlin2 layout，contiguous和low-latency路径推荐使用W6 packed layout

4.contiguous接口input按照256的倍数对齐

5.k_group_contiguous kernel暂不支持

## 调试

### 算子size打印

调试算子调用链路时，可以通过环境变量开启size打印。该功能默认关闭，关闭时不影响正常性能路径。

开启方式：
    DEEPGEMM_PRINT_OP_SIZE=1 python3 your_script.py

例如：
    DEEPGEMM_PRINT_OP_SIZE=1 python3 op_test/test_deepgemm_contiguous_w8a8_fp8.py

可选过滤指定算子：
    DEEPGEMM_PRINT_OP_SIZE=1 DEEPGEMM_PRINT_OP_SIZE_FILTER=w8a8 python3 your_script.py

masked场景如需打印masked_m的sum/min/max：
    DEEPGEMM_PRINT_OP_SIZE=1 DEEPGEMM_PRINT_OP_SIZE_STATS=1 python3 your_script.py

注意：DEEPGEMM_PRINT_OP_SIZE_STATS=1会触发GPU同步，只建议调试时使用，性能测试时请关闭。


## License

本项目名称：deepgemm-das。

Copyright (c) 2025 DeepSeek

HCU adaptations, modifications, and original contributions by Hygon Information Technology Co., Ltd.

上游 MIT 许可证全文见 [LICENSE](LICENSE)。文件自身带有 Apache-2.0 声明时，
该文件适用其原有声明，许可证全文见 [LICENSES/Apache-2.0](LICENSES/Apache-2.0)。
第三方文件的版权、许可证和来源声明应完整保留；根 MIT 许可证不覆盖第三方的独立声明。
HYGON 仓库级贡献归属见 [COPYRIGHT.md](COPYRIGHT.md)。

## Third-Party

组件、依赖及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
