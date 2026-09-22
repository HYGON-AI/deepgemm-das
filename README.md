# deepgemm
## 简介
DeepGEMM是一个专为简洁高效的通用矩阵相乘（GEMM）而设计的库。它支持普通和混合专家（MoE）分组场景的FP8/INT8/BF16.

## 上游归属

本仓库基于 DeepGEMM 进行适配和扩展。

- 上游仓库：https://github.com/deepseek-ai/DeepGEMM.git
- 上游分支：main
- 上游 Commit：559d79fb6994a58b8a15b4b93bf13ccc16edf247
- 上游许可证：MIT
- Modified by Hygon Information Technology Co., Ltd.

## 安装
环境依赖：
    pytorch2.0以上
    dtk24.04\dtk25.04\dtk26.04 


安装脚本：
    python3 setup.py bdist_wheel

## 算子介绍
### 核心算子

说明：下表列出推荐使用的 Python 对外接口。`deepgemm.op.*` 中保留了部分底层接口，主要用于兼容、ASM/HIP 路径对比和内部调试；框架侧集成建议优先调用下表接口。

INT8/FP8 group GEMM 的权重布局按算子路径区分：masked marlin 路径使用 marlin/marlin2 layout；contiguous 和 low-latency 路径推荐使用 W6 packed layout。

<table>
    <th>算子接口</th><th>算子介绍</th><th>架构路径</th>
    <tr>
        <td>m_grouped_w4a8_gemm_nt_masked</td>
        <td>input 为 int8，weight 为 int4，输出 bf16，masked group GEMM，weight 使用 marlin/marlin2 layout</td>
        <td>gfx936 / gfx938</td>
    </tr>
    <tr>
        <td>m_grouped_w4a8_gemm_nt_masked_hipc</td>
        <td>input 为 int8，weight 为 int4，输出 bf16，masked group GEMM，weight 需要先使用 pack_w4a8_moe_hipc_weight 转成 HIPC 专用 layout</td>
        <td>gfx936 / gfx938 HIPC 路径</td>
    </tr>
    <tr>
        <td>m_grouped_w8a8_gemm_nt_masked</td>
        <td>input 为 int8，weight 为 int8，输出 bf16，masked group GEMM，weight 使用 marlin layout</td>
        <td>gfx938 marlin 路径</td>
    </tr>
    <tr>
        <td>m_grouped_w8a8_gemm_nt_masked_gfx936</td>
        <td>input 为 int8，weight 为 int8，输出 bf16，masked group GEMM，weight 使用 W6 packed layout，由 pack_w8a8_gfx936_weight_to_w6 生成</td>
        <td>gfx936 HIPC 路径</td>
    </tr>
    <tr>
        <td>m_grouped_bf16_gemm_nt_masked</td>
        <td>input 为 bf16，weight 为 bf16，输出 bf16，masked group GEMM，不涉及量化权重重排</td>
        <td>gfx936 / gfx938</td>
    </tr>
    <tr>
        <td>m_grouped_fp8_gemm_nt_masked_ll</td>
        <td>input 为 fp8，weight 为 fp8，输出 bf16，low-latency masked group GEMM，weight 使用 W6 packed layout</td>
        <td>gfx938 low-latency 路径</td>
    </tr>
    <tr>
        <td>m_grouped_fp8_gemm_nt_masked_ll_tensor_wise</td>
        <td>FP8 low-latency masked group GEMM，tensor-wise scale 版本，weight 使用 W6 packed layout</td>
        <td>gfx938 low-latency 路径</td>
    </tr>
    <tr>
        <td>m_grouped_i8_gemm_nt_contiguous</td>
        <td>input 为 int8，weight 为 int8，输出 bf16，contiguous group GEMM；weight 使用 pack_i8_contiguous_weight 预处理，shuffle_unique=1 默认走 shuffle-unique/HIPC 路径，shuffle_unique=0 在 gfx936/gfx92a 上走 ASM 路径；gfx938 不区分 0/1</td>
        <td>gfx936 / gfx92a / gfx938 contiguous 路径</td>
    </tr>
    <tr>
        <td>m_grouped_w4a8_gemm_nt_contiguous_hipc</td>
        <td>input 为 int8，weight 为 int4，输出 bf16/fp16，contiguous group GEMM；weight 需要先使用 pack_w4a8_moe_hipc_weight 转成 HIPC 专用 layout，m_indices 使用 -1 标记 padding 行</td>
        <td>gfx936 / gfx938 HIPC contiguous 路径</td>
    </tr>
    <tr>
        <td>m_grouped_fp8_gemm_nt_contiguous</td>
        <td>input 为 fp8，weight 为 fp8，输出 bf16，contiguous group GEMM，weight 使用 W6 packed layout</td>
        <td>gfx938 contiguous 路径</td>
    </tr>
    <tr>
        <td>m_grouped_fp8_gemm_nt_contiguous_tensor_wise</td>
        <td>FP8 contiguous group GEMM，tensor-wise scale 版本，weight 使用 W6 packed layout</td>
        <td>gfx938 contiguous 路径</td>
    </tr>
    <tr>
        <td>fp8_gemm</td>
        <td>普通 FP8 GEMM，NT layout，输出 bf16</td>
        <td>gfx938 hipBLASLt 路径</td>
    </tr>
    <tr>
        <td>einsum / fp8_einsum</td>
        <td>常用 einsum 表达式封装；BF16 走 einsum/bmm 路径，FP8 走 fp8_gemm 路径</td>
        <td>gfx938 路径</td>
    </tr>
</table>

历史底层接口说明：`deepgemm.op.*` 下部分旧接口仍保留，例如旧的 contiguous ASM 对比路径。这些接口可能使用不同的权重 layout，主要服务于历史兼容和测试验证；不建议框架侧直接依赖。

详细的算子参数、权重 layout、shape 约束和调用示例见：

[deepgemm API 接口文档](<doc/deepgemm API接口文档.md>)

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

本项目名称：deepgemm-das

本仓库基于上游 DeepGEMM 进行 HCU 平台适配和扩展，上游项目采用 MIT License 开源。

Copyright (c) 2025 DeepSeek

HCU adaptations, modifications, and original contributions by Hygon Information Technology Co., Ltd.

- 许可证全文：[LICENSE](LICENSE)
- 上游仓库：https://github.com/deepseek-ai/DeepGEMM.git
- 上游分支：main
- 上游 Commit：559d79fb6994a58b8a15b4b93bf13ccc16edf247

## Third-Party

第三方依赖、来源和许可证信息以后续合规确认清单为准。
