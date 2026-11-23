## 说明

该文件夹用于生成 DeepGEMM GROUPED GEEM 算子在不同架构，不同量化方式，不同输入大小下的最佳底层内核实现，输出整数 mode 表示最佳内核。

#### tune脚本调用

`tune_xxxx.py`脚本用于生成最佳实现表

```bash
python3 tune_xxxx.py
```

最佳实现表由json文件表示，包括：

- `XXXX_Ee_Nn_Kk.json`
- `XXXX_Nn_Kk.json`
- `XXXX_DEFAULT.json`

表中数据如下所示：

```json
{
    "configs": {
        "16": {
            "MODE": 1161
        },
        "32": {
            "MODE": 1322
        },
        "128": {
            "MODE": 1642
        },
        "2048": {
            "MODE": 1644
        }
    }
}
```

最佳实现表生成逻辑

- `XXXX_Ee_Nn_Kk.json`实验测试内核对比产生

- `XXXX_Nn_Kk.json` 提炼`XXXX_Ee_Nn_Kk.json`取众数
- `XXXX_DEFAULT.json`提炼`XXXX_Nn_Kk.json`取众数

内核查找逻辑：

- 算子输入 `e` 、`m`、 `k`、`expected_m` 进行查找
- 算子查找路径：`XXX_Ee_Nn_Kk.json` →（无命中 fallback）→ `XXX_Nn_Kk.json` →（无命中 fallback）→ `XXX_DEFAULT.json`
- 算子接口会选择第一个大于 `expected_m` 的键值对应的内核编号 `MODE`。

#### 算子侧改动

在调用C++算子前最后一层python层，要求按照当前架构、硬件CU数量（可能）、矩阵形状大小进行内核mode选择, 调用`deepgemm/configs/xx.py` 中方法 `select_xxxx_masked_mode` 进行选择。

## TODO

当前 fp8 与 int8 共用一个内核，也暂时都使用 w8a8 下的最佳实现表。
## Contiguous FP8 pair32_v2

```bash
python3 tune_contiguous_fp8.py --config tune_contiguous_fp8_cfg.yml
```

需要先用 AICC 编译并加载包含最新 mode 编号的 gfx938 扩展。脚本打印实际
`deepgemm` 和扩展路径；可通过 `PYTHONPATH` 指向目标构建目录。

- 候选均为持久化内核：1000=BM256/BN256，1001=BM128/BN256，1002=BM128/BN384。
- `contiguous_fp8_cases.yml` 的 `m_list` 是总 M；JSON 中的键同样表示总 M。
  按 M256 块分配连续 expert 段，所有候选使用同一份输入和路由；M 很小时可能只有部分 expert 有数据。
- 自动跳过 M/N 不满足 tile 整除要求的候选；运行或编译错误会终止，不生成虚假赢家。
- 计时使用 HIP Graph 和 GPU event，交替候选顺序，取多轮中位数，不含打包时间。
  该脚本只调优性能，不检查量化精度或数值正确性；请在空闲设备运行。
- `resume` 根据 CSV 续跑；`emit_only` 只从 CSV 生成配置；`aggregate_only` 只聚合已有 E 级 JSON。
- 输出目录为 `deepgemm/configs/gfx938/w8a8_contiguous/hipc_pair32_v2/`，文件为：
  `DEEPGEMM_W8A8_FP8_CONTIGUOUS_E{E}_N{N}_K{K}_CU{CU}.json`、
  `DEEPGEMM_W8A8_FP8_CONTIGUOUS_N{N}_K{K}_CU{CU}.json`、
  `DEEPGEMM_W8A8_FP8_CONTIGUOUS_DEFAULT_BN384_CU{CU}.json`、
  `DEEPGEMM_W8A8_FP8_CONTIGUOUS_DEFAULT_BN256_CU{CU}.json`。
  后两级分别按 E、N/K 对每个总 M 取 mode 众数，平票取较小 mode。

DEFAULT 按 N 能否被 384 整除分别聚合：BN384 组包含所有候选，BN256 组排除 mode 1002。
两组只在对应扫描数据存在时生成，不使用旧的单份 DEFAULT 文件。

pair32_v2 Python 入口未显式传 MODE 时，调用 `configs/w8a8_contiguous.py`：
按 E/N/K → N/K → 对应 DEFAULT 查表，各级优先使用当前 CU 数文件，再尝试无 CU 后缀文件。
表内取大于等于总 M 的最小键；缺少 M 覆盖或 mode 不适用时继续下一张表。
最终回退到 BM256/BN256（M%256=0）或 BM128/BN256。显式 MODE 始终优先。
可用 `DEEPGEMM_W8A8_FP8_CONTIGUOUS_CONFIG` 指定单份配置文件覆盖；仍会检查 tile 整除条件。
配置文件通过 LRU 缓存加载，更新文件后应重新启动进程。
