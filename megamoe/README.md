# megamoe

HCU gfx938 上的 MegaMoE（FP8 W8A8）实现：把 dispatch → grouped GEMM → SiLU/quant → grouped GEMM → combine 融合成单个 kernel，LL / HT 共用一套 split-block 主体。对称内存支持**节点内 HIP IPC** 与**超节点 HIP fabric** 两条传输路径，运行时切换。

fabric 依赖 DTK 26.10+ 才提供的 `<hip/hip_hsa_ext.h>`，构建时用 `__has_include` 探测：头文件存在则 fabric 编入 `_C`，不存在则整段跳过，MegaMoE 退化为仅 HIP IPC。旧 DTK（如 26.04）因此仍可正常出包。

本文档覆盖代码结构、编译安装、测试脚本与超节点 EP8 / EP16 / EP32 测试。

## 1. 代码结构

### 1.1 目录树（当前生效）

```
DeepGEMM/
├── setup.py                          # 主包构建，含 deepgemm.mega._C 扩展
├── csrc/megamoe/
│   ├── mega_moe.cu                   # host launcher + pybind11（IPC/fabric buffer、launch、build config）
│   └── kernels/
│       ├── params.h                  # 运行时参数、常量、debug 宏
│       ├── mega_moe_kernel_unified.cuh  # 统一 LL/HT split-block 融合 kernel 主体
│       ├── mega_moe_preprocess.cuh  # bf16→FP8/INT8 channel-wise 量化 + topk 拷贝
│       ├── scheduler.cuh            # expert 级 token 调度器
│       ├── workspace.cuh            # workspace 元数据布局（IPC 细粒度区 + local 区）
│       ├── buffer_layout.cuh        # sym_buffer / pool 布局与切片
│       ├── comm.cuh                 # rank barrier / 子集 barrier / grid sync
│       ├── gemm_device.cuh          # LL MFMA FP8 GEMM 设备函数
│       ├── gemm_device_ht.cuh       # HT GEMM 设备函数（Marlin w6 / LDS）
│       ├── activation_device.cuh    # SwiGLU + per-token channel-wise FP8 量化
│       └── warp_reduce.cuh          # 64-lane warp 归约原语（DeepEP utils.cuh 等价实现）
├── deepgemm/mega/
│   ├── __init__.py                   # 对外导出 9 个符号
│   └── mega_moe.py                   # SymmBuffer、权重变换、fp8_mega_moe
├── megamoe/
│   └── README.md                     # 本文档
└── op_test/
    ├── test_mega_moe_fp8.py          # 主测试：正确性 + ref/fused 性能对比
    ├── test_mega_moe_int8.py         # INT8 W8A8（需 DG_MEGA_BUILD_INT8=1 编译）
    ├── stress_balanced_8h.sh         # 8h 随机压测（均衡 / 不均衡 + GPU 干扰）
    ├── interfere_spin_rotate.sh      # GPU 干扰程序（压测脚本内部调用）
    ├── spin_occupier.hip             # 干扰 kernel 源
    ├── test_deepgemm_*.py            # GEMM 单算子用例
    └── unified_test/  tune_test/     # 单算子用例与调优配置
```

### 1.2 模块职责

| 层 | 入口 | 职责 |
|---|---|---|
| Python API | `deepgemm/mega/mega_moe.py` | `SymmBuffer` 分配缓冲 / 交换句柄 / 安装 peer 指针；权重变换；`fp8_mega_moe` |
| Host launcher | `csrc/megamoe/mega_moe.cu` | 计算 buffer 尺寸、IPC/fabric 分配、按 `expected_tokens_per_expert` 选 LL/HT 分档、launch |
| Device kernel | `kernels/mega_moe_kernel_unified.cuh` | dispatch-pull → L1 SwiGLU/quant → L2 → combine-reduce，跨 rank system-scope 同步 |
| 测试 | `op_test/test_mega_moe_fp8.py` | DeepEP 参考路径 vs 融合 kernel，精度 + 耗时 |

`deepgemm/mega/__init__.py` 导出：`SymmBuffer`、`get_symm_buffer_for_mega_moe`、`transform_weights_for_mega_moe`、`transform_fp8_weights_for_mega_moe`、`transform_int8_weights_for_mega_moe`、`interleave_gate_up_weights`、`interleave_gate_up_scales`、`fp8_mega_moe`、`mega_moe_preprocess`。

### 1.3 外部依赖（不在本仓库）

- `deep_ep`（HCU 移植版）、`lightop`、`rocshmem`：预装在容器 `site-packages`，仓库内只有调用、环境变量和少量等价移植代码，**没有源码**。DeepEP 参考路径与 rocshmem 的 MNVL 行为由这些包决定。
- `torch` HIP backend；DTK 26.10+ 的 `<hip/hip_hsa_ext.h>`（提供 `hipFabricHandle_t` 与 `hipHsaExtRpcMemoryCreate` / `AttachDevice` / `Detach`）——fabric 路径的可选依赖，**仅 DTK 26.10 起提供**，构建时用 `__has_include` 探测，缺失只影响 fabric。

## 2. 编译与安装

### 2.1 环境要求

| 项 | 要求 |
|---|---|
| GPU | HCU gfx938（BW1100 72CU / BW1101 / BW1301_LC 64CU） |
| 工具链 | DTK / ROCm（`/opt/dtk`）、`aicc` 或 `hipcc`、`ninja` |
| DTK 版本 | **26.10+ 才有 fabric**（`<hip/hip_hsa_ext.h>`）；旧 DTK（26.04 等）照常构建，`_C` 不含 fabric 绑定，MegaMoE 只能走 HIP IPC |
| Python | torch with HIP backend |
| 外部包 | `deep_ep`、`lightop`（跑参考路径必需） |

### 2.2 构建

`deepgemm.mega._C` 随主包一起构建。目标架构由 **`PYTORCH_ROCM_ARCH`** 指定（`setup.py:234`）；`GPU_ARCHS` 只被 `deepgemm/utils.py` 用于运行时 JIT 探测，**不影响本扩展构建**。

```bash
cd DeepGEMM
PYTORCH_ROCM_ARCH=gfx938 python3 setup.py bdist_wheel
pip3 install --force-reinstall --no-deps dist/deepgemm-*.whl

# 或原地编译（配合 PYTHONPATH=. 跑测试）
PYTORCH_ROCM_ARCH=gfx938 python3 setup.py build_ext --inplace
```

> 不设架构时 `setup.py` 探测当前 GPU 并过滤可编译架构，兜底 `gfx936`；多架构用分号分隔，例如 `PYTORCH_ROCM_ARCH="gfx936;gfx938"`。

### 2.3 构建开关（全部可选，实现见 `setup.py:771-835`）

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `PYTORCH_ROCM_ARCH` | 自动探测 | 目标架构，生成 `--offload-arch=` |
| `SAVE_TEMPS` | `1` | `=0` 关闭 `--save-temps` |
| `DG_MEGAMOE_DBUG` + `DG_DBUG_LEVEL` | `0` / `2` | 编译 device 侧 printf 诊断（`1`=TRACE，`2`=全量） |
| `DG_MEGA_BUILD_LL=0` / `DG_MEGA_SKIP_LL=1` | 编译 LL | 跳过 low-latency 实例 |
| `DG_MEGA_BUILD_HT=0` / `DG_MEGA_SKIP_HT=1` | 编译 HT | 跳过 high-throughput 实例 |
| `DG_MEGA_BUILD_INT8=1` | `0`（默认跳过） | 编译 INT8 实例 |
| `DG_MEGA_SKIP_INT8=1` | — | 强制跳过 INT8 |
| `DG_MEGA_PROF` / `DG_MEGA_PROF_BUILD` | `0` | per-block 阶段耗时 profiling |
| `DG_MEGA_HANG_PRINT` 等诊断宏 | `0` | 另有 `DG_MEGA_GW_NORESCALE`、`DG_MEGA_L1_NOQUANT`、`DG_MEGA_GW_SWAPMG`、`DG_MEGA_GWDUMP`、`DG_MEGA_HT_MG`、`DG_MEGA_HT_MG2`、`DG_MEGA_DUMP_SILU`、`DG_MEGA_W12P_DBG` |
| `MAX_JOBS` / `AICC` / `HIPCC` / `ROCM_HOME` | 自动 | 工具链与并行度 |

> **注意**：历史构建脚本里的 `DG_MEGA_BUILD_72CU=0`、`DG_MEGA_BUILD_V41_ONLY=1`、`DG_MEGA_SKIP_NONSPLIT=1` 属于**本地 patch** 引入的 shape / 实例裁剪，当前 `setup.py` 未包含，要用必须先 apply 对应 patch。BW1301_LC 原生 64CU，不需要 `ROC_GLOBAL_CU_MASK`。
>
> 切换任何编译开关重编前，先删 `build/temp.linux-*/csrc/megamoe/mega_moe.o`，否则 ninja 会复用旧对象。

### 2.4 验证安装

```bash
cd /tmp && python3 -c "import deepgemm.mega as m, deepgemm.mega._C as C; \
print(m.__file__); print(C.get_mega_moe_hip_build_config()); \
print('fabric bindings:', hasattr(C, 'allocate_hip_fabric_buffer'))"
```

必须在 `/tmp`（仓库外）验证，避免仓库内源码抢先加载。**DTK 26.10+** 上应输出 `fabric_peer_memory: True`、`fabric bindings: True`；在缺 `<hip/hip_hsa_ext.h>` 的旧 DTK（如 26.04）上两者均为 `False`，属预期，不是安装失败。头文件存在时 **fabric 路径默认编译，没有构建开关、不需要链接 `hsa-runtime64`**。`get_mega_moe_hip_build_config()` 还返回 `operator`、`package`、三种 `*_scale_mode`、`has_fp8_api`、`has_fp4_api`、`fusion_boundary`。

## 3. 测试脚本 `op_test/test_mega_moe_fp8.py`

端到端对比 **DeepEP 参考路径**（dispatch → grouped GEMM → SiLU+quant → grouped GEMM → combine）与 **融合 kernel**（`fp8_mega_moe`）的精度和耗时。

### 3.1 运行方式与参数

```bash
cd DeepGEMM
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 op_test/test_mega_moe_fp8.py <num_ranks> <local_tokens> [--print-recv-counts]

# 未安装 whl、用仓库内 inplace .so 时
PYTHONPATH=. HIP_VISIBLE_DEVICES=0,1,2,3 python3 op_test/test_mega_moe_fp8.py 4 512
```

| 参数 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `num_ranks` | argv[1] | `4` | 全局 world size（多机时也填全局值） |
| `local_tokens` | argv[2] | `512` | 每 rank 均衡 token 数；`DG_TOKEN_COUNTS` 设置后仅作上限校验 |
| `--print-recv-counts` | argv[3:] | — | 打印各 rank 收到的 token 分布 |

### 3.2 默认配置与 shape

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hidden` | `5120` | V4.1 生产配置 |
| `intermediate_hidden` | `2304` | |
| `num_topk` | `6` | |
| `num_experts` | `num_ranks * 48` | 不设 `DG_TEST_SHAPE` 时，每 rank 恒为 48 |
| `num_local_experts` | `num_experts // num_ranks` | |
| `DG_TEST_SHAPE` | 空 | `"hidden,intermediate_hidden,num_topk,num_experts"`，要求 `num_experts % num_ranks == 0` |

超节点用 `DG_TEST_SHAPE=5120,2304,6,384` 把总专家固定为 384，于是 EP8 / EP16 / EP32 分别是 48 / 24 / 12 本地专家（见 4.3）。

### 3.3 LL / HT 模式

**ref 路径与 fused kernel 各选各的**，不要混淆：

- **ref 路径**由 `DG_REF_MODE` 决定：

  | 值 | 行为 |
  |---|---|
  | `framework`（默认） | `tokens_mode_sel <= DG_DEEPEP_LL_CAP`（默认 128）走 LL，否则 normal |
  | `ll` | 强制 LL ref（token 数超过 `DG_DEEPEP_LL_CAP` 会报错） |
  | `normal` | 强制 normal / HT ref |

  日志 `[REF-FW] DG_REF_MODE=ll -> low_latency` 可确认实际选择。
- **fused kernel** 由 C++ host launcher 按 `expected_tokens_per_expert` 自动分档（`mega_moe.cu:580-631`）：`<=12` → LL BM16、`<=24` → LL BM32、`<=56` → HT BM64、`>56` → HT BM128；覆盖阈值 `DG_MEGA_HT_EXPECTED` 默认 **13.0**，Python 侧没有阈值。

因此 `32` token 走 LL、`4096` token 走 HT。

### 3.4 环境变量（按读取方分类）

**A. 测试脚本读取**（`op_test/test_mega_moe_fp8.py`）

| 变量 | 默认 | 用途 |
|---|---|---|
| `HIP_VISIBLE_DEVICES` | 全部可见 | 指定可见卡 |
| `DG_REF_MODE` / `DG_DEEPEP_LL_CAP` | `framework` / `128` | ref 路径选择 |
| `DG_TEST_SHAPE` | 空 | shape 覆盖 |
| `DG_DEEPEP_BUFFER_BYTES` | `2e9` | DeepEP normal NVL buffer；32 卡必须 `4e9` |
| `DG_DEEPEP_QPS` | `48` | normal ref 的 `num_qps_per_rank` |
| `DG_TOKEN_COUNTS` / `DG_MAX_TOKENS` | 空 / `0` | 不均衡 token 配置与容量上限 |
| `MASTER_ADDR` / `MASTER_PORT` | `localhost` / `29500` | 分布式 rendezvous |
| `DG_MEGA_NNODES` / `DG_MEGA_NODE_RANK` | `1` / `0` | 多机启动（见 4.5） |
| `BENCH_WARMUP` / `BENCH_ITERS` | `5` / `20` | warmup / 计时轮数 |
| `BENCH_ITER_BARRIER` | `1` | 每轮前 `dist.barrier()` |
| `BENCH_REF_CUDAGRAPH_LL` / `_NORMAL` | `1` / `0` | ref 路径 CUDA graph |
| `BENCH_FUSED_CUDAGRAPH` / `BENCH_KINETO` | `1` / `1` | fused 计时方式 |
| `SKIP_REF` / `CORRECTNESS_ITERS` | `0` / `1` | 跳过 ref / 重复精度对比 |
| `DG_HANG_WATCHDOG_S` / `DG_WATCHDOG_DIR` | 不开 | hang 现场快照 |
| `DG_INIT_MODE` / `DG_RANDN_INIT` / `DG_SEED` | `randn` / `1234` | 数据分布与随机种子 |
| `DG_TEST_ATOL` / `DG_TEST_ATOL_FRAC` / `DG_TEST_RTOL` | 见 3.5 | 精度容差 |

**B. 外部 DeepEP / rocshmem 读取（测试脚本本身不读）**

| 变量 | 用途 |
|---|---|
| `ROCSHMEM_IPC_MNVL=1` | **LL ref 的 rocshmem MNVL**（超节点必需） |
| `ROCSHMEM_IPC_MNVL_NPES` | 默认 32；`group_size <= NPES` 时 DeepEP 自动设 `ROCSHMEM_BACKEND=ipc`，8 / 16 / 32 卡都合法 |
| `DEEP_EP_NORMAL_MNVL=1` | **16 / 32 卡 HT(normal) ref 必需**，否则 DeepEP 误判成多节点走 internode 而 VMFault |
| `ROCSHMEM_HEAP_SIZE` / `ROCSHMEM_GDR_DISABLE_XDP` | LL heap 大小（常用 `7000000000`）/ 关 XDP |
| `ROCSHMEM_DISABLE_HDP_FLUSH` | LL 分支里脚本会 `setdefault` 为 `1` |
| `DEEPEP_ENABLE_LL_LAYERED_OPT` | LL 分支里脚本 `setdefault` 为 `1`；**EP16 / EP32 的 LL ref 必须显式设 `0`**（layered LL kernel 假设 8 GPU/节点，否则 VMFault） |

**C. `deepgemm/mega/mega_moe.py` 读取**

| 变量 | 用途 |
|---|---|
| `DG_MEGA_PEER_MEMORY` | 选择 IPC / fabric（见 4.2） |
| `MEGAMOE_HCU_PEER_MEMORY` | 上者的兼容别名 |
| `DG_MEGA_CFG_ALLGATHER` / `DG_MEGA_TOPK_STATS` | token 总数 all-gather / topk 统计 |

C++ host launcher 另有一批**运行时**调优开关（`DG_MEGA_HT_EXPECTED`、`DG_MEGA_HT_BM*`、`DG_MEGA_LL_LB`、`DG_MEGA_PROF` 等，见 `mega_moe.cu:577-910`），与构建期 `DG_MEGA_BUILD_*` 是两套东西。

### 3.5 输出与判定

每个 rank 打印：

1. `DeepEP + GEMM + SILU + GEMM ref total ... ms`
2. `fused_mega_moe total ... ms`
3. 精度统计（`abs_diff`、`Per-row max abs_diff`、`ratio`、worst row）
4. `speedup (ref_median / fused_median)` 与 `speedup (ref_min / fused_min)`
5. `test rank N async mode True PASS` / `FAIL`

判定公式 `|ref - y| <= atol + atol_frac * amax|ref| + rtol * |ref|`，并要求两侧无 NaN/Inf。默认 randn 分布下 `atol=0.00355`、`atol_frac=0.03`、`rtol=0.02`（NV 分布下 `atol=0`、`atol_frac=0.02`），可由 `DG_TEST_ATOL*` 覆盖。一个用例的 PASS 行数应等于 `num_ranks`。

### 3.6 其它测试脚本

| 脚本 | 用途 |
|---|---|
| `test_mega_moe_int8.py` | INT8 W8A8（需 `DG_MEGA_BUILD_INT8=1` 编译） |
| `test_deepgemm_*.py` | GEMM 单算子用例 |

### 3.7 压测

```bash
bash op_test/stress_balanced_8h.sh [hours=8] [log_root]
```

8 卡随机循环：3 种 shape（默认 / V4 `4096,2048,6,256` / V4.1 `5120,2304,6,384`）× 均衡或不均衡 token，全程跑 GPU 干扰程序并开看门狗。最大 token `<=128` 时容量为 256、`<=16384` 为 16384，其余按实际值。

| 环境变量 | 默认值 |
|---|---|
| `HIP_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` |
| `DG_HANG_WATCHDOG_S` | `60` |
| `BENCH_ITER_BARRIER` | `0` |
| `ROCSHMEM_GDR_DISABLE_XDP` / `ROCSHMEM_HEAP_SIZE` | `1` / `7000000000` |

判定：cfg 最大 token ≥8192 → `timeout 900s`，否则 `400s`；`rc=124` → `REPRO-HANG`；`rc=0` 且 ≥8 个 PASS → `PASS`；其余 → `BAD` 并自动重试 1 次。

输出落在脚本 `log_root` 下按 session 建的子目录：`hangs.txt`、`interfere/`、每轮 `t<轮>_<kind>_<shape>_<max>_try<N>/{run.log,result.txt,wd/}`。脚本退出（含 Ctrl-C）自动清理干扰与残留进程。独立跑干扰程序用 `bash op_test/interfere_spin_rotate.sh [log_dir]`（首次自动 hipcc 编译 `spin_occupier`），复用前用 `hy-smi` 确认 HCU% 归 0。

## 4. 超节点测试（EP8 / EP16 / EP32）

### 4.1 环境与前置约定

适用环境：两台 16 卡节点（下文记作 `node0` / `node1`），每台 16 × BW1301_LC（64CU、gfx938），容器名统一记作 `your_docker`；两机代码路径保持一致（下文以 `/root/yuwenjun/DeepGEMM` 为例，主机与容器同路径共享，`scp` 直达），跨机 rendezvous 用 `node0` 的管理网 IP。两机合起来是 32 卡超节点，日志**分别**落在两机本地，不要只在一台汇总。

**跑前必须**在**每个节点**做 `hy-smi` 双采样（宿主机 PATH 里没有，写全路径），要使用的卡全部 `HCU% ≤ 5` 且 `VRAM% ≤ 15` 才开跑：

```bash
/opt/hyhal/bin/hy-smi
sleep 6
/opt/hyhal/bin/hy-smi
```

> 只用部分卡时，判定**实际要用的那几张**即可，例如 `node0` 物理卡 8–15 空闲时设 `HIP_VISIBLE_DEVICES=8,9,...,15`（映射为逻辑 0–7）。
> 共享机上 VRAM 会被常驻推理服务静态预留（sglang `--mem-fraction-static 0.94` 可从 39% 涨到 97%），此时即使 `HCU% = 0` 也没有可用显存，强行跑 16 卡 / EP32 会 OOM 并可能挤崩他人服务。

### 4.2 peer memory：默认 IPC 与强制 fabric

| | 默认 IPC | 强制 fabric（超节点） |
|---|---|---|
| 环境变量 | 不设 `DG_MEGA_PEER_MEMORY` 或 `=ipc` | `DG_MEGA_PEER_MEMORY=rpc`（等价 `fabric` / `mnvl` / `1` / `true` / `on` / `yes`） |
| 兼容别名 | — | `MEGAMOE_HCU_PEER_MEMORY=rpc` |
| 底层 API | `hipIpcGetMemHandle` / `hipIpcOpenMemHandle` | `hipHsaExtRpcMemoryCreate` / `AttachDevice` / `Detach` |
| 内存属性 | `hipDeviceMallocUncached` | 同样 `hipDeviceMallocUncached` |
| 适用范围 | 同一节点 | 跨节点必需；同节点也可用于验证 fabric 语义 |
| 日志标记 | `SymmBuffer ipc:` | `SymmBuffer fabric:` |

要点：

- fabric **在 DTK 26.10+ 上默认编译进 `deepgemm.mega._C`**，不需要构建开关、不链接 `hsa-runtime64`；此时 `get_mega_moe_hip_build_config()['fabric_peer_memory']` 为 `True`。旧 DTK（26.04 等）没有 `<hip/hip_hsa_ext.h>`，构建时整段 fabric 代码被跳过，该值为 `False`。
- 传输由**环境变量显式选择**，EP / rank 数不会自动切换；`_require_fabric_support()` 发现 `.so` 里没有 `allocate_hip_fabric_buffer`（旧 whl，或由旧 DTK 构建的包）会直接抛错，**不会静默回退 IPC**。
- 两条路径的数据缓冲与信号缓冲都用同一套 `dist.all_gather_object` 交换句柄，kernel 侧的 system-scope 屏障不变。

### 4.3 EP 规模参数矩阵

用 `DG_TEST_SHAPE=5120,2304,6,384` 固定总专家 384：

| 档 | `num_ranks` | 节点数 | 本地专家 | `DG_DEEPEP_BUFFER_BYTES` | HT(normal) ref | LL ref |
|---|---|---|---|---|---|---|
| EP8 | 8 | 1 | 48 | 默认 `2e9` | 不需要 MNVL | `ROCSHMEM_IPC_MNVL=1` |
| EP16 | 16 | 1 | 24 | `2e9`（`4e9` 亦可） | **`DEEP_EP_NORMAL_MNVL=1` 必须** | `ROCSHMEM_IPC_MNVL=1` |
| EP32 | 32 | 2 | 12 | **`4e9` 必须** | `DEEP_EP_NORMAL_MNVL=1` | `ROCSHMEM_IPC_MNVL=1` |

> 32 卡时 DeepEP 的 `num_channels * num_ranks * ...` 随 rank 线性翻倍，默认 `2e9` 会触发 NVL buffer 容量断言（报错落在 `intranode_combine`）。
> **EP16 / EP32 跑 LL 必须额外设 `DEEPEP_ENABLE_LL_LAYERED_OPT=0`**：测试脚本在 LL 分支把该变量 `setdefault` 成 `1`，而 layered LL kernel 假设 8 GPU/节点，16 / 32 卡会在 `deep_ep.Buffer` 创建后 VMFault；实测单设 `DEEP_EP_NORMAL_MNVL=1` 无效。
> 不设 `DG_TEST_SHAPE` 时 `num_experts = num_ranks * 48`，本地专家恒为 48，EP16 / EP32 会变成 768 / 1536 专家，**不是**上表的生产配置。

### 4.4 单机 EP8 / EP16：默认 IPC 与强制 fabric

公共环境（以 EP16 为例；EP8 把 `HIP_VISIBLE_DEVICES` 换成 `0..7`、`num_ranks` 换成 `8`）：

```bash
cd /root/yuwenjun/DeepGEMM
export DG_TEST_SHAPE=5120,2304,6,384      # 总专家 384 → EP16 24/rank
export DG_DEEPEP_BUFFER_BYTES=4e9
export HIP_VISIBLE_DEVICES=$(seq -s, 0 15)
export ROCSHMEM_HEAP_SIZE=7000000000
export ROCSHMEM_GDR_DISABLE_XDP=1
```

**HT 4096 token**（默认 `DG_REF_MODE=framework` → normal ref，fused 也走 HT）：

```bash
# 默认 IPC（16 卡 HT ref 必须开 MNVL）
DEEP_EP_NORMAL_MNVL=1 timeout 240 python3 op_test/test_mega_moe_fp8.py 16 4096

# 强制 fabric：只多一个环境变量
DG_MEGA_PEER_MEMORY=rpc DEEP_EP_NORMAL_MNVL=1 \
  timeout 240 python3 op_test/test_mega_moe_fp8.py 16 4096
```

**LL 32 token**（ref 与 fused 都走 LL）：

```bash
export DG_REF_MODE=ll
export ROCSHMEM_IPC_MNVL=1                # LL ref 的 rocshmem MNVL
export DEEPEP_ENABLE_LL_LAYERED_OPT=0     # EP16 LL 必需（layered LL 假设 8 GPU/节点）

# 默认 IPC
timeout 240 python3 op_test/test_mega_moe_fp8.py 16 32

# 强制 fabric
DG_MEGA_PEER_MEMORY=rpc timeout 240 python3 op_test/test_mega_moe_fp8.py 16 32
```

> EP8 的 HT 用例不需要 `DEEP_EP_NORMAL_MNVL`；EP8 的 LL 用例也不需要 `DEEPEP_ENABLE_LL_LAYERED_OPT=0`（8 卡实测可直接跑）。
> `ROCSHMEM_IPC_MNVL=1` 时 DeepEP `buffer.py` 在 `group_size <= ROCSHMEM_IPC_MNVL_NPES(32)` 下自动设 `ROCSHMEM_BACKEND=ipc`，8 / 16 / 32 卡都合法。

### 4.5 双机 EP32（强制 fabric）

`num_ranks=32` 是**全局** world size，每台 launcher 只 spawn `32 / 2 = 16` 个本地进程；两机 `HIP_VISIBLE_DEVICES` 都是 `0..15`，`DG_MEGA_NODE_RANK` 负责全局 rank 偏移。两机必须**同时**启动，装**同一个 whl**。

```bash
# ===== 两机共同环境 =====
cd /root/yuwenjun/DeepGEMM
export DG_MEGA_NNODES=2
export MASTER_ADDR=<node0_ip>              # node0 管理网 IP
export MASTER_PORT=29532
export DG_MEGA_PEER_MEMORY=rpc           # 跨节点必须强制 fabric
export DG_TEST_SHAPE=5120,2304,6,384     # 32 卡 → 12 experts/rank
export DG_DEEPEP_BUFFER_BYTES=4e9        # 32 卡必需
export HIP_VISIBLE_DEVICES=$(seq -s, 0 15)
export ROCSHMEM_HEAP_SIZE=7000000000
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEP_EP_NORMAL_MNVL=1

# ===== HT 4096：两机同时 =====
DG_MEGA_NODE_RANK=0 timeout 300 python3 op_test/test_mega_moe_fp8.py 32 4096   # node0
DG_MEGA_NODE_RANK=1 timeout 300 python3 op_test/test_mega_moe_fp8.py 32 4096   # node1

# ===== LL 32：两机各再加三个变量，同时跑 =====
export DG_REF_MODE=ll ROCSHMEM_IPC_MNVL=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=0     # EP32 LL 必需
DG_MEGA_NODE_RANK=0 timeout 300 python3 op_test/test_mega_moe_fp8.py 32 32     # node0
DG_MEGA_NODE_RANK=1 timeout 300 python3 op_test/test_mega_moe_fp8.py 32 32     # node1
```

从本地后台同时拉起（把上面的命令存成两机同一份 `/root/yuwenjun/ep32.sh`，`DG_MEGA_NODE_RANK` 由 `docker exec -e` 传入）：

```bash
ssh node0 'docker exec -d -e DG_MEGA_NODE_RANK=0 your_docker bash /root/yuwenjun/ep32.sh'
ssh node1 'docker exec -d -e DG_MEGA_NODE_RANK=1 your_docker bash /root/yuwenjun/ep32.sh'
```

约束：`num_ranks % DG_MEGA_NNODES == 0`；两机 `MASTER_PORT` 相同且未被占用；日志分别落两机本地（`master_node0.log` / `master_node1.log`），不要只在一台汇总。跨机 LL 记得同时设 `DEEPEP_ENABLE_LL_LAYERED_OPT=0`；`DEEP_EP_NORMAL_MNVL=1` 是 HT ref 的拓扑开关，实测单独设置并不能解决 LL 的 VMFault。

### 4.6 结果核对与常见坑

每个用例检查：

- `SymmBuffer ipc:` / `SymmBuffer fabric:` —— 确认实际传输路径；
- `[REF-FW] DG_REF_MODE=... -> low_latency/normal` —— 确认 ref 模式；
- `test rank N async mode True PASS` 共 `num_ranks` 条，`FAIL` 为 0；
- `speedup (ref_min / fused_min)` 用于性能对比，共享机有干扰时以 `ref_min` 为准。

| 现象 | 原因 / 处理 |
|---|---|
| `DG_MEGA_PEER_MEMORY=rpc` 抛 "no fabric buffer bindings" | whl 里的 `_C` 没有 fabric 绑定：旧 whl，或由旧 DTK（26.04 等，缺 `<hip/hip_hsa_ext.h>`）构建。`/tmp` 下看 `fabric_peer_memory`：`False` 时换 DTK 26.10+ 重编；旧 DTK 上只能走 IPC |
| 16 / 32 卡 HT ref `Invalid address access: (nil)` VMFault | 缺 `DEEP_EP_NORMAL_MNVL=1` |
| 32 卡 DeepEP NVL buffer 容量断言 | `DG_DEEPEP_BUFFER_BYTES=4e9` |
| LL ref rocshmem 初始化失败 | `ROCSHMEM_IPC_MNVL=1` + `ROCSHMEM_HEAP_SIZE=7000000000` |
| 16 / 32 卡 LL 在 `deep_ep.Buffer` 创建后 `Invalid address access` VMFault | 缺 `DEEPEP_ENABLE_LL_LAYERED_OPT=0`；单设 `DEEP_EP_NORMAL_MNVL=1` 无效 |
| 专家数意外翻倍 | 没设 `DG_TEST_SHAPE`，默认 `num_ranks*48` |
| 多机 NCCL 卡在 init | 两机 `MASTER_ADDR` / `MASTER_PORT` 不一致、端口被占、或 launcher 没同时起 |
| 显存 OOM / 挤崩他人服务 | `hy-smi` 显示 VRAM 被静态预留，等空闲再跑（见 4.1） |

## 附：查证索引

| 想查什么 | 位置 |
|---|---|
| 构建开关 / 架构解析 | `setup.py:771-835`、`setup.py:233-241` |
| fabric build config / buffer API | `csrc/megamoe/mega_moe.cu:1635-1653`、`:229-309`、`:1700-1711` |
| LL/HT 分档与运行时开关 | `csrc/megamoe/mega_moe.cu:577-631` |
| peer memory 选择与校验 | `deepgemm/mega/mega_moe.py:79-140` |
| 测试脚本默认值 / 环境变量 | `op_test/test_mega_moe_fp8.py:30-115` |
| 多机启动 / shape 覆盖 | `op_test/test_mega_moe_fp8.py:1730-1780` |
