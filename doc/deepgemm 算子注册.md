# DeepGEMM 算子注册流程

> 架构：1 个主库（`deepgemm.op`）+ 3 个按 GPU 架构划分的 kernel 库（`deepgemm._k_gfx92a` / `_k_gfx936` / `_k_gfx938`）。主库不含任何 kernel 定义，运行时按设备架构加载对应 kernel 库。

## 文件分工

| 文件 | 作用 |
|---|---|
| `csrc/common/deepgemm_ops.cpp` | 主库：唯一的注册入口，全部算子通过 pybind 绑定为 `deepgemm.op.xxx`，**全量无条件绑定**，无 stub、无 arch 宏 |
| `csrc/include/deepgemm_ops.hpp` | pybind 宏组（`*_PYBIND`），供 deepgemm_ops.cpp 使用 |
| `csrc/include/kernel_registry_decls.h` | 跨库算子函数声明，供主库 pybind 绑定与 kernel 库 stub 共用（签名必须与定义逐字一致） |
| `csrc/include/m_grouped_gemm_nt_masked.h` 等 | 各算子族的 `deepgemm::` 函数声明 |
| `csrc/arch/<arch>/...` | kernel 库：真实实现（host 入口 + HIP kernel），**禁止出现 TORCH_LIBRARY / TORCH_LIBRARY_FRAGMENT**（注册只在主库 pybind） |
| `csrc/common/kernel_stubs_<arch>.cpp` | kernel 库：该架构**没有实现**的算子的占位定义（调用时 TORCH_CHECK 报错）；每个 kernel 库靠它凑齐全量符号集 |
| `csrc/common/kernel_module_main.cpp` | kernel 库：空的 `PYBIND11_MODULE`，仅为打包成 Python extension |
| `setup.py` | 拆分主库/kernel 库源码清单、per-source `--offload-arch`、按 extension 切换 arch 映射 |
| `deepgemm/__init__.py` | 运行时：检测 `gcnArchName` → `ctypes.CDLL(kernel.so, RTLD_GLOBAL)` → 再 import `.op` |

## 注册流程（新增一个算子）

```
① 实现函数        放 csrc/arch/<arch>/xxx.cu（需要哪个 arch 就放哪个 arch 的目录）
                  若部分 arch 无实现 → 对应 kernel_stubs_<arch>.cpp 补 stub

② 声明            已有头文件就用现有头；否则加进 kernel_registry_decls.h
                  （签名与定义逐字一致，返回类型/const/引用都不能差）

③ pybind          deepgemm_ops.cpp 的 PYBIND11_MODULE 里（通常经 deepgemm_ops.hpp 宏组）：
                  m.def("op_name", &deepgemm::函数, ...);

④ Python 封装     deepgemm/ 下对应模块封装入口函数，__init__.py 导出

⑤ 构建            setup.py：新源文件加入 SOURCE_GROUPS 对应 arch 组
                  （common 下的新 .cpp 需归类，否则 _validate_source_groups 报错）
```

## 运行时加载顺序（为什么这样设计）

```
import deepgemm
 → __init__.py: ctypes.CDLL("deepgemm/_k_gfx936.*.so", RTLD_GLOBAL)   # 先加载 kernel 库
 → import deepgemm.op                                                 # Python 用 RTLD_NOW 加载
 → 主库所有 deepgemm::* / pair32 命名空间符号从全局作用域解析，
   命中当前 arch 的 kernel 库实现；缺失算子命中 stub → 调用时抛 TORCH_CHECK
```

要点：
- **Python 默认 RTLD_NOW 加载扩展**，主库引用的每个符号都必须在加载时可解析 → 每个 kernel 库必须导出全量符号（实现 + stub）。
- **必须在 import `.op` 之前**用 RTLD_GLOBAL 预加载 kernel 库，否则符号解析不到。
- 同名符号在三个 kernel 库中并存是安全的：单进程内只会加载匹配当前 GPU 架构的一个库（不支持单进程混合架构）。

## 常见错误

| 现象 | 原因 |
|---|---|
| 链接报 multiple definition | arch 源文件里残留函数定义 + stub 重复定义，或同名文件进了同一库 |
| import op 报 undefined symbol | kernel 库缺符号：新算子漏加 stub，或签名与定义不一致导致 mangled name 不匹配 |
| setup.py 报 unclassified source | `csrc/common/` 下新文件未归入任何 `SOURCE_GROUPS` 组 |
