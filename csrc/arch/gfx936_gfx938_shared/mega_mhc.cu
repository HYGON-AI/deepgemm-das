// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// mega_mhc 的共享实例化入口，编译进 gfx936 与 gfx938 两个 kernel 库
// （各自带本架构的 --offload-arch）。MMAC 的架构差异由 mega_mhc.cuh 里的
// __gfx936__ / __gfx938__ 预处理分支选择，因此这里只需要一对架构无关入口。

#include "mega_mhc.cuh"

namespace deepgemm::mega_mhc {

void mega_mhc_launch(const MegaMhcLaunchArgs& args) {
  mega_mhc_launch_common(args);
}

void mega_mhc_pack_fn(const MegaMhcLaunchArgs& args) {
  pack_fn_only(args);
}

}  // namespace deepgemm::mega_mhc
