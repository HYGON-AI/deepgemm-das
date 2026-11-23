// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// FP8 config resolution and dense/compact launch entry points.
// Instantiates both kernel families through selectors.hpp.

#include "auto.hpp"
#include "selectors.hpp"

namespace deepgemm::mqa::fp8 {
KernelConfig resolve(const MqaFp8Params &p) {
    KernelConfig config;
    mqa_auto_config(p, config, true);
    return config;
}

KernelConfig resolve_compact(const MqaFp8Params &p) {
    MqaFp8Params selection = p;
    selection.out_stride_m = selection.n;
    selection.clean_logits = 0;
    selection.output_compressed = 0;
    selection.output_width = 0;
    KernelConfig config;
    mqa_auto_config(selection, config, false);
    return config;
}

hipError_t launch(const MqaFp8Params &p, const KernelConfig &config, hipStream_t stream,
                  uint32_t max_grid_x, uint32_t max_grid_y) {
    if (p.output_compressed)
        return launch_compact(p, config, stream, max_grid_x, max_grid_y);
    LaunchGrid grid;
    if (!make_grid(p, config, max_grid_x, max_grid_y, grid)) {
        return hipErrorInvalidValue;
    }
    if (!grid.x || !grid.y)
        return hipSuccess;
    const Kernel kernel = dense::select_kernel(p, config);
    if (!kernel)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(kernel, dim3(grid.x, grid.y), dim3(256), 0, stream, p);
    return hipGetLastError();
}

hipError_t launch_compact(const MqaFp8Params &p, const KernelConfig &config, hipStream_t stream,
                         uint32_t max_grid_x, uint32_t max_grid_y) {
    LaunchGrid grid;
    if (!make_grid(p, config, max_grid_x, max_grid_y, grid))
        return hipErrorInvalidValue;
    if (!grid.x || !grid.y)
        return hipSuccess;
    const Kernel kernel = compact::select_kernel(p, config);
    if (!kernel)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(kernel, dim3(grid.x, grid.y), dim3(256), 0, stream, p);
    return hipGetLastError();
}
} // namespace deepgemm::mqa::fp8
