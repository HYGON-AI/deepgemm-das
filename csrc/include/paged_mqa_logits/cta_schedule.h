// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include <algorithm>
#include <hip/hip_runtime.h>
#include <c10/util/Exception.h>

namespace deepgemm {
// Cache the most recent kernel/device resource configuration on this thread.
// No device tensor contents affect launch geometry or graph replay.
template <int R, typename Kernel>
inline dim3 paged_mqa_persistent_grid(dim3 full, Kernel kernel, int threads,
                                      size_t shared_bytes, int device, bool clean_logits) {
    if constexpr (R != 1) return full;
    if (clean_logits) return full;
    struct Capacity {
        int device = -1;
        const void* function = nullptr;
        size_t shared_bytes = 0;
        int threads = 0;
        int ctas = 0;
    };
    static thread_local Capacity cached;
    const auto function = reinterpret_cast<const void*>(kernel);
    if (cached.device != device || cached.function != function ||
        cached.shared_bytes != shared_bytes || cached.threads != threads) {
        hipDeviceProp_t properties{};
        int per_cu = 0;
        TORCH_CHECK(hipGetDeviceProperties(&properties, device) == hipSuccess,
                    "paged_mqa_logits: cannot query device resources");
        TORCH_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                        &per_cu, function, threads, shared_bytes) == hipSuccess && per_cu > 0,
                    "paged_mqa_logits: cannot determine persistent occupancy");
        cached = {device, function, shared_bytes, threads, per_cu * properties.multiProcessorCount};
    }
    // One resident wave across the device. The full grid still caps capacity
    // for small shapes; it need not be rounded to a multiple of the CU count.
    full.x = unsigned(std::min(uint64_t(full.x) * full.y, uint64_t(cached.ctas)));
    full.y = 1;
    return full;
}
} // namespace deepgemm
