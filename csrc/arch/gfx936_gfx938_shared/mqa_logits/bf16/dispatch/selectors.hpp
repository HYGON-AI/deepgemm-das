// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// BF16 host selectors for both output modes; included only by launch.cu.

#pragma once
#include "config.hpp"
#include "../kernels/dense/small_kv.hpp"
#include "../kernels/dense/split_head.hpp"
#include "../kernels/dense/tiled.hpp"
#include "../kernels/compact/small_kv.hpp"
#include "../kernels/compact/split_head.hpp"
#include "../kernels/compact/tiled.hpp"

namespace deepgemm::mqa::bf16 {
namespace dense {
namespace {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
template <int H, int TN, int Waves, bool Pair, bool CleanLogits>
Kernel select_small(const MqaBf16Params &p) {
    constexpr int BM = Waves * (Pair ? 2 : 1);
    if (p.m % BM == 0) {
        if (p.n % TN == 0)
            return small_kv<H, TN, Waves, Pair, true, true, CleanLogits>;
        return small_kv<H, TN, Waves, Pair, true, false, CleanLogits>;
    }
    if (p.n % TN == 0)
        return small_kv<H, TN, Waves, Pair, false, true, CleanLogits>;
    return small_kv<H, TN, Waves, Pair, false, false, CleanLogits>;
}

template <int H, bool CleanLogits>
Kernel select_small_n(const MqaBf16Params &p, const KernelConfig &c) {
    if constexpr (H == 8) {
        if (c.paired_queries)
            return select_small<8, 16, 1, true, CleanLogits>(p);
    }
    if (c.block_n == 16)
        return select_small<H, 16, 1, false, CleanLogits>(p);
    if (c.block_n == 32)
        return select_small<H, 32, 1, false, CleanLogits>(p);
    return select_small<H, 64, 1, false, CleanLogits>(p);
}

template <bool CleanLogits>
Kernel select_small_kv_kernel(const MqaBf16Params &p, const KernelConfig &c) {
    switch (p.h) {
    case 1:
        return select_small_n<1, CleanLogits>(p, c);
    case 2:
        return select_small_n<2, CleanLogits>(p, c);
    case 4:
        return select_small_n<4, CleanLogits>(p, c);
    case 8:
        return select_small_n<8, CleanLogits>(p, c);
    case 16:
        return select_small_n<16, CleanLogits>(p, c);
    case 32:
        return select_small_n<32, CleanLogits>(p, c);
    case 64:
        return select_small_n<64, CleanLogits>(p, c);
    case 128:
        return select_small_n<128, CleanLogits>(p, c);
    }
    return nullptr;
}

template <int BM, int BN, int GH, bool CleanLogits>
Kernel select_tiled_tile(const MqaBf16Params &p) {
    if (p.m % BM == 0) {
        if (p.n % BN == 0)
            return tiled<BM, BN, GH, true, true, CleanLogits>;
        return tiled<BM, BN, GH, true, false, CleanLogits>;
    }
    if (p.n % BN == 0)
        return tiled<BM, BN, GH, false, true, CleanLogits>;
    return tiled<BM, BN, GH, false, false, CleanLogits>;
}

template <int BM, int GH, bool CleanLogits>
Kernel select_tiled_n(const MqaBf16Params &p, int bn) {
    switch (bn) {
    case 64:
        return select_tiled_tile<BM, 64, GH, CleanLogits>(p);
    case 128:
        return select_tiled_tile<BM, 128, GH, CleanLogits>(p);
    case 256:
        if constexpr (BM == 64 && GH == 2)
            return nullptr;
        else
            return select_tiled_tile<BM, 256, GH, CleanLogits>(p);
    default:
        return nullptr;
    }
}

template <int BM, bool CleanLogits>
Kernel select_tiled_group(const MqaBf16Params &p, const KernelConfig &c) {
    switch (c.group_heads) {
    case 1:
        return select_tiled_n<BM, 1, CleanLogits>(p, c.block_n);
    case 2:
        return select_tiled_n<BM, 2, CleanLogits>(p, c.block_n);
    case 4:
        if constexpr (BM <= 32)
            return select_tiled_n<BM, 4, CleanLogits>(p, c.block_n);
    }
    return nullptr;
}

template <bool CleanLogits>
Kernel select_split_head_tile(const MqaBf16Params &p) {
    if (p.m % 16 == 0) {
        if (p.n % 64 == 0)
            return split_head<16, 64, 4, true, true, CleanLogits>;
        return split_head<16, 64, 4, true, false, CleanLogits>;
    }
    if (p.n % 64 == 0)
        return split_head<16, 64, 4, false, true, CleanLogits>;
    return split_head<16, 64, 4, false, false, CleanLogits>;
}

template <bool CleanLogits>
Kernel select_kernel(const MqaBf16Params &p, const KernelConfig &c) {
    if (c.kind == KernelKind::SmallKV)
        return select_small_kv_kernel<CleanLogits>(p, c);
    if (c.kind == KernelKind::SplitHead)
        return select_split_head_tile<CleanLogits>(p);
    switch (c.block_m) {
    case 16:
        return select_tiled_group<16, CleanLogits>(p, c);
    case 32:
        return select_tiled_group<32, CleanLogits>(p, c);
    case 64:
        return select_tiled_group<64, CleanLogits>(p, c);
    }
    return nullptr;
}
// The Tensor entry requires ks/ke, so every specialization uses row windows.
Kernel select_kernel(const MqaBf16Params &p, const KernelConfig &config) {
    return p.clean_logits ? select_kernel<true>(p, config) : select_kernel<false>(p, config);
}
#endif
} // namespace
} // namespace dense

namespace compact {
namespace {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
template <int H, int TN, int Waves, bool Pair>
Kernel select_small(const MqaBf16Params &p) {
    constexpr int BM = Waves * (Pair ? 2 : 1);
    if (p.m % BM == 0) {
        if (p.n % TN == 0)
            return small_kv<H, TN, Waves, Pair, true, true>;
        return small_kv<H, TN, Waves, Pair, true, false>;
    }
    if (p.n % TN == 0)
        return small_kv<H, TN, Waves, Pair, false, true>;
    return small_kv<H, TN, Waves, Pair, false, false>;
}

template <int H>
Kernel select_small_n(const MqaBf16Params &p, const KernelConfig &c) {
    if constexpr (H == 8) {
        if (c.paired_queries)
            return select_small<8, 16, 1, true>(p);
    }
    if (c.block_n == 16)
        return select_small<H, 16, 1, false>(p);
    if (c.block_n == 32)
        return select_small<H, 32, 1, false>(p);
    return select_small<H, 64, 1, false>(p);
}

Kernel select_small_kv_kernel(const MqaBf16Params &p, const KernelConfig &c) {
    switch (p.h) {
    case 1:
        return select_small_n<1>(p, c);
    case 2:
        return select_small_n<2>(p, c);
    case 4:
        return select_small_n<4>(p, c);
    case 8:
        return select_small_n<8>(p, c);
    case 16:
        return select_small_n<16>(p, c);
    case 32:
        return select_small_n<32>(p, c);
    case 64:
        return select_small_n<64>(p, c);
    case 128:
        return select_small_n<128>(p, c);
    }
    return nullptr;
}

template <int BM, int BN, int GH>
Kernel select_tiled_tile(const MqaBf16Params &p) {
    if (p.m % BM == 0) {
        if (p.n % BN == 0)
            return tiled<BM, BN, GH, true, true>;
        return tiled<BM, BN, GH, true, false>;
    }
    if (p.n % BN == 0)
        return tiled<BM, BN, GH, false, true>;
    return tiled<BM, BN, GH, false, false>;
}

template <int BM, int GH>
Kernel select_tiled_n(const MqaBf16Params &p, int bn) {
    switch (bn) {
    case 64:
        return select_tiled_tile<BM, 64, GH>(p);
    case 128:
        return select_tiled_tile<BM, 128, GH>(p);
    case 256:
        if constexpr (BM == 64 && GH == 2)
            return nullptr;
        else
            return select_tiled_tile<BM, 256, GH>(p);
    default:
        return nullptr;
    }
}

template <int BM>
Kernel select_tiled_group(const MqaBf16Params &p, const KernelConfig &c) {
    switch (c.group_heads) {
    case 1:
        return select_tiled_n<BM, 1>(p, c.block_n);
    case 2:
        return select_tiled_n<BM, 2>(p, c.block_n);
    case 4:
        if constexpr (BM <= 32)
            return select_tiled_n<BM, 4>(p, c.block_n);
    }
    return nullptr;
}

Kernel select_split_head_tile(const MqaBf16Params &p) {
    if (p.m % 16 == 0) {
        if (p.n % 64 == 0)
            return split_head<16, 64, 4, true, true>;
        return split_head<16, 64, 4, true, false>;
    }
    if (p.n % 64 == 0)
        return split_head<16, 64, 4, false, true>;
    return split_head<16, 64, 4, false, false>;
}

Kernel select_kernel(const MqaBf16Params &p, const KernelConfig &c) {
    if (c.kind == KernelKind::SmallKV)
        return select_small_kv_kernel(p, c);
    if (c.kind == KernelKind::SplitHead)
        return select_split_head_tile(p);
    switch (c.block_m) {
    case 16:
        return select_tiled_group<16>(p, c);
    case 32:
        return select_tiled_group<32>(p, c);
    case 64:
        return select_tiled_group<64>(p, c);
    }
    return nullptr;
}
#endif
} // namespace
} // namespace compact
} // namespace deepgemm::mqa::bf16
