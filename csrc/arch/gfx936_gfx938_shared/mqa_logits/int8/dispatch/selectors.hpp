// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// INT8 host selectors for both output modes; included only by launch.cu.

#pragma once
#include "config.hpp"
#include "../kernels/dense/m16.hpp"
#include "../kernels/dense/m32.hpp"
#include "../kernels/dense/m64.hpp"
#include "../kernels/dense/small_kv.hpp"
#include "../kernels/compact/m16.hpp"
#include "../kernels/compact/m32.hpp"
#include "../kernels/compact/m64.hpp"
#include "../kernels/compact/small_kv.hpp"

namespace deepgemm::mqa::int8 {
namespace dense {
namespace {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
template <int Heads, bool CleanLogits>
Kernel select_m64(const MqaInt8Params &p) {
    if (p.m % 64 == 0) {
        if (p.n % 256 == 0)
            return m64<Heads, true, true, CleanLogits>;
        return m64<Heads, true, false, CleanLogits>;
    }
    if (p.n % 256 == 0)
        return m64<Heads, false, true, CleanLogits>;
    return m64<Heads, false, false, CleanLogits>;
}

template <int Heads, int BN, int GroupHeads, bool CleanLogits>
Kernel select_m16(const MqaInt8Params &p) {
    if (p.m % 16 == 0) {
        if (p.n % BN == 0)
            return m16<Heads, BN, GroupHeads, true, true, CleanLogits>;
        return m16<Heads, BN, GroupHeads, true, false, CleanLogits>;
    }
    if (p.n % BN == 0)
        return m16<Heads, BN, GroupHeads, false, true, CleanLogits>;
    return m16<Heads, BN, GroupHeads, false, false, CleanLogits>;
}

template <int Heads, int BN, int GroupHeads, bool CleanLogits>
Kernel select_m32(const MqaInt8Params &p) {
    if (p.m % 32 == 0) {
        if (p.n % BN == 0)
            return m32<Heads, BN, GroupHeads, true, true, CleanLogits>;
        return m32<Heads, BN, GroupHeads, true, false, CleanLogits>;
    }
    if (p.n % BN == 0)
        return m32<Heads, BN, GroupHeads, false, true, CleanLogits>;
    return m32<Heads, BN, GroupHeads, false, false, CleanLogits>;
}

template <int Heads, int BM, int GroupHeads, bool CleanLogits>
Kernel select_tiled_n(const MqaInt8Params &p, int bn) {
    static_assert(BM == 16 || BM == 32);
    switch (bn) {
    case 64:
        if constexpr (BM == 16)
            return select_m16<Heads, 64, GroupHeads, CleanLogits>(p);
        else
            return select_m32<Heads, 64, GroupHeads, CleanLogits>(p);
    case 128:
        if constexpr (BM == 16)
            return select_m16<Heads, 128, GroupHeads, CleanLogits>(p);
        else
            return select_m32<Heads, 128, GroupHeads, CleanLogits>(p);
    case 256:
        if constexpr (BM == 16)
            return select_m16<Heads, 256, GroupHeads, CleanLogits>(p);
        else
            return select_m32<Heads, 256, GroupHeads, CleanLogits>(p);
    default:
        return nullptr;
    }
}

template <int H, int TN, int Waves, bool Pair, bool CleanLogits>
Kernel select_small(const MqaInt8Params &p) {
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

template <int Heads, bool CleanLogits>
Kernel select_tiled(const MqaInt8Params &p, const KernelConfig &c) {
    constexpr int GH4 = Heads < 4 ? Heads : 4;
    constexpr int GH2 = Heads < 2 ? Heads : 2;
    if (c.block_m == 16 && c.group_heads == GH4)
        return select_tiled_n<Heads, 16, GH4, CleanLogits>(p, c.block_n);
    if (c.block_m == 32 && c.group_heads == GH2)
        return select_tiled_n<Heads, 32, GH2, CleanLogits>(p, c.block_n);
    // Preserve the 64x256 config while using the standalone M64 kernel.
    if (c.block_m == 64 && c.group_heads == 1 && c.block_n == 256)
        return select_m64<Heads, CleanLogits>(p);
    return nullptr;
}

template <bool CleanLogits>
Kernel select_kernel(const MqaInt8Params &p, const KernelConfig &c) {
    if (c.kind == KernelKind::SmallKV) {
        if (p.h == 8 && c.block_n == 16) {
            if (c.paired_queries && c.waves == 1)
                return select_small<8, 16, 1, true, CleanLogits>(p);
            if (!c.paired_queries && c.waves == 2)
                return select_small<8, 16, 2, false, CleanLogits>(p);
            if (!c.paired_queries && c.waves == 1)
                return select_small<8, 16, 1, false, CleanLogits>(p);
        }
        if (p.h == 64 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<64, 16, 1, false, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<64, 32, 1, false, CleanLogits>(p);
        }
        if (p.h == 16 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<16, 16, 1, false, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<16, 32, 1, false, CleanLogits>(p);
        }
        if (p.h == 32 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<32, 16, 1, false, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<32, 32, 1, false, CleanLogits>(p);
        }
        if (p.h == 128 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<128, 16, 1, false, CleanLogits>(p);
            if (c.block_n == 32)
                return select_small<128, 32, 1, false, CleanLogits>(p);
        }
        return nullptr;
    }
    if (c.kind != KernelKind::Tiled)
        return nullptr;
    switch (p.h) {
    case 1:
        return select_tiled<1, CleanLogits>(p, c);
    case 2:
        return select_tiled<2, CleanLogits>(p, c);
    case 4:
        return select_tiled<4, CleanLogits>(p, c);
    case 8:
        return select_tiled<8, CleanLogits>(p, c);
    case 16:
        return select_tiled<16, CleanLogits>(p, c);
    case 32:
        return select_tiled<32, CleanLogits>(p, c);
    case 64:
        return select_tiled<64, CleanLogits>(p, c);
    case 128:
        return select_tiled<128, CleanLogits>(p, c);
    default:
        return nullptr;
    }
}
// The Tensor entry requires ks/ke, so every specialization uses row windows.
Kernel select_kernel(const MqaInt8Params &p, const KernelConfig &config) {
    return p.clean_logits ? select_kernel<true>(p, config) : select_kernel<false>(p, config);
}
#endif
} // namespace
} // namespace dense

namespace compact {
namespace {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
template <int Heads>
Kernel select_m64(const MqaInt8Params &p) {
    if (p.m % 64 == 0) {
        if (p.n % 256 == 0)
            return m64<Heads, true, true>;
        return m64<Heads, true, false>;
    }
    if (p.n % 256 == 0)
        return m64<Heads, false, true>;
    return m64<Heads, false, false>;
}

template <int Heads, int BN, int GroupHeads>
Kernel select_m16(const MqaInt8Params &p) {
    if (p.m % 16 == 0) {
        if (p.n % BN == 0)
            return m16<Heads, BN, GroupHeads, true, true>;
        return m16<Heads, BN, GroupHeads, true, false>;
    }
    if (p.n % BN == 0)
        return m16<Heads, BN, GroupHeads, false, true>;
    return m16<Heads, BN, GroupHeads, false, false>;
}

template <int Heads, int BN, int GroupHeads>
Kernel select_m32(const MqaInt8Params &p) {
    if (p.m % 32 == 0) {
        if (p.n % BN == 0)
            return m32<Heads, BN, GroupHeads, true, true>;
        return m32<Heads, BN, GroupHeads, true, false>;
    }
    if (p.n % BN == 0)
        return m32<Heads, BN, GroupHeads, false, true>;
    return m32<Heads, BN, GroupHeads, false, false>;
}

template <int Heads, int BM, int GroupHeads>
Kernel select_tiled_n(const MqaInt8Params &p, int bn) {
    static_assert(BM == 16 || BM == 32);
    switch (bn) {
    case 64:
        if constexpr (BM == 16)
            return select_m16<Heads, 64, GroupHeads>(p);
        else
            return select_m32<Heads, 64, GroupHeads>(p);
    case 128:
        if constexpr (BM == 16)
            return select_m16<Heads, 128, GroupHeads>(p);
        else
            return select_m32<Heads, 128, GroupHeads>(p);
    case 256:
        if constexpr (BM == 16)
            return select_m16<Heads, 256, GroupHeads>(p);
        else
            return select_m32<Heads, 256, GroupHeads>(p);
    default:
        return nullptr;
    }
}

template <int H, int TN, int Waves, bool Pair>
Kernel select_small(const MqaInt8Params &p) {
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

template <int Heads>
Kernel select_tiled(const MqaInt8Params &p, const KernelConfig &c) {
    constexpr int GH4 = Heads < 4 ? Heads : 4;
    constexpr int GH2 = Heads < 2 ? Heads : 2;
    if (c.block_m == 16 && c.group_heads == GH4)
        return select_tiled_n<Heads, 16, GH4>(p, c.block_n);
    if (c.block_m == 32 && c.group_heads == GH2)
        return select_tiled_n<Heads, 32, GH2>(p, c.block_n);
    // Preserve the 64x256 config while using the standalone M64 kernel.
    if (c.block_m == 64 && c.group_heads == 1 && c.block_n == 256)
        return select_m64<Heads>(p);
    return nullptr;
}

Kernel select_kernel(const MqaInt8Params &p, const KernelConfig &c) {
    if (c.kind == KernelKind::SmallKV) {
        if (p.h == 8 && c.block_n == 16) {
            if (c.paired_queries && c.waves == 1)
                return select_small<8, 16, 1, true>(p);
            if (!c.paired_queries && c.waves == 2)
                return select_small<8, 16, 2, false>(p);
            if (!c.paired_queries && c.waves == 1)
                return select_small<8, 16, 1, false>(p);
        }
        if (p.h == 64 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<64, 16, 1, false>(p);
            if (c.block_n == 32)
                return select_small<64, 32, 1, false>(p);
        }
        if (p.h == 16 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<16, 16, 1, false>(p);
            if (c.block_n == 32)
                return select_small<16, 32, 1, false>(p);
        }
        if (p.h == 32 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<32, 16, 1, false>(p);
            if (c.block_n == 32)
                return select_small<32, 32, 1, false>(p);
        }
        if (p.h == 128 && c.waves == 1 && !c.paired_queries) {
            if (c.block_n == 16)
                return select_small<128, 16, 1, false>(p);
            if (c.block_n == 32)
                return select_small<128, 32, 1, false>(p);
        }
        return nullptr;
    }
    if (c.kind != KernelKind::Tiled)
        return nullptr;
    switch (p.h) {
    case 1:
        return select_tiled<1>(p, c);
    case 2:
        return select_tiled<2>(p, c);
    case 4:
        return select_tiled<4>(p, c);
    case 8:
        return select_tiled<8>(p, c);
    case 16:
        return select_tiled<16>(p, c);
    case 32:
        return select_tiled<32>(p, c);
    case 64:
        return select_tiled<64>(p, c);
    case 128:
        return select_tiled<128>(p, c);
    default:
        return nullptr;
    }
}
#endif
} // namespace
} // namespace compact
} // namespace deepgemm::mqa::int8
