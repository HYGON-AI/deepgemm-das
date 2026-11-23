// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// FP8 host selectors for both output modes; included only by launch.cu.

#pragma once
#include "config.hpp"
#include "static_switch.hpp"
#include "../kernels/dense/m64.hpp"
#include "../kernels/dense/m16.hpp"
#include "../kernels/dense/m16_grouped.hpp"
#include "../kernels/dense/qh.hpp"
#include "../kernels/compact/m64.hpp"
#include "../kernels/compact/m16.hpp"
#include "../kernels/compact/qh.hpp"

namespace deepgemm::mqa::fp8 {
namespace dense {
Kernel mqa_select_m16_grouped(const MqaFp8Params &params);

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
Kernel mqa_select_m64(const MqaFp8Params &p) {
    // NMode=0 means a complete N tile; 4/2/1 are tail tiles with that alignment.
    const int n_mode = p.n % 256 == 0 ? 0 : p.n % 4 == 0 ? 4 : p.n % 2 == 0 ? 2 : 1;
    return MQA_BOOL_SWITCH(p.m % 64 == 0, EvenM, [&] {
        return MQA_SIZE_SWITCH_4(n_mode, NMode, 0, 4, 2, 1, [&] {
            constexpr bool EvenN = NMode == 0;
            constexpr int AlignN = EvenN ? 4 : NMode;
            // The Tensor entry requires ks/ke, so all kernels use row windows.
            if (!p.kv_scale)
                return MQA_BOOL_SWITCH(p.clean_logits, Clean, [&] {
                    return static_cast<Kernel>(mqa_fp8_64x256_kernel<EvenM, EvenN, AlignN,
                                                                     Clean, true>);
                });
            return MQA_BOOL_SWITCH(p.clean_logits, Clean, [&] {
                return static_cast<Kernel>(mqa_fp8_64x256_kernel<EvenM, EvenN, AlignN,
                                                                 Clean, false>);
            });
        });
    });
}

Kernel mqa_select_m16(const MqaFp8Params &p, int bn) {
    if (bn == 128)
        if (Kernel grouped = mqa_select_m16_grouped(p))
            return grouped;
    return MQA_SIZE_SWITCH_3(bn, BN, 64, 128, 256, [&] {
        // Complete tiles and vector stores are separate from KS/KE window coverage.
        const bool aligned = p.m % 16 == 0 && p.n % BN == 0 &&
                             ((reinterpret_cast<uintptr_t>(p.out) |
                               (uint64_t(p.out_stride_m) * sizeof(float))) & 15) == 0;
        // Mode 0: tail, 1: full, 2: specialized full tile.
        int mode = aligned ? 1 : 0;
        if constexpr (BN == 64) {
            if (aligned && p.m == 16 && p.h == 8 && p.n >= 2048 && p.n <= 8192)
                mode = 2;
        } else if constexpr (BN == 128) {
            if (aligned && p.h == 8 && p.n >= 4096 && p.n <= 16384)
                mode = 2;
        }
        return MQA_SIZE_SWITCH_3(mode, Mode, 0, 1, 2, [&] {
            if constexpr (BN == 256 && Mode == 2) {
                return static_cast<Kernel>(nullptr);
            } else {
                constexpr bool FullTile = Mode != 0;
                constexpr bool PreloadH8 = BN == 64 && Mode == 2;
                constexpr bool UseK64 = BN == 128 && Mode == 2;
                return MQA_BOOL_SWITCH(p.clean_logits, Clean, [&] {
                    return MQA_BOOL_SWITCH(!p.kv_scale, UnitScale, [&] {
                        return static_cast<Kernel>(mqa_m16<BN, Clean, UnitScale, FullTile,
                                                          PreloadH8, UseK64>);
                    });
                });
            }
        });
    });
}

Kernel mqa_select_m16_grouped(const MqaFp8Params &p) {
    // Select the dense grouped kernel using shape, strides, alignment,
    // the cleaning flag and scale presence. KS/KE values stay on the device.
    if (p.d != 128 || p.clean_logits != 1 || !p.kv_scale || p.m % 16 != 0 || p.n % 128 != 0 ||
        p.q_stride_m != p.h * 128 || p.q_stride_h != 128 || p.kv_stride_n != 128 ||
        p.weights_stride_m != p.h || p.out_stride_m != p.n ||
        (reinterpret_cast<uintptr_t>(p.out) & 15) != 0)
        return nullptr;
    // Restrict the grouped path to the complete-tile ranges checked by the
    // causal crossover benchmark. The existing M16 selector handles the rest.
    if (p.h == 8 &&
        (((p.m == 16 || p.m == 32) && p.n >= 8192 && p.n <= 32768) ||
         (p.m == 64 && p.n >= 4096 && p.n < 8192))) {
        if (p.n >= 4096 && p.n <= 16384)
            return mqa_m16_grouped<128, true, false, true, false, true, 8>;
        return mqa_m16_grouped<128, true, false, true, false, false, 8>;
    }
    if (p.h == 64 &&
        ((p.m == 16 && p.n >= 32768 && p.n < 49152) ||
         (p.m == 32 && p.n >= 12288 && p.n < 16384)))
        return mqa_m16_grouped<128, true, false, true, false, false, 16>;
    return nullptr;
}

Kernel mqa_select_qh(const MqaFp8Params &p, int bn) {
    return MQA_SIZE_SWITCH_4(p.h, H, 8, 16, 32, 64, [&] {
        return MQA_SIZE_SWITCH_3(bn, BN, 64, 128, 256, [&] {
            const bool full_tile = p.m % (64 / H) == 0 && p.n % BN == 0;
            return MQA_BOOL_SWITCH(full_tile, FullTile, [&] {
                return MQA_BOOL_SWITCH(p.clean_logits, Clean, [&] {
                    return MQA_BOOL_SWITCH(!p.kv_scale, UnitScale, [&] {
                        return static_cast<Kernel>(mqa_qh<H, BN, Clean, UnitScale, FullTile>);
                    });
                });
            });
        });
    });
}
#endif

namespace {
Kernel select_kernel(const MqaFp8Params &p, const KernelConfig &config) {
    Kernel kernel = nullptr;
    switch (config.kind) {
    case KernelKind::M64:
        kernel = mqa_select_m64(p);
        break;
    case KernelKind::M16:
        kernel = mqa_select_m16(p, config.block_n);
        break;
    case KernelKind::QH:
        kernel = mqa_select_qh(p, config.block_n);
        break;
    default:
        return nullptr;
    }
    return kernel;
}
} // namespace
} // namespace dense

namespace compact {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
Kernel mqa_select_m64_compact(const MqaFp8Params &p) {
    const int n_mode = p.n % 256 == 0 ? 0 : p.n % 4 == 0 ? 4 : p.n % 2 == 0 ? 2 : 1;
    return MQA_BOOL_SWITCH(p.m % 64 == 0, EvenM, [&] {
        return MQA_SIZE_SWITCH_4(n_mode, NMode, 0, 4, 2, 1, [&] {
            constexpr bool EvenN = NMode == 0;
            constexpr int AlignN = EvenN ? 4 : NMode;
            return MQA_BOOL_SWITCH(!p.kv_scale, UnitScale, [&] {
                return static_cast<Kernel>(mqa_fp8_64x256_kernel<EvenM, EvenN, AlignN, UnitScale>);
            });
        });
    });
}

// Grouped M16 is available only in the dense selector.
Kernel mqa_select_m16_compact(const MqaFp8Params &p, int bn) {
    return MQA_SIZE_SWITCH_3(bn, BN, 64, 128, 256, [&] {
        // Complete tiles and vector stores are separate from KS/KE window coverage.
        const bool aligned = p.m % 16 == 0 && p.n % BN == 0;
        // Mode 0: tail, 1: full, 2: specialized full tile.
        int mode = aligned ? 1 : 0;
        if constexpr (BN == 64) {
            if (aligned && p.m == 16 && p.h == 8 && p.n >= 2048 && p.n <= 8192)
                mode = 2;
        } else if constexpr (BN == 128) {
            if (aligned && p.h == 8 && p.n >= 4096 && p.n <= 16384)
                mode = 2;
        }
        return MQA_SIZE_SWITCH_3(mode, Mode, 0, 1, 2, [&] {
            if constexpr (BN == 256 && Mode == 2) {
                return static_cast<Kernel>(nullptr);
            } else {
                constexpr bool FullTile = Mode != 0;
                constexpr bool PreloadH8 = BN == 64 && Mode == 2;
                constexpr bool UseK64 = BN == 128 && Mode == 2;
                return MQA_BOOL_SWITCH(!p.kv_scale, UnitScale, [&] {
                    return static_cast<Kernel>(mqa_m16<BN, UnitScale, FullTile, PreloadH8, UseK64>);
                });
            }
        });
    });
}

Kernel mqa_select_qh_compact(const MqaFp8Params &p, int bn) {
    return MQA_SIZE_SWITCH_4(p.h, H, 8, 16, 32, 64, [&] {
        return MQA_SIZE_SWITCH_3(bn, BN, 64, 128, 256, [&] {
            const bool full_tile = p.m % (64 / H) == 0 && p.n % BN == 0;
            return MQA_BOOL_SWITCH(full_tile, FullTile, [&] {
                return MQA_BOOL_SWITCH(!p.kv_scale, UnitScale, [&] {
                    return static_cast<Kernel>(mqa_qh<H, BN, UnitScale, FullTile>);
                });
            });
        });
    });
}
#endif

namespace {
Kernel select_kernel(const MqaFp8Params &p, const KernelConfig &config) {
    Kernel kernel = nullptr;
    switch (config.kind) {
    case KernelKind::M64:
        kernel = mqa_select_m64_compact(p);
        break;
    case KernelKind::M16:
        kernel = mqa_select_m16_compact(p, config.block_n);
        break;
    case KernelKind::QH:
        kernel = mqa_select_qh_compact(p, config.block_n);
        break;
    default:
        return nullptr;
    }
    return kernel;
}
} // namespace
} // namespace compact
} // namespace deepgemm::mqa::fp8
