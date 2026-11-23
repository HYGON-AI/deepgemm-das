// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "config.hpp"
#include <cstddef>
#include <limits>

namespace deepgemm::mqa::fp8 {

// Retune flags: M16/M32/M64, N64/N128/N256, packed rows, output16,
// scale, clean, both bounds, N4/N2, short-M32-tail, M64 offsets, M8/M4/M2, N16/N32,
// Q16, K16 and exactly one of the measured dense/guarded-stride layouts.
static inline uint32_t auto_retune_flags(const common::MqaParams &p) {
    const bool dense = p.h >= 1 && p.h <= 128 && p.q_stride_m == p.h * 128 &&
        p.q_stride_h == 128 && p.kv_stride_n == 128 &&
        p.weights_stride_m == p.h && p.out_stride_m == p.n;
    const auto out_mod = reinterpret_cast<uintptr_t>(p.out) & 15;
    const bool known_layout = p.h >= 1 && p.h <= 128 && p.n >= 0 && p.n <= INT32_MAX &&
        ((p.weights_stride_m == p.h && p.out_stride_m == p.n && out_mod == 0) ||
         (p.weights_stride_m == p.h + 5 && p.out_stride_m == p.n + 8 && out_mod == 4));
    return (uint32_t(p.m % 16 == 0) << 0) | (uint32_t(p.m % 32 == 0) << 1) |
        (uint32_t(p.m % 64 == 0) << 2) | (uint32_t(p.n % 64 == 0) << 3) |
        (uint32_t(p.n % 128 == 0) << 4) | (uint32_t(p.n % 256 == 0) << 5) |
        (uint32_t(dense) << 6) | (uint32_t((reinterpret_cast<uintptr_t>(p.out) & 15) == 0) << 7) |
        (uint32_t(p.kv_scale != nullptr) << 8) | (uint32_t(p.clean_logits == 1) << 9) |
        (uint32_t(p.ks != nullptr && p.ke != nullptr) << 10) |
        (uint32_t(p.n % 4 == 0) << 11) | (uint32_t(p.n % 2 == 0) << 12) |
        (uint32_t(p.m % 32 > 0 && p.m % 32 <= 16) << 13) |
        (uint32_t(p.weights_stride_m >= p.h && p.weights_stride_m <= (int64_t(1) << 23)) << 14) |
        (uint32_t(p.m % 8 == 0) << 15) | (uint32_t(p.m % 4 == 0) << 16) |
        (uint32_t(p.m % 2 == 0) << 17) | (uint32_t(p.n % 16 == 0) << 18) |
        (uint32_t(p.n % 32 == 0) << 19) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.q) & 15) == 0) << 20) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.kv) & 15) == 0) << 21) |
        (uint32_t(known_layout) << 22);
}


// Frozen selection after the 2026-09-27 causal-mask measurements. M/N bounds
// are inclusive. The first matching row within an H/clean group wins. Untuned
// regions retain the previous selection or the M64:256 fallback.
// A true layout flag requires D=128, N divisible by 256, and tightly packed
// Q, K, weights, and output rows. Rules tuned for grouped M16 also require
// KV scales and 16-byte output alignment. The dense selector checks
// grouped eligibility separately. Compact uses its own M16 selector.
// KS/KE values remain on the device.
struct AutoRule {
    int32_t m_min, m_max, n_min, n_max;
    KernelKind kind;
    uint16_t block_n;
    bool aligned_contiguous;
    bool requires_grouped_m16 = false;
    // Measured subregions retain the original row constraints and fallback.
    uint32_t tune_mask = 0, tune_value = 0;
    int64_t tune_area_min = 0, tune_area_max = std::numeric_limits<int64_t>::max();
};

static constexpr AutoRule kAutoRules[] = {
    // H=1, clean_logits=0
    // fp8_refined_010; original default region
    {1, 32, 1, 24, KernelKind::M16, 64, false, false, 7342016, 7341504, 1, 192},
    // fp8_refined_012; original default region
    {1, 8, 25, 416, KernelKind::M16, 64, false, false, 7342016, 7341504, 25, 416},
    // fp8_refined_013; original default region
    {1, 256, 53, 950272, KernelKind::M16, 128, false, false, 7342016, 7341504, 417, 950272},
    // H=1, clean_logits=1
    // fp8_refined_011; original default region
    {1, 32, 1, 63488, KernelKind::M16, 64, false, false, 7342016, 7342016, 1, 9217},
    // fp8_refined_014; original default region
    {1, 3, 65536, 851968, KernelKind::M16, 256, false, false, 7342048, 7342048, 131072, 851968},
    // fp8_refined_015; original default region
    {1, 1, 851969, 1048576, KernelKind::M16, 256, false, false, 7342016, 7342016, 851969, 1048576},
    // H=2, clean_logits=0
    // fp8_refined_016; original default region
    {1, 16, 1, 15870, KernelKind::M16, 64, false, false, 7342016, 7341504, 1, 63488},
    // fp8_refined_019; original default region
    {1, 5, 15873, 720896, KernelKind::M16, 256, false, false, 7342016, 7341504, 63489, 720896},
    // fp8_refined_024; original default region
    {6, 2046, 32, 22235, KernelKind::M16, 64, false, false, 7342016, 7341504, 63490, 155645},
    // fp8_refined_025; original default region
    {6, 16, 22236, 102985, KernelKind::M16, 128, false, false, 7342016, 7341504, 155652, 720896},
    // H=2, clean_logits=1
    // fp8_refined_017; original default region
    {1, 24, 1, 65536, KernelKind::M16, 128, false, false, 7342016, 7341824, 24, 524288},
    // fp8_refined_018; original default region
    {1, 512, 1, 1152, KernelKind::M16, 64, false, false, 7342016, 7342016, 1, 53248},
    // fp8_refined_020; original default region
    {1, 16, 38910, 622592, KernelKind::M16, 256, false, false, 7342016, 7342016, 311297, 622592},
    // fp8_refined_021; original default region
    {1, 20, 38913, 1048576, KernelKind::M16, 256, false, false, 7342016, 7342016, 622593, 4456448},
    // fp8_refined_023; original default region
    {1, 1024, 53250, 311296, KernelKind::M16, 128, false, false, 7342016, 7342016, 53249, 311296},
    // H=4, clean_logits=0
    // fp8_refined_026; original default region
    {1, 16, 1, 10240, KernelKind::M16, 64, false, false, 7342016, 7341504, 1, 81920},
    // fp8_refined_030; original default region
    {1, 8, 10241, 118784, KernelKind::M16, 128, false, false, 7342016, 7341504, 10241, 118784},
    // fp8_refined_033; original default region
    {1, 8, 24577, 425984, KernelKind::M16, 256, false, false, 7342016, 7341504, 118785, 425984},
    // fp8_refined_036; original default region
    {1, 2048, 53250, 622592, KernelKind::M16, 256, false, false, 7342016, 7341504, 425985, 622592},
    // H=4, clean_logits=1
    // fp8_refined_027; original default region
    {1, 1536, 1, 7168, KernelKind::M16, 64, false, false, 7342016, 7342016, 1, 73728},
    // fp8_refined_029; original default region
    {1, 8, 7169, 14336, KernelKind::M16, 64, false, false, 7342016, 7342016, 7169, 77824},
    // fp8_refined_031; original default region
    {1, 8, 14334, 311296, KernelKind::M16, 128, false, false, 7342016, 7342016, 106497, 311296},
    // fp8_refined_032; original default region
    {1, 4, 14337, 77824, KernelKind::M16, 128, false, false, 7342016, 7342016, 14337, 77824},
    // H=8, clean_logits=0
    {16, 16, 32768, 1048576, KernelKind::M16, 256, true},
    {16, 16, 12288, 32767, KernelKind::M16, 128, true},
    {16, 16, 4096, 12287, KernelKind::M16, 64, true},
    {16, 16, 2048, 4095, KernelKind::QH, 64, true},
    {32, 32, 12288, 1048576, KernelKind::M16, 256, true},
    {32, 32, 6144, 12287, KernelKind::M16, 128, true},
    {32, 32, 5888, 6143, KernelKind::M16, 64, true},
    {32, 32, 2048, 4095, KernelKind::QH, 64, true},
    {64, 64, 16384, 1048576, KernelKind::M64, 256, true},
    {64, 64, 8192, 16383, KernelKind::M16, 256, true},
    {64, 64, 3072, 8191, KernelKind::M16, 128, true},
    {64, 64, 2048, 3071, KernelKind::M16, 64, true},
    {33, 63, 1449, 2048, KernelKind::QH, 64, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    {129, 512, 182, 1448, KernelKind::M16, 128, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // fp8_refined_037; original default region
    {1, 33, 1, 63, KernelKind::QH, 64, false, false, 7342016, 7341504, 1, 896},
    // fp8_refined_039; original default region
    {1, 6, 65537, 1048576, KernelKind::QH, 256, false, false, 7342016, 7341504, 290834, 6291456},
    // fp8_refined_040; original default region
    {1, 4, 65537, 290833, KernelKind::QH, 256, false, false, 7342016, 7341504, 65537, 290833},
    // fp8_refined_043; original default region
    {7, 10, 65538, 1048576, KernelKind::M16, 128, false, false, 7342016, 7341504, 458766, 10485760},
    // H=8, clean_logits=1
    {16, 16, 32768, 32768, KernelKind::M16, 128, true, true},
    {32, 32, 16384, 24576, KernelKind::M16, 128, true, true},
    {16, 16, 3072, 4095, KernelKind::M16, 64, true},
    {16, 16, 2048, 3071, KernelKind::QH, 64, true},
    {64, 64, 8192, 32767, KernelKind::M16, 256, true},
    {64, 64, 3072, 8191, KernelKind::M16, 128, true},
    {64, 64, 2048, 3071, KernelKind::M16, 64, true},
    {16, 16, 32768, 1048576, KernelKind::M16, 256, true},
    {16, 16, 12288, 32767, KernelKind::M16, 128, true},
    {16, 16, 4096, 12287, KernelKind::M16, 64, true},
    {32, 32, 12288, 1048576, KernelKind::M16, 256, true},
    {32, 32, 6144, 12287, KernelKind::M16, 128, true},
    {32, 32, 4608, 6143, KernelKind::M16, 64, true},
    {32, 32, 2048, 4095, KernelKind::QH, 64, true},
    {64, 64, 16384, 1048576, KernelKind::M64, 256, true},
    {33, 63, 1449, 2048, KernelKind::QH, 64, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    {129, 512, 182, 1448, KernelKind::M16, 64, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // fp8_refined_038; original default region
    {1, 512, 1, 63, KernelKind::QH, 64, false, false, 7342016, 7342016, 1, 3072},
    // fp8_refined_042; original default region
    {1, 4, 65537, 270385, KernelKind::QH, 256, false, false, 7342016, 7342016, 65537, 270385},
    // H=16, clean_logits=0
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {33, 64, 1449, 2048, KernelKind::QH, 128, false},
    {33, 64, 2049, 4096, KernelKind::QH, 256, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    // fp8_refined_001; unit-step boundary validated
    {5, 8, 8193, 29490, KernelKind::QH, 128, false, false, 7342016, 7341504, 40965, 147450},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    {129, 512, 182, 1448, KernelKind::M16, 128, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // fp8_refined_045; original default region
    {1, 35, 1, 63, KernelKind::QH, 64, false, false, 7342016, 7341504, 1, 189},
    // fp8_refined_047; original default region
    {1, 4, 65537, 262406, KernelKind::QH, 256, false, false, 7342016, 7341504, 65537, 262406},
    // fp8_refined_048; original default region
    {1, 30, 65537, 1048576, KernelKind::M16, 256, false, false, 7342016, 7341504, 282659, 33554432},
    // fp8_refined_051; original default region
    {1, 4, 65602, 282658, KernelKind::M16, 128, false, false, 7342016, 7341504, 262407, 282658},
    // H=16, clean_logits=1
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {129, 512, 182, 1448, KernelKind::M16, 128, false},
    {33, 64, 2049, 8192, KernelKind::M16, 256, false},
    {65, 256, 1449, 8192, KernelKind::M16, 256, false},
    {33, 64, 1449, 2048, KernelKind::QH, 64, false},
    {1, 32, 64, 8192, KernelKind::QH, 64, false},
    {1, 4, 8193, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::M16, 128, false},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    // fp8_refined_046; original default region
    {1, 2046, 1, 63, KernelKind::QH, 64, false, false, 7342016, 7342016, 1, 65472},
    // fp8_refined_049; original default region
    {1, 4, 65537, 1048576, KernelKind::QH, 256, false, false, 7342016, 7342016, 93291, 4194304},
    // fp8_refined_050; original default region
    {1, 1, 65537, 93290, KernelKind::QH, 128, false, false, 7342016, 7342016, 65537, 93290},
    // fp8_refined_052; original default region
    {5, 29, 65537, 1048576, KernelKind::M16, 256, false, false, 7342016, 7342016, 327685, 30408704},
    // H=32, clean_logits=0
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {1, 4, 1, 4, KernelKind::QH, 64, false},
    {33, 64, 1449, 2048, KernelKind::QH, 256, false},
    {33, 48, 2049, 4096, KernelKind::QH, 256, false},
    {1, 64, 64, 1448, KernelKind::QH, 64, false},
    {65, 512, 64, 181, KernelKind::QH, 64, false},
    {65, 512, 182, 1448, KernelKind::QH, 256, false},
    {1, 4, 1449, 8192, KernelKind::QH, 64, false},
    {5, 32, 1449, 8192, KernelKind::QH, 128, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    {1, 4, 8193, 65536, KernelKind::QH, 128, false},
    {5, 32, 8193, 32768, KernelKind::QH, 256, false},
    // fp8_refined_002; unit-step boundary validated
    {5, 6, 32769, 65536, KernelKind::QH, 256, false, false, 7342016, 7341504, 163845, 393216},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    // fp8_refined_053; original default region
    {1, 48, 1, 61, KernelKind::QH, 64, false, false, 7342016, 7341504, 5, 183},
    // fp8_refined_056; original default region
    {1, 514, 182, 290833, KernelKind::QH, 256, false, false, 7342016, 7341504, 65537, 290833},
    // fp8_refined_057; original default region
    {1, 6, 65537, 1048576, KernelKind::QH, 256, false, false, 7342016, 7341504, 290834, 6291456},
    // fp8_refined_059; original default region
    {7, 12, 32769, 1048576, KernelKind::M16, 256, false, false, 7342016, 7341504, 458766, 35651584},
    // H=32, clean_logits=1
    {16, 16, 65536, 524288, KernelKind::M16, 256, true},
    {32, 32, 65536, 262144, KernelKind::M16, 256, true},
    {1, 4, 1, 4, KernelKind::QH, 64, false},
    {33, 64, 1449, 2048, KernelKind::QH, 128, false},
    {33, 48, 2049, 4096, KernelKind::QH, 128, false},
    {1, 128, 64, 1448, KernelKind::QH, 64, false},
    {129, 512, 64, 181, KernelKind::QH, 64, false},
    // fp8_refined_003; unit-step boundary validated
    {510, 512, 768, 1448, KernelKind::M16, 256, false, false, 7342016, 7342016, 391680, 741376},
    {129, 512, 182, 1448, KernelKind::QH, 256, false},
    {1, 4, 1449, 32768, KernelKind::QH, 64, false},
    {1, 4, 32769, 65536, KernelKind::QH, 128, false},
    {5, 32, 1449, 8192, KernelKind::QH, 128, false},
    // fp8_refined_004; unit-step boundary validated
    {5, 9, 8193, 18432, KernelKind::QH, 128, false, false, 7342016, 7342016, 40965, 92160},
    {5, 32, 8193, 65536, KernelKind::M16, 256, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // fp8_refined_054; original default region
    {1, 515, 1, 61, KernelKind::QH, 64, false, false, 7342016, 7342016, 5, 515},
    // fp8_refined_055; original default region
    {1, 515, 156, 139264, KernelKind::QH, 256, false, false, 7342016, 7342016, 65537, 139264},
    // fp8_refined_058; original default region
    {1, 6, 65537, 1048576, KernelKind::QH, 256, false, false, 7342016, 7342016, 139265, 5242880},
    // fp8_refined_060; original default region
    {7, 32, 65537, 1048576, KernelKind::M16, 256, false, false, 7342016, 7342016, 1048586, 16777216},
    // fp8_refined_061; original default region
    {7, 8, 65537, 149797, KernelKind::QH, 256, false, false, 7342016, 7342016, 458759, 1048579},
    // H=64, clean_logits=0
    {16, 16, 40960, 1048576, KernelKind::M16, 256, true},
    {16, 16, 4096, 40959, KernelKind::QH, 256, true},
    {16, 16, 2048, 4095, KernelKind::QH, 128, true},
    {32, 32, 14336, 1048576, KernelKind::M16, 256, true},
    {32, 32, 2048, 14335, KernelKind::QH, 256, true},
    {64, 64, 24576, 1048576, KernelKind::M64, 256, true},
    {64, 64, 8192, 24575, KernelKind::M16, 256, true},
    {64, 64, 2048, 8191, KernelKind::QH, 256, true},
    {33, 64, 1449, 4096, KernelKind::QH, 256, false},
    {1, 4, 64, 8192, KernelKind::QH, 64, false},
    {1, 1, 8193, 65536, KernelKind::QH, 128, false},
    {2, 4, 8193, 65536, KernelKind::QH, 256, false},
    {5, 32, 64, 1448, KernelKind::QH, 64, false},
    {5, 32, 1449, 32768, KernelKind::QH, 256, false},
    // fp8_refined_006; unit-step boundary validated
    {5, 14, 32769, 65536, KernelKind::QH, 256, false, false, 7342016, 7341504, 163845, 507904},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 512, 64, 181, KernelKind::QH, 64, false},
    {33, 512, 182, 1448, KernelKind::QH, 256, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // fp8_refined_062; original default region
    {1, 1536, 1, 153, KernelKind::QH, 64, false, false, 7866304, 7341504, 1, 235008},
    // fp8_refined_064; original default region
    {1, 130, 1449, 480239, KernelKind::QH, 256, false, false, 7342016, 7341504, 65538, 480239},
    // fp8_refined_066; original default region
    {1, 10, 131072, 1048576, KernelKind::QH, 256, false, false, 7342017, 7341504, 480240, 10485760},
    // fp8_refined_067; original default region
    {11, 19, 8192, 1048576, KernelKind::M16, 256, false, false, 7342017, 7341504, 901251, 32505856},
    // H=64, clean_logits=1
    {16, 16, 32768, 45056, KernelKind::M16, 128, true, true},
    {32, 32, 12288, 12544, KernelKind::M16, 128, true, true},
    {16, 16, 40960, 1048576, KernelKind::M16, 256, true},
    {16, 16, 4096, 40959, KernelKind::QH, 256, true},
    {16, 16, 2048, 4095, KernelKind::QH, 128, true},
    {32, 32, 14336, 1048576, KernelKind::M16, 256, true},
    {32, 32, 2048, 14335, KernelKind::QH, 256, true},
    {64, 64, 24576, 1048576, KernelKind::M64, 256, true},
    {64, 64, 8192, 24575, KernelKind::M16, 256, true},
    {64, 64, 2048, 8191, KernelKind::QH, 256, true},
    {33, 64, 1449, 4096, KernelKind::QH, 256, false},
    {1, 4, 64, 8192, KernelKind::QH, 64, false},
    {1, 1, 8193, 65536, KernelKind::QH, 128, false},
    {2, 4, 8193, 65536, KernelKind::QH, 256, false},
    {5, 32, 64, 1448, KernelKind::QH, 64, false},
    // fp8_refined_007; unit-step boundary validated
    {5, 8, 1449, 2662, KernelKind::QH, 64, false, false, 7342016, 7342016, 7245, 13312},
    {5, 32, 1449, 32768, KernelKind::QH, 256, false},
    // fp8_refined_009; unit-step boundary validated
    {5, 10, 32769, 65536, KernelKind::QH, 256, false, false, 7342016, 7342016, 163845, 425984},
    {5, 32, 32769, 65536, KernelKind::M16, 256, false},
    {33, 512, 64, 181, KernelKind::QH, 64, false},
    {33, 512, 182, 1448, KernelKind::QH, 256, false},
    {33, 128, 1449, 8192, KernelKind::M16, 256, false},
    // fp8_refined_063; original default region
    {1, 26, 1, 63, KernelKind::QH, 64, false, false, 7342016, 7342016, 1, 104},
    // fp8_refined_065; original default region
    {1, 10, 65537, 1048576, KernelKind::QH, 256, false, false, 7342016, 7342016, 65538, 10485760},
    // fp8_refined_068; original default region
    {11, 26, 131072, 1048576, KernelKind::M16, 256, false, false, 7342016, 7342016, 1441792, 27262976},
    // fp8_refined_069; original default region
    {27, 32, 1, 63, KernelKind::M16, 64, false, false, 7342016, 7342016, 27, 2016},
    // fp8_refined_070; original default region
    {27, 31, 262144, 1048576, KernelKind::M16, 256, false, false, 7342017, 7342016, 7077888, 32505856},
    // H=128, clean_logits=0
    // fp8_refined_071; original default region
    {1, 32, 1, 475136, KernelKind::M16, 128, false, false, 7342016, 7341504, 1, 475136},
    // fp8_refined_073; original default region
    {1, 32, 14849, 1048576, KernelKind::M16, 256, false, false, 7342016, 7341504, 475137, 16777216},
    // H=128, clean_logits=1
    // fp8_refined_072; original default region
    {1, 32, 1, 475136, KernelKind::M16, 128, false, false, 7342016, 7342016, 1, 475136},
    // fp8_refined_074; original default region
    {1, 32, 14849, 1048576, KernelKind::M16, 256, false, false, 7342016, 7342016, 475137, 16777216},
};

static constexpr uint16_t kAutoRuleOffsets[] = {0, 3, 6, 10, 15, 19, 23, 48, 75, 93, 110, 129, 149, 172, 199, 201, 203};
static_assert(kAutoRuleOffsets[16] == sizeof(kAutoRules) / sizeof(kAutoRules[0]));

static inline void mqa_legacy_auto_config(const MqaFp8Params& p, KernelConfig& c) {
    c = {KernelKind::M64, 256};
    if (p.clean_logits > 1) return;
    unsigned h_group;
    switch (p.h) {
    case 1: h_group = 0; break;
    case 2: h_group = 1; break;
    case 4: h_group = 2; break;
    case 8: h_group = 3; break;
    case 16: h_group = 4; break;
    case 32: h_group = 5; break;
    case 64: h_group = 6; break;
    case 128: h_group = 7; break;
    default: return;
    }
    const unsigned group = 2 * h_group + p.clean_logits;
    const bool aligned_contiguous = p.d == 128 && p.n % 256 == 0 &&
        p.q_stride_m == p.h * 128 && p.q_stride_h == 128 &&
        p.kv_stride_n == 128 && p.weights_stride_m == p.h &&
        p.out_stride_m == p.n;
    const uint32_t tune_flags = auto_retune_flags(p);
    const int64_t tune_area = p.m >= 0 && p.n >= 0 && p.m <= INT32_MAX && p.n <= INT32_MAX ? p.m * p.n : -1;
    for (uint16_t i = kAutoRuleOffsets[group]; i < kAutoRuleOffsets[group + 1]; ++i) {
        const AutoRule& r = kAutoRules[i];
        if (p.m < r.m_min || p.m > r.m_max || p.n < r.n_min || p.n > r.n_max)
            continue;
        if ((tune_flags & r.tune_mask) != r.tune_value ||
            tune_area < r.tune_area_min || tune_area > r.tune_area_max)
            continue;
        if (r.aligned_contiguous && !aligned_contiguous) continue;
        if (r.requires_grouped_m16 &&
            (!p.kv_scale || (reinterpret_cast<uintptr_t>(p.out) & 15) != 0)) continue;
        c = {r.kind, r.block_n};
        return;
    }
}


// Exact-shape overrides for aligned, packed inputs. Compact lookup uses
// clean_logits=0 and skips the dense-only table below. Other shapes and
// layouts keep the configuration selected by the legacy table.
struct RetunedRule {
    int64_t m, n, h;
    bool clean, unit_scale;
    KernelKind previous_kind;
    uint32_t previous_n;
    KernelKind kind;
    uint32_t block_n;
};
static constexpr RetunedRule kRetunedRules[] = {
    {1, 65, 128, true, false, KernelKind::M16, 128, KernelKind::M16, 64},
    {16, 31, 16, false, false, KernelKind::M64, 256, KernelKind::QH, 64},
    {16, 31, 32, false, false, KernelKind::M64, 256, KernelKind::QH, 64},
    {16, 31, 64, true, false, KernelKind::M64, 256, KernelKind::QH, 64},
    {16, 31, 128, true, false, KernelKind::M16, 128, KernelKind::M16, 64},
    {16, 512, 1, false, false, KernelKind::M16, 128, KernelKind::M16, 64},
    {16, 512, 128, true, false, KernelKind::M16, 128, KernelKind::M16, 64},
    {16, 512, 128, false, false, KernelKind::M16, 128, KernelKind::M16, 64},
    {17, 65, 128, true, false, KernelKind::M16, 128, KernelKind::M16, 64},
    {31, 32768, 1, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {31, 32768, 1, false, false, KernelKind::M64, 256, KernelKind::M16, 128},
    {31, 32768, 2, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {31, 32768, 2, false, false, KernelKind::M64, 256, KernelKind::M16, 128},
    {31, 32768, 4, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {31, 32768, 4, false, false, KernelKind::M64, 256, KernelKind::M16, 128},
    {31, 32768, 8, true, false, KernelKind::M16, 128, KernelKind::M16, 256},
    {31, 32768, 16, true, false, KernelKind::M16, 128, KernelKind::M16, 256},
    {31, 32768, 32, false, false, KernelKind::QH, 256, KernelKind::M16, 128},
    {64, 384, 64, true, false, KernelKind::QH, 256, KernelKind::QH, 128},
    {64, 384, 64, false, false, KernelKind::QH, 256, KernelKind::QH, 128},
    {64, 384, 128, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {64, 384, 128, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {64, 16384, 1, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {64, 16384, 2, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {64, 16384, 4, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {64, 16384, 16, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {64, 16384, 32, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {64, 16384, 128, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {127, 511, 32, true, false, KernelKind::QH, 64, KernelKind::QH, 128},
    {127, 511, 32, false, false, KernelKind::QH, 256, KernelKind::QH, 128},
    {127, 511, 128, true, false, KernelKind::M64, 256, KernelKind::M16, 64},
    {127, 511, 128, false, false, KernelKind::M64, 256, KernelKind::M16, 64},
    {128, 128, 128, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {128, 128, 128, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {128, 256, 32, false, false, KernelKind::QH, 256, KernelKind::QH, 64},
    {128, 256, 64, true, false, KernelKind::QH, 256, KernelKind::QH, 128},
    {128, 256, 64, false, false, KernelKind::QH, 256, KernelKind::QH, 128},
    {128, 256, 128, true, false, KernelKind::M64, 256, KernelKind::M16, 64},
    {128, 256, 128, false, false, KernelKind::M64, 256, KernelKind::M16, 64},
    {128, 4096, 1, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {128, 4096, 2, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {128, 4096, 2, false, false, KernelKind::M64, 256, KernelKind::M16, 128},
    {128, 4096, 4, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {128, 4096, 4, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {128, 4096, 128, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {128, 4096, 128, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {512, 1024, 1, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {512, 1024, 1, false, false, KernelKind::M64, 256, KernelKind::M16, 128},
    {512, 1024, 2, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {512, 1024, 2, false, false, KernelKind::M64, 256, KernelKind::M16, 128},
    {512, 1024, 4, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {512, 1024, 4, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {512, 1024, 8, true, false, KernelKind::M16, 64, KernelKind::M16, 256},
    {512, 1024, 8, false, false, KernelKind::M16, 128, KernelKind::M16, 256},
    {512, 1024, 16, true, false, KernelKind::M16, 128, KernelKind::M16, 256},
    {512, 1024, 16, false, false, KernelKind::M16, 128, KernelKind::M16, 256},
    {512, 1024, 32, false, false, KernelKind::QH, 256, KernelKind::M16, 256},
    {512, 1024, 128, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {512, 1024, 128, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {16, 4096, 8, false, false, KernelKind::M16, 64, KernelKind::QH, 64},
    {65, 16386, 64, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {65, 16386, 64, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {65, 16387, 64, true, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {65, 16387, 64, false, false, KernelKind::M64, 256, KernelKind::M16, 256},
    {16, 4096, 8, false, true, KernelKind::M16, 64, KernelKind::QH, 64},
    {65, 4099, 32, false, true, KernelKind::M16, 256, KernelKind::QH, 256},
    {65, 16387, 32, false, true, KernelKind::M64, 256, KernelKind::M16, 128},
    {64, 4096, 8, true, true, KernelKind::M16, 128, KernelKind::M16, 256},
    {16, 12288, 8, true, true, KernelKind::M16, 128, KernelKind::M16, 256},
    {65, 4099, 32, true, true, KernelKind::M16, 256, KernelKind::QH, 256},
};

// Additional overrides for dense output, tuned with builtin waitcnt(0).
static constexpr RetunedRule kDenseRetunedRules[] = {
    {16, 12288, 8, false, false, KernelKind::M16, 128, KernelKind::QH, 128},
    {16, 12288, 8, false, true, KernelKind::M16, 128, KernelKind::QH, 128},
    {64, 4096, 8, false, true, KernelKind::M16, 128, KernelKind::QH, 128},
};

template <std::size_t N>
static inline void mqa_apply_retuned_rules(const MqaFp8Params &p, KernelConfig &c,
                                          const RetunedRule (&rules)[N]) {
    for (const auto &r : rules) {
        if (p.m == r.m && p.n == r.n && p.h == r.h &&
            bool(p.clean_logits) == r.clean && (!p.kv_scale) == r.unit_scale &&
            c.kind == r.previous_kind && c.block_n == r.previous_n) {
            c = {r.kind, r.block_n};
            return;
        }
    }
}

static inline void mqa_auto_config(const MqaFp8Params &p, KernelConfig &c, bool dense_output) {
    mqa_legacy_auto_config(p, c);
    if (p.d != 128 || p.h < 1 || p.h > 128 || p.clean_logits > 1 || !p.ks || !p.ke ||
        p.q_stride_m != p.h * 128 || p.q_stride_h != 128 || p.kv_stride_n != 128 ||
        p.weights_stride_m != p.h || p.out_stride_m != p.n ||
        ((reinterpret_cast<uintptr_t>(p.q) | reinterpret_cast<uintptr_t>(p.kv) |
          reinterpret_cast<uintptr_t>(p.out)) & 15) != 0)
        return;
    mqa_apply_retuned_rules(p, c, kRetunedRules);
    if (dense_output)
        mqa_apply_retuned_rules(p, c, kDenseRetunedRules);
}

} // namespace deepgemm::mqa::fp8
