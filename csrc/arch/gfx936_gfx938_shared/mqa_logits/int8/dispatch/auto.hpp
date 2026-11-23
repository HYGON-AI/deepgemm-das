// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// INT8 auto rule table shared by dense and compact dispatch.
// Host only; reads shapes, strides and pointers, never tensor values.

#pragma once
#include "config.hpp"
#include <limits>

namespace deepgemm::mqa::int8 {

// Retune flags retain the bit positions used by kAutoRules.
// Only bits consumed by rule masks or the final override are computed.
static inline uint32_t auto_retune_flags(const common::MqaParams &p) {
    const bool dense = p.h >= 1 && p.h <= 128 && p.q_stride_m == p.h * 128 &&
        p.q_stride_h == 128 && p.kv_stride_n == 128 &&
        p.weights_stride_m == p.h && p.out_stride_m == p.n;
    const auto out_mod = reinterpret_cast<uintptr_t>(p.out) & 15;
    const bool known_layout = p.h >= 1 && p.h <= 128 && p.n >= 0 && p.n <= INT32_MAX &&
        ((p.weights_stride_m == p.h && p.out_stride_m == p.n && out_mod == 0) ||
         (p.weights_stride_m == p.h + 5 && p.out_stride_m == p.n + 8 && out_mod == 4));
    return (uint32_t(p.m % 16 == 0) << 0) |
        (uint32_t(p.n % 64 == 0) << 3) |
        (uint32_t(dense) << 6) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.out) & 15) == 0) << 7) |
        (uint32_t(p.kv_scale != nullptr) << 8) |
        (uint32_t(p.clean_logits == 1) << 9) |
        (uint32_t(p.ks != nullptr && p.ke != nullptr) << 10) |
        (uint32_t(p.n % 4 == 0) << 11) |
        (uint32_t(p.weights_stride_m >= p.h && p.weights_stride_m <= (int64_t(1) << 23)) << 14) |
        (uint32_t(p.m % 8 == 0) << 15) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.q) & 15) == 0) << 20) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.kv) & 15) == 0) << 21) |
        (uint32_t(known_layout) << 22);
}

enum class AutoArea : uint8_t { Logical, PaddedM16, PaddedM32, PaddedM64 };
enum AutoFlag : uint32_t {
    N128Aligned = 1 << 0,
    N128Tail = 1 << 1,
    N256Aligned = 1 << 2,
    ShortMTail = 1 << 3,
    LargeLowHeadGrid = 1 << 4,
    // Padding M to 64 rows adds no rows beyond padding it to 32.
    M64NoExtraPad = 1 << 5,
};

inline constexpr int32_t kMaxDim = std::numeric_limits<int32_t>::max();
inline constexpr int64_t kMaxArea = std::numeric_limits<int64_t>::max();

// M/N/H and area bounds are inclusive. As in FP8, the first matching row wins.
// Area and alignment flags preserve the measured INT8 tail-tile choices.
// Tiled group_heads is capped by H after matching; small-KV configs are exact.
// Retuned entries may distinguish cleaning modes. Compact lookup uses clean_logits=0.
// The host uses KS/KE pointer presence without reading the window bounds.
struct AutoRule {
    int32_t h_min, h_max, m_min, m_max, n_min, n_max;
    KernelConfig config;
    int64_t area_min = 0, area_max = kMaxArea;
    AutoArea area = AutoArea::Logical;
    uint32_t flags = 0;
    // Measured subregions retain the original row constraints and fallback.
    uint32_t tune_mask = 0, tune_value = 0;
    int64_t tune_area_min = 0, tune_area_max = kMaxArea;
};

static constexpr AutoRule kAutoRules[] = {
    // Small KV: H8 uses one wave, two waves, or paired queries as M grows.
    {8, 8, 16, 16, 0, 128, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    // int8_refined_000; unit-step boundary validated
    {8, 8, 17, 17, 1, 128, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 17, 2176},
    {8, 8, 17, 32, 0, 128, {KernelKind::SmallKV, 2, 16, 1, 2, false}},
    // int8_refined_001; unit-step boundary validated
    {8, 8, 33, 111, 1, 48, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 33, 1816},
    // int8_refined_002; unit-step boundary validated
    {8, 8, 64, 128, 57, 128, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 7232, 16384},
    // int8_refined_003; unit-step boundary validated
    {8, 8, 112, 128, 1, 21, {KernelKind::SmallKV, 2, 16, 1, 2, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 112, 6144},
    {8, 8, 33, 128, 0, 128, {KernelKind::SmallKV, 2, 16, 1, 1, true}},
    // int8_refined_004; unit-step boundary validated
    {64, 64, 16, 32, 114, 128, {KernelKind::SmallKV, 1, 32, 1, 1, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 1584, 4096},
    {64, 64, 16, 32, 0, 128, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    // int8_refined_005; unit-step boundary validated
    {64, 64, 33, 128, 1, 8, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 33, 2079},
    {64, 64, 33, 128, 0, 128, {KernelKind::SmallKV, 1, 32, 1, 1, false}},
    // H16 leaves M>96 and N>96 on the tiled path.
    // int8_refined_006; unit-step boundary validated
    {16, 16, 16, 128, 17, 31, {KernelKind::SmallKV, 1, 32, 1, 1, false}, 0, kMaxArea, AutoArea::Logical, 0, 7344064, 7342016, 272, 2208},
    // int8_refined_007; unit-step boundary validated
    {16, 16, 96, 128, 18, 32, {KernelKind::SmallKV, 1, 32, 1, 1, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 2304, 4096},
    {16, 16, 16, 128, 16, 32, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {16, 16, 16, 32, 33, 96, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {16, 16, 16, 96, 33, 128, {KernelKind::SmallKV, 1, 32, 1, 1, false}},
    {16, 16, 97, 128, 33, 96, {KernelKind::SmallKV, 1, 32, 1, 1, false}},
    // H64 was handled above; these rows select the H32/H128 schedules.
    // int8_refined_009; unit-step boundary validated
    {128, 128, 64, 128, 16, 32, {KernelKind::SmallKV, 1, 32, 1, 1, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 2048, 4096},
    {32, 128, 16, 128, 16, 32, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {32, 128, 16, 32, 33, 96, {KernelKind::SmallKV, 1, 16, 1, 1, false}},
    {32, 128, 16, 128, 33, 128, {KernelKind::SmallKV, 1, 32, 1, 1, false}},

    // Tiled 64x256 needs at least ~1.5 waves of CTAs (480 at 4 CTAs/CU). H<=8
    // gains 6-43% for N in [8K, 64K]; at N=128K (K > L2) it loses up to 22%.
    // H>=16 gains 2-19%, which a partial last 64-row tile can erase (M=321: -3%).
    // H8 keeps M=65..96 on BM32: BM64 rounds 96 rows up to 128.
    // int8_refined_011; unit-step boundary validated
    {1, 1, 96, 2048, 62065, 65536, {KernelKind::Tiled, 32, 256, 1, 4, false}, 7500000, kMaxArea, AutoArea::PaddedM64, 0, 7358400, 7358400, 127112193, 134217728},
    {1, 4, 0, kMaxDim, 8192, 65536, {KernelKind::Tiled, 64, 256, 1}, 7500000, kMaxArea,
     AutoArea::PaddedM64},
    {8, 8, 97, kMaxDim, 8192, 65536, {KernelKind::Tiled, 64, 256, 1}, 7500000, kMaxArea,
     AutoArea::PaddedM64},
    // int8_refined_016; unit-step boundary validated
    {16, 16, 960, 2048, 8192, 14336, {KernelKind::Tiled, 32, 128, 2, 4, false}, 11000000, kMaxArea, AutoArea::PaddedM64, 32, 7358400, 7358400, 11000640, 20961280},
    // int8_refined_018; unit-step boundary validated
    {64, 64, 48, 1024, 10743, 262144, {KernelKind::Tiled, 16, 256, 4, 4, false}, 11000000, kMaxArea, AutoArea::PaddedM64, 32, 7358400, 7358400, 11000064, 14940160},
    // int8_refined_019; unit-step boundary validated
    {128, 128, 48, 2048, 8192, 262144, {KernelKind::Tiled, 16, 256, 4, 4, false}, 11000000, kMaxArea, AutoArea::PaddedM64, 32, 7358400, 7358400, 11000064, 16777216},
    {16, 128, 0, kMaxDim, 8192, kMaxDim, {KernelKind::Tiled, 64, 256, 1}, 11000000, kMaxArea,
     AutoArea::PaddedM64, M64NoExtraPad},
    // H<=8, M>=32, N>=4096, and padded-M32 area>=524288.
    // Match the narrow-grid and short-M-tail exceptions before wider tiles.
    {8, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 16, 64, 4},
     524288, 524288, AutoArea::PaddedM32, N128Aligned},
    {8, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     0, 1048576, AutoArea::PaddedM16, LargeLowHeadGrid | ShortMTail},
    {1, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 32, 64, 2},
     524288, 1048576, AutoArea::PaddedM32},
    {8, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 32, 64, 2},
     524288, 1310720, AutoArea::PaddedM32, N128Tail},
    {1, 8, 64, kMaxDim, 131072, kMaxDim, {KernelKind::Tiled, 32, 256, 2}},
    {1, 8, 1024, kMaxDim, 65536, kMaxDim, {KernelKind::Tiled, 32, 256, 2},
     0, kMaxArea, AutoArea::Logical, N256Aligned},
    // int8_refined_020; unit-step boundary validated
    {1, 1, 32, 128, 131072, 1048576, {KernelKind::Tiled, 32, 256, 1, 4, false}, 524288, kMaxArea, AutoArea::PaddedM32, 0, 7342016, 7342016, 4194240, 34603008},
    {1, 8, 32, kMaxDim, 4096, kMaxDim, {KernelKind::Tiled, 32, 128, 2},
     524288, kMaxArea, AutoArea::PaddedM32},

    // Short M grids: avoid redundant rows in a partially filled BM32 tile.
    {64, 128, 24, 128, 0, kMaxDim, {KernelKind::Tiled, 16, 256, 4}, 3145728, 8388608},
    {32, 32, 24, 128, 0, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     4194305, 8388608, AutoArea::Logical, ShortMTail},
    {16, 16, 24, 80, 0, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     4194305, 8388608, AutoArea::Logical, ShortMTail},

    // General tiled rules, after all small-KV and padded-grid exceptions.
    // int8_refined_024; unit-step boundary validated
    {1, 1, 1023, 2048, 112, 256, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 229152, 524288},
    // int8_refined_025; unit-step boundary validated
    {2, 2, 1023, 2048, 128, 256, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342024, 7342024, 260865, 524288},
    {1, 128, 0, kMaxDim, 0, 256, {KernelKind::Tiled, 16, 64, 4}},
    {32, 32, 1024, kMaxDim, 131072, kMaxDim, {KernelKind::Tiled, 32, 256, 2}},
    {1, 128, 1024, kMaxDim, 8192, kMaxDim, {KernelKind::Tiled, 32, 256, 2},
     0, kMaxArea, AutoArea::Logical, N256Aligned},
    // int8_refined_026; unit-step boundary validated
    {1, 1, 1024, 2047, 257, 321, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 263168, 657408},
    // int8_refined_027; unit-step boundary validated
    {2, 2, 1024, 2048, 257, 321, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 263168, 657408},
    // int8_refined_028; unit-step boundary validated
    {4, 4, 1024, 2048, 257, 321, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 263168, 657408},
    // int8_refined_033; unit-step boundary validated
    {32, 32, 1024, 2048, 384, 1536, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7341504, 524288, 6291456},
    // int8_refined_035; unit-step boundary validated
    {64, 64, 1024, 2048, 384, 4096, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7341504, 589824, 6291456},
    // int8_refined_036; unit-step boundary validated
    {64, 64, 1024, 2048, 577, 4415, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 590848, 9691136},
    // int8_refined_037; unit-step boundary validated
    {64, 64, 1282, 2048, 257, 576, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 329474, 1179648},
    // int8_refined_038; unit-step boundary validated
    {128, 128, 1024, 2048, 257, 8194, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 263168, 16775168},
    // int8_refined_039; unit-step boundary validated
    {128, 128, 1024, 2048, 384, 6144, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7341504, 393216, 12582912},
    {1, 128, 1024, kMaxDim, 257, kMaxDim, {KernelKind::Tiled, 32, 128, 2}},
    {1, 4, 32, 1023, 257, kMaxDim, {KernelKind::Tiled, 32, 128, 2}, 524288},
    {8, 8, 32, 1023, 257, kMaxDim, {KernelKind::Tiled, 32, 128, 2},
     524288, kMaxArea, AutoArea::Logical, N128Aligned},
    // int8_refined_041; unit-step boundary validated
    {1, 1, 7, 31, 54614, 655360, {KernelKind::Tiled, 32, 128, 1, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7374784, 7374784, 1310722, 20316160},
    // int8_refined_043; unit-step boundary validated
    {4, 4, 1, 1, 32769, 1048576, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 32769, 1048576},
    // int8_refined_044; unit-step boundary validated
    {4, 4, 13, 31, 196609, 1048576, {KernelKind::Tiled, 32, 256, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342017, 7342016, 3145744, 32505856},
    // int8_refined_046; unit-step boundary validated
    {8, 8, 1, 31, 196609, 1048576, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 196609, 32505856},
    {1, 8, 0, 31, 32769, kMaxDim, {KernelKind::Tiled, 16, 128, 4}},
    // int8_refined_047; unit-step boundary validated
    {1, 1, 1, 128, 1025, 6143, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxArea, AutoArea::Logical, 2, 7342016, 7342016, 1025, 196576},
    // int8_refined_048; unit-step boundary validated
    {1, 1, 17, 32, 6145, 32767, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxArea, AutoArea::Logical, 2, 7342016, 7342016, 139230, 1015777},
    // int8_refined_049; unit-step boundary validated
    {2, 2, 1, 128, 1025, 6143, {KernelKind::Tiled, 16, 64, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 2, 7342016, 7342016, 1025, 196576},
    // int8_refined_050; unit-step boundary validated
    {2, 2, 17, 32, 6145, 32767, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 2, 7342016, 7342016, 139230, 1015777},
    // int8_refined_051; unit-step boundary validated
    {4, 4, 1, 128, 1025, 8189, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 2, 7342016, 7342016, 1025, 131456},
    {1, 8, 0, 1023, 1024, kMaxDim, {KernelKind::Tiled, 16, 128, 4},
     0, kMaxArea, AutoArea::Logical, N128Tail},
    {16, 128, 0, 1023, 257, 65535, {KernelKind::Tiled, 16, 64, 4}, 0, 524288},
    // int8_refined_053; unit-step boundary validated
    {32, 32, 1, 24, 171874, 1048576, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, 4194304, AutoArea::Logical, 0, 7342016, 7342016, 171874, 4194302},
    // int8_refined_054; unit-step boundary validated
    {64, 64, 1, 16, 262144, 1048576, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, 4194304, AutoArea::Logical, 0, 7342016, 7342016, 171876, 4194304},
    {16, 128, 0, 1023, 257, kMaxDim, {KernelKind::Tiled, 16, 128, 4}, 0, 4194304},
    // int8_refined_056; unit-step boundary validated
    {32, 32, 7, 31, 430667, 1048576, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 4194312, 32505856},
    // int8_refined_057; unit-step boundary validated
    {64, 64, 7, 31, 349526, 1048576, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 7340032, 32505856},
    {16, 128, 0, 31, 257, kMaxDim, {KernelKind::Tiled, 16, 128, 4}},
    // int8_refined_058; unit-step boundary validated
    {16, 16, 32, 1023, 4101, 27126, {KernelKind::Tiled, 32, 128, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 4194336, 74250432},
    // int8_refined_060; unit-step boundary validated
    {32, 32, 32, 768, 6144, 262144, {KernelKind::Tiled, 32, 128, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7341504, 4718592, 8388608},
    // int8_refined_063; unit-step boundary validated
    {32, 32, 161, 1023, 4101, 38165, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 4194432, 7327744},
    // int8_refined_064; unit-step boundary validated
    {32, 32, 192, 1023, 7163, 65536, {KernelKind::Tiled, 32, 128, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 7327749, 14680064},
    // int8_refined_065; unit-step boundary validated
    {64, 64, 32, 1023, 7163, 1048576, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 7327749, 33554432},
    // int8_refined_066; unit-step boundary validated
    {64, 64, 192, 1023, 4101, 38165, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 4195323, 7327680},
    {16, 128, 32, 1023, 257, kMaxDim, {KernelKind::Tiled, 32, 256, 2}},
    // int8_refined_067; original default region
    {1, 1, 16, 512, 384, 16384, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7341504, 147456, 393216},
    // int8_refined_068; original default region
    {1, 1, 449, 1023, 257, 768, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7342016, 246430, 523776},
    // int8_refined_069; original default region
    {2, 2, 24, 768, 384, 16384, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, kMaxArea, AutoArea::Logical, 0, 7342016, 7341504, 294912, 393216},
};

static inline void mqa_auto_config(const MqaInt8Params& p, KernelConfig& c) {
    c = {KernelKind::Tiled, 16, 64, 4};
    const int64_t areas[] = {p.m * p.n, ((p.m + 15) / 16) * 16 * p.n,
                             ((p.m + 31) / 32) * 32 * p.n, ((p.m + 63) / 64) * 64 * p.n};
    const uint32_t flags = (p.n % 128 == 0 ? N128Aligned : N128Tail) |
        (p.n % 256 == 0 ? N256Aligned : 0u) |
        (p.m % 32 > 0 && p.m % 32 <= 16 ? ShortMTail : 0u) |
        (p.h <= 8 && p.m >= 32 && p.n >= 4096 && areas[2] >= 524288 ? LargeLowHeadGrid : 0u) |
        (areas[3] == areas[2] ? M64NoExtraPad : 0u);
    // make_grid accepts BM=64 only when a 64-row tile's weight offsets fit in 32 bits.
    const bool m64_ok = p.weights_stride_m >= p.h && p.weights_stride_m <= (int64_t(1) << 23);
    const uint32_t tune_flags = auto_retune_flags(p);
    const int64_t tune_area = areas[0];
    for (const AutoRule &r : kAutoRules) {
        if (p.h < r.h_min || p.h > r.h_max || p.m < r.m_min || p.m > r.m_max ||
            p.n < r.n_min || p.n > r.n_max)
            continue;
        if ((tune_flags & r.tune_mask) != r.tune_value ||
            tune_area < r.tune_area_min || tune_area > r.tune_area_max)
            continue;
        const int64_t area = areas[static_cast<unsigned>(r.area)];
        if (area < r.area_min || area > r.area_max || (flags & r.flags) != r.flags)
            continue;
        if (r.config.block_m == 64 && !m64_ok)
            continue;
        c = r.config;
        break;
    }
    // The one-wave TN16 specialization also handles fewer than 16 KV tokens.
    // Keep its supported M/H range and the measured input/output layouts.
    if ((p.h == 16 || p.h == 32 || p.h == 128) && p.m >= 16 && p.m <= 128 &&
        p.n >= 1 && p.n < 16 && (tune_flags & (1u << 22)))
        c = {KernelKind::SmallKV, 1, 16, 1, 1, false};
    if (c.kind == KernelKind::Tiled) {
        // Tiled kernels never pair queries; keep this explicit after rule selection.
        c.paired_queries = false;
        if (p.h < c.group_heads)
            c.group_heads = static_cast<int>(p.h);
    }
    return;
}

} // namespace deepgemm::mqa::int8
