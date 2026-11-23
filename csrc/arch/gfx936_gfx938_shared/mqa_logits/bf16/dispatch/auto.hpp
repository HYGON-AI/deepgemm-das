// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// BF16 auto rule table shared by dense and compact dispatch.
// Host only; reads shapes, strides and pointers, never tensor values.

#pragma once
#include "config.hpp"
#include <limits>

namespace deepgemm::mqa::bf16 {

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
        (uint32_t(p.n % 128 == 0) << 4) |
        (uint32_t(dense) << 6) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.out) & 15) == 0) << 7) |
        (uint32_t(p.kv_scale != nullptr) << 8) |
        (uint32_t(p.clean_logits == 1) << 9) |
        (uint32_t(p.ks != nullptr && p.ke != nullptr) << 10) |
        (uint32_t(p.m % 32 > 0 && p.m % 32 <= 16) << 13) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.q) & 15) == 0) << 20) |
        (uint32_t((reinterpret_cast<uintptr_t>(p.kv) & 15) == 0) << 21) |
        (uint32_t(known_layout) << 22);
}

enum class AutoGrid : uint8_t { None, M16N64, M16N128, M16N256, M32N128 };

inline constexpr int32_t kMaxDim = std::numeric_limits<int32_t>::max();
inline constexpr int64_t kMaxGrid = std::numeric_limits<int64_t>::max();

// M/N/H and CTA-count bounds are inclusive; the first matching row wins, as in
// INT8. Grid counts use padded tiles, not M*N, so partial tiles keep their route.
// Refined with the 2026-10-19 gfx936 M/N sweep; stable existing schedules are
// retained outside those intervals. These ranges describe tile utilization,
// CTA occupancy and K reuse, rather than an exact-shape benchmark lookup.
// KS/KE values are never read on the host, but the tune flags include KS/KE
// presence (bit 10) and clean_logits==1 (bit 9): refined rules whose tune_mask
// covers bit 9 apply only to that cleaning mode. resolve_compact passes 0.
struct AutoRule {
    int32_t h_min, h_max, m_min, m_max, n_min, n_max;
    KernelConfig config;
    int64_t grid_min = 0, grid_max = kMaxGrid;
    AutoGrid grid = AutoGrid::None;
    // Measured subregions retain the original row constraints and fallback.
    uint32_t tune_mask = 0, tune_value = 0;
    int64_t tune_area_min = 0, tune_area_max = kMaxGrid;
};

// Measured on gfx936 with the 2026-10-20 9,745-shape M/N/H sweep using
// causal and full windows. Rule order and thresholds preserve that measured table.
static constexpr AutoRule kAutoRules[] = {
    // Short H8 windows: grouped BM16 amortizes cooperative Q staging.
    {8, 8, 24, 512, 256, 512, {KernelKind::Tiled, 16, 64, 4}},
    // Larger H8 batches: BN128 reduces the number of short-window CTAs.
    // bf16_refined_000; unit-step boundary validated
    {8, 8, 513, 768, 256, 446, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 131328, 228864},
    {8, 8, 513, kMaxDim, 256, 512, {KernelKind::Tiled, 16, 128, 2}},
    // Short H16 windows: grouped BM16 covers the moderate-M interval.
    {16, 16, 32, 1023, 256, 512, {KernelKind::Tiled, 16, 64, 4}},
    // Larger H16 batches use BN128 with two heads per group.
    {16, 16, 1024, kMaxDim, 256, 512, {KernelKind::Tiled, 16, 128, 2}},
    // High H and enough rows amortize the grouped tiled schedule.
    // bf16_refined_002; unit-step boundary validated
    {32, 32, 128, 256, 256, 512, {KernelKind::SplitHead, 16, 64, 1, 8, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 32768, 70144},
    // bf16_refined_003; unit-step boundary validated
    {32, 32, 1021, 2048, 341, 512, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 458429, 1048576},
    // bf16_refined_004; unit-step boundary validated
    {64, 64, 128, 256, 256, 512, {KernelKind::SplitHead, 16, 64, 1, 8, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 32768, 70144},
    // bf16_refined_005; unit-step boundary validated
    {64, 64, 1025, 2048, 257, 512, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 524800, 1048576},
    // bf16_refined_006; unit-step boundary validated
    {128, 128, 128, 256, 256, 512, {KernelKind::SplitHead, 16, 64, 1, 8, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 32768, 70144},
    // bf16_refined_007; unit-step boundary validated
    {128, 128, 1025, 2048, 257, 512, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 785281, 1048576},
    // bf16_refined_008; unit-step boundary validated
    {128, 128, 1025, 2048, 257, 512, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 524800, 785280},
    {32, 128, 128, kMaxDim, 256, 512, {KernelKind::Tiled, 16, 64, 4}},
    // High H with fewer rows benefits from partitioning heads within each CTA.
    {32, 128, 64, 127, 384, 512, {KernelKind::SplitHead, 16, 64, 1, 8}},
    // Very short queries use one-wave small-KV tiles.
    {1, 16, 0, 4, 513, 768, {KernelKind::SmallKV, 1, 16, 1, 1}},
    // Tiny low-H queries use narrower BN64 tiles for this KV interval.
    {1, 8, 2, 7, 4096, 8192, {KernelKind::Tiled, 16, 64, 4}},
    // Longer KV intervals amortize BN128 even for tiny query batches.
    // bf16_refined_009; unit-step boundary validated
    {4, 4, 2, 7, 8193, 14336, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 16386, 100352},
    // bf16_refined_010; unit-step boundary validated
    {8, 8, 2, 7, 8193, 14336, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 16386, 100352},
    // bf16_refined_011; unit-step boundary validated
    {16, 16, 2, 7, 8193, 14336, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 16386, 100352},
    {1, 16, 2, 7, 8193, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Tiny H32 queries use BN128 once KV is sufficiently long.
    {32, 32, 2, 7, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // H64/H128 tiny batches use BN128 in the long-KV interval.
    // bf16_refined_013; unit-step boundary validated
    {64, 64, 4, 7, 32768, 196608, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 131072, 1376256},
    // bf16_refined_014; unit-step boundary validated
    {128, 128, 4, 7, 32768, 49153, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 131072, 344071},
    {64, 128, 4, 7, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Single-row low-H batches retain the measured BN128 schedule.
    // bf16_refined_015; unit-step boundary validated
    {4, 4, 1, 1, 8192, 10240, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 8192, 10240},
    {1, 4, 0, 1, 8192, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Single-row H8 batches need a longer KV interval to amortize BN128.
    {8, 8, 0, 1, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // H8 moderate batches use BM32/BN128 in this KV interval.
    {8, 8, 64, 256, 16384, 32768, {KernelKind::Tiled, 32, 128, 1}},
    // Large H8 batches reuse K across BM64 rows.
    {8, 8, 1024, kMaxDim, 16384, 65535, {KernelKind::Tiled, 64, 256, 1}},
    // H8 short batches use BM32 in the measured long-KV interval.
    {8, 8, 32, 63, 49152, 98304, {KernelKind::Tiled, 32, 128, 1}},
    // Large H16 batches and long KV amortize BM64/BN256.
    {16, 16, 1536, kMaxDim, 32768, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Moderate H16 batches use BM32/BN256 for long KV.
    {16, 16, 256, 1535, 16384, kMaxDim, {KernelKind::Tiled, 32, 256, 1}},
    // High H moderate batches use BM32/BN256 for long KV.
    {32, 128, 256, 1023, 32768, kMaxDim, {KernelKind::Tiled, 32, 256, 1}},
    // Short high-H query batches use BN128 in this KV interval.
    {16, 128, 8, 16, 8193, 16384, {KernelKind::Tiled, 16, 128, 4}},
    // High-H moderate batches retain the measured BN128 schedule.
    {16, 128, 49, 96, 2049, 4096, {KernelKind::Tiled, 16, 128, 4}},
    // Low H with 17-32 rows uses BM32 to reuse K.
    {1, 4, 17, 32, 32768, kMaxDim, {KernelKind::Tiled, 32, 128, 1}},
    // Low H with 33-63 rows amortizes BM64 when KV is long.
    {1, 4, 33, 63, 65536, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Remaining queries below eight rows use one-wave small-KV tiles.
    // bf16_refined_018; unit-step boundary validated
    {1, 1, 1, 7, 513, 8191, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 2307, 28665},
    // bf16_refined_019; unit-step boundary validated
    {1, 1, 1, 2, 769, 2306, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 769, 2306},
    // bf16_refined_020; unit-step boundary validated
    {2, 2, 1, 7, 513, 3594, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 769, 3594},
    // bf16_refined_021; unit-step boundary validated
    {2, 2, 1, 7, 514, 8191, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 3595, 28665},
    // bf16_refined_022; unit-step boundary validated
    {4, 4, 1, 7, 513, 3968, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 769, 3968},
    // bf16_refined_023; unit-step boundary validated
    {4, 4, 1, 7, 567, 8191, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 3969, 28665},
    // bf16_refined_025; unit-step boundary validated
    {8, 8, 2, 7, 951, 4095, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 6657, 13307},
    // bf16_refined_026; unit-step boundary validated
    {8, 8, 4, 7, 1902, 32767, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 13313, 32767},
    // bf16_refined_027; unit-step boundary validated
    {16, 16, 1, 7, 513, 5760, {KernelKind::SmallKV, 1, 32, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 769, 5760},
    // bf16_refined_028; unit-step boundary validated
    {32, 32, 1, 3, 513, 1792, {KernelKind::SmallKV, 1, 32, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 513, 1792},
    // Packed clean=false: use the bounded alternative and retain the fallback.
    {2, 2, 1, 7, 513, 3594, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 769, 3594},
    {1, 128, 0, 7, 513, kMaxDim, {KernelKind::SmallKV, 1, 64, 1, 1}},
    // Remaining high-H short windows use wider small-KV tiles.
    // bf16_refined_030; unit-step boundary validated
    {32, 32, 48, 640, 47, 512, {KernelKind::SplitHead, 16, 64, 1, 8, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 24048, 163200},
    // bf16_refined_031; unit-step boundary validated
    {32, 32, 641, 2048, 12, 173, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 24576, 132990},
    // bf16_refined_032; unit-step boundary validated
    {32, 32, 641, 2048, 65, 255, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 133120, 522240},
    // bf16_refined_033; unit-step boundary validated
    {64, 64, 48, 2048, 11, 512, {KernelKind::SplitHead, 16, 64, 1, 8, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 21600, 196608},
    // bf16_refined_035; unit-step boundary validated
    {64, 64, 641, 2048, 12, 173, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 24576, 132990},
    // bf16_refined_036; unit-step boundary validated
    {64, 64, 641, 2048, 65, 255, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 133120, 522240},
    // bf16_refined_037; unit-step boundary validated
    {128, 128, 1, 64, 319, 512, {KernelKind::SmallKV, 1, 64, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 319, 24528},
    // bf16_refined_038; unit-step boundary validated
    {128, 128, 48, 125, 197, 512, {KernelKind::SmallKV, 1, 64, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 24576, 32256},
    // bf16_refined_039; unit-step boundary validated
    {128, 128, 126, 895, 32, 383, {KernelKind::SplitHead, 16, 64, 1, 8, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 24570, 228225},
    // bf16_refined_041; unit-step boundary validated
    {128, 128, 896, 2048, 12, 130, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 24552, 132990},
    // bf16_refined_042; unit-step boundary validated
    {128, 128, 896, 2048, 65, 255, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 133120, 522240},
    {32, 128, 0, kMaxDim, 0, 512, {KernelKind::SmallKV, 1, 32, 1, 1}},
    // H8 pairs queries when the short-window batch has enough rows.
    // bf16_refined_046; unit-step boundary validated
    {8, 8, 34, 2048, 4, 255, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 7136, 227328},
    // bf16_refined_047; unit-step boundary validated
    {8, 8, 96, 1536, 6, 128, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 12241, 98304},
    // bf16_refined_049; unit-step boundary validated
    {8, 8, 2046, 2048, 112, 255, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 229376, 522240},
    {8, 8, 32, kMaxDim, 0, 512, {KernelKind::SmallKV, 2, 16, 1, 1, true}},
    // Remaining H8/H16 short windows use one query per wave.
    // bf16_refined_051; unit-step boundary validated
    {16, 16, 144, 2048, 6, 512, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 10776, 140151},
    // bf16_refined_052; unit-step boundary validated
    {16, 16, 1023, 2048, 69, 255, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 141174, 522240},
    {8, 16, 0, kMaxDim, 0, 512, {KernelKind::SmallKV, 1, 16, 1, 1}},
    // Tiny low-H short windows use one query per wave.
    // bf16_refined_053; unit-step boundary validated
    {1, 1, 8, 15, 257, 512, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 3848, 7680},
    // bf16_refined_054; unit-step boundary validated
    {2, 2, 8, 15, 257, 512, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 3848, 7680},
    // bf16_refined_055; unit-step boundary validated
    {4, 4, 8, 15, 257, 512, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 4093, 7680},
    {1, 4, 0, 15, 0, 512, {KernelKind::SmallKV, 1, 16, 1, 1}},
    // At most 80 CTAs on gfx936: partition heads within each CTA.
    // bf16_refined_056; unit-step boundary validated
    {32, 32, 8, 16, 513, 1728, {KernelKind::SmallKV, 1, 64, 1, 1, false}, 0, 80, AutoGrid::M16N64, 7342016, 7341760, 4104, 13824},
    // bf16_refined_057; unit-step boundary validated
    {64, 64, 8, 32, 513, 2182, {KernelKind::SmallKV, 1, 64, 1, 1, false}, 0, 80, AutoGrid::M16N64, 7342016, 7341760, 4104, 17456},
    {32, 128, 0, kMaxDim, 0, kMaxDim,
     {KernelKind::SplitHead, 16, 64, 1, 8}, 0, 80, AutoGrid::M16N64},
    // H8 long KV reuses K across ordinary BM64 rows.
    {8, 8, 64, kMaxDim, 65536, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Large high-H batches with short KV use grouped BM16/BN128.
    // bf16_refined_061; unit-step boundary validated
    {64, 64, 1024, 1281, 513, 736, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 525312, 1317120},
    {16, 128, 1024, kMaxDim, 0, 736, {KernelKind::Tiled, 16, 128, 2}},
    // The remaining large high-H batches use BM32/BN256.
    // bf16_refined_063; unit-step boundary validated
    {16, 16, 1024, 2048, 737, 2688, {KernelKind::Tiled, 32, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 754688, 2752512},
    // bf16_refined_064; unit-step boundary validated
    {32, 32, 1024, 1026, 737, 896, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 754688, 917504},
    // bf16_refined_065; unit-step boundary validated
    {32, 32, 1024, 2048, 737, 2176, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 918270, 2096640},
    // bf16_refined_066; unit-step boundary validated
    {64, 64, 1024, 2048, 737, 1824, {KernelKind::Tiled, 16, 256, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 918270, 4456448},
    // bf16_refined_067; unit-step boundary validated
    {64, 64, 1024, 1026, 737, 896, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 754688, 917504},
    // bf16_refined_069; unit-step boundary validated
    {128, 128, 1024, 1026, 737, 896, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 754688, 917504},
    // Packed clean=false: use the bounded alternative and retain the fallback.
    {32, 32, 1024, 1027, 737, 897, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 754688, 918528},
    {16, 128, 1024, kMaxDim, 737, kMaxDim, {KernelKind::Tiled, 32, 256, 1}},
    // Remaining H8 long-window batches use BM32/BN128.
    // bf16_refined_070; unit-step boundary validated
    {8, 8, 449, 1023, 16384, 65535, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 10486273, 67042305},
    {8, 8, 64, kMaxDim, 16384, kMaxDim, {KernelKind::Tiled, 32, 128, 1}},
    // Short-window rules and the BM16/BN64 tiled fallback.
    // bf16_refined_071; unit-step boundary validated
    {1, 1, 16, 96, 1, 7, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 16, 112},
    // bf16_refined_073; unit-step boundary validated
    {1, 1, 1024, 2048, 256, 512, {KernelKind::Tiled, 64, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 524288, 784384},
    // bf16_refined_074; unit-step boundary validated
    {2, 2, 16, 2048, 1, 512, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 64, 81920},
    // bf16_refined_076; unit-step boundary validated
    {2, 2, 16, 2048, 1, 512, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 336, 523264},
    // bf16_refined_078; unit-step boundary validated
    {4, 4, 16, 224, 1, 80, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 48, 1280},
    // bf16_refined_079; unit-step boundary validated
    {4, 4, 16, 192, 1, 14, {KernelKind::SmallKV, 1, 16, 1, 1, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 16, 224},
    // bf16_refined_081; unit-step boundary validated
    {4, 4, 1536, 2048, 257, 512, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 526336, 1048576},
    {1, 128, 0, kMaxDim, 0, 512, {KernelKind::Tiled, 16, 64, 4}},
    // Low-H long KV amortizes ordinary BM64/BN256.
    {1, 4, 64, kMaxDim, 65536, kMaxDim, {KernelKind::Tiled, 64, 256, 1}},
    // Shrink BN128 to BN64 when the larger tile would leave fewer than 80 CTAs.
    // bf16_refined_082; unit-step boundary validated
    {1, 1, 64, 256, 513, 2432, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, 79, AutoGrid::M32N128, 7342016, 7341760, 32832, 155648},
    // bf16_refined_083; unit-step boundary validated
    {2, 2, 64, 256, 513, 2432, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, 79, AutoGrid::M32N128, 7342016, 7341760, 32832, 155648},
    // bf16_refined_084; unit-step boundary validated
    {2, 2, 64, 384, 513, 4992, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, 79, AutoGrid::M32N128, 7342016, 7341760, 155712, 319488},
    // bf16_refined_085; unit-step boundary validated
    {4, 4, 64, 384, 513, 3456, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, 79, AutoGrid::M32N128, 7342016, 7341760, 32832, 221184},
    // bf16_refined_086; unit-step boundary validated
    {4, 4, 64, 384, 577, 4992, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, 79, AutoGrid::M32N128, 7342016, 7341760, 221248, 319488},
    // bf16_refined_087; unit-step boundary validated
    {8, 8, 64, 384, 513, 4352, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, 79, AutoGrid::M32N128, 7342016, 7341760, 32832, 278528},
    // bf16_refined_088; unit-step boundary validated
    {8, 8, 64, 384, 726, 4992, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, 79, AutoGrid::M32N128, 7342016, 7341760, 278592, 319488},
    {1, 8, 64, kMaxDim, 513, kMaxDim, {KernelKind::Tiled, 32, 64, 1}, 0, 79, AutoGrid::M32N128},
    // Otherwise low-H batches of at least 64 rows use BM32/BN128.
    // bf16_refined_089; unit-step boundary validated
    {1, 1, 64, 512, 513, 5634, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342017, 7341761, 262143, 360576},
    // bf16_refined_090; unit-step boundary validated
    {1, 1, 128, 2048, 2689, 57342, {KernelKind::Tiled, 64, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 5505216, 7339776},
    // bf16_refined_091; unit-step boundary validated
    {1, 1, 128, 2048, 49151, 65535, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 7339904, 134215680},
    // bf16_refined_092; unit-step boundary validated
    {1, 1, 192, 2048, 897, 32768, {KernelKind::Tiled, 64, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 1835136, 50331648},
    // bf16_refined_093; unit-step boundary validated
    {1, 1, 192, 2048, 3584, 49150, {KernelKind::Tiled, 64, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 7339968, 100659200},
    // bf16_refined_094; unit-step boundary validated
    {2, 2, 64, 512, 609, 6912, {KernelKind::Tiled, 32, 64, 2, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342032, 7341760, 311808, 442368},
    // bf16_refined_095; unit-step boundary validated
    {2, 2, 128, 2048, 24577, 65535, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 8388352, 134215680},
    // bf16_refined_096; unit-step boundary validated
    {2, 2, 192, 2048, 2561, 32768, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 5242944, 33554432},
    // bf16_refined_097; unit-step boundary validated
    {2, 2, 512, 2048, 4096, 24576, {KernelKind::Tiled, 64, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 8388608, 50331648},
    // bf16_refined_098; unit-step boundary validated
    {4, 4, 128, 2048, 7169, 32768, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341248, 3670144, 25165824},
    // bf16_refined_099; unit-step boundary validated
    {4, 4, 512, 2048, 7169, 65535, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 14680576, 134215680},
    // bf16_refined_101; unit-step boundary validated
    {8, 8, 768, 2048, 5377, 16383, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 11011072, 33552384},
    {1, 8, 64, kMaxDim, 513, kMaxDim, {KernelKind::Tiled, 32, 128, 1}},
    // Low-H batches below 32 rows retain GH2 for long KV.
    {1, 8, 0, 31, 32768, kMaxDim, {KernelKind::Tiled, 16, 128, 2}},
    // Short low-H batches shrink BN when BN128 would leave fewer than 80 CTAs.
    // bf16_refined_102; unit-step boundary validated
    {2, 2, 8, 63, 513, 8193, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, 79, AutoGrid::M16N128, 7342016, 7341760, 4104, 153216},
    {1, 8, 0, 63, 513, kMaxDim, {KernelKind::Tiled, 16, 64, 4}, 0, 79, AutoGrid::M16N128},
    // Otherwise short low-H batches use BM16/BN128 with grouped heads.
    // bf16_refined_103; unit-step boundary validated
    {1, 1, 8, 48, 24577, 65535, {KernelKind::Tiled, 64, 256, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7350208, 7349952, 1310771, 3145680},
    // bf16_refined_105; unit-step boundary validated
    {1, 1, 32, 63, 2433, 5119, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7350208, 7341760, 153279, 223744},
    // bf16_refined_106; unit-step boundary validated
    {1, 1, 49, 63, 5120, 65535, {KernelKind::Tiled, 64, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7350208, 7341760, 322655, 4128705},
    // bf16_refined_107; unit-step boundary validated
    {2, 2, 16, 63, 7087, 32767, {KernelKind::Tiled, 32, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7350208, 7341760, 446471, 2064321},
    // bf16_refined_108; unit-step boundary validated
    {2, 2, 32, 63, 2433, 5633, {KernelKind::Tiled, 16, 64, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 148413, 217104},
    // bf16_refined_109; unit-step boundary validated
    {2, 2, 33, 63, 32768, 65535, {KernelKind::Tiled, 64, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 1081344, 4128705},
    // bf16_refined_110; unit-step boundary validated
    {4, 4, 26, 63, 24575, 49150, {KernelKind::Tiled, 32, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 262120, 3096450},
    // bf16_refined_111; unit-step boundary validated
    {4, 4, 32, 63, 2433, 5633, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 153279, 217104},
    // bf16_refined_112; unit-step boundary validated
    {4, 4, 33, 63, 49151, 65535, {KernelKind::Tiled, 64, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 1621983, 4128705},
    // bf16_refined_114; unit-step boundary validated
    {8, 8, 55, 63, 24575, 49151, {KernelKind::Tiled, 32, 128, 1, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 1499075, 3096513},
    {1, 8, 0, 63, 513, kMaxDim, {KernelKind::Tiled, 16, 128, 4}},
    // High H: shrink BN128 below 40 CTAs.
    // bf16_refined_115; unit-step boundary validated
    {16, 16, 8, 10, 513, 1152, {KernelKind::SmallKV, 1, 32, 1, 1, false}, 0, 39, AutoGrid::M16N128, 7342016, 7341760, 4104, 9216},
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 64, 4}, 0, 39, AutoGrid::M16N128},
    // High H: shrink BN256 below 64 CTAs.
    // bf16_refined_116; unit-step boundary validated
    {16, 16, 8, 256, 513, 8192, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, 63, AutoGrid::M16N256, 7342016, 7341760, 49152, 196608},
    // bf16_refined_118; unit-step boundary validated
    {64, 64, 8, 256, 640, 8192, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, 63, AutoGrid::M16N256, 7342016, 7341760, 49152, 196608},
    // bf16_refined_119; unit-step boundary validated
    {128, 128, 8, 256, 640, 8192, {KernelKind::Tiled, 16, 64, 4, 4, false}, 0, 63, AutoGrid::M16N256, 7342016, 7341760, 49152, 164352},
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 128, 4}, 0, 63, AutoGrid::M16N256},
    // Up to 128 BN256 CTAs retain GH4.
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 256, 4}, 0, 128, AutoGrid::M16N256},
    // Larger BN256 grids use GH2 to reduce register and LDS pressure.
    // bf16_refined_121; unit-step boundary validated
    {32, 32, 895, 1023, 513, 766, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 491520, 783618},
    // bf16_refined_123; unit-step boundary validated
    {64, 64, 895, 1023, 513, 766, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 491520, 783618},
    // bf16_refined_125; unit-step boundary validated
    {128, 128, 895, 1023, 513, 766, {KernelKind::Tiled, 16, 128, 4, 4, false}, 0, kMaxGrid, AutoGrid::None, 7342016, 7341760, 491520, 783618},
    {16, 128, 0, 1023, 513, kMaxDim, {KernelKind::Tiled, 16, 256, 2}},
};

static inline void mqa_auto_config(const MqaBf16Params &p, KernelConfig &c) {
    c = {KernelKind::Tiled, 16, 64, 4};
    // Shared params store int64 dimensions; the public BF16 entry is int32.
    // Reject larger internal inputs in make_grid before any kernel launch.
    if (p.m < 0 || p.n < 0 || p.m > kMaxDim || p.n > kMaxDim)
        return;
    // The checked range keeps rounded CTA counts and their products in int64.
    const int64_t m16 = (int64_t(p.m) + 15) / 16;
    const int64_t m32 = (int64_t(p.m) + 31) / 32;
    const int64_t n64 = (int64_t(p.n) + 63) / 64;
    const int64_t n128 = (int64_t(p.n) + 127) / 128;
    const int64_t n256 = (int64_t(p.n) + 255) / 256;
    const int64_t grids[] = {0, m16 * n64, m16 * n128, m16 * n256, m32 * n128};
    const uint32_t tune_flags = auto_retune_flags(p);
    const int64_t tune_area = p.m * p.n;
    for (const AutoRule &r : kAutoRules) {
        if (p.h < r.h_min || p.h > r.h_max || p.m < r.m_min || p.m > r.m_max ||
            p.n < r.n_min || p.n > r.n_max)
            continue;
        if ((tune_flags & r.tune_mask) != r.tune_value ||
            tune_area < r.tune_area_min || tune_area > r.tune_area_max)
            continue;
        const int64_t grid = grids[static_cast<unsigned>(r.grid)];
        if (grid < r.grid_min || grid > r.grid_max)
            continue;
        c = r.config;
        break;
    }
    // Short KV batches with enough tiled CTAs can reuse K across 16 queries.
    // Limit this override to the measured layouts and shape range; narrower
    // windows and smaller grids retain their one-wave SmallKV schedules.
    if (c.kind == KernelKind::SmallKV && p.h >= 8 && p.m <= 2048 &&
        p.n >= 64 && p.n <= 512 && (tune_flags & (1u << 22)) &&
        grids[static_cast<unsigned>(AutoGrid::M16N64)] >= 128)
        c = {KernelKind::Tiled, 16, 64, 4, 4, false};
    if (c.kind == KernelKind::Tiled && p.h < c.group_heads)
        c.group_heads = p.h;
    return;
}

} // namespace deepgemm::mqa::bf16
