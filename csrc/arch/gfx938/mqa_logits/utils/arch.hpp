// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "mqa_logits/common/types.hpp"
#include <cstdint>

namespace deepgemm::mqa::fp8::arch {
using common::I2;
using common::I4;
using common::F2;
using common::F4;
using common::LdsInt;

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
// One MLS instruction and its matching DS format. A kernel owns the LDS layout,
// including how several MLS requests are combined and when their data is ready.
enum class MLSKind { B8_64x32_T1_F1, B8_128x16_T1_F0, B8_64x16_T1_F0 };

struct MLSB8TransDS {
    static constexpr int kDSRow = 3;
    static constexpr int kDSCol = 1;
    static constexpr int kDSAlt = 0;
    static constexpr int kNonmajorFilterShift = 8;
};

template <MLSKind Kind> struct MLSTraits;

template <> struct MLSTraits<MLSKind::B8_64x32_T1_F1> : MLSB8TransDS {
    static constexpr int kRows = 32;
    static constexpr int kLdsBytesPerIssue = 2048;
    static constexpr int kControlWord = 1 << 17;
    static constexpr bool kSupportsTail = true;
};

template <> struct MLSTraits<MLSKind::B8_128x16_T1_F0> : MLSB8TransDS {
    static constexpr int kRows = 16;
    static constexpr int kLdsBytesPerIssue = 2048;
    static constexpr int kControlWord = 0;
    static constexpr bool kSupportsTail = true;
};

template <> struct MLSTraits<MLSKind::B8_64x16_T1_F0> : MLSB8TransDS {
    static constexpr int kRows = 16;
    static constexpr int kLdsBytesPerIssue = 1024;
    static constexpr int kControlWord = 0;
    // The current two-request K path is used only for complete N16 tiles.
    static constexpr bool kSupportsTail = false;
};

template <MLSKind Kind> struct MLSAccessor {
    using Traits = MLSTraits<Kind>;
    using Descriptor = I4;

    // valid_nonmajor is in [1, kRows]. A zero-row tile is handled by the
    // kernel: the filter does not promise to initialize its LDS destination.
    template <bool Tail = false>
    __device__ __forceinline__ static Descriptor make_descriptor(const uint8_t *source,
                                                                  int stride_bytes,
                                                                  int valid_nonmajor = Traits::kRows) {
        static_assert(!Tail || Traits::kSupportsTail, "MLS tail mode is not verified");
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        int control = Traits::kControlWord;
        if constexpr (Tail)
            control |= (Traits::kRows - valid_nonmajor) << Traits::kNonmajorFilterShift;
        return {int(address), int(address >> 32), stride_bytes, control};
    }

    __device__ __forceinline__ static void matrix_load(Descriptor descriptor, uint32_t destination) {
        const uint32_t offset = destination | 0x80000000u;
        using LdsByte = char __attribute__((address_space(3)));
        auto *dst = reinterpret_cast<LdsByte *>(uintptr_t(offset));
        // Preserve the transposed LDS address bit and t=1, r=glc=slc=0.
        // The gfx938 compiler emits the MLS instruction's leading s_nop 0.
        if constexpr (Kind == MLSKind::B8_64x32_T1_F1)
            __builtin_hcu_matrix_load_64x32_b8(descriptor, dst, 0, true, false, false, false);
        else if constexpr (Kind == MLSKind::B8_128x16_T1_F0)
            __builtin_hcu_matrix_load_128x16_b8(descriptor, dst, 0, true, false, false, false);
        else
            __builtin_hcu_matrix_load_64x16_b8(descriptor, dst, 0, true, false, false, false);
    }

    // OffsetBytes is an immediate relative to the caller's LDS base. It may
    // select data written by a different MLS request in the same stage.
    template <int OffsetBytes = 0>
    __device__ __forceinline__ static I4 ds_read_format(uint32_t source) {
        auto *p = reinterpret_cast<LdsInt *>(uintptr_t(source));
        return __builtin_bit_cast(
            I4, __builtin_hcu_ds_read_matrix_trans_format_i8(
                    p, OffsetBytes, Traits::kDSRow, Traits::kDSCol, Traits::kDSAlt));
    }
};

template <int BytesPerAccess> struct BufferValue;
template <> struct BufferValue<4> { using type = float; };
template <> struct BufferValue<8> { using type = F2; };
template <> struct BufferValue<16> { using type = F4; };

template <int BytesPerAccess> struct BufferLoadAccessor {
    static_assert(BytesPerAccess == 4 || BytesPerAccess == 8 || BytesPerAccess == 16);
    using Value = typename BufferValue<BytesPerAccess>::type;
    using Descriptor = I4;
    static constexpr uint32_t kOOBOffset = UINT32_MAX;

    // Raw gfx938 resource descriptor. The UINT32_MAX offset is the tested
    // invalid-request sentinel; every other issued access must be fully valid.
    __device__ __forceinline__ static Descriptor make_descriptor(const void *source) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        return {int(address), int(address >> 32), -1, 0x00020000};
    }

    // Builtins expose their result dependency to the compiler.
    __device__ __forceinline__ static Value load(Descriptor descriptor, uint32_t offset) {
        if constexpr (BytesPerAccess == 16)
            return __builtin_bit_cast(
                F4, __builtin_amdgcn_buffer_load_dwordx4(descriptor, 0, offset, false, false));
        else if constexpr (BytesPerAccess == 8)
            return __builtin_bit_cast(
                F2, __builtin_amdgcn_buffer_load_dwordx2(descriptor, 0, offset, false, false));
        else
            return __builtin_bit_cast(
                float, __builtin_amdgcn_buffer_load_dword(descriptor, 0, offset, false, false));
    }

    // The builtin takes a byte offset, not separate voffset/soffset operands.
    // Preserve the OOB sentinel when adding the head offset: UINT32_MAX plus
    // a nonzero head offset would wrap into an otherwise valid weight row.
    __device__ __forceinline__ static Value load(Descriptor descriptor, uint32_t row_offset,
                                                 uint32_t head_offset) {
        static_assert(BytesPerAccess == 4, "Head offsets are used only for scalar weights");
        const uint32_t offset = row_offset == kOOBOffset ? kOOBOffset : row_offset + head_offset;
        return load(descriptor, offset);
    }
};

enum class MMADataType { FP8E4M3FN };
enum class MMAEmitMode { InlineAsm, Builtin };

// A and B are selected pairs of adjacent VGPRs. Preserve the FP8/LIT layout
// while exposing MMAC dependencies to aicc, including independent first D/C.
template <MMADataType DataType, bool Lit, bool Lts, MMAEmitMode Mode>
__device__ __forceinline__ F4 mmac(I2 a, I2 b, F4 c) {
    static_assert(DataType == MMADataType::FP8E4M3FN && Lit && !Lts &&
                      Mode == MMAEmitMode::Builtin,
                  "Unverified MMAC type, layout or emission mode");
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, c, Lit, Lts);
}

__device__ __forceinline__ float shuffle_weight(float x, int row) {
    return __builtin_bit_cast(float,
                              __builtin_amdgcn_ds_bpermute(row * 4, __builtin_bit_cast(int, x)));
}

// A single-output ASM avoids this DTK's incorrect extraction of multiple
// FP32 ASM outputs. Native max semantics intentionally replace fmaxf's
// extra input canonicalization; non-finite values follow the ISA operation.
__device__ __forceinline__ float relu(float value) {
    float result;
    asm volatile("v_max_f32 %0, 0, %1" : "=&v"(result) : "v"(value));
    return result;
}

// Synchronize the workgroup after the caller satisfies pending-memory
// and LDS ownership requirements. Scheduling barriers constrain ordering.
__device__ __forceinline__ void barrier() {
    __builtin_hcu_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_hcu_sched_barrier(0);
}

// The zero wait immediate drains the current wave's VM, EXP and LGKM counters.
__device__ __forceinline__ void wait_vm() {
    __builtin_hcu_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_hcu_sched_barrier(0);
}

// Complete the wave's pending operations before the workgroup barrier.
__device__ __forceinline__ void wait_vm_barrier() {
    wait_vm();
    __builtin_amdgcn_s_barrier();
    __builtin_hcu_sched_barrier(0);
}

// Finish this wave's matrix DS reads before the caller reuses the LDS half.
// Scheduling boundaries keep requests ahead of the VM, EXP and LGKM drain.
__device__ __forceinline__ void finish_kv_half(I4 x0, I4 x1, I4 x2, I4 x3) {
    __builtin_hcu_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_hcu_sched_barrier(0);
}

// Drain VM, EXP and LGKM before the weight handoff.
__device__ __forceinline__ void finish_weight(float &x) {
    __builtin_hcu_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_hcu_sched_barrier(0);
}

// Drain VM, EXP and LGKM before the scale group is consumed.
__device__ __forceinline__ void finish_scales(F4 (&x)[4]) {
    __builtin_hcu_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_hcu_sched_barrier(0);
}
#endif
} // namespace deepgemm::mqa::fp8::arch
