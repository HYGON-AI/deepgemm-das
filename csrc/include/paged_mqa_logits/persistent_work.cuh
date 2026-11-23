// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include <stdint.h>
#include <hip/hip_runtime.h>

namespace deepgemm {

// Each wave independently maps this CTA's balanced, contiguous range of valid
// tiles onto requests. Device lengths are read on every graph replay; no host
// readback, external metadata, atomics or extra LDS are required.
class PagedMqaWorkRange {
    const int* context_;
    uint64_t max_len_;
    int tile_, batch_, request_, first_, end_request_, end_tile_;

    __device__ __forceinline__ unsigned tiles(int batch) const {
        const int length = batch < batch_ ? context_[batch] : 0;
        const uint64_t valid = length > 0 ? uint64_t(length) : 0;
        return unsigned(((valid < max_len_ ? valid : max_len_) + tile_ - 1) / tile_);
    }

    // Callers use tile >= 64. A wave sum is at most
    // 64 * ceil(INT32_MAX / 64) = 2^31, so uint32 is sufficient here.
    // Accumulation across waves' 64-request chunks remains uint64.
    __device__ __forceinline__ unsigned scan(unsigned value) const {
        const int lane = threadIdx.x & 63;
        for (int offset = 1; offset < 64; offset <<= 1) {
            const unsigned left = __shfl_up(value, offset, 64);
            if (lane >= offset) value += left;
        }
        return value;
    }

    __device__ __forceinline__ uint64_t uniform(uint64_t value) const {
        const unsigned lo = __builtin_amdgcn_readfirstlane(unsigned(value));
        const unsigned hi = __builtin_amdgcn_readfirstlane(unsigned(value >> 32));
        return uint64_t(lo) | (uint64_t(hi) << 32);
    }

    __device__ __forceinline__ void locate(uint64_t prefix, uint64_t previous,
                                         int base, uint64_t target,
                                         int& request, int& first) const {
        const int lane = threadIdx.x & 63;
        const auto mask = __ballot(base + lane < batch_ && prefix > target);
        if (mask && request == batch_) {
            const int owner = __ffsll(mask) - 1;
            request = base + owner;
            first = int(target - __shfl(previous, owner, 64));
        }
    }

public:
    __device__ __forceinline__ PagedMqaWorkRange(const int* context,
                                              uint64_t max_len, int tile, int batch)
        : context_(context), max_len_(max_len), tile_(tile), batch_(batch),
          request_(batch), first_(0), end_request_(batch), end_tile_(0) {
        const int lane = threadIdx.x & 63;
        const unsigned own = tiles(lane), first_prefix = scan(own);
        const unsigned first_sum = __shfl(first_prefix, 63, 64);
        uint64_t total = first_sum;
        for (int base = 64; base < batch_; base += 64)
            total += __shfl(scan(tiles(base + lane)), 63, 64);
        total = uniform(total);

        const unsigned count = gridDim.x, id = blockIdx.x;
        uint64_t quotient, remainder;
        if (total <= UINT32_MAX) {
            quotient = unsigned(total) / count;
            remainder = unsigned(total) % count;
        } else {
            quotient = total / count;
            remainder = total % count;
        }
        // The first remainder CTAs receive one extra tile; all ranges are
        // adjacent and cover [0,total). Empty requests have no tile ownership.
        const uint64_t begin = uint64_t(id) * quotient + (id < remainder ? id : remainder);
        const uint64_t next = uint64_t(id) + 1;
        const uint64_t end = next * quotient + (next < remainder ? next : remainder);
        locate(first_prefix, first_prefix - own, 0, begin, request_, first_);
        locate(first_prefix, first_prefix - own, 0, end, end_request_, end_tile_);
        uint64_t offset = first_sum;
        for (int base = 64; base < batch_ && (request_ == batch_ || end_request_ == batch_); base += 64) {
            const unsigned own = tiles(base + lane), local = scan(own);
            const uint64_t prefix = offset + local, previous = offset + (local - own);
            locate(prefix, previous, base, begin, request_, first_);
            locate(prefix, previous, base, end, end_request_, end_tile_);
            offset += __shfl(local, 63, 64);
        }
        // Preserve scalar Q/KV address arithmetic in the original math helpers.
        request_ = __builtin_amdgcn_readfirstlane(request_);
        first_ = __builtin_amdgcn_readfirstlane(first_);
        end_request_ = __builtin_amdgcn_readfirstlane(end_request_);
        end_tile_ = __builtin_amdgcn_readfirstlane(end_tile_);
    }

    __device__ __forceinline__ bool next(int& request, int& first, int& last) {
        while (request_ < batch_ && request_ <= end_request_) {
            request = request_++;
            first = first_;
            first_ = 0;
            last = request == end_request_ ? end_tile_ : int(tiles(request));
            if (first < last) return true;
        }
        return false;
    }
};

} // namespace deepgemm
