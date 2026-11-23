// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "paged_mqa_logits/cta_schedule.h"
#include "paged_sparse_mqa_fp8_impl.hpp"

namespace deepgemm {

template <bool FP4, int HeadTiles>
static void launch_paged_sparse_mqa_fp8_typed(PagedSparseMqaKernelArgs args, int device, hipStream_t stream) {
    const auto& a = args.opt.base;
    dim3 grid(uint64_t(a.mapping.rows) * ((a.tiles_per_row + 15) / 16));
    if constexpr (FP4) {
        grid =
            paged_mqa_persistent_grid<1>(grid, paged_sparse_mqa_fp4_rescaled_kernel<HeadTiles>, 256, 0, device, false);
        paged_sparse_mqa_fp4_rescaled_kernel<HeadTiles><<<grid, 256, 0, stream>>>(args);
        dim3 fallback(uint64_t(a.mapping.rows) * ((a.tiles_per_row + 15) / 16));
        fallback = paged_mqa_persistent_grid<1>(fallback, paged_sparse_mqa_fp8_group_kernel<true, HeadTiles, true>, 256,
                                                0, device, false);
        paged_sparse_mqa_fp8_group_kernel<true, HeadTiles, true>
            <<<fallback, 256, 0, stream>>>(args.opt, args.exceptional);
    } else {
        grid = paged_mqa_persistent_grid<1>(grid, paged_sparse_mqa_fp8_group_kernel<false, HeadTiles>, 256, 0, device,
                                            false);
        paged_sparse_mqa_fp8_group_kernel<false, HeadTiles><<<grid, 256, 0, stream>>>(args.opt);
    }
}

void launch_paged_sparse_mqa_fp8(PagedSparseMqaKernelArgs args, bool fp4, int device, hipStream_t stream) {
    if (fp4) {
        if (args.opt.base.heads <= 16)
            launch_paged_sparse_mqa_fp8_typed<true, 1>(args, device, stream);
        else
            launch_paged_sparse_mqa_fp8_typed<true, 2>(args, device, stream);
    } else {
        if (args.opt.base.heads <= 16)
            launch_paged_sparse_mqa_fp8_typed<false, 1>(args, device, stream);
        else
            launch_paged_sparse_mqa_fp8_typed<false, 2>(args, device, stream);
    }
}
} // namespace deepgemm
