// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "paged_mqa_logits/cta_schedule.h"
#include "paged_sparse_mqa_impl.hpp"

namespace deepgemm {

void launch_paged_sparse_mqa_metadata(const PagedSparseMqaMapping& mapping, int64_t* metadata, hipStream_t stream) {
    paged_sparse_mqa_metadata_kernel<kPagedSparseMqaReuseTiles>
        <<<mapping.rows, kPagedSparseMqaThreads, 0, stream>>>(mapping, metadata);
}

template <bool FP4, int HeadTiles>
static void launch_paged_sparse_mqa_bf16_typed(PagedSparseMqaKernelArgs args, int device, hipStream_t stream) {
    const auto& a = args.opt.base;
    dim3 grid(uint64_t(a.mapping.rows) * ((a.tiles_per_row + 15) / 16));
    grid = paged_mqa_persistent_grid<1>(grid, paged_sparse_mqa_bf16_kernel<FP4, HeadTiles>, 256, 0, device, false);
    paged_sparse_mqa_bf16_kernel<FP4, HeadTiles><<<grid, 256, 0, stream>>>(args);
    dim3 fallback(uint64_t(a.mapping.rows) * ((a.tiles_per_row + 7) / 8));
    fallback = paged_mqa_persistent_grid<1>(fallback, paged_sparse_mqa_bf16_exception_kernel<FP4, HeadTiles>, 256, 0,
                                            device, false);
    paged_sparse_mqa_bf16_exception_kernel<FP4, HeadTiles><<<fallback, 256, 0, stream>>>(args);
}

void launch_paged_sparse_mqa_bf16(PagedSparseMqaKernelArgs args, bool fp4, int device, hipStream_t stream) {
    if (fp4) {
        if (args.opt.base.heads <= 16)
            launch_paged_sparse_mqa_bf16_typed<true, 1>(args, device, stream);
        else
            launch_paged_sparse_mqa_bf16_typed<true, 2>(args, device, stream);
    } else {
        if (args.opt.base.heads <= 16)
            launch_paged_sparse_mqa_bf16_typed<false, 1>(args, device, stream);
        else
            launch_paged_sparse_mqa_bf16_typed<false, 2>(args, device, stream);
    }
}
} // namespace deepgemm
