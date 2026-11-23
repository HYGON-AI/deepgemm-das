// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// mega_mhc 的 host 入口：输入契约检查、workspace 与 split 选择。
// 这里只放 host 代码；device 侧的实现体在
// csrc/arch/gfx936_gfx938_shared/mega_mhc.cuh，由各架构 kernel 库导出
// mega_mhc_launch / mega_mhc_pack_fn。

#include "arch_dispatch.h"

#include "mega_mhc.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <string>

namespace deepgemm::mega_mhc {

namespace {

constexpr int kChunkHHost = 32;      // 必须与 mega_mhc.cuh 的 kChunkH 一致
constexpr int kNumHCOutputsHost = 24;  // R * (R + 2)，R = 4
constexpr int kThreadsHost = 256;    // 必须与 kNumThreads 一致
constexpr int kMaxHiddenHost = 32768;  // 仅作合理性上界；现在已经没有任何东西把 H 暂存进 LDS
constexpr int kMaxSlotsHost = 160;   // split 上限；epilogue 按
                                     // kMaxEpilogueSlots 分块暂存每 token 的
                                     // partial 块，所以这不占 LDS

// 从 hidden/kChunkHHost 的合法除子中选择 split，使CTA数量覆盖设备并摊薄每个CTA的启动成本。
int choose_splits(int hidden, int num_m_blocks, int num_sms) {
  const int total = hidden / kChunkHHost;
  // 一个 CTA 是 256 线程（4 个 wave）；目标为每 CU 八个 CTA，以兼顾小 T 覆盖与尾部平衡。
  const int target = std::max(1, (8 * num_sms + num_m_blocks - 1) / num_m_blocks);
  // 小token块保留更高的split下限，以增加独立CTA；最终选择仍满足完整chunk整除约束。
  const int floor_splits = num_m_blocks <= 2 ? 80 : (num_m_blocks <= 64 ? 40 : 1);
  const int want = std::max(target, floor_splits);
  // 只选完整 chunk 的等分，且每个 CTA 至少处理两个 chunk。
  const int cap = std::min(kMaxSlotsHost, std::max(1, total / 2));
  int best = 1;
  for (int d = 1; d <= total && d <= cap; ++d) {
    if (total % d != 0) continue;
    best = d;
    if (d >= want) break;
  }
  return best;
}

}  // namespace

void mega_mhc(const torch::Tensor& x,
              const torch::Tensor& residual,
              const std::optional<torch::Tensor>& shifted_prev_mix,
              const torch::Tensor& post_mix,
              const torch::Tensor& comb_res_mix,
              const torch::Tensor& fn,
              const torch::Tensor& mix_scales,
              const torch::Tensor& mix_bases,
              const int& hc_mult,
              const float& hc_norm_eps,
              const float& hc_pre_eps,
              const float& hc_post_scale,
              const float& sinkhorn_eps,
              const int& num_sinkhorn_iters,
              const torch::Tensor& rmsnorm_weight,
              const float& rmsnorm_eps,
              const float& rmsnorm_scale,
              const torch::Tensor& new_residual,
              const std::optional<torch::Tensor>& new_prev_mix,
              const torch::Tensor& new_post_mix,
              const torch::Tensor& new_comb_res_mix,
              const std::optional<torch::Tensor>& y_bf16,
              const std::optional<torch::Tensor>& y_fp8,
              const std::optional<torch::Tensor>& y_gemm_sf,
              const std::optional<torch::Tensor>& y_routed_sf,
              const std::optional<torch::Tensor>& y_shared_sf,
              const std::optional<torch::Tensor>& fn_packed,
              const int& shared_sf_block_m) {
  // ---- 输出组合契约（与官方 API 同一套规则）---------------------------------
  const bool is_shifted = shifted_prev_mix.has_value();
  TORCH_CHECK(is_shifted == new_prev_mix.has_value(),
              "mega_mhc: shifted_prev_mix and new_prev_mix must both be given or both be None");
  TORCH_CHECK(y_routed_sf.has_value() == y_shared_sf.has_value(),
              "mega_mhc: y_routed_sf and y_shared_sf must be given together");
  TORCH_CHECK(!(y_gemm_sf.has_value() && y_routed_sf.has_value()),
              "mega_mhc: supply either y_gemm_sf or (y_routed_sf, y_shared_sf)");
  TORCH_CHECK(y_bf16.has_value() || y_fp8.has_value(),
              "mega_mhc: at least one of y_bf16 / y_fp8 must be requested");
  TORCH_CHECK(y_fp8.has_value() == (y_gemm_sf.has_value() || y_routed_sf.has_value()),
              "mega_mhc: the FP8 output requires its scaling factors");
  TORCH_CHECK(y_shared_sf.has_value() ? shared_sf_block_m > 0 : shared_sf_block_m == 0,
              "mega_mhc: shared_sf_block_m must be > 0 exactly when y_shared_sf is given");

  const int num_tokens = static_cast<int>(x.size(0));
  const int hidden = static_cast<int>(x.size(1));
  // 只接受 HIP 张量：否则一组 CPU 张量会通过下面的 dtype/shape/device 检查
  // （它们都"在同一设备上"），然后把 CPU 指针交给 kernel。
  TORCH_CHECK(x.is_cuda(), "mega_mhc: tensors must be on a HIP device");
  TORCH_CHECK(x.dim() == 2, "mega_mhc: x must be [T, H]");
  TORCH_CHECK(hidden > 0, "mega_mhc: H must be positive");
  if (num_tokens == 0) return;

  // 在任何设备属性查询、workspace 或 stream 查询之前，先把输入所在设备设为当前设备，
  // 这样 SM 数和 launch stream 都指向张量真正所在的设备。
  const c10::cuda::CUDAGuard device_guard(x.device());

  // ---- 形状契约 -------------------------------------------------------------
  TORCH_CHECK(hc_mult == 4, "mega_mhc: hc_mult must be 4, got ", hc_mult);
  TORCH_CHECK(residual.dim() == 3 && residual.size(0) == num_tokens &&
                  residual.size(1) == hc_mult && residual.size(2) == hidden,
              "mega_mhc: residual must be [T, hc_mult, H] matching x");
  TORCH_CHECK(post_mix.sizes() == torch::IntArrayRef({num_tokens, hc_mult, 1}),
              "mega_mhc: post_mix must be [T, hc_mult, 1]");
  TORCH_CHECK(comb_res_mix.sizes() ==
                  torch::IntArrayRef({num_tokens, hc_mult, hc_mult}),
              "mega_mhc: comb_res_mix must be [T, hc_mult, hc_mult]");
  TORCH_CHECK(fn.sizes() == torch::IntArrayRef({hc_mult * (hc_mult + 2), hc_mult * hidden}),
              "mega_mhc: fn must be [hc_mult*(hc_mult+2), hc_mult*H]");
  TORCH_CHECK(mix_scales.numel() == 3, "mega_mhc: mix_scales must be [3]");
  TORCH_CHECK(mix_bases.numel() == hc_mult * (hc_mult + 2),
              "mega_mhc: mix_bases must be [hc_mult*(hc_mult+2)]");
  TORCH_CHECK(rmsnorm_weight.numel() == hidden, "mega_mhc: rmsnorm_weight must be [H]");
  TORCH_CHECK(num_sinkhorn_iters >= 1, "mega_mhc: num_sinkhorn_iters must be >= 1");
  TORCH_CHECK(hidden % kChunkHHost == 0,
              "mega_mhc: H must be a multiple of ", kChunkHHost, ", got ", hidden);
  TORCH_CHECK(hidden <= kMaxHiddenHost, "mega_mhc: H must be <= ", kMaxHiddenHost,
              " (sanity bound), got ", hidden);
  TORCH_CHECK(new_residual.sizes() == residual.sizes(),
              "mega_mhc: new_residual must have the residual shape");
  TORCH_CHECK(new_post_mix.sizes() == post_mix.sizes(),
              "mega_mhc: new_post_mix must have the post_mix shape");
  TORCH_CHECK(new_comb_res_mix.sizes() == comb_res_mix.sizes(),
              "mega_mhc: new_comb_res_mix must have the comb_res_mix shape");
  if (is_shifted) {
    const auto& prev = shifted_prev_mix.value();
    TORCH_CHECK(prev.dim() == 3 && prev.size(0) == num_tokens &&
                    prev.size(1) == hc_mult && prev.size(2) == 1,
                "mega_mhc: shifted_prev_mix must be [T, hc_mult, 1]");
    TORCH_CHECK(new_prev_mix->sizes() == prev.sizes(),
                "mega_mhc: new_prev_mix must have the shifted_prev_mix shape");
  }

  const auto device = x.device();
  const auto check_dense = [&](const torch::Tensor& tensor, const torch::ScalarType& dtype,
                               const char* name) {
    TORCH_CHECK(tensor.scalar_type() == dtype, "mega_mhc: ", name, " has the wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), "mega_mhc: ", name, " must be contiguous");
    TORCH_CHECK(tensor.device() == device, "mega_mhc: ", name, " is on another device");
  };
  check_dense(x, torch::kBFloat16, "x");
  check_dense(residual, torch::kBFloat16, "residual");
  check_dense(new_residual, torch::kBFloat16, "new_residual");
  check_dense(post_mix, torch::kFloat, "post_mix");
  check_dense(comb_res_mix, torch::kFloat, "comb_res_mix");
  check_dense(fn, torch::kFloat, "fn");
  check_dense(mix_scales, torch::kFloat, "mix_scales");
  check_dense(mix_bases, torch::kFloat, "mix_bases");
  check_dense(new_post_mix, torch::kFloat, "new_post_mix");
  check_dense(new_comb_res_mix, torch::kFloat, "new_comb_res_mix");
  check_dense(rmsnorm_weight, torch::kBFloat16, "rmsnorm_weight");
  if (is_shifted) {
    check_dense(shifted_prev_mix.value(), torch::kFloat, "shifted_prev_mix");
    check_dense(new_prev_mix.value(), torch::kFloat, "new_prev_mix");
  }
  if (y_bf16.has_value()) {
    TORCH_CHECK(y_bf16->sizes() == x.sizes(), "mega_mhc: y_bf16 must be [T, H]");
    check_dense(y_bf16.value(), torch::kBFloat16, "y_bf16");
  }
  if (y_fp8.has_value()) {
    TORCH_CHECK(y_fp8->sizes() == x.sizes(), "mega_mhc: y_fp8 must be [T, H]");
    check_dense(y_fp8.value(), torch::kFloat8_e4m3fn, "y_fp8");
    TORCH_CHECK(hidden % kThreadsHost == 0 && hidden % 128 == 0,
                "mega_mhc: the FP8 path needs H divisible by 256 and 128, got ", hidden);
  }
  const auto check_sf = [&](const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == num_tokens &&
                    tensor.size(1) == hidden / 128,
                "mega_mhc: ", name, " must be [T, H/128]");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt, "mega_mhc: ", name, " must be int32");
    TORCH_CHECK(tensor.device() == device, "mega_mhc: ", name, " is on another device");
  };
  if (y_gemm_sf.has_value()) {
    check_sf(y_gemm_sf.value(), "y_gemm_sf");
    TORCH_CHECK(y_gemm_sf->stride(0) == 1 && y_gemm_sf->stride(1) >= num_tokens,
                "mega_mhc: y_gemm_sf must be column-major with stride(1) >= T");
  }
  if (y_routed_sf.has_value()) {
    check_sf(y_routed_sf.value(), "y_routed_sf");
    TORCH_CHECK(y_routed_sf->is_contiguous(), "mega_mhc: y_routed_sf must be contiguous");
  }
  int shared_sf_stride1 = 0;
  if (y_shared_sf.has_value()) {
    check_sf(y_shared_sf.value(), "y_shared_sf");
    const int64_t align_block = ((shared_sf_block_m + 127) / 128) * 128;
    const int64_t required_rows =
        ((num_tokens + shared_sf_block_m - 1) / shared_sf_block_m) * align_block;
    TORCH_CHECK(y_shared_sf->stride(0) == 1 && y_shared_sf->stride(1) >= required_rows,
                "mega_mhc: y_shared_sf must be column-major with stride(1) >= required rows (",
                required_rows, ")");
    const int64_t required_elements =
        y_shared_sf->storage_offset() + required_rows +
        (hidden / 128 - 1) * y_shared_sf->stride(1);
    TORCH_CHECK(static_cast<uint64_t>(required_elements) * y_shared_sf->element_size() <=
                    y_shared_sf->storage().nbytes(),
                "mega_mhc: y_shared_sf storage is too small for the shared SF layout");
    shared_sf_stride1 = static_cast<int>(y_shared_sf->stride(1));
  }

  // ---- launch 几何 ----------------------------------------------------------
  const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int num_m_blocks = (num_tokens + 15) / 16;
  const int num_splits = choose_splits(hidden, num_m_blocks, num_sms);
  TORCH_CHECK(hidden % (num_splits * kChunkHHost) == 0,
              "mega_mhc: internal split selection produced an illegal split count");

  // ---- workspace ------------------------------------------------------------
  const int num_slots = num_splits;
  const int64_t partial_bytes =
      static_cast<int64_t>(num_tokens) * num_slots * 32 * sizeof(float);
  const bool needs_bf16_scratch = !y_bf16.has_value();
  const int64_t bf16_scratch_bytes =
      needs_bf16_scratch ? static_cast<int64_t>(num_tokens) * hidden * 2 : 0;
  // 打包后的 BF16 hi/lo 投影权重。调用方给了 `fn_packed` 就说明它已经打包过一次
  // （见 `mega_mhc_prepare_fn`），本次调用完全不打包；否则从 scratch 里切一块，
  // 每次调用重建。
  const int64_t fn_half = static_cast<int64_t>(hc_mult * (hc_mult + 2)) * hc_mult * hidden;
  const int64_t fn_packed_bytes = fn_packed.has_value() ? 0 : 2 * fn_half * 2;
  if (fn_packed.has_value()) {
    const auto& packed = fn_packed.value();
    TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kBFloat16 &&
                    packed.is_contiguous() && packed.numel() == 2 * fn_half,
                "mega_mhc: fn_packed must be the contiguous BF16 buffer returned by "
                "mega_mhc_prepare_fn for this fn (expected ", 2 * fn_half, " elements)");
    TORCH_CHECK(packed.device() == x.device(),
                "mega_mhc: fn_packed is on another device");
  }
  auto scratch = torch::empty({partial_bytes + bf16_scratch_bytes + 16 + fn_packed_bytes},
                             x.options().dtype(torch::kByte));
  void* partial_ptr = scratch.data_ptr();
  void* u_io_ptr = y_bf16.has_value()
                       ? y_bf16->data_ptr()
                       : static_cast<void*>(static_cast<char*>(scratch.data_ptr()) + partial_bytes);

  // 默认采用 main + reduce；可选 fused 在最后抵达的 CTA 内完成该 m-block 的 epilogue。
  bool fused = false;
  if (const char* mode = std::getenv("MEGA_MHC_LAUNCH")) {
    if (std::string(mode) == "fused") fused = true;
  }
  torch::Tensor arrive;
  if (fused) {
    arrive = torch::zeros({num_m_blocks}, x.options().dtype(torch::kInt));
  }

  MegaMhcLaunchArgs args{};
  args.x = x.data_ptr();
  args.residual = residual.data_ptr();
  args.prev_mix = is_shifted ? shifted_prev_mix->data_ptr() : nullptr;
  args.post_mix = post_mix.data_ptr();
  args.comb_mix = comb_res_mix.data_ptr();
  args.fn = fn.data_ptr();
  args.mix_scales = mix_scales.data_ptr();
  args.mix_bases = mix_bases.data_ptr();
  args.rmsnorm_weight = rmsnorm_weight.data_ptr();
  args.new_residual = new_residual.data_ptr();
  args.u_io = u_io_ptr;
  args.new_prev_mix = is_shifted ? new_prev_mix->data_ptr() : nullptr;
  args.new_post_mix = new_post_mix.data_ptr();
  args.new_comb_mix = new_comb_res_mix.data_ptr();
  args.y_fp8 = y_fp8.has_value() ? y_fp8->data_ptr() : nullptr;
  args.y_gemm_sf = y_gemm_sf.has_value() ? y_gemm_sf->data_ptr() : nullptr;
  args.y_routed_sf = y_routed_sf.has_value() ? y_routed_sf->data_ptr() : nullptr;
  args.y_shared_sf = y_shared_sf.has_value() ? y_shared_sf->data_ptr() : nullptr;
  const int64_t fn_off = ((partial_bytes + bf16_scratch_bytes + 15) / 16) * 16;  // 保持 8 字节 fragment 读取的对齐
  args.fn_packed = fn_packed.has_value()
                      ? const_cast<void*>(fn_packed->data_ptr())
                      : static_cast<void*>(static_cast<char*>(scratch.data_ptr()) + fn_off);
  args.fn_half = fn_half;
  args.pack_fn = fn_packed.has_value() ? 0 : 1;
  args.partial = partial_ptr;
  args.arrive = fused ? arrive.data_ptr() : nullptr;
  args.fused = fused;
  args.num_tokens = num_tokens;
  args.hidden = hidden;
  args.num_splits = num_splits;
  args.num_slots = num_slots;
  args.gemm_sf_stride1 = y_gemm_sf.has_value() ? static_cast<int>(y_gemm_sf->stride(1)) : 0;
  args.shared_sf_block_m = shared_sf_block_m;
  args.shared_sf_stride1 = shared_sf_stride1;
  args.num_sinkhorn_iters = num_sinkhorn_iters;
  args.hc_norm_eps = hc_norm_eps;
  args.hc_pre_eps = hc_pre_eps;
  args.hc_post_scale = hc_post_scale;
  args.sinkhorn_eps = sinkhorn_eps;
  args.rmsnorm_eps = rmsnorm_eps;
  args.rmsnorm_scale = rmsnorm_scale;
  args.is_shifted = is_shifted;
  args.flags = y_fp8.has_value() ? 1 : 0;
  args.stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  // ---- 发射（实现由当前加载的 kernel 库提供）--------------------------------
#if DEEPGEMM_HAS_GFX93X
  mega_mhc_launch(args);
#else
  ::deepgemm::arch::unsupported("mega_mhc");
#endif
}

// ---------------------------------------------------------------------------
// 一次性打包入口
// ---------------------------------------------------------------------------

torch::Tensor mega_mhc_prepare_fn(const torch::Tensor& fn) {
  TORCH_CHECK(fn.is_cuda(), "mega_mhc_prepare_fn: fn must be on a HIP device");
  TORCH_CHECK(fn.scalar_type() == torch::kFloat && fn.is_contiguous(),
              "mega_mhc_prepare_fn: fn must be a contiguous float32 tensor");
  TORCH_CHECK(fn.dim() == 2 && fn.size(0) == kNumHCOutputsHost,
              "mega_mhc_prepare_fn: fn must be [", kNumHCOutputsHost, ", 4H]");
  const int hc_mult = 4;
  const int hidden = static_cast<int>(fn.size(1) / hc_mult);
  TORCH_CHECK(hidden * hc_mult == fn.size(1) && hidden > 0,
              "mega_mhc_prepare_fn: fn.shape[1] must be 4H");
  const c10::cuda::CUDAGuard device_guard(fn.device());
  const int64_t fn_half = static_cast<int64_t>(fn.size(0)) * fn.size(1);
  auto packed = torch::empty({2 * fn_half}, fn.options().dtype(torch::kBFloat16));
  MegaMhcLaunchArgs args{};
  args.fn = static_cast<const void*>(fn.data_ptr());
  args.fn_packed = static_cast<void*>(packed.data_ptr());
  args.fn_half = fn_half;
  args.pack_fn = 1;
  args.stream = at::cuda::getCurrentCUDAStream(fn.get_device()).stream();
#if DEEPGEMM_HAS_GFX93X
  mega_mhc_pack_fn(args);
#else
  ::deepgemm::arch::unsupported("mega_mhc_prepare_fn");
#endif
  return packed;
}

}  // namespace deepgemm::mega_mhc
