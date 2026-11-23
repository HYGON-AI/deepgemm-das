// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <torch/extension.h>

namespace deepgemm::mega_mhc {

// 一次 mega_mhc 调用的 host 侧描述。
// 这里只放裸指针和标量，目的是让本头文件也能被"持有 pybind 模块的非 HIP 编译单元"使用；
// 各架构的源文件再把它翻译成自己的 device 参数结构体。
struct MegaMhcLaunchArgs {
  const void* x;
  const void* residual;
  const void* prev_mix;        // 空指针 => normal 模式
  const void* post_mix;
  const void* comb_mix;
  const void* fn;
  const void* mix_scales;
  const void* mix_bases;
  const void* rmsnorm_weight;
  void* new_residual;
  void* u_io;                  // [T,H] bf16：y_bf16，或替代它的暂存区
  void* new_prev_mix;          // 空指针 => normal 模式
  void* new_post_mix;
  void* new_comb_mix;
  void* y_fp8;                 // 可为空
  void* y_gemm_sf;             // 可为空
  void* y_routed_sf;           // 可为空
  void* y_shared_sf;           // 可为空
  void* partial;               // [T, num_slots, 32] fp32
  void* arrive;                // [num_m_blocks] int32，已清零；split 模式下为空
  void* fn_packed;             // [2][24][4H] bf16，每个 k 组内 hi4/lo4 交错
  int pack_fn;                 // 1 = 启动 pack kernel；0 = fn_packed 已就绪
  int64_t fn_half;             // 每一半的元素个数
  bool fused;                  // true：单次 launch，用 last-arrival 做 epilogue
  int num_tokens;
  int hidden;
  int num_splits;
  int num_slots;
  int gemm_sf_stride1;
  int shared_sf_block_m;
  int shared_sf_stride1;
  int num_sinkhorn_iters;
  float hc_norm_eps;
  float hc_pre_eps;
  float hc_post_scale;
  float sinkhorn_eps;
  float rmsnorm_eps;
  float rmsnorm_scale;
  bool is_shifted;
  int flags;                   // bit0：请求输出 E4M3FN + SF
  void* stream;
};

// 共享实现入口（定义在 csrc/arch/gfx936_gfx938_shared/mega_mhc.cu，编译进
// gfx936 / gfx938 kernel 库；gfx92a 由 kernel_stubs_gfx92a.cpp 提供同名 stub）。
void mega_mhc_launch(const MegaMhcLaunchArgs& args);
void mega_mhc_pack_fn(const MegaMhcLaunchArgs& args);

// 面向 pybind 的入口（host 侧检查 + workspace）。
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
              const int& shared_sf_block_m);

// 把 `fn` 一次性打包成 BF16 hi/lo 布局，供多次调用复用。
//
// 失效是**显式**的：返回的缓冲区与这批权重值绑定，`mega_mhc(..., fn_packed=...)`
// 会原样消费它。这里没有隐藏缓存，也没有按 data_ptr 做的查找，
// 所以权重变了就必须重新调用一次 prepare。
torch::Tensor mega_mhc_prepare_fn(const torch::Tensor& fn);

}  // namespace deepgemm::mega_mhc

// 本算子的绑定定义，供统一扩展入口注册。
#define MEGA_MHC_PYBIND \
    m.def("mega_mhc", &deepgemm::mega_mhc::mega_mhc, "mega_mhc (Mega mHC seam)", \
        py::arg("x"), \
        py::arg("residual"), \
        py::arg("shifted_prev_mix"), \
        py::arg("post_mix"), \
        py::arg("comb_res_mix"), \
        py::arg("fn"), \
        py::arg("mix_scales"), \
        py::arg("mix_bases"), \
        py::arg("hc_mult"), \
        py::arg("hc_norm_eps"), \
        py::arg("hc_pre_eps"), \
        py::arg("hc_post_scale"), \
        py::arg("sinkhorn_eps"), \
        py::arg("num_sinkhorn_iters"), \
        py::arg("rmsnorm_weight"), \
        py::arg("rmsnorm_eps"), \
        py::arg("rmsnorm_scale"), \
        py::arg("new_residual"), \
        py::arg("new_prev_mix"), \
        py::arg("new_post_mix"), \
        py::arg("new_comb_res_mix"), \
        py::arg("y_bf16") = std::nullopt, \
        py::arg("y_fp8") = std::nullopt, \
        py::arg("y_gemm_sf") = std::nullopt, \
        py::arg("y_routed_sf") = std::nullopt, \
        py::arg("y_shared_sf") = std::nullopt, \
        py::arg("fn_packed") = std::nullopt, \
        py::arg("shared_sf_block_m") = 0); \
    m.def("mega_mhc_prepare_fn", &deepgemm::mega_mhc::mega_mhc_prepare_fn, \
        "pack mega_mhc projection weights once for reuse", \
        py::arg("fn"));
