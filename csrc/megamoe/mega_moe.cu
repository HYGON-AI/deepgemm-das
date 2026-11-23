// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Host launcher + torch bindings for HCU gfx938 MegaMoE.
//
// Provides:
//   - IPC buffer allocation (allocate_hip_ipc_buffer, open_hip_ipc_handles)
//   - IPC signal buffer allocation (allocate_hip_ipc_signal_buffer)
//   - Supernode Fabric/RPC buffer allocation (allocate_hip_fabric_*)
//   - Peer pointer installation (set_mega_moe_peer_ptrs)
//   - Buffer size calculation (get_symm_buffer_size_for_mega_moe)
//   - Fused kernel launch (fp8_mega_moe)
//   - Combine kernel launch (combine)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

// Supernode (cross-node) peer memory: DTK HIP fabric shareable handles.
// <hip/hip_hsa_ext.h> only ships with newer DTKs (26.10+). When it is missing
// the fabric bindings are omitted, DG_MEGA_PEER_MEMORY=rpc fails loudly from
// Python, and HIP IPC stays the only transport.
#include <cstring>
#if defined(__has_include)
#  if __has_include(<hip/hip_hsa_ext.h>)
#    include <hip/hip_hsa_ext.h>
#    define DG_MEGA_HAS_HIP_FABRIC 1
#  endif
#endif

#include "kernels/params.h"
#include "kernels/workspace.cuh"
#include "kernels/buffer_layout.cuh"
#include "kernels/activation_device.cuh"

// Kernel build selection: skip low-latency or high-throughput template
// instantiations via DG_MEGA_SKIP_LL / DG_MEGA_SKIP_HT (set by setup.py
// from DG_MEGA_BUILD_LL/HT).
#if !defined(DG_MEGA_SKIP_LL) || !defined(DG_MEGA_SKIP_HT)
// Unified split kernel: blockM <= 48 instantiates the LL GEMM family,
// blockM >= 64 instantiates the HT gemm_ht_tile family.
#include "kernels/mega_moe_kernel_unified.cuh"
#endif
#include "kernels/mega_moe_preprocess.cuh"

namespace deep_gemm::mega::hcu {

// ============================================================================
// Runtime debug level --set from DG_DBUG_LEVEL env var
// ============================================================================
#if DG_MEGAMOE_DBUG
static void init_dbug_level_from_env() {
    const char* env = std::getenv("DG_DBUG_LEVEL");
    int lvl = env ? atoi(env) : 2;
    dg_dbug_level = lvl;
    printf("[HOST] DG_DBUG_LEVEL=%d\n", lvl);
    fflush(stdout);
}
#else
static void init_dbug_level_from_env() {}
#endif

// ============================================================================
// Phase profiling buffer (DG_MEGA_PROF=1): [num_sms * 16] int64 clock64
// timestamps/counters written by thread 0 of each block. Fetch from Python via
// mega_moe_prof_fetch(); slots are documented in the split kernels.
// ============================================================================
static torch::Tensor g_mega_prof_buf;
static uint64_t* mega_moe_prof_ptr(int num_sms, const torch::TensorOptions& opts) {
    static const bool enabled = [] {
        const char* e = std::getenv("DG_MEGA_PROF");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (!enabled) return nullptr;
    if (!g_mega_prof_buf.defined() || g_mega_prof_buf.numel() < num_sms * 32)
        g_mega_prof_buf = torch::zeros({num_sms * 32}, opts.dtype(torch::kLong));
    else
        g_mega_prof_buf.zero_();
    return reinterpret_cast<uint64_t*>(g_mega_prof_buf.data_ptr());
}
static torch::Tensor mega_moe_prof_fetch() {
    if (!g_mega_prof_buf.defined())
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong));
    return g_mega_prof_buf.cpu().clone();
}

// ============================================================================
//   - allocate_hip_ipc_buffer:        hipExtMallocWithFlags + hipDeviceMallocUncached
//   - allocate_hip_ipc_signal_buffer: hipExtMallocWithFlags + hipDeviceMallocUncached
//   - IPC handles use HIP_IPC_HANDLE_SIZE + handle.reserved
//   - free_hip_ipc_buffer actually calls hipFree (not no-op)
// ============================================================================

#define DG_HIP_CHECK(expr) do { \
    const hipError_t _status = (expr); \
    if (_status != hipSuccess) { \
        TORCH_CHECK(false, #expr " failed: ", hipGetErrorString(_status)); \
    } \
} while (0)

// Compile-time INT8 kernel instantiation guard.
// setup.py with DG_MEGA_BUILD_INT8=0 passes -DDG_MEGA_SKIP_INT8=1: int8
// instantiations are not compiled (roughly halves build time) and any
// is_int8 launch fails fast instead of dispatching to a missing kernel.
// NOTE: arguments must be *macro calls* (LAUNCH_*_K(...)), never raw
// kernel<<<...>>>(...) text --aicc rewrites <<< >>> inside macro arguments.
#if defined(DG_MEGA_SKIP_INT8)
#define DG_MEGA_IF_INT8_ELSE(INT8_LAUNCH, FP8_LAUNCH) \
    do { \
        TORCH_CHECK(!(is_int8), "MegaMoE INT8 kernels not compiled (DG_MEGA_BUILD_INT8=0)"); \
        FP8_LAUNCH; \
    } while (0)
#else
#define DG_MEGA_IF_INT8_ELSE(INT8_LAUNCH, FP8_LAUNCH) \
    do { \
        if (is_int8) { INT8_LAUNCH; } else { FP8_LAUNCH; } \
    } while (0)
#endif

static std::tuple<torch::Tensor, int64_t, pybind11::bytes>
allocate_hip_ipc_buffer(int64_t num_bytes) {
    TORCH_CHECK(num_bytes > 0, "num_bytes must be positive");
    void* ptr = nullptr;
    // Uncached memory for cross-GPU access on gfx938. Fine-grained buffers let
    // local writes linger in the writer's L2 (remote reads then see stale DRAM
    // data) and have shown dropped remote stores/atomics under load; uncached
    // writes reach the coherence point directly.
    DG_HIP_CHECK(hipExtMallocWithFlags(&ptr, static_cast<size_t>(num_bytes),
                                       hipDeviceMallocUncached));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(num_bytes)));

    hipIpcMemHandle_t handle{};
    DG_HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
    auto tensor = torch::from_blob(
        ptr,
        {num_bytes},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
    return {tensor, reinterpret_cast<int64_t>(ptr),
            pybind11::bytes(handle.reserved, HIP_IPC_HANDLE_SIZE)};
}

static std::vector<int64_t>
open_hip_ipc_handles(const std::vector<pybind11::bytes>& handles, int rank_idx) {
    TORCH_CHECK(rank_idx >= 0 && rank_idx < static_cast<int>(handles.size()),
                "rank_idx out of bounds for HIP IPC handles");
    std::vector<int64_t> ptrs(handles.size(), 0);
    for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
        if (i == rank_idx) continue;  // local handle already open
        auto handle_str = handles[i].cast<std::string>();
        TORCH_CHECK(handle_str.size() == HIP_IPC_HANDLE_SIZE, "Invalid HIP IPC handle size");
        hipIpcMemHandle_t handle{};
        std::memcpy(handle.reserved, handle_str.data(), HIP_IPC_HANDLE_SIZE);
        void* ptr = nullptr;
        DG_HIP_CHECK(hipIpcOpenMemHandle(&ptr, handle, hipIpcMemLazyEnablePeerAccess));
        ptrs[i] = reinterpret_cast<int64_t>(ptr);
    }
    return ptrs;
}

static void close_hip_ipc_handles(const std::vector<int64_t>& ptrs) {
    for (int64_t ptr : ptrs) {
        if (ptr != 0)
            DG_HIP_CHECK(hipIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
    }
}

static void free_hip_ipc_buffer(int64_t ptr) {
    if (ptr != 0)
        DG_HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
}

// Local (non-P2P) buffer: regular hipMalloc with L2 cache for GEMM areas.
// These buffers are only accessed on the local GPU (no sym.map() calls),
// so they benefit from L2 cache for GEMM performance.
static std::tuple<torch::Tensor, int64_t>
allocate_hip_local_buffer(int64_t num_bytes) {
    TORCH_CHECK(num_bytes > 0, "local num_bytes must be positive");
    void* ptr = nullptr;
    DG_HIP_CHECK(hipMalloc(&ptr, static_cast<size_t>(num_bytes)));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(num_bytes)));
    auto tensor = torch::from_blob(
        ptr,
        {num_bytes},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
    return {tensor, reinterpret_cast<int64_t>(ptr)};
}

static void free_hip_local_buffer(int64_t ptr) {
    if (ptr != 0)
        DG_HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
}

// IPC signal buffer (separate from sym_buffer, for rank barrier)
// Uses hipDeviceMallocUncached --uncached memory for signal/coherence
static std::tuple<int64_t, pybind11::bytes>
allocate_hip_ipc_signal_buffer(int64_t num_bytes) {
    TORCH_CHECK(num_bytes > 0, "signal num_bytes must be positive");
    void* ptr = nullptr;
    DG_HIP_CHECK(hipExtMallocWithFlags(&ptr, static_cast<size_t>(num_bytes), hipDeviceMallocUncached));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(num_bytes)));

    hipIpcMemHandle_t handle{};
    DG_HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
    return {reinterpret_cast<int64_t>(ptr),
            pybind11::bytes(handle.reserved, HIP_IPC_HANDLE_SIZE)};
}

static void free_hip_ipc_signal_buffer(int64_t ptr) {
    if (ptr != 0)
        DG_HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
}

// ============================================================================
// Supernode peer memory (DTK HIP fabric handles)
//
// HIP IPC handles only reach GPUs of one node; on a supernode the ranks span
// hosts, so buffers are exported with hipHsaExtRpcMemoryCreate() and attached
// per peer with hipHsaExtRpcMemoryAttachDevice() (<hip/hip_hsa_ext.h>), so no
// HSA headers or hsa-runtime64 linkage are needed. The result is plain device
// pointers, so set_mega_moe_peer_ptrs() and comm.cuh are unchanged.
//
// Compiled only when the toolchain provides <hip/hip_hsa_ext.h> (DTK 26.10+).
// HIP IPC is the default; DG_MEGA_PEER_MEMORY=rpc selects this path at run
// time. Both transports allocate uncached device memory.
// ============================================================================

#if defined(DG_MEGA_HAS_HIP_FABRIC)

// Fabric exports must be 2 MiB aligned; exporting the unaligned size makes
// remote attaches fail (observed on the supernode branch).
static int64_t align_fabric_bytes(int64_t num_bytes) {
    constexpr int64_t kFabricAlignmentBytes = 2 * 1024 * 1024;
    return (num_bytes + kFabricAlignmentBytes - 1) & ~(kFabricAlignmentBytes - 1);
}

static pybind11::bytes make_fabric_handle_bytes(void* ptr, int64_t num_bytes) {
    TORCH_CHECK(num_bytes > 0, "fabric buffer size must be positive");
    hipFabricHandle_t handle{};
    DG_HIP_CHECK(hipHsaExtRpcMemoryCreate(
        ptr, static_cast<size_t>(num_bytes), &handle));
    return pybind11::bytes(reinterpret_cast<const char*>(&handle),
                           sizeof(hipFabricHandle_t));
}

// Both IPC and fabric use hipDeviceMallocUncached. Fine-grained buffers let
// local writes linger in the writer's L2, so remote reads see stale DRAM data
// and remote stores/atomics can be dropped under load.
static void* alloc_fabric_memory(int64_t* alloc_bytes, int64_t num_bytes) {
    *alloc_bytes = align_fabric_bytes(num_bytes);
    void* ptr = nullptr;
    DG_HIP_CHECK(hipExtMallocWithFlags(
        &ptr, static_cast<size_t>(*alloc_bytes), hipDeviceMallocUncached));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(*alloc_bytes)));
    return ptr;
}

static std::tuple<torch::Tensor, int64_t, pybind11::bytes>
allocate_hip_fabric_buffer(int64_t num_bytes) {
    TORCH_CHECK(num_bytes > 0, "num_bytes must be positive");
    int64_t alloc_bytes = 0;
    void* ptr = alloc_fabric_memory(&alloc_bytes, num_bytes);
    auto tensor = torch::from_blob(
        ptr, {num_bytes},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
    return {tensor, reinterpret_cast<int64_t>(ptr),
            make_fabric_handle_bytes(ptr, alloc_bytes)};
}

static std::tuple<int64_t, pybind11::bytes>
allocate_hip_fabric_signal_buffer(int64_t num_bytes) {
    TORCH_CHECK(num_bytes > 0, "signal num_bytes must be positive");
    int64_t alloc_bytes = 0;
    void* ptr = alloc_fabric_memory(&alloc_bytes, num_bytes);
    return {reinterpret_cast<int64_t>(ptr),
            make_fabric_handle_bytes(ptr, alloc_bytes)};
}

static std::vector<int64_t>
open_hip_fabric_handles(const std::vector<pybind11::bytes>& handles, int rank_idx) {
    TORCH_CHECK(rank_idx >= 0 && rank_idx < static_cast<int>(handles.size()),
                "rank_idx out of bounds for HIP fabric handles");
    int device = 0;
    DG_HIP_CHECK(hipGetDevice(&device));
    std::vector<int64_t> ptrs(handles.size(), 0);
    for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
        if (i == rank_idx) continue;  // local buffer is already mapped
        auto handle_str = handles[i].cast<std::string>();
        TORCH_CHECK(handle_str.size() == sizeof(hipFabricHandle_t),
                    "Invalid HIP fabric handle size");
        hipFabricHandle_t handle{};
        std::memcpy(&handle, handle_str.data(), sizeof(hipFabricHandle_t));
        void* ptr = nullptr;
        DG_HIP_CHECK(hipHsaExtRpcMemoryAttachDevice(&handle, 1, &device, &ptr));
        TORCH_CHECK(ptr != nullptr, "HIP fabric attach returned a null pointer");
        ptrs[i] = reinterpret_cast<int64_t>(ptr);
    }
    return ptrs;
}

static void close_hip_fabric_handles(const std::vector<int64_t>& ptrs) {
    for (int64_t ptr : ptrs) {
        if (ptr != 0)
            DG_HIP_CHECK(hipHsaExtRpcMemoryDetach(reinterpret_cast<void*>(ptr)));
    }
}

#endif  // DG_MEGA_HAS_HIP_FABRIC

// Write peer buffer + signal pointers to sym_buffer header
static void set_mega_moe_peer_ptrs(
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const std::vector<int64_t>& signal_ptrs) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    TORCH_CHECK(signal_ptrs.size() == sym_buffer_ptrs.size(), "ptr size mismatch");
    TORCH_CHECK(sym_buffer.is_cuda(), "sym_buffer must be CUDA");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8, "sym_buffer must be int8");

    auto* base = static_cast<uint8_t*>(sym_buffer.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Write peer buffer ptrs
    uint64_t buf_ptrs_offset = 0;
    hipMemcpyAsync(base + buf_ptrs_offset, sym_buffer_ptrs.data(),
                   sizeof(int64_t) * num_ranks, hipMemcpyHostToDevice, stream);

    // Write peer signal ptrs
    uint64_t sig_ptrs_offset = align_u64(static_cast<uint64_t>(num_ranks) * sizeof(uint8_t*), 16);
    hipMemcpyAsync(base + sig_ptrs_offset, signal_ptrs.data(),
                   sizeof(int64_t) * num_ranks, hipMemcpyHostToDevice, stream);
}

// ============================================================================
// Buffer size calculation
// ============================================================================

// L1 ring-buffer enable policy, host side. Every call site that derives the
// local-buffer layout must go through this so sizing and launch always agree.
// Env overrides (testing knobs):
//   DG_MEGA_DISABLE_RING: force off
//   DG_MEGA_ENABLE_RING:  force on (still gated by the pool activation bound)
// Default: heavy-GEMM shapes (hidden*ih > V4's 4096*2048) enable the ring —
// their longer L1 tiles hide the producer sync (measured within noise).
// Light shapes enable it only up to a tokens/rank capacity bound: beyond
// that the sync costs a measured ~3-8% (see megamoe/l1_ring_buffer.md).
// Below a small-capacity bound the pool is too small to bother ringing.
// Threshold envs:
//   DG_MEGA_RING_MIN_TOKENS:        below -> off           (default 1024)
//   DG_MEGA_RING_LIGHT_MAX_TOKENS:  light-shape cap bound  (default 16384)
static bool l1_ring_enabled_for_shape(int64_t hidden, int64_t intermediate_hidden,
                                      int64_t num_max_tokens_per_rank) {
    if (std::getenv("DG_MEGA_DISABLE_RING"))
        return false;
    if (std::getenv("DG_MEGA_ENABLE_RING"))
        return true;
    const int64_t min_tokens = std::getenv("DG_MEGA_RING_MIN_TOKENS")
        ? std::atoll(std::getenv("DG_MEGA_RING_MIN_TOKENS")) : 1024;
    const int64_t light_max_tokens = std::getenv("DG_MEGA_RING_LIGHT_MAX_TOKENS")
        ? std::atoll(std::getenv("DG_MEGA_RING_LIGHT_MAX_TOKENS")) : 16384;
    if (num_max_tokens_per_rank < min_tokens)
        return false;
    if (hidden * intermediate_hidden > 4096LL * 2048)
        return true;
    return num_max_tokens_per_rank <= light_max_tokens;
}

// MegaMoE kernels are gfx938-only (HCU builtins, wave64).  Under multi-arch
// builds the device side compiles empty stubs for other targets, so check
// the runtime arch on the host side as well.
static void mega_moe_check_arch() {
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const std::string arch =
        props && props->gcnArchName ? props->gcnArchName : "";
    TORCH_CHECK(arch.rfind("gfx938", 0) == 0,
                "deepgemm.mega (MegaMoE) is only supported on HCU gfx938, got '",
                arch.empty() ? "<unknown>" : arch, "'");
}

static std::tuple<int64_t, int64_t, pybind11::object>
get_symm_buffer_size_for_mega_moe(
    int num_ranks, int num_experts,
    int num_max_tokens_per_rank, int num_topk,
    int hidden, int intermediate_hidden,
    bool use_fp8_dispatch, const std::string& activation) {
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    TORCH_CHECK(use_fp8_dispatch, "Only FP8 dispatch is supported");
    TORCH_CHECK(activation == "swiglu", "Only swiglu activation is supported");

    // Runtime CU detection: L2 output scratch has one slot per block.
    // Buffer must be sized for the actual CU count, not the default kNumSMs=64.
    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    TORCH_CHECK(num_sms == 64 || num_sms == 72,
                "Unsupported CU count: ", num_sms, " (only 64 or 72 supported)");

    const auto ipc_bytes = get_total_ipc_buffer_bytes(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden);
    const auto local_bytes = get_local_buffer_total_bytes(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_sms,
        l1_ring_enabled_for_shape(hidden, intermediate_hidden,
                                  num_max_tokens_per_rank));

    // Slice function (returns tensors that view into ipc_buffer + local_buffer)
    auto slice_fn = pybind11::cpp_function([=](const torch::Tensor& ipc_buffer,
                                                const torch::Tensor& local_buffer) {
        auto* ipc_base = static_cast<uint8_t*>(ipc_buffer.data_ptr());
        auto* local_base = static_cast<uint8_t*>(local_buffer.data_ptr());
        uint8_t* workspace_base = get_workspace_base(ipc_base, num_ranks);
        auto slices = slice_sym_buffers(workspace_base, local_base, num_ranks, num_experts,
                                         num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate_hidden, num_sms,
                                         l1_ring_enabled_for_shape(hidden, intermediate_hidden,
                                                                   num_max_tokens_per_rank));

        auto device = ipc_buffer.device();
        auto fp8_opts = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

        auto x = torch::from_blob(slices.input_token_buffer.base,
                                  {num_max_tokens_per_rank, hidden}, fp8_opts);
        auto x_sf = torch::from_blob(slices.input_sf_buffer.base,
                                     {num_max_tokens_per_rank}, f32_opts);
        auto topk_idx = torch::from_blob(slices.input_topk_idx_buffer.base,
                                          {num_max_tokens_per_rank, num_topk}, i64_opts);
        auto topk_weights = torch::from_blob(slices.input_topk_weights_buffer.base,
                                              {num_max_tokens_per_rank, num_topk}, f32_opts);

        const uint32_t num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts / num_ranks);

        auto l1_acts = torch::from_blob(slices.l1_token_buffer.base,
                                         {static_cast<int64_t>(slices.l1_token_buffer.num_max_tokens), hidden}, fp8_opts);
        auto l1_acts_sf = torch::from_blob(slices.l1_sf_buffer.base,
                                            {static_cast<int64_t>(slices.l1_token_buffer.num_max_tokens)}, f32_opts);
        auto l2_acts = torch::from_blob(slices.l2_token_buffer.base,
                                         {static_cast<int64_t>(num_max_pool_tokens), intermediate_hidden}, fp8_opts);
        auto l2_acts_sf = torch::from_blob(slices.l2_sf_buffer.base,
                                            {static_cast<int64_t>(num_max_pool_tokens)}, f32_opts);

        return pybind11::make_tuple(x, x_sf, topk_idx, topk_weights,
                                     l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    });

    return {static_cast<int64_t>(ipc_bytes), static_cast<int64_t>(local_bytes), slice_fn};
}

// ============================================================================
// ============================================================================
// Fused MegaMoE kernel launch (low-latency)
// ============================================================================
// ============================================================================
// Fused MegaMoE launch -- single unified flow for all token counts.
//
// One launcher serves both GEMM families; mode selection is capacity-based
// (num_max_tokens_per_rank) so every rank picks the same path under
// imbalanced loads:
//   max_tokens <  DG_MEGA_HT_THRESHOLD (default 192) -> LL config
//                                                    (BM16/32/48, BN256,
//                                                    gemm_channelwise family)
//   max_tokens >= threshold                          -> HT config
//                                                    (BM64/128, BN256/512,
//                                                    gemm_ht_tile family)
// Both consume the same Marlin w6 weight layout [E, K/64, N/16, 4, 16, 16]
// (a flat 3D view of the same storage is also accepted).
// ============================================================================
#if !defined(DG_MEGA_SKIP_LL) || !defined(DG_MEGA_SKIP_HT)
static void fp8_mega_moe_launch(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights_sf,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& local_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const std::vector<int64_t>& signal_ptrs,
    int rank_idx,
    int num_max_tokens_per_rank,
    int num_experts,
    int num_topk,
    float activation_clamp,
    bool fast_math,
    bool is_int8 = false,
    int64_t tokens_all_ranks = -1) {
    init_dbug_level_from_env();
    TORCH_CHECK(y.is_cuda() && l1_weights.is_cuda() && l2_weights.is_cuda(),
                "Tensors must be CUDA");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "y must be BF16");
    // NV convention: clamp >= 0, +inf disables clamping.
    TORCH_CHECK(activation_clamp >= 0.0f, "activation_clamp must be >= 0");
    if (is_int8) {
        TORCH_CHECK(l1_weights.scalar_type() == torch::kInt8, "l1_weights must be INT8 for int8 mode");
        TORCH_CHECK(l2_weights.scalar_type() == torch::kInt8, "l2_weights must be INT8 for int8 mode");
    } else {
        TORCH_CHECK(l1_weights.scalar_type() == torch::kFloat8_e4m3fn, "l1_weights must be FP8");
        TORCH_CHECK(l2_weights.scalar_type() == torch::kFloat8_e4m3fn, "l2_weights must be FP8");
    }

    const int num_tokens = static_cast<int>(y.size(0));
    const int hidden = static_cast<int>(y.size(1));
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    TORCH_CHECK(num_experts % num_ranks == 0,
                "num_experts must be divisible by num_ranks");
    TORCH_CHECK(hidden % 8 == 0,
                "hidden must be divisible by 8 (vectorized bf16 access)");
    const int num_experts_per_rank = num_experts / num_ranks;

    // Weights are Marlin w6 permuted [E, K/64, N/16, 4, 16, 16]; a flat 3D
    // view [E, N, K] of the same storage is also accepted (the packer is
    // in-place). The kernel uses flat data pointers, so only the derived
    // (N, K) extents per expert matter.
    const auto l1_sizes = l1_weights.sizes();
    const auto l2_sizes = l2_weights.sizes();
    const int l1_dim = static_cast<int>(l1_sizes.size());
    const int l2_dim = static_cast<int>(l2_sizes.size());

    int ep_r1, ih2, hid1, ep_r2, hid2, ih;
    if (l1_dim == 3) {
        ep_r1 = l1_sizes[0]; ih2 = l1_sizes[1]; hid1 = l1_sizes[2];
    } else if (l1_dim == 6) {
        ep_r1 = l1_sizes[0];
        ih2 = static_cast<int>(l1_weights.numel()) / ep_r1 / hidden;
        hid1 = hidden;
    } else {
        TORCH_CHECK(false, "l1_weights must be 3D or 6D, got ", l1_dim, "D");
    }
    if (l2_dim == 3) {
        ep_r2 = l2_sizes[0]; hid2 = l2_sizes[1]; ih = l2_sizes[2];
    } else if (l2_dim == 6) {
        ep_r2 = l2_sizes[0];
        ih = static_cast<int>(l2_weights.numel()) / ep_r2 / hidden;
        hid2 = hidden;
    } else {
        TORCH_CHECK(false, "l2_weights must be 3D or 6D, got ", l2_dim, "D");
    }

    const int intermediate_hidden = static_cast<int>(ih);
    TORCH_CHECK(static_cast<int>(hid1) == hidden, "hidden mismatch");
    TORCH_CHECK(static_cast<int>(ih2) == 2 * intermediate_hidden, "intermediate mismatch");
    TORCH_CHECK(static_cast<int>(hid2) == hidden, "l2 hidden mismatch");
    TORCH_CHECK(static_cast<int>(ep_r1) == num_experts_per_rank, "l1 expert count mismatch");
    TORCH_CHECK(static_cast<int>(ep_r2) == num_experts_per_rank, "l2 expert count mismatch");
    TORCH_CHECK(intermediate_hidden % kGroupQuantSize == 0,
                "intermediate_hidden must be divisible by the l2 group-quant size (",
                kGroupQuantSize, ")");

    // Signal ptrs are already stored in the sym_buffer IPC header by
    // set_mega_moe_peer_ptrs at init --no per-launch H2D copy needed.
    int** d_signal_ptrs = get_peer_signal_ptrs(
        static_cast<uint8_t*>(sym_buffer.data_ptr()), num_ranks);
    (void)signal_ptrs;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto* sym_buf_ptr = static_cast<uint8_t*>(sym_buffer.data_ptr());
    auto* local_buf_ptr = static_cast<uint8_t*>(local_buffer.data_ptr());
    const int num_threads = 1024;  // 16 warps * 64 lanes

    const size_t expert_count_bytes = static_cast<size_t>(num_experts) * sizeof(uint32_t);
    const size_t expert_count_aligned = ((expert_count_bytes + 255) / 256) * 256;

    // Scatter address fast path is permanently enabled (LDS peer_delta[]):
    // verified no regression and 2~3% win across 1~4096 tokens.

    // Shared kernel argument list for all launches below.
    #define DG_MEGA_LAUNCH_ARGS \
        reinterpret_cast<hip_bfloat16*>(y.data_ptr()), \
        reinterpret_cast<const char*>(l1_weights.data_ptr()), \
        reinterpret_cast<const char*>(l2_weights.data_ptr()), \
        l1_weights_sf.data_ptr<float>(), \
        l2_weights_sf.data_ptr<float>(), \
        sym_buf_ptr, local_buf_ptr, d_signal_ptrs, rank_idx, \
        num_experts_per_rank, num_ranks, \
        num_tokens, num_max_tokens_per_rank, \
        activation_clamp, fast_math

    // ---- Config-selection token source (shared by family + tile) ---------
    //   tokens_all_ranks >= 0: exact cross-rank total from a caller-side
    //     allgather -- identical config on every rank, exact density.
    //   default (like NV): this rank's actual num_tokens -- per-rank config
    //     (safe: the family/block_m choice is receiver-local).
    //   DG_MEGA_CFG_ACTUAL=0: num_max_tokens_per_rank -- uniform config
    //     under imbalanced loads, but overestimates density.
    const bool cfg_actual =
        !std::getenv("DG_MEGA_CFG_ACTUAL") ||
        std::atoi(std::getenv("DG_MEGA_CFG_ACTUAL")) != 0;
    const float expected_tokens_per_expert =
        (tokens_all_ranks >= 0)
            ? static_cast<float>(tokens_all_ranks) * num_topk / num_experts
            : static_cast<float>(cfg_actual ? num_tokens
                                            : num_max_tokens_per_rank) *
                  num_ranks * num_topk / num_experts;

    // ---- Unified mode + GEMM config selection ----------------------------
    // One axis drives the whole launch: expected_tokens_per_expert selects
    // both the GEMM family and the block-M tile inside it:
    //   expected <= 12      -> LL BM16 | <= 24 -> LL BM32
    //   expected <= 56      -> HT BM64 | >  56 -> HT BM128
    // DG_MEGA_HT_EXPECTED overrides the family threshold in expected units;
    // legacy DG_MEGA_HT_THRESHOLD (per-rank capacity tokens) is converted via
    // the tokens->expected ratio so its meaning is preserved for the default
    // (num_max) source. Default 13: crossover measured on the default shape
    // (7168/3072/topk6) -- LL wins expected<=12 (fused 1.91-2.03ms vs HT
    // 2.08-2.09ms), HT wins expected>=14 (fused 2.12 vs LL 2.25-2.31ms).
    // Ranks need not agree on the family: it is receiver-local (pool layout
    // is family-agnostic) and the send_count marker no longer encodes the
    // dispatch-SM count, so heterogeneous per-rank configs are legal.
#if !defined(DG_MEGA_SKIP_LL) && !defined(DG_MEGA_SKIP_HT)
    const float tok2exp = static_cast<float>(num_ranks) * num_topk / num_experts;
    const float ht_expected_threshold = std::getenv("DG_MEGA_HT_EXPECTED")
        ? std::atof(std::getenv("DG_MEGA_HT_EXPECTED"))
        : (std::getenv("DG_MEGA_HT_THRESHOLD")
              ? std::atof(std::getenv("DG_MEGA_HT_THRESHOLD")) * tok2exp
              : 13.0f);
    // A rank with zero local tokens still receives tokens routed to its
    // local experts; the tiny-BM LL path starves that receive work and the
    // mixed LL+HT call regressed to 0.5-0.96x in the decode-16k vectors.
    // Force HT on empty senders so every rank runs the pull path -- but
    // only when the receive-side load can actually exceed the HT
    // threshold: with a known cluster total that is expected itself; else
    // the capacity worst case num_max_tokens_per_rank * tok2exp.  Small
    // capacities (e.g. decode-sized buffers) keep the empty rank on LL.
    // Default on; DG_MEGA_HT_ON_EMPTY=0 restores the threshold-only rule.
    const bool ht_on_empty =
        !std::getenv("DG_MEGA_HT_ON_EMPTY") ||
        std::atoi(std::getenv("DG_MEGA_HT_ON_EMPTY")) != 0;
    const float empty_recv_expected =
        (tokens_all_ranks >= 0)
            ? expected_tokens_per_expert
            : static_cast<float>(num_max_tokens_per_rank) * tok2exp;
    const bool use_ht = (expected_tokens_per_expert >= ht_expected_threshold) ||
                        (ht_on_empty && num_tokens == 0 &&
                         empty_recv_expected >= ht_expected_threshold);
#elif defined(DG_MEGA_SKIP_HT)
    const bool use_ht = false;
#else
    const bool use_ht = true;
#endif

    // Runtime CU detection (shared). Family-scoped opt-outs:
    // DG_MEGA_HT_72CU / DG_MEGA_LL_72CU, both default enabled.
    const bool cu72_enabled = use_ht
        ? (!std::getenv("DG_MEGA_HT_72CU") || std::atoi(std::getenv("DG_MEGA_HT_72CU")) != 0)
        : (!std::getenv("DG_MEGA_LL_72CU") || std::atoi(std::getenv("DG_MEGA_LL_72CU")) != 0);
    mega_moe_check_arch();
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    if (!cu72_enabled) num_sms = 64;
    if (num_sms != 64 && num_sms != 72) {
        TORCH_CHECK(false, "Unsupported CU count: ", num_sms, " (only 64 or 72 supported)");
    }
    uint64_t* prof_ptr = mega_moe_prof_ptr(num_sms, y.options());

    // Ring mode: the pull -> L1 staging buffers are capped at the live
    // pool-block bound and reused as a ring; below that bound the full staging
    // area is smaller than the cap anyway, so the zero-overhead non-ring path
    // is kept.  This predicate must stay identical to
    // HCUMegaMoeWorkspace::has_l1_ring (fixed-SM activation bound), because
    // the kernel side derives its ring geometry from
    // workspace.num_l1_pool_tokens.  Computed early: the dispatch-SM heuristic
    // below prefers more pull warps when the ring protocol is active.
    const uint32_t num_max_pool_tokens = get_num_max_pool_tokens(
        num_ranks, static_cast<uint32_t>(num_max_tokens_per_rank), num_topk,
        num_experts_per_rank);
    // Same policy as the host-side workspace ctor (shape gate + env
    // overrides) so allocation and launch stay consistent.
    const bool use_ring_buffer =
        l1_ring_enabled_for_shape(hidden, intermediate_hidden,
                                  num_max_tokens_per_rank) &&
        num_max_pool_tokens > get_l1_ring_activation_pool_tokens();

    // Dispatch SM count. DG_MEGA_NDISP_SMS env override wins; otherwise a
    // family-specific heuristic from the 8-card sweeps. Per-rank values need
    // not match anymore (recv_sum carries one marker per source rank), so
    // heterogeneous dispatch counts are supported.
    // sel_tokens: the ACTUAL per-rank token load (same source policy as the
    // family/bm heuristic above) -- dispatch work scales with tokens in
    // flight, not with the capacity buffers.
    const int64_t sel_tokens =
        (tokens_all_ranks >= 0)
            ? (tokens_all_ranks + num_ranks - 1) / num_ranks
            : (cfg_actual ? static_cast<int64_t>(num_tokens)
                          : static_cast<int64_t>(num_max_tokens_per_rank));
    // Light-GEMM gate for the dispatch-SM tables: pull payload per routed
    // entry scales with hidden and the smaller shapes measured on V4.1 all
    // favor fewer dispatch SMs. hidden*ih <= 5120*2304 covers v4
    // (4096*2048), v4.1 and the topk-3 sibling; 7168*3072 and 6144*2560
    // stay on the heavy-shape tables.
    const bool light_shape =
        static_cast<int64_t>(hidden) * intermediate_hidden <= 5120LL * 2304;
    int n_disp_sms;
    if (std::getenv("DG_MEGA_NDISP_SMS")) {
        n_disp_sms = std::atoi(std::getenv("DG_MEGA_NDISP_SMS"));
    } else if (use_ht) {
        // 72-CU sweep (8-card, fused L1 + groupwise L2, kineto timing):
        //   >=2048 tokens: ndisp=4 wins; <2048: ndisp=8 wins at 1024.
        // Light-shape (hidden*ih <= 5120*2304) 8-card sweep on actual
        // tokens (10-21, measured on V4.1, log ndisp_sweep_v41_1021_1127):
        //   <=512 -> ndisp=2 wins (1.262 @128; 1.252 vs 1.383 @256;
        //   1.823 vs 1.877 @512); >=1024 -> ndisp=4 wins
        //   (2.89/4.69/8.19 vs nd8 3.00/4.9/8.5 and nd10 2.99/4.85/8.53).
        //   ndisp=8/10 never win on V4.1; lighter shapes pay the same or
        //   less dispatch cost, so they inherit this table.
        // Ring mode flips >=32K capacity to 8 for light-GEMM shapes: producers
        // stall on slot gates and extra pull warps hide the wait (v4@64K:
        // 96.7 vs 100.3 ms, v4@32K: 45.2 vs 46-56 ms). For heavy-GEMM shapes
        // or 16K capacity the dispatch blocks steal consumer throughput
        // instead (v41@64K: 131 vs 125 ms, v4@16K: 22.4 vs 22.0 ms), so keep 4.
        const bool ring_wants_8 = use_ring_buffer &&
            static_cast<int64_t>(hidden) * intermediate_hidden <= 4096 * 2048 &&
            num_max_tokens_per_rank >= 32768;
        if (num_sms == 72 && cu72_enabled) {
            if (light_shape) {
                // ndisp=8 was pruned from the instantiation set (never wins on
                // V4.1).  It used to fall back to 3 here, but 3 is not
                // instantiated either (kept set is {2,4,6}), so that path
                // aborted the launch; 4 is the closest kept config.
                n_disp_sms = ring_wants_8 ? 4
                                          : (sel_tokens <= 512 ? 2 : 4);
            } else {
                // Same: the pruned ndisp=3 is not instantiated (kept set is
                // {2, 4, 6}), so asking for it is a hard launch error.
                n_disp_sms = 4;
            }
        } else if (light_shape && num_sms == 64 && num_ranks >= 8) {
            // Post-12-wave sweep (2026-10-31/31(2), see megamoe/perf_history.md):
            // ndisp=2 wins the tiny band, 4 wins from 512 tokens/rank up at
            // every wave count; 3/5-8 never beat the winner outside noise and
            // stay instantiated (env-reachable) for re-sweeps only.
            n_disp_sms = (sel_tokens <= 256) ? 2 : 4;
        } else if (num_ranks >= 8 ? (sel_tokens < 512) : (sel_tokens <= 128 * num_ranks)) {
            n_disp_sms = 2;
        } else if (num_sms == 64 && num_ranks >= 8 && sel_tokens < 2048) {
            // 64-CU 8-card sweep previously preferred ndisp=8 here; 8 is
            // pruned.  The runner-up 3 is *also* not instantiated (kept set is
            // {2, 4, 6}), so returning it aborted every non-light 64-CU shape
            // in this band.  Snap to the nearest kept config.
            n_disp_sms = 4;
        } else if (sel_tokens <= 512 * num_ranks) {
            n_disp_sms = 4;
        } else {
            // 8/10 pruned: ndisp=4 measured best (or tied) in every band.
            n_disp_sms = 4;
        }
    } else {
        // 72-CU kineto sweep (ndisp_sweep_full_8card):
        //   tokens<=8: 4;  8<t<=64: 2;  tokens>64: 4.
        // Light-shape (hidden*ih <= 5120*2304) sweep on actual tokens,
        // measured on V4.1: ndisp=4 wins the whole LL band
        // (t<=~103 expected<13; 0.795/0.991/1.105/1.118 vs nd2 at 8-64).
        // t=128 is already HT internally (expected=16>=13), so no LL>=128
        // branch is needed here.
        if (num_sms == 72 && cu72_enabled) {
            if (light_shape) {
                n_disp_sms = 4;
            } else {
                n_disp_sms = (sel_tokens > 8 && sel_tokens <= 64) ? 2 : 4;
            }
        } else if (num_sms == 64 && num_ranks >= 8 && sel_tokens <= 16) {
            // 64-CU 8-card sweep: small tokens favor more dispatch SMs.
            n_disp_sms = 4;
        } else if (num_ranks >= 8 ? (sel_tokens < 512) : (sel_tokens <= 128 * num_ranks)) {
            n_disp_sms = 2;
        } else if (sel_tokens <= 512 * num_ranks) {
            n_disp_sms = 4;
        } else {
            n_disp_sms = (num_ranks >= 8) ? 4 : 2;
        }
    }

    // ---- Per-family GEMM config -----------------------------------------
#if !defined(DG_MEGA_SKIP_HT)
    // Launch bounds for BN=512 to reduce VGPR pressure (DG_MEGA_HT_LB:
    // 768 = 12 warps default; 512/1024 env-only).
    const int ht_lb = std::getenv("DG_MEGA_HT_LB")
        ? std::atoi(std::getenv("DG_MEGA_HT_LB")) : 768;
    // HT GEMM tile: expected <= bm64_max_expected -> BM64, else BM128.
    // BM64 uses the V2 path (BM=64 tile through the proven BM128/BK64 GEMM
    // pipeline); DG_MEGA_HT_BM64_V1=1 falls back to the original BK=128
    // GEMM.  The crossover is shape dependent; with the 12-wave L1 pass the
    // weight-stream-bound band crosses over to BM128 earlier, hence the
    // lowered 32 for V4.1 (other shapes keep the legacy 56; crossover data
    // in megamoe/perf_history.md, 2026-10-31).
    // Env overrides: DG_MEGA_HT_BM=64/128 picks the tile directly;
    // DG_MEGA_HT_BM64_MAX re-tunes the crossover in expected units.
    const float bm64_max_expected_default =
        (hidden == 5120 && intermediate_hidden == 2304) ? 32.0f : 56.0f;
    const float bm64_max_expected = std::getenv("DG_MEGA_HT_BM64_MAX")
        ? std::atof(std::getenv("DG_MEGA_HT_BM64_MAX"))
        : bm64_max_expected_default;
    int block_m_ht = std::getenv("DG_MEGA_HT_BM")
        ? std::atoi(std::getenv("DG_MEGA_HT_BM"))
        : (expected_tokens_per_expert <= bm64_max_expected ? 64 : 128);
    // Empty-sender ranks forced to HT (ht_on_empty) may still receive a
    // heavy routed load for their local experts.  DG_MEGA_HT_BM_EMPTY
    // overrides the HT tile for exactly those ranks; non-empty ranks keep
    // the normal expected-based selection.
    if (num_tokens == 0 && std::getenv("DG_MEGA_HT_BM_EMPTY"))
        block_m_ht = std::atoi(std::getenv("DG_MEGA_HT_BM_EMPTY"));
    const bool ht_bm64_v1 = std::getenv("DG_MEGA_HT_BM64_V1") &&
        std::atoi(std::getenv("DG_MEGA_HT_BM64_V1")) != 0;
    // BN512 default for BM64/BM128; BN256 for the env-only BM144/160/256.
    int block_n_ht = std::getenv("DG_MEGA_HT_BN")
        ? std::atoi(std::getenv("DG_MEGA_HT_BN")) : 512;
    // L2-W12X: BN768 + 12 math warps for the L2 groupwise GEMM.  Two layouts:
    //   V4.1 (hidden 5120 = 6x768 + 512, 2*ih 4608 % 768 == 0):
    //     L1 also runs the 768-column / 12-warp W12-P pass.
    //   hy4  (hidden 6144 = 8x768 exactly, 2*ih 4096 % 768 != 0):
    //     L1 stays BN512 / 8 warps, only the L2 GEMM widens to 768 / 12 warps.
    // DG_MEGA_HT_L2W12=0 falls back to the BN512 / 8-warp L2 for A/B.
    const char* l2w12_env = std::getenv("DG_MEGA_HT_L2W12");
    // hy4 is the shape whose default kernel path is W16 (see use_l1w16 below),
    // so its other widening knobs must stay off unless asked for explicitly.
    const bool is_hy4_shape = (num_topk == 8 && hidden == 6144 &&
                               intermediate_hidden == 2048);
    // L2-W12X stays OFF for hy4 by default: the exact-tiling 6144 = 8x768 shape
    // measured neutral to negative here, and switching block_n_ht to 768 would
    // bypass W16 entirely.  DG_MEGA_HT_L2W12=1 turns it on for A/B.  V4.1
    // (hidden % 768 == 512) keeps its tuned default-on behaviour.
    const bool l2w12_default_on = !is_hy4_shape && (hidden % 768 == 512);
    const bool use_l2w12 = (l2w12_env != nullptr)
        ? (std::atoi(l2w12_env) != 0) : l2w12_default_on;
    // L2 side tiles exactly as either 6x768 + 512 (hidden % 768 == 512, V4.1)
    // or 8x768 (hidden % 768 == 0, hy4).
    const bool l2w12_l2_ok = (hidden % 768 == 0) || (hidden % 768 == 512);
    // L1 side: the 768-column (12-warp) pass needs 2*ih % 768 == 0 and a 384
    // group; when only 2*ih % 512 == 0 holds (hy4: 2*ih = 4096) L1 keeps the
    // BN512 / 8-warp pass and only L2 widens (group stays 256).
    const bool l1w12_ok =
        (2 * intermediate_hidden) % 768 == 0 && intermediate_hidden % 384 == 0;
    const bool l1_bn512_ok =
        (2 * intermediate_hidden) % 512 == 0 && intermediate_hidden % 256 == 0;
    // Capacity gate, same policy as megamoe/patches/minimax_m31_support_20261104.patch:
    // the exactly-tiling shapes measured 1.1-1.5% faster from 2048 tokens/rank
    // up but 2.6% / 4.2% slower at 1024 / 512 (few M-blocks per expert, so
    // eight 768-wide tiles idle more warps than twelve 512-wide ones).
    // DG_MEGA_HT_L2W12_MIN_TOKENS=0 forces it on everywhere for A/B.
    const bool l2_tiles_exactly = (hidden % 768 == 0);
    const int64_t l2w12_min_tokens = std::getenv("DG_MEGA_HT_L2W12_MIN_TOKENS")
        ? std::atoll(std::getenv("DG_MEGA_HT_L2W12_MIN_TOKENS"))
        : (l2_tiles_exactly ? 2048 : 0);
    if (block_m_ht == 128 && use_l2w12 && sel_tokens >= l2w12_min_tokens &&
        l2w12_l2_ok && (l1w12_ok || l1_bn512_ok))
        block_n_ht = 768;
    if (block_m_ht != 128 && block_m_ht != 64) block_n_ht = 256;
    // The pull/ring paths divide the pool capacity by BLOCK_M; the capacity
    // is aligned to the candidate-BM LCM, so catch env-chosen tiles that fall
    // outside it here instead of silently truncating in-kernel.
    TORCH_CHECK(num_max_pool_tokens % block_m_ht == 0 &&
                get_l1_ring_pool_tokens(num_sms) % block_m_ht == 0,
                "HT BLOCK_M=", block_m_ht,
                " must divide the pool/ring token capacity");
    // 4-warp GEMM mode (DG_MEGA_HT_4W), BM128/BK64/BN256 only.
    bool use_4warp_gemm = std::getenv("DG_MEGA_HT_4W") &&
        std::atoi(std::getenv("DG_MEGA_HT_4W")) != 0;
    if (block_m_ht != 128 || block_n_ht != 256) use_4warp_gemm = false;
    // W12-P (ht_gemm_12wave_plan.md §3, single-pass form): the L1 fused GEMM
    // runs all 12 warps over a 768-column window (kNFragments = 768/(12*16) = 4,
    // so each wave keeps the proven acc[8][4] shape and acc VGPR count), while
    // the L2 groupwise GEMM keeps BN512 with 8 math warps + 4 mirror warps.
    // The quant group becomes kL1BlockN/2 = 384 columns: still a multiple of
    // kGroupQuantSize=128, K_IN_ITERS = 384/64 = 6 (even) and l2_sf rows keep
    // their 18-float allocation (6 used).  Needs the 768-thread (12W) launch
    // bound; instantiated for every kept dispatch count {2,4,6} (BM128 and
    // BM64) and default-on for all of them.  Shapes whose 2*IH does not tile
    // by 768
    // automatically fall back to the BN512/8-warp instance.  BM64 uses the
    // same pass on its 64-row tile (acc[4][4] = 64 VGPR, A-LDS 48KB/k-tile).
    // DG_MEGA_HT_W12P=0 falls back to the 8-warp L1 everywhere (A/B escape).
    const char* w12p_env = std::getenv("DG_MEGA_HT_W12P");
    bool use_w12p = (w12p_env != nullptr)
        ? (std::atoi(w12p_env) != 0)
        : true;
    if ((block_m_ht != 128 && block_m_ht != 64) ||
        block_n_ht != 512 || ht_lb != 768 || is_int8)
        use_w12p = false;
    // L1-W12: 12 math warps over a 768-wide L1 tile for shapes where 2*ih does
    // not divide by 768 but the trailing block still tiles 8 warps
    // (hy4: 4096 = 5x768 + 256).  The 768-wide tile produces 384 SwiGLU
    // channels and 2048 % 384 != 0, so the group is pinned to 128 (3 groups in
    // a main tile, 1 in the tail, 16 total == ih/128) -- that also doubles the
    // L2 groupwise rescale count, so it is env-gated and never default-on.
    // Does not apply where the plain W12-P path already fits (2*ih % 768 == 0).
    const char* l1w12_env = std::getenv("DG_MEGA_HT_L1W12");
    bool use_l1w12 = (l1w12_env != nullptr)
        ? (std::atoi(l1w12_env) != 0) : false;
    if (block_m_ht != 128 || ht_lb != 768 || is_int8 ||
        (2 * intermediate_hidden) % 768 == 0 ||
        (2 * intermediate_hidden) % 128 != 0 ||
        intermediate_hidden % 128 != 0)
        use_l1w12 = false;
    // DG_MEGA_HT_L1W12_TAIL12=1 additionally runs the 256-wide tail on all 12
    // math warps (non-uniform fragment split) instead of 8.  That removes the
    // tail's 2x A-LDS-per-column and its 2-wave occupancy -- the dominant skew
    // cost identified in ht_l1_12wave_skew_regression_analysis.md (P2).
    const char* l1w12_tail_env = std::getenv("DG_MEGA_HT_L1W12_TAIL12");
    const bool use_l1w12_tail12 = use_l1w12 && (l1w12_tail_env != nullptr) &&
        (std::atoi(l1w12_tail_env) != 0);
    // Mixed L1 tiling: 4x768 (12 warp) + 2x512 (8 warp) instead of 5x768 +
    // a 256-wide tail.  Keeps the 6-tile A-reread count but removes the narrow
    // tail that dominates the skew regression (perf_history 2026-11-10 续3).
    // The layout requires 2*ih == 4096; other shapes keep the tail variant.
    const char* l1w12_mix_env = std::getenv("DG_MEGA_HT_L1W12_MIX");
    const bool use_l1w12_mix = use_l1w12 && !use_l1w12_tail12 &&
        (2 * intermediate_hidden == 4096) &&
        (l1w12_mix_env != nullptr) && (std::atoi(l1w12_mix_env) != 0);
    // W16: 16 math warps over a 1024-wide L1 tile (4 tiles for 2*ih = 4096).
    // Unlike the 768-wide 12-warp tile, 1024/2 = 512 divides ih = 2048, so the
    // quant group stays 512 and the L2 groupwise GEMM has 4 rescale boundaries
    // instead of 16 (group=128) or 8 (baseline group=256).  That removes the
    // per-m-block rescale cost which the skew measurements point to as the real
    // regression source (perf_history 2026-11-10 续3/续6/续8).  A-rereads drop
    // 8 -> 4 with unchanged A-LDS per column (16 warps x 4 fragments = 1024).
    const char* l1w16_env = std::getenv("DG_MEGA_HT_L1W16");
    // Default-on for hy4: measured -11.5/-2.0/-8.2% on balanced 2K/4K/8K and
    // only ~+1% under hot2 (vs +5~6% for the old 12-wave), see perf_history
    // 续8/续9.  DG_MEGA_HT_L1W16=0 restores the 8-wave L1.  Shape gates are
    // unconditional so an env value can only switch a valid instance off/on.
    const bool l1w16_shape_ok = is_hy4_shape && (block_m_ht == 128) &&
        !is_int8 && (2 * intermediate_hidden) % 1024 == 0 &&
        intermediate_hidden % 512 == 0;
    // Capacity gate: the 4 x 1024-wide tiles need enough m-blocks per expert to
    // fill 16 warps.  Same-batch A/B (perf_history 续10): +21% at 128, +22% at
    // 256, -1.8% at 512, -4.9% at 1024, -11.5% at 2048 (balanced), so 1024 is
    // the first clearly positive bucket.  DG_MEGA_HT_L1W16_MIN_TOKENS overrides.
    const int64_t l1w16_min_tokens = std::getenv("DG_MEGA_HT_L1W16_MIN_TOKENS")
        ? std::atoll(std::getenv("DG_MEGA_HT_L1W16_MIN_TOKENS")) : 1024;
    const bool use_l1w16 = l1w16_shape_ok &&
        (sel_tokens >= l1w16_min_tokens) &&
        ((l1w16_env != nullptr) ? (std::atoi(l1w16_env) != 0) : true);
    // W16 + L2-W12X: L1 1024/16w (4 tiles) and L2 768/12w (8 tiles, 6144 =
    // 8x768, no tail).  L2 widening measured -1.5..-3.1% under skew (续6) and
    // should cancel the last ~1% W16 loss at 2048/4096; group stays 512.
    const char* w16l2_env = std::getenv("DG_MEGA_HT_W16_L2W12");
    const bool use_w16_l2w12 = (w16l2_env != nullptr) &&
        (std::atoi(w16l2_env) != 0) && (block_m_ht == 128) && !is_int8 &&
        (2 * intermediate_hidden) % 1024 == 0 &&
        intermediate_hidden % 512 == 0 && hidden % 768 == 0;
    // W16 + L2-W16: L1 1024/16w and L2 1024/16w (hidden 6144 = 6x1024, no
    // tail).  All 16 warps now take part in L2 as well -- this needs the
    // l2_math_warps fix in mega_moe_kernel_unified.cuh, otherwise only the
    // first 8 warps compute and the remaining columns hold stale scratch.
    // Shares the W16 shape/capacity gates; env-only and default off.
    const char* w16l2w16_env = std::getenv("DG_MEGA_HT_W16_L2W16");
    const bool use_w16_l2w16 = use_l1w16 &&
        (w16l2w16_env != nullptr) && (std::atoi(w16l2w16_env) != 0) &&
        (hidden % 1024 == 0);

    // SMEM: static smem_silu_result overlays the expert table; dynamic SMEM
    // only when the table outgrows it (BM256's dedicated 2048-entry scratch).
    const size_t ht_static_capacity = (block_m_ht == 256)
        ? 2048 * sizeof(uint32_t)
        : static_cast<size_t>(intermediate_hidden) * 4;
    const size_t smem_bytes_ht =
        (expert_count_bytes <= ht_static_capacity ? 0 : expert_count_aligned);
#endif

#if !defined(DG_MEGA_SKIP_LL)
    // LL GEMM tile (env DG_MEGA_BLOCK_M=16|32|48):
    //   <= 12 tokens/expert -> BM16; 8-rank & >128 -> BM48; else BM32.
    const int block_m_tile = std::getenv("DG_MEGA_BLOCK_M")
        ? std::atoi(std::getenv("DG_MEGA_BLOCK_M"))
        : (expected_tokens_per_expert <= 12.0f) ? 16 :
          (num_ranks >= 8 && expected_tokens_per_expert > 128.0f) ? 48 :
          (expected_tokens_per_expert <= 512.0f) ? 32 : 48;
    TORCH_CHECK(num_max_pool_tokens % block_m_tile == 0 &&
                get_l1_ring_pool_tokens(num_sms) % block_m_tile == 0,
                "LL BLOCK_M=", block_m_tile,
                " must divide the pool/ring token capacity");
    // Launch bounds (DG_MEGA_LL_LB): 768 = 12 warps default; 512/1024 env-only.
    const int ll_lb = std::getenv("DG_MEGA_LL_LB")
        ? std::atoi(std::getenv("DG_MEGA_LL_LB")) : 768;
    // Math warps for the split kernel (default 8, env DG_MEGA_SPLIT_NMATH).
    const int n_math = std::getenv("DG_MEGA_SPLIT_NMATH") ?
        std::atoi(std::getenv("DG_MEGA_SPLIT_NMATH")) : 8;
    TORCH_CHECK(256 % (n_math * 16) == 0, "BLOCK_N/(n_math*16) must be integer, got 256/", n_math*16);
    // PIPE_STAGE for L1/L2 GEMMs (default 4/4; envs DG_MEGA_L1_PIPE/_L2_PIPE,
    // L1>=8 selects the warp-K variant). Only (4,4) is instantiated.
    const int l1_pipe_env = std::getenv("DG_MEGA_L1_PIPE") ?
        std::atoi(std::getenv("DG_MEGA_L1_PIPE")) : 4;
    const int l2_pipe_env = std::getenv("DG_MEGA_L2_PIPE") ?
        std::atoi(std::getenv("DG_MEGA_L2_PIPE")) : 4;
    // Dynamic SMEM: expert table + the [BM x 256] L2 output staging tile.
    const size_t smem_bytes_ll = expert_count_aligned
        + static_cast<size_t>(block_m_tile) * 256 * sizeof(hip_bfloat16);
#endif

    // Resolved launch config for logging (family-specific fields unified).
    int cfg_bm, cfg_bn, cfg_lb, cfg_math;
    size_t cfg_smem;
#if !defined(DG_MEGA_SKIP_HT) && !defined(DG_MEGA_SKIP_LL)
    if (use_ht) {
        cfg_bm = block_m_ht; cfg_bn = block_n_ht; cfg_lb = ht_lb;
        cfg_math = (block_m_ht == 64 && block_n_ht == 256) ? 4
                 : (block_n_ht == 768) ? 12 : 8;
        cfg_smem = smem_bytes_ht;
    } else {
        cfg_bm = block_m_tile; cfg_bn = 256; cfg_lb = ll_lb;
        cfg_math = n_math; cfg_smem = smem_bytes_ll;
    }
#elif !defined(DG_MEGA_SKIP_HT)
    cfg_bm = block_m_ht; cfg_bn = block_n_ht; cfg_lb = ht_lb;
    cfg_math = (block_m_ht == 64 && block_n_ht == 256) ? 4
             : (block_n_ht == 768) ? 12 : 8;
    cfg_smem = smem_bytes_ht;
#else
    cfg_bm = block_m_tile; cfg_bn = 256; cfg_lb = ll_lb;
    cfg_math = n_math; cfg_smem = smem_bytes_ll;
#endif

    // Config log: DG_MEGA_CFG_VERBOSE -> stderr; DG_MEGA_CFG_LOG=1 ->
    // per-rank file megamoe_cfg_rank<rank>.log in the current directory
    // (one appended line per launch).
    static FILE* cfg_log = nullptr;
    static bool cfg_log_tried = false;
    if (!cfg_log_tried && std::getenv("DG_MEGA_CFG_LOG") &&
        std::atoi(std::getenv("DG_MEGA_CFG_LOG")) != 0) {
        cfg_log_tried = true;
        char cfg_path[256];
        snprintf(cfg_path, sizeof(cfg_path),
                 "megamoe_cfg_rank%d.log", rank_idx);
        cfg_log = fopen(cfg_path, "a");
    }
    FILE* cfg_out = cfg_log
        ? cfg_log
        : (std::getenv("DG_MEGA_CFG_VERBOSE") ? stderr : nullptr);
    if (cfg_out) {
        fprintf(cfg_out,
                "[cfg] rank=%d/%d tokens=%lld max=%lld all=%lld expected=%.1f "
                "experts=%d(%d/rank) topk=%d hidden=%d ih=%d int8=%d "
                "clamp=%.4g fast_math=%d -> %s BM%d%s BN%d ndisp=%d sms=%d "
                "lb=%d mathw=%d smem=%zu\n",
                rank_idx, num_ranks, static_cast<long long>(num_tokens),
                static_cast<long long>(num_max_tokens_per_rank),
                static_cast<long long>(tokens_all_ranks),
                expected_tokens_per_expert,
                num_experts, num_experts_per_rank, num_topk, hidden,
                intermediate_hidden, static_cast<int>(is_int8),
                activation_clamp, static_cast<int>(fast_math),
                use_ht ? "HT" : "LL", cfg_bm,
                (use_ht && use_w16_l2w16) ? "w16l2w16"
                : (use_ht && use_w16_l2w12) ? "w16l2"
                : (use_ht && use_l1w16) ? "l1w16"
                : (use_ht && use_l1w12_mix) ? "l1w12mix"
                : (use_ht && use_l1w12_tail12) ? "l1w12t12"
                : (use_ht && use_l1w12) ? "l1w12"
                : (use_ht && cfg_bm == 64 && cfg_bn == 512 && !ht_bm64_v1 &&
                 !use_w12p) ? "v2"
                : (use_ht && cfg_bn == 768) ? "l2w12"
                : (use_ht && use_w12p && (2 * intermediate_hidden) % 768 == 0 &&
                   intermediate_hidden % 384 == 0) ? "w12p"
                : "",
                cfg_bn, n_disp_sms, num_sms,
                cfg_lb, cfg_math, cfg_smem);
        if (cfg_out != stderr) fflush(cfg_out);
    }

    // ---- Launch macros (both families; only the selected family is taken) -
#if !defined(DG_MEGA_SKIP_HT)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM64_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 64, 128, NUM_SMS, NDISP_SMS, 16, 4, 2, 2, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, num_threads, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM64(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM64_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM64_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // BM64 with 8 math warps + BN=512: same N-split/fragment organization as
    // the BM128/BN512 path (each warp owns kNFragments=4 N-fragments; the 4
    // M-subtiles are loaded by warps 0-3 only).
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 64, 128, NUM_SMS, NDISP_SMS, 16, 8, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, num_threads, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // BM64 V2 (BK=64): runs the BM128-proven GEMM pipeline on a 64-row
    // M-tile -- 4 M-subtiles, kAStagesHT=4, NTW12/768 threads identical to
    // the BM128/BN512 instantiation.  Default for DG_MEGA_HT_BM=64.
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM64V2_BN512_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 64, 64, NUM_SMS, NDISP_SMS, 12, 8, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM64V2_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM64V2_BN512_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM64V2_BN512_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // BM64 + W12-P: the same 768-column / 12-math-warp L1 pass on the 64-row M
    // tile.  kMSubtiles = 4 -> acc[4][4] = 64 VGPR (half of BM128's), A-LDS
    // reads 48KB/k-tile instead of 96KB, and the L1/L2 weight stream is
    // unchanged, so the small-batch band (one m-block per expert) keeps BM64's
    // cheaper L2 while still giving L1 the 3-waves-per-SIMD shape.
    #define LAUNCH_BN512_BM64_W12P_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 64, 64, NUM_SMS, NDISP_SMS, 12, 12, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING, 768, 8><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN512_BM64_W12P_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN512_BM64_W12P_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN512_BM64_W12P_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM128_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 16, 8, 2, 2, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, num_threads, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM128(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM128_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM128_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // BN=512 with selectable launch bounds (DG_MEGA_HT_LB)
    // 1024 threads = 16 warps (4 waves/SIMD, 192 VGPR/thread, 328B private_seg --may spill)
    // 768  threads = 12 warps (3 waves/SIMD, 256 VGPR/thread, 36B private_seg --no spill)
    // 512  threads = 8 warps  (2 waves/SIMD, 256 VGPR/thread, 36B private_seg --no spill)
    // GEMM always uses 8 math warps (kNFragments=4); extra warps help dispatch/SiLU/combine.
    // NOTE: NMATH=16 measured @16384 = 57.6ms vs 48.6ms (NMATH=8) -- N-split
    // doubles A-LDS reads (16 warps x 8KB/ktile), same structural loss as
    // the BM256 experiment.  Keep 8.
    #define LAUNCH_BN512_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, NMATH, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, NTW, NMATH, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, NTH, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN512_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, NMATH, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN512_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, NMATH, true, EPR, NR), \
            LAUNCH_BN512_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, NMATH, false, EPR, NR))

    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM128_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        if (ht_lb == 768) { LAUNCH_BN512_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, 12, 768, 8, EPR, NR); } \
        else { TORCH_CHECK(false, "Unsupported (or commented-out) ht_lb: ", ht_lb); }

    // W12-P: L1 tile width 768 / 12 math warps; L2 stays BN512 / 8 math warps
    // (kNumMathWarpsL2Arg = 8, the remaining 4 warps mirror L2's stage bars).
    #define LAUNCH_BN512_W12P_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 12, 12, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING, 768, 8><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN512_W12P_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN512_W12P_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN512_W12P_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))
    #define LAUNCH_W12P_SEL(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        LAUNCH_BN512_W12P_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR)

    // BM=144/BK=64 (mult-of-16 M tiles >128): BN256, selectable launch bounds.
    // SMEM: NTW12 = ~57KB, NTW16 = ~62KB (amax scratch overlaid on silu_result).
    #define LAUNCH_BM144_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 144, 64, NUM_SMS, NDISP_SMS, NTW, 8, 2, 2, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, NTH, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BM144_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BM144_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, true, EPR, NR), \
            LAUNCH_BM144_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, false, EPR, NR))

    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM144(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        if (ht_lb == 512) { LAUNCH_BM144_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, 8, 512, EPR, NR); } \
        else if (ht_lb == 1024) { LAUNCH_BM144_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, 16, 1024, EPR, NR); } \
        else { LAUNCH_BM144_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, 12, 768, EPR, NR); }

    // BM=160/BK=64: SMEM only fits NTW<=12 (a_lds 30KB + silu 15KB + ec 12KB).
    #define LAUNCH_BM160_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 160, 64, NUM_SMS, NDISP_SMS, NTW, 8, 2, 2, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, NTH, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM160_K2(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BM160_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, true, EPR, NR), \
            LAUNCH_BM160_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, NTW, NTH, false, EPR, NR))

    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM160(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        if (ht_lb == 512) { LAUNCH_MEGA_MOE_HT_SPLIT_BM160_K2(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, 8, 512, EPR, NR); } \
        else { LAUNCH_MEGA_MOE_HT_SPLIT_BM160_K2(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, 12, 768, EPR, NR); }

    // BM=256/BK=64 dedicated GEMM (gemm_ht_tile_bm256_bk64_* -- NOT a BM128
    // template widening).  16 math warps, N-split: each warp owns a
    // distinct 16-col fragment and computes all 256 rows, so acc[16][1]
    // = 64 VGPRs (the BM128 accumulator profile).  B is fetched once per
    // column set -- no panel duplication.  NTW=NMATH=16 (1024 threads).
    #define LAUNCH_BM256_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 256, 64, NUM_SMS, NDISP_SMS, 16, 16, 2, 2, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, 1024, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM256(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BM256_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BM256_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // 4-warp variant: BM=128/BK=64 with 4 math warps (kNFragments=4, matching reference)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM128_4W_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 16, 4, 2, 2, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, num_threads, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_MEGA_MOE_HT_SPLIT_BM128_4W(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM128_4W_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_MEGA_MOE_HT_SPLIT_BM128_4W_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // L2-W12X: BN768 / 12 math warps for the L2 groupwise GEMM; the L1 pass
    // keeps its 768-column W12-P tile.  5120 = 6x768 + 512, so each m-block
    // runs six 12-warp tiles plus one 512-wide tile on the proven 8-warp +
    // 4-mirror layout -- exact coverage, nothing computed twice.
    // Default-on for fitting shapes; DG_MEGA_HT_L2W12=0 reverts to BN512.
    #define LAUNCH_BN768_L2W12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 12, 12, 2, 2, 768, IS_INT8, EPR, NR, DG_MEGA_KRING, 768, 12><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN768_L2W12_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN768_L2W12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN768_L2W12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // L2-W12X with the L1 pass kept at BN512 / 8 math warps.  Used when the
    // 768-column L1 tile cannot tile 2*ih (hy4: 2*ih = 4096 has no factor 3)
    // but 2*ih % 512 == 0.  kL1BlockNTile = 512 keeps the group at 256 so the
    // groupwise L2 GEMM (K = ih = 2048 = 8 groups) still tiles, while the L2
    // 768-wide tile spans three groups and runs all 12 math warps.
    #define LAUNCH_BN768_L2W12_L1BN512_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 12, 8, 2, 2, 768, IS_INT8, EPR, NR, DG_MEGA_KRING, 512, 12><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN768_L2W12_L1BN512_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN768_L2W12_L1BN512_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN768_L2W12_L1BN512_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // L1-W12 (DG_MEGA_HT_L1W12=1): 12 math warps over a 768-wide L1 tile with a
    // 256-wide trailing block on 8 warps.  kHTGroupSize is pinned to 128 for
    // these instances (2048 % 384 != 0), so one main tile emits 3 groups and the
    // tail emits 1.  Two L2 companions: BN512/8-warp and BN768/12-warp.
    #define LAUNCH_BN512_L1W12_G128_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 12, 12, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING, 768, 8, 128><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN512_L1W12_G128_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN512_L1W12_G128_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN512_L1W12_G128_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // Same L1-W12 layout with the 256-wide tail also running all 12 math warps
    // (non-uniform 2/2/2/2 + 8x1 fragment split) instead of 8.  Only the tail
    // specialization changes; the 768-wide main tile is identical.
    #define LAUNCH_BN512_L1W12_G128_TAIL12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 12, 12, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING, 768, 8, 128, true><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN512_L1W12_G128_TAIL12_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN512_L1W12_G128_TAIL12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN512_L1W12_G128_TAIL12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // Mixed L1 tiling (4 wide 768-column tiles + 2 standard 512-column tiles).
    // kL1WideTilesArg = 4 selects the mixed partition; kL1TailAllWarps stays
    // false because the mixed layout has no narrow tail at all.
    #define LAUNCH_BN512_L1W12_MIX_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 12, 12, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING, 768, 8, 128, false, 4><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN512_L1W12_MIX_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN512_L1W12_MIX_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN512_L1W12_MIX_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // W16: 16-warp / 1024-wide L1 tile, group defaults to kL1BlockN/2 = 512.
    // L2 keeps BN512 / 8 math warps; the other 8 warps mirror L2's barriers.
    #define LAUNCH_BN512_L1W16_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 16, 16, 2, 2, 512, IS_INT8, EPR, NR, DG_MEGA_KRING, 1024, 8><<<NUM_SMS, 1024, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN512_L1W16_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN512_L1W16_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN512_L1W16_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // W16 + L2-W12X: L1 1024/16w, L2 768/12w, group 512 (default).
    #define LAUNCH_W16_L2W12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 16, 16, 2, 2, 768, IS_INT8, EPR, NR, DG_MEGA_KRING, 1024, 12><<<NUM_SMS, 1024, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_W16_L2W12_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_W16_L2W12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_W16_L2W12_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    // W16 + L2-W16: L1 1024/16w (4 tiles), L2 1024/16w (6 tiles for
    // hidden 6144), group 512.  All 16 launched warps participate in both
    // phases, so there are no mirror warps in L2 anymore.
    #define LAUNCH_W16_L2W16_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 16, 16, 2, 2, 1024, IS_INT8, EPR, NR, DG_MEGA_KRING, 1024, 16><<<NUM_SMS, 1024, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_W16_L2W16_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_W16_L2W16_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_W16_L2W16_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))



    #define LAUNCH_BN768_L1W12_G128_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, 128, 64, NUM_SMS, NDISP_SMS, 12, 12, 2, 2, 768, IS_INT8, EPR, NR, DG_MEGA_KRING, 768, 12, 128><<<NUM_SMS, 768, smem_bytes_ht, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_BN768_L1W12_G128_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_BN768_L1W12_G128_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, true, EPR, NR), \
            LAUNCH_BN768_L1W12_G128_KERN_K(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, false, EPR, NR))

    #define LAUNCH_MEGA_MOE_HT(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR) \
        if (block_m_ht == 128 && block_n_ht == 768) { \
            /* L2-W12X: L2 runs 768-wide / 12 warps.  The L1 pass is 768 wide \
               when 2*ih tiles by it (V4.1), otherwise it stays BN512/8 warps \
               (hy4: 2*ih = 4096). */ \
            if constexpr ((2 * (IH)) % 768 == 0 && (IH) % 384 == 0) { \
                LAUNCH_BN768_L2W12_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
            else if constexpr ((2 * (IH)) % 128 == 0 && (IH) % 128 == 0) { \
                if (use_l1w12) { \
                    LAUNCH_BN768_L1W12_G128_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else if constexpr ((2 * (IH)) % 512 == 0 && (IH) % 256 == 0) { \
                    LAUNCH_BN768_L2W12_L1BN512_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else { TORCH_CHECK(false, "L2W12 BN768 needs 2*ih % 512 == 0 or DG_MEGA_HT_L1W12=1"); } } \
            else { TORCH_CHECK(false, "L2W12 BN768 requires 2*ih % 768 == 0 (L1 768) or 2*ih % 512 == 0 (L1 512)"); } } \
        else if (block_m_ht == 128 && block_n_ht == 512) { \
            /* The 768-column L1 pass only tiles ih=2304/3072 (2*ih % 768 == 0). \
               Other shapes must not even instantiate it -- a runtime if() would \
               still instantiate both sides and trip the tile asserts. */ \
            /* W12P is default-on for every dispatch count LAUNCH_HT_FOR_SMS \
               uses ({2,4,6}); no per-NDISP gate is needed here -- the \
               caller macro is the only source of NDISP values. */ \
            if constexpr ((2 * (IH)) % 768 == 0 && (IH) % 384 == 0) { \
                if (use_w12p) { LAUNCH_W12P_SEL(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else { LAUNCH_MEGA_MOE_HT_SPLIT_BM128_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } } \
            else if constexpr ((2 * (IH)) % 128 == 0 && (IH) % 128 == 0) { \
                if (use_w16_l2w16) { \
                    LAUNCH_W16_L2W16_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else if (use_w16_l2w12) { \
                    LAUNCH_W16_L2W12_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else if (use_l1w16) { \
                    LAUNCH_BN512_L1W16_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else if (use_l1w12_mix) { \
                    LAUNCH_BN512_L1W12_MIX_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else if (use_l1w12_tail12) { \
                    LAUNCH_BN512_L1W12_G128_TAIL12_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else if (use_l1w12) { \
                    LAUNCH_BN512_L1W12_G128_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else { LAUNCH_MEGA_MOE_HT_SPLIT_BM128_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } } \
            else { LAUNCH_MEGA_MOE_HT_SPLIT_BM128_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } } \
        /* env-only variants (DG_MEGA_HT_BM / DG_MEGA_HT_4W / DG_MEGA_HT_BN), */ \
        /* never auto-selected -- commented out to cut template instantiations: */ \
        /* else if (block_m_ht == 128 && use_4warp_gemm) { LAUNCH_MEGA_MOE_HT_SPLIT_BM128_4W(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } */ \
        /* BN256 variants trimmed: BN512 is the default everywhere: */ \
        /* else if (block_m_ht == 128) { LAUNCH_MEGA_MOE_HT_SPLIT_BM128(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } */ \
        /* else if (block_m_ht == 144) { LAUNCH_MEGA_MOE_HT_SPLIT_BM144(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } */ \
        /* else if (block_m_ht == 160) { LAUNCH_MEGA_MOE_HT_SPLIT_BM160(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } */ \
        /* else if (block_m_ht == 256) { LAUNCH_MEGA_MOE_HT_SPLIT_BM256(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } */ \
        else if (block_m_ht == 64 && block_n_ht == 512) { \
            /* BM64 also gets the 768-column / 12-warp L1 pass where it tiles \
               (default-on for every dispatch count, same as BM128). */ \
            if constexpr ((2 * (IH)) % 768 == 0 && (IH) % 384 == 0) { \
                if (use_w12p && !ht_bm64_v1) { LAUNCH_BN512_BM64_W12P_KERN(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else if (ht_bm64_v1) { LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
                else { LAUNCH_MEGA_MOE_HT_SPLIT_BM64V2_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } } \
            else if (ht_bm64_v1) { LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } \
            else { LAUNCH_MEGA_MOE_HT_SPLIT_BM64V2_BN512(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } } \
        /* else if (block_m_ht == 64) { LAUNCH_MEGA_MOE_HT_SPLIT_BM64(NUM_SMS, TOPK, HIDDEN, IH, NDISP_SMS, EPR, NR); } */ \
        else { TORCH_CHECK(false, "HT compiled config is BM128/BN512 or BM64/BN512; got bm=", block_m_ht, " bn=", block_n_ht); }

    // HT dispatch SM counts kept instantiated: {2, 4, 6} (post-12-wave sweep
    // in megamoe/perf_history.md: the heuristic auto-selects 2/4; 6 is kept
    // as the env-reachable runner-up band; 3/5/7/8 never beat 4 outside
    // noise and are pruned).  Every kept value has BOTH the BN512/8-warp
    // instance and (where the 768-column pass tiles) the W12P/12-wave one.
    // Values outside {2,4,6} TORCH_CHECK.
    #define LAUNCH_HT_FOR_SMS(NUM_SMS, TOPK, HIDDEN, IH, EPR, NR) \
        do { \
            if (n_disp_sms == 2) { LAUNCH_MEGA_MOE_HT(NUM_SMS, TOPK, HIDDEN, IH, 2, EPR, NR); } \
            else if (n_disp_sms == 4) { LAUNCH_MEGA_MOE_HT(NUM_SMS, TOPK, HIDDEN, IH, 4, EPR, NR); } \
            else if (n_disp_sms == 6) { LAUNCH_MEGA_MOE_HT(NUM_SMS, TOPK, HIDDEN, IH, 6, EPR, NR); } \
            else { TORCH_CHECK(false, "HT dispatch SM count not instantiated (kept: 2/4/6): ", n_disp_sms); } \
        } while (0)
#endif // DG_MEGA_SKIP_HT

#if !defined(DG_MEGA_SKIP_LL)
    #define LAUNCH_SPLIT_16W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, BLOCK_M, 64, NUM_SMS, NDISP_SMS, 16, NMATH, L1PS, L2PS, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, 1024, smem_bytes_ll, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_SPLIT_16W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, true, EPR, NR), \
            LAUNCH_SPLIT_16W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, false, EPR, NR))

    #define LAUNCH_SPLIT_12W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, BLOCK_M, 64, NUM_SMS, NDISP_SMS, 12, NMATH, L1PS, L2PS, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, 768, smem_bytes_ll, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_SPLIT_12W(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_SPLIT_12W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, true, EPR, NR), \
            LAUNCH_SPLIT_12W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, false, EPR, NR))

    #define LAUNCH_SPLIT_8W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, IS_INT8, EPR, NR) \
        mega_moe_fp8_kernel_unified<TOPK, HIDDEN, IH, BLOCK_M, 64, NUM_SMS, NDISP_SMS, 8, NMATH, L1PS, L2PS, 256, IS_INT8, EPR, NR, DG_MEGA_KRING><<<NUM_SMS, 512, smem_bytes_ll, stream>>>(DG_MEGA_LAUNCH_ARGS, prof_ptr)
    #define LAUNCH_SPLIT_8W(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, EPR, NR) \
        DG_MEGA_IF_INT8_ELSE( \
            LAUNCH_SPLIT_8W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, true, EPR, NR), \
            LAUNCH_SPLIT_8W_K(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, false, EPR, NR))

    // Default macro: dispatch based on ll_lb at runtime (compile-time template selection)
    // Note: NMATH must be <= 12 for 12W variant and <= 8 for 8W variant.
    // For n_math == 16, use LAUNCH_SPLIT_16W directly (see DISPATCH_SPLIT_INNER).
    #define LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, EPR, NR) \
        do { \
            if (ll_lb == 768) { LAUNCH_SPLIT_12W(NUM_SMS, TOPK, HIDDEN, IH, BLOCK_M, NDISP_SMS, NMATH, L1PS, L2PS, EPR, NR); } \
            /* 512/1024-thread (8W/16W) instantiations trimmed: 768/12W is */ \
            /* the only auto-selected LL launch bound. */ \
            else { TORCH_CHECK(false, "Unsupported (or commented-out) LL launch bound: ", ll_lb); } \
        } while (0)

    // Inner dispatch with compile-time PIPE_STAGE values.
    // LL kernel: n_disp_sms in {2, 4} and n_math in {8}.
    // Other combinations are commented out to limit compile time and binary size.
    #define DISPATCH_SPLIT_INNER(NUM_SMS, TOPK, HIDDEN, IH, L1PS, L2PS, EPR, NR) \
        do { \
            if (block_m_tile == 16) { \
                if (n_disp_sms == 4 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 4, 8, L1PS, L2PS, EPR, NR); } \
                else if (n_disp_sms == 2 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 2, 8, L1PS, L2PS, EPR, NR); } \
                /* env-only combos, never auto-selected: */ \
                /* else if (n_disp_sms == 8 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 8, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 12 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 12, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 16 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 16, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 20 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 20, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 1 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 1, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 4 && n_math == 4) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 4, 4, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 2 && n_math == 4) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 2, 4, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 1 && n_math == 4) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 16, 1, 4, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 2 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 16, 2, 16, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 4 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 16, 4, 16, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 8 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 16, 8, 16, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 1 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 16, 1, 16, L1PS, L2PS, EPR, NR); } */ \
                else { TORCH_CHECK(false, "Unsupported (or commented-out) split config: disp=", n_disp_sms, " math=", n_math); } \
            } else if (block_m_tile == 32) { \
                if (n_disp_sms == 2 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 2, 8, L1PS, L2PS, EPR, NR); } \
                else if (n_disp_sms == 4 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 4, 8, L1PS, L2PS, EPR, NR); } \
                /* env-only combos, never auto-selected: */ \
                /* else if (n_disp_sms == 8 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 8, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 12 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 12, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 16 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 16, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 20 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 20, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 1 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 1, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 2 && n_math == 4) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 32, 2, 4, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 2 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 32, 2, 16, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 4 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 32, 4, 16, L1PS, L2PS, EPR, NR); } */ \
                else { TORCH_CHECK(false, "Unsupported (or commented-out) split config: disp=", n_disp_sms, " math=", n_math); } \
            } else { \
                /* block_m_tile == 48: large-token specialization (M_TILES=3) */ \
                if (n_disp_sms == 2 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 48, 2, 8, L1PS, L2PS, EPR, NR); } \
                else if (n_disp_sms == 4 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 48, 4, 8, L1PS, L2PS, EPR, NR); } \
                /* env-only combos, never auto-selected: */ \
                /* else if (n_disp_sms == 8 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 48, 8, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 12 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 48, 12, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 16 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 48, 16, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 20 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 48, 20, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 1 && n_math == 8) { LAUNCH_MEGA_MOE_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, 48, 1, 8, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 2 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 48, 2, 16, L1PS, L2PS, EPR, NR); } */ \
                /* else if (n_disp_sms == 4 && n_math == 16) { LAUNCH_SPLIT_16W(NUM_SMS, TOPK, HIDDEN, IH, 48, 4, 16, L1PS, L2PS, EPR, NR); } */ \
                else { TORCH_CHECK(false, "Unsupported (or commented-out) split config: disp=", n_disp_sms, " math=", n_math); } \
            } \
        } while (0)

    // Outer dispatch: select compile-time PIPE_STAGE combination.
    // Only the (4,4) default is compiled; env-only combos are commented out
    // to cut build time (they were never auto-selected anyway).
    #define DISPATCH_TOPK_SPLIT(NUM_SMS, TOPK, HIDDEN, IH, EPR, NR) \
        do { \
            /* L1PS >= 8 selects the warp-K L1 variant (inner pipe = L1PS-8) */ \
            /* if (l1_pipe_env == 10 && l2_pipe_env == 4) { DISPATCH_SPLIT_INNER(NUM_SMS, TOPK, HIDDEN, IH, 10, 4, EPR, NR); } */ \
            /* else if (l1_pipe_env == 11 && l2_pipe_env == 4) { DISPATCH_SPLIT_INNER(NUM_SMS, TOPK, HIDDEN, IH, 11, 4, EPR, NR); } */ \
            /* else if (l1_pipe_env == 3 && l2_pipe_env == 3) { DISPATCH_SPLIT_INNER(NUM_SMS, TOPK, HIDDEN, IH, 3, 3, EPR, NR); } */ \
            if (l1_pipe_env == 4 && l2_pipe_env == 4) { DISPATCH_SPLIT_INNER(NUM_SMS, TOPK, HIDDEN, IH, 4, 4, EPR, NR); } \
            /* else if (l1_pipe_env == 2 && l2_pipe_env == 2) { DISPATCH_SPLIT_INNER(NUM_SMS, TOPK, HIDDEN, IH, 2, 2, EPR, NR); } */ \
            /* else { DISPATCH_SPLIT_INNER(NUM_SMS, TOPK, HIDDEN, IH, 2, 3, EPR, NR); } */ \
            else { TORCH_CHECK(false, "Unsupported (or commented-out) LL pipe: l1=", l1_pipe_env, " l2=", l2_pipe_env); } \
        } while (0)
#endif // DG_MEGA_SKIP_LL

    // ---- Single dispatch + host sync -------------------------------------
    // Launch one config family for a given (topk, hidden, ih) shape and a
    // compile-time topology (EPR experts/rank x NR ranks; 0,0 = runtime
    // fallback that reads the values from kernel arguments).
#if !defined(DG_MEGA_SKIP_HT) && !defined(DG_MEGA_SKIP_LL)
    #define DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR_, NR_) \
        do { \
            if (use_ht) { \
                if (num_sms == 64) { LAUNCH_HT_FOR_SMS(64, TOPK_, H_, IH_, EPR_, NR_); } \
                else { LAUNCH_HT_FOR_SMS(72, TOPK_, H_, IH_, EPR_, NR_); } \
            } else { \
                if (num_sms == 64) { DISPATCH_TOPK_SPLIT(64, TOPK_, H_, IH_, EPR_, NR_); } \
                else { DISPATCH_TOPK_SPLIT(72, TOPK_, H_, IH_, EPR_, NR_); } \
            } \
        } while (0)
#elif !defined(DG_MEGA_SKIP_HT)
    #define DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR_, NR_) \
        do { \
            if (num_sms == 64) { LAUNCH_HT_FOR_SMS(64, TOPK_, H_, IH_, EPR_, NR_); } \
            else { LAUNCH_HT_FOR_SMS(72, TOPK_, H_, IH_, EPR_, NR_); } \
        } while (0)
#else
    #define DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR_, NR_) \
        do { \
            if (num_sms == 64) { DISPATCH_TOPK_SPLIT(64, TOPK_, H_, IH_, EPR_, NR_); } \
            else { DISPATCH_TOPK_SPLIT(72, TOPK_, H_, IH_, EPR_, NR_); } \
        } while (0)
#endif

    // One compiled (shape, topology) row: specialized <EPR_,8> kernels when the
    // runtime topology matches, generic <0,0> otherwise.
    #define DG_MEGA_DISPATCH_SHAPE(TOPK_, H_, IH_, EPR_) \
        do { \
            if (num_experts_per_rank == (EPR_) && num_ranks == 8) { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR_, 8); \
            } else { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, 0, 0); \
            } \
        } while (0)
    // DG_MEGA_DISPATCH_SHAPE plus one extra EPLB-redundant 8-rank expert count
    // for the SAME (topk, h, ih).  Plain <EPR_,8> stays the default; EPLB
    // deployments get their own <EPR_EPLB_,8>; every other topology (including
    // any num_ranks != 8) keeps the original generic <0,0> fallback, so the
    // non-8-rank behaviour is unchanged.
    #define DG_MEGA_DISPATCH_SHAPE_EPLB(TOPK_, H_, IH_, EPR_, EPR_EPLB_) \
        do { \
            if (num_experts_per_rank == (EPR_) && num_ranks == 8) { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR_, 8); \
            } else if (num_experts_per_rank == (EPR_EPLB_) && num_ranks == 8) { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR_EPLB_, 8); \
            } else { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, 0, 0); \
            } \
        } while (0)
    // Same as DG_MEGA_DISPATCH_SHAPE, but additionally specializes the
    // two-node <EPR16_,16> topology. The 8-rank and 16-rank deployments of a
    // model can carry different experts-per-rank, so they take separate EPRs
    // (hy4: 32/rank x 8 = 256 experts; 17/rank x 16 = 272 experts).
    #define DG_MEGA_DISPATCH_SHAPE_NR16(TOPK_, H_, IH_, EPR8_, EPR16_) \
        do { \
            if (num_experts_per_rank == (EPR8_) && num_ranks == 8) { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR8_, 8); \
            } else if (num_experts_per_rank == (EPR16_) && num_ranks == 16) { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, EPR16_, 16); \
            } else { \
                DG_MEGA_LAUNCH_KERNELS(TOPK_, H_, IH_, 0, 0); \
            } \
        } while (0)
    // Compile-time (shape, topology) specialization table.  Each row compiles
    // <EPR,8> kernels that fold all expert/rank math to constants; unmatched
    // topologies take the <0,0> generic instantiation (runtime args).
    // Shape format: (num_topk, hidden, intermediate_hidden, epr_for_nr8).
#if DG_MEGAMOE_DBUG
    {
        size_t dbg_smem = 0;
#if !defined(DG_MEGA_SKIP_HT) && !defined(DG_MEGA_SKIP_LL)
        dbg_smem = use_ht ? smem_bytes_ht : smem_bytes_ll;
#elif !defined(DG_MEGA_SKIP_HT)
        dbg_smem = smem_bytes_ht;
#else
        dbg_smem = smem_bytes_ll;
#endif
        DG_DBUG_PRINTF_HOST("[HOST] launching mega_moe kernel: num_sms=%u smem=%zu rank=%d nranks=%d\n",
               num_sms, dbg_smem, rank_idx, num_ranks);
        fflush(stdout);
    }
#endif

    // Shape x topology dispatch table.  Rows compile the listed combos plus a
    // generic <0,0> topology fallback per shape (runtime epr/nr args).  The
    // (8,7168,2048) shape carries two expert totals -> two specialized rows.
    // Expanded twice below: once with DG_MEGA_KRING=true (ring-buffered L1
    // staging) and once with false — the leaf *_K macros append it as the
    // kernel's kUseRingBuffer template argument.
    #define DG_MEGA_DISPATCH_TABLE() \
        do { \
            if (num_topk == 6 && hidden == 5120 && intermediate_hidden == 2304) { \
                /* v4.1: plain 384 (48/rank) + EPLB-r32 416 (52/rank) */ \
                DG_MEGA_DISPATCH_SHAPE_EPLB(6, 5120, 2304, 48, 52); \
            } else if (num_topk == 3 && hidden == 5120 && intermediate_hidden == 2304) { \
                /* dspark-mark: plain 128 (16/rank) + EPLB-r32 160 (20/rank) */ \
                DG_MEGA_DISPATCH_SHAPE_EPLB(3, 5120, 2304, 16, 20); \
            } else if (num_topk == 6 && hidden == 7168 && intermediate_hidden == 3072) { \
                DG_MEGA_DISPATCH_SHAPE(6, 7168, 3072, 48);   /* 384 experts / 8 ranks */ \
            } else if (num_topk == 6 && hidden == 4096 && intermediate_hidden == 2048) { \
                DG_MEGA_DISPATCH_SHAPE(6, 4096, 2048, 32);   /* deepseek-v4: 256 experts / 8 ranks */ \
            } else if (num_topk == 8 && hidden == 6144 && intermediate_hidden == 2048) { \
                DG_MEGA_DISPATCH_SHAPE_NR16(8, 6144, 2048, 32, 17);   /* hy4: 256 (8x32) / 272 (16x17) experts */ \
            } else { \
                TORCH_CHECK(false, "Unsupported (topk, hidden, ih) = (", \
                            num_topk, ", ", hidden, ", ", intermediate_hidden, ")"); \
            } \
        } while (0)

    // use_ring_buffer / num_max_pool_tokens were computed above (needed by
    // the dispatch-SM heuristic).
    if (use_ring_buffer) {
        #define DG_MEGA_KRING true
        DG_MEGA_DISPATCH_TABLE();
        #undef DG_MEGA_KRING
    } else {
        #define DG_MEGA_KRING false
        DG_MEGA_DISPATCH_TABLE();
        #undef DG_MEGA_KRING
    }
    #undef DG_MEGA_DISPATCH_TABLE

    #undef DG_MEGA_LAUNCH_KERNELS
    #undef DG_MEGA_DISPATCH_SHAPE
    #undef DG_MEGA_DISPATCH_SHAPE_EPLB
    #undef DG_MEGA_DISPATCH_SHAPE_NR16

    // No post-launch host sync by default (matches the NV entry point).
    // DG_MEGA_HOST_SYNC=1 restores the legacy hipStreamSynchronize for
    // debugging; the sync holds the HIP context lock in some ROCm versions,
    // blocks concurrent hipMemcpyAsync from watchdog/diagnostic threads,
    // and would invalidate HIP graph capture, so it stays opt-in.
    static const bool host_sync =
        std::getenv("DG_MEGA_HOST_SYNC") != nullptr
        && std::string(std::getenv("DG_MEGA_HOST_SYNC")) == "1";
    if (host_sync) {
        hipStreamSynchronize(stream);
    }
#if DG_MEGAMOE_DBUG
    DG_DBUG_PRINTF_HOST("[HOST] kernel sync done\n");
    fflush(stdout);
#endif

    #undef LAUNCH_MEGA_MOE_HT
    #undef LAUNCH_HT_FOR_SMS
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM64
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM128
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM128_BN512
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM128_4W
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM144
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM160
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM256
    #undef LAUNCH_BN512_KERN
    #undef LAUNCH_BN512_KERN_K
    #undef LAUNCH_BM144_KERN
    #undef LAUNCH_BM144_KERN_K
    #undef LAUNCH_BM160_KERN
    #undef LAUNCH_BM160_KERN_K
    #undef LAUNCH_BM256_KERN
    #undef LAUNCH_BM256_KERN_K
    #undef LAUNCH_MEGA_MOE_SPLIT
    #undef LAUNCH_SPLIT_16W
    #undef LAUNCH_SPLIT_16W_K
    #undef LAUNCH_SPLIT_12W
    #undef LAUNCH_SPLIT_12W_K
    #undef LAUNCH_SPLIT_8W
    #undef LAUNCH_SPLIT_8W_K
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM64_K
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM64_BN512_K
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM128_K
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM128_BN512_K
    #undef LAUNCH_MEGA_MOE_HT_SPLIT_BM128_4W_K
    #undef DISPATCH_SPLIT_INNER
    #undef DISPATCH_TOPK_SPLIT

    #undef DG_MEGA_LAUNCH_ARGS
}
#endif

// ============================================================================
// mega_moe_preprocess: fused bf16 -> FP8 quant + topk copy into sym buffer
// ============================================================================

static void mega_moe_preprocess_launch(
    torch::Tensor x_bf16,
    torch::Tensor topk_idx,
    torch::Tensor topk_weights,
    torch::Tensor out_fp8,
    torch::Tensor out_scale,
    torch::Tensor out_topk_idx,
    torch::Tensor out_topk_weights,
    int64_t rows,
    bool is_int8 = false) {
    mega_moe_check_arch();
    TORCH_CHECK(x_bf16.dim() == 2, "x_bf16 must be 2D");
    TORCH_CHECK(x_bf16.scalar_type() == torch::kBFloat16, "x_bf16 must be BF16");
    const int hidden = static_cast<int>(x_bf16.size(1));
    const int topk = static_cast<int>(topk_idx.size(1));
    const int n_rows = static_cast<int>(rows);

    const bool topk_idx_i64 = (topk_idx.scalar_type() == torch::kInt64);
    const bool topk_weights_bf16 = (topk_weights.scalar_type() == torch::kBFloat16);

    launch_mega_moe_preprocess(
        x_bf16.data_ptr(),
        topk_idx.data_ptr(),
        topk_weights.data_ptr(),
        out_fp8.data_ptr(),
        out_scale.data_ptr<float>(),
        out_topk_idx.data_ptr<int64_t>(),
        out_topk_weights.data_ptr<float>(),
        n_rows, hidden, topk,
        topk_idx_i64, topk_weights_bf16,
        is_int8);
}

// ============================================================================
// Python bindings
// ============================================================================

static int get_token_alignment_for_mega_moe() {
    return kLCMCandidateBlockM;
}

static pybind11::dict get_mega_moe_hip_build_config() {
    pybind11::dict config;
    config["operator"] = "fp8_w8a8_mega_moe";
    config["package"] = "megamoe_hcu";
    config["weight_scale_mode"] = "channelwise_fp32";
    config["input_scale_mode"] = "channelwise_fp32";
    config["l2_act_scale_mode"] = "channelwise_fp32";
    config["has_fp8_api"] = true;
    config["has_fp4_api"] = false;
#if defined(DG_MEGA_HAS_HIP_FABRIC)
    config["fabric_peer_memory"] = true;
#else
    // Toolchain lacks <hip/hip_hsa_ext.h> (DTK < 26.10): fabric bindings are
    // not compiled in, so report the capability honestly.
    config["fabric_peer_memory"] = false;
#endif
    config["fusion_boundary"] = "dispatch_pool_l1_swiglu_quant_l2_combine";
    return config;
}

} // namespace deep_gemm::mega::hcu

// Device VA of the dg_hang_records array (params.h).  The watchdog reads it
// with ctypes.string_at — no GPU queue involved, so it stays reachable even
// when a kernel is wedged.  Valid in every build (array is always emitted).
static int64_t get_hang_record_ptr() {
    void* p = nullptr;
    const hipError_t err = hipGetSymbolAddress(&p, dg_hang_records);
    if (err != hipSuccess || p == nullptr)
        return 0;
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(p));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace deep_gemm::mega::hcu;

    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe,
          pybind11::arg("num_ranks"), pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_topk"),
          pybind11::arg("hidden"), pybind11::arg("intermediate_hidden"),
          pybind11::arg("use_fp8_dispatch"), pybind11::arg("activation"));
    m.def("get_mega_moe_hip_build_config", &get_mega_moe_hip_build_config);

    m.def("allocate_hip_ipc_buffer", &allocate_hip_ipc_buffer,
          pybind11::arg("num_bytes"));
    m.def("open_hip_ipc_handles", &open_hip_ipc_handles,
          pybind11::arg("handles"), pybind11::arg("rank_idx"));
    m.def("close_hip_ipc_handles", &close_hip_ipc_handles,
          pybind11::arg("ptrs"));
    m.def("free_hip_ipc_buffer", &free_hip_ipc_buffer,
          pybind11::arg("ptr"));
    m.def("allocate_hip_local_buffer", &allocate_hip_local_buffer,
          pybind11::arg("num_bytes"));
    m.def("free_hip_local_buffer", &free_hip_local_buffer,
          pybind11::arg("ptr"));
    m.def("allocate_hip_ipc_signal_buffer", &allocate_hip_ipc_signal_buffer,
          pybind11::arg("num_bytes"));
    m.def("free_hip_ipc_signal_buffer", &free_hip_ipc_signal_buffer,
          pybind11::arg("ptr"));
    m.def("set_mega_moe_peer_ptrs", &set_mega_moe_peer_ptrs,
          pybind11::arg("sym_buffer"),
          pybind11::arg("sym_buffer_ptrs"),
          pybind11::arg("signal_ptrs"));

#if defined(DG_MEGA_HAS_HIP_FABRIC)
    // Omitted on toolchains without <hip/hip_hsa_ext.h> (DTK < 26.10); the
    // Python layer then rejects DG_MEGA_PEER_MEMORY=rpc with a clear message.
    m.def("allocate_hip_fabric_buffer", &allocate_hip_fabric_buffer,
          pybind11::arg("num_bytes"));
    m.def("allocate_hip_fabric_signal_buffer", &allocate_hip_fabric_signal_buffer,
          pybind11::arg("num_bytes"));
    m.def("open_hip_fabric_handles", &open_hip_fabric_handles,
          pybind11::arg("handles"), pybind11::arg("rank_idx"));
    m.def("close_hip_fabric_handles", &close_hip_fabric_handles,
          pybind11::arg("ptrs"));
#endif  // DG_MEGA_HAS_HIP_FABRIC

#if !defined(DG_MEGA_SKIP_LL) || !defined(DG_MEGA_SKIP_HT)
    // Single unified entry: auto-selects LL vs HT config by
    // num_max_tokens_per_rank (>= DG_MEGA_HT_THRESHOLD, default 192 -> HT).
    // Both modes consume the same Marlin w6 weight layout.
    m.def("fp8_mega_moe", &fp8_mega_moe_launch,
          pybind11::arg("y"),
          pybind11::arg("l1_weights"), pybind11::arg("l2_weights"),
          pybind11::arg("l1_weights_sf"), pybind11::arg("l2_weights_sf"),
          pybind11::arg("sym_buffer"),
          pybind11::arg("local_buffer"),
          pybind11::arg("sym_buffer_ptrs"), pybind11::arg("signal_ptrs"),
          pybind11::arg("rank_idx"), pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_experts"), pybind11::arg("num_topk"),
          pybind11::arg("activation_clamp"), pybind11::arg("fast_math"),
          pybind11::arg("is_int8") = false,
          pybind11::arg("tokens_all_ranks") = -1,
          pybind11::call_guard<pybind11::gil_scoped_release>());
#endif

    m.def("mega_moe_prof_fetch", &mega_moe_prof_fetch,
          "Fetch the DG_MEGA_PROF per-block phase buffer (int64[num_sms*16])");
    m.def("get_hang_record_ptr", &get_hang_record_ptr,
          "Device VA of dg_hang_records (u32[1024]) for host-direct reads");
    m.def("mega_moe_preprocess", &mega_moe_preprocess_launch,
          pybind11::arg("x_bf16"),
          pybind11::arg("topk_idx"), pybind11::arg("topk_weights"),
          pybind11::arg("out_fp8"), pybind11::arg("out_scale"),
          pybind11::arg("out_topk_idx"), pybind11::arg("out_topk_weights"),
          pybind11::arg("rows"),
          pybind11::arg("is_int8") = false);
}
