// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Runtime GPU architecture detection for the operator dispatchers in
// csrc/arch/wrapper.  Every shared operator file switches on the architecture
// of the current device and forwards to the implementation namespace of the
// matching arch folder (gfx928 / gfx92a / gfx936 / gfx938 / gfx936_gfx938_shared).
#pragma once

#include <torch/extension.h>

#include <hip/hip_runtime.h>

#include <cstring>

// gfx936/gfx938 share a large part of their kernels; the host dispatch files
// only take this branch when at least one of the two was built.
#if defined(DEEPGEMM_ENABLE_GFX936) || defined(DEEPGEMM_ENABLE_GFX938)
#define DEEPGEMM_HAS_GFX93X 1
#else
#define DEEPGEMM_HAS_GFX93X 0
#endif

namespace deepgemm {
namespace arch {

enum class Arch { kGfx928, kGfx92a, kGfx936, kGfx938, kUnknown };

inline Arch from_name(const char* name) {
  if (name == nullptr) {
    return Arch::kUnknown;
  }
  if (std::strncmp(name, "gfx928", 6) == 0) {
    return Arch::kGfx928;
  }
  if (std::strncmp(name, "gfx92a", 6) == 0) {
    return Arch::kGfx92a;
  }
  if (std::strncmp(name, "gfx936", 6) == 0) {
    return Arch::kGfx936;
  }
  if (std::strncmp(name, "gfx938", 6) == 0) {
    return Arch::kGfx938;
  }
  return Arch::kUnknown;
}

inline Arch current() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    return Arch::kUnknown;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
    return Arch::kUnknown;
  }
  return from_name(prop.gcnArchName);
}

inline const char* name(Arch arch) {
  switch (arch) {
    case Arch::kGfx928:
      return "gfx928";
    case Arch::kGfx92a:
      return "gfx92a";
    case Arch::kGfx936:
      return "gfx936";
    case Arch::kGfx938:
      return "gfx938";
    default:
      return "unknown";
  }
}

[[noreturn]] inline void unsupported(const char* op) {
  TORCH_CHECK(false, op, " is not supported on the current GPU architecture (",
              name(current()), ")");
}

}  // namespace arch
}  // namespace deepgemm
