// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <hip/hip_runtime.h>

namespace deepgemm::mqa::common {
using I2 = int __attribute__((ext_vector_type(2)));
using I4 = int __attribute__((ext_vector_type(4)));
using F2 = float __attribute__((ext_vector_type(2)));
using F4 = float __attribute__((ext_vector_type(4)));
using LdsInt = int __attribute__((address_space(3)));
} // namespace deepgemm::mqa::common
