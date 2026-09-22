#pragma once

#include <torch/extension.h>

#include "exception.h"

namespace deepgemm {

template <int N>
static auto get_shape(const torch::Tensor& t) {
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    LIGHTOP_HOST_ASSERT(dim == 2 or dim == 3);
    if (dim == 3)
        LIGHTOP_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    LIGHTOP_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

static void check_major_type_cd(const torch::Tensor& t) {
    major_check(t);
    LIGHTOP_HOST_ASSERT(t.stride(-1) == 1);
}

} // namespace deepgemm
