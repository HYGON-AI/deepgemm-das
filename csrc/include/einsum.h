#pragma once

#include <torch/extension.h>

#include <string>
#include <tuple>
#include <utility>

namespace deepgemm::einsum {

void einsum(const std::string& expr,
            const torch::Tensor& a,
            const torch::Tensor& b,
            const torch::Tensor& d,
            const std::optional<torch::Tensor>& c = std::nullopt);

void fp8_einsum(const std::string& expr,
                const std::pair<torch::Tensor, torch::Tensor>& a,
                const std::pair<torch::Tensor, torch::Tensor>& b,
                const torch::Tensor& d,
                const std::optional<torch::Tensor>& c = std::nullopt,
                const std::tuple<int, int, int>& recipe = {1, 128, 128});

} // namespace deepgemm::einsum
