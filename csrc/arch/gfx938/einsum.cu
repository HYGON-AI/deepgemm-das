#include <cstdlib>
#include <sstream>
#include <string>

#include <torch/extension.h>

#include "einsum.h"
#include "exception.h"
#include "fp8_gemm_hipc.h"
#include "gemm.h"
#include "tensor_utils.h"

namespace deepgemm::einsum {
namespace {

// Backend for the fp8_bmm inner GEMM.
//
// Default (kHipc): the warp-specialized hipc kernel
// run_fp8_gemm_hipc. It shares deepgemm::fp8_gemm's signature,
// but unlike the hipBLASLt path (csrc/arch/gfx938/fp8_gemm.cu, which never
// sets a scale batch stride) it applies per-batch (per-head) row scales
// correctly when batch > 1 — exactly what the head-major FP8 einsum
// expressions need.
//
// Set DEEPGEMM_EINSUM_FP8_ENGINE=hipblaslt to force the hipBLASLt reference
// back in (e.g. for A/B accuracy or performance comparisons).
enum class Fp8Engine { kHipc, kHipblaslt };

Fp8Engine fp8_engine() {
    static const Fp8Engine engine = [] {
        const char* env = std::getenv("DEEPGEMM_EINSUM_FP8_ENGINE");
        return (env != nullptr && std::string(env) == "hipblaslt")
                   ? Fp8Engine::kHipblaslt
                   : Fp8Engine::kHipc;
    }();
    return engine;
}

bool early_return(int m, int n, int k) {
    return m == 0 || n == 0 || k == 0;
}

void bmk_bnk_mn(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                const std::optional<torch::Tensor>& c) {
    LIGHTOP_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    LIGHTOP_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    LIGHTOP_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (d.scalar_type() == torch::kFloat) {
        LIGHTOP_HOST_ASSERT(c.has_value());
        LIGHTOP_HOST_ASSERT(c->data_ptr() == d.data_ptr());
    } else {
        LIGHTOP_HOST_ASSERT(not c.has_value());
    }

    const auto [s, m, k] = deepgemm::get_shape<3>(a);
    const auto [s_, n, k_] = deepgemm::get_shape<3>(b);
    LIGHTOP_HOST_ASSERT(s == s_ and k == k_);

    if (early_return(m, n, k)) {
        return;
    }

    const auto reduced = at::bmm(a, b.transpose(1, 2)).sum(0);

    if (d.scalar_type() == torch::kFloat) {
        d.copy_(c.value().add(reduced));
    } else {
        d.copy_(reduced.to(torch::kBFloat16));
    }
}

// bhr,hdr->bhd: permute to head-major batch GEMM, then permute back.
void bhr_hdr_bhd(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    const auto [b_sz, h, r] = deepgemm::get_shape<3>(a);
    const auto [h_, d_sz, r_] = deepgemm::get_shape<3>(b);
    const auto [b_, h__, d_] = deepgemm::get_shape<3>(d);
    LIGHTOP_HOST_ASSERT(b_sz == b_ and h == h_ and r == r_ and d_sz == d_ and h == h__);

    LIGHTOP_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and a.stride(2) == 1);
    LIGHTOP_HOST_ASSERT(b.scalar_type() == torch::kBFloat16 and b.stride(2) == 1);
    LIGHTOP_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 and d.stride(2) == 1);

    if (early_return(b_sz, d_sz, r)) {
        return;
    }

    const auto a_hbr = a.permute({1, 0, 2}).contiguous();
    const auto out_hbd = at::bmm(a_hbr, b.transpose(-1, -2));
    d.copy_(out_hbd.permute({1, 0, 2}));
}

// bhd,hdr->bhr
void bhd_hdr_bhr(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    const auto [b_sz, h, d_sz] = deepgemm::get_shape<3>(a);
    const auto [h_, d_, r] = deepgemm::get_shape<3>(b);
    const auto [b_, h__, r_] = deepgemm::get_shape<3>(d);
    LIGHTOP_HOST_ASSERT(b_sz == b_ and h == h_ and r == r_ and d_sz == d_ and h == h__);

    LIGHTOP_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and a.stride(2) == 1);
    LIGHTOP_HOST_ASSERT(b.scalar_type() == torch::kBFloat16 and b.stride(2) == 1);
    LIGHTOP_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 and d.stride(2) == 1);

    if (early_return(b_sz, r, d_sz)) {
        return;
    }

    const auto a_hbd = a.permute({1, 0, 2}).contiguous();
    const auto out_hbr = at::bmm(a_hbd, b);
    d.copy_(out_hbr.permute({1, 0, 2}));
}

void fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
             const torch::Tensor& b, const torch::Tensor& sfb,
             const torch::Tensor& d,
             const std::optional<torch::Tensor>& c) {
    LIGHTOP_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    LIGHTOP_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    LIGHTOP_HOST_ASSERT(d.stride(-1) == 1);

    const auto [batch_size, m, k] = deepgemm::get_shape<3>(a);
    const auto [batch_size_, n, k_] = deepgemm::get_shape<3>(b);
    const auto [batch_size__, m_, n_] = deepgemm::get_shape<3>(d);
    LIGHTOP_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    LIGHTOP_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    LIGHTOP_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    LIGHTOP_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    LIGHTOP_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    if (batch_size == 0 or early_return(m, n, k)) {
        return;
    }

    const auto alpha = torch::tensor(1.f, torch::TensorOptions().dtype(torch::kFloat32));
    const auto beta = torch::tensor(c.has_value() ? 1.f : 0.f, torch::TensorOptions().dtype(torch::kFloat32));
    const auto output = c.has_value() ? c.value() : d;

    // Both engines share the deepgemm::fp8_gemm argument order, so the call
    // below is identical either way (see fp8_engine() above).
    if (fp8_engine() == Fp8Engine::kHipc) {
        run_fp8_gemm_hipc(b, a, sfb, sfa, output,
                          n, m, k, batch_size, "TN",
                          alpha, beta, std::nullopt);
    } else {
        deepgemm::fp8_gemm(b, a, sfb, sfa, output,
                           n, m, k, batch_size, "TN",
                           alpha, beta, std::nullopt);
    }
}

torch::Tensor permute_scale_bh(const torch::Tensor& scale_bh) {
    LIGHTOP_HOST_ASSERT(scale_bh.dim() == 2);
    return scale_bh.permute({1, 0}).contiguous();
}

} // namespace

void einsum(const std::string& expr,
            const torch::Tensor& a,
            const torch::Tensor& b,
            const torch::Tensor& d,
            const std::optional<torch::Tensor>& c) {
    LIGHTOP_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    LIGHTOP_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    LIGHTOP_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        LIGHTOP_HOST_ASSERT(c->scalar_type() == torch::kFloat);
        LIGHTOP_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    }

    if (expr == "bmk,bnk->mn") {
        bmk_bnk_mn(a, b, d, c);
    } else if (expr == "bhr,hdr->bhd") {
        LIGHTOP_HOST_ASSERT(not c.has_value());
        bhr_hdr_bhd(a, b, d);
    } else if (expr == "bhd,hdr->bhr") {
        LIGHTOP_HOST_ASSERT(not c.has_value());
        bhd_hdr_bhr(a, b, d);
    } else {
        std::ostringstream oss;
        oss << "Unsupported einsum expression: " << expr;
        LIGHTOP_HOST_UNREACHABLE(oss.str().c_str());
    }
}

void fp8_einsum(const std::string& expr,
                const std::pair<torch::Tensor, torch::Tensor>& a,
                const std::pair<torch::Tensor, torch::Tensor>& b,
                const torch::Tensor& d,
                const std::optional<torch::Tensor>& c,
                const std::tuple<int, int, int>& /*recipe*/) {
    if (expr == "bhr,hdr->bhd") {
        const auto perm_a = a.first.permute({1, 0, 2}).contiguous();
        const auto perm_sfa = permute_scale_bh(a.second);
        auto perm_d = d.permute({1, 0, 2}).contiguous();
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2}).contiguous()) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, b.first.contiguous(), b.second.contiguous(), perm_d, perm_c);
        d.copy_(perm_d.permute({1, 0, 2}));
    } else if (expr == "bhd,hdr->bhr") {
        const auto perm_a = a.first.permute({1, 0, 2}).contiguous();
        const auto perm_sfa = permute_scale_bh(a.second);
        const auto perm_b = b.first.permute({0, 2, 1}).contiguous();
        LIGHTOP_HOST_ASSERT(b.second.dim() == 2);
        const auto perm_sfb = b.second.contiguous();
        auto perm_d = d.permute({1, 0, 2}).contiguous();
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2}).contiguous()) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, perm_d, perm_c);
        d.copy_(perm_d.permute({1, 0, 2}));
    } else if (expr == "bhd,bhr->hdr") {
        const auto perm_a = a.first.permute({1, 2, 0}).contiguous();
        const auto perm_b = b.first.permute({1, 2, 0}).contiguous();
        LIGHTOP_HOST_ASSERT(a.second.dim() == 2 and b.second.dim() == 2);
        const auto perm_sfa = a.second.contiguous();
        const auto perm_sfb = b.second.contiguous();
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, d, c);
    } else {
        std::ostringstream oss;
        oss << "Unsupported einsum expression: " << expr;
        LIGHTOP_HOST_UNREACHABLE(oss.str().c_str());
    }
}

} // namespace deepgemm::einsum
