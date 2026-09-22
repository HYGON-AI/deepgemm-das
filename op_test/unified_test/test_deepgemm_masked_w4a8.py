import csv
from pathlib import Path

import torch
from deepgemm import m_grouped_w4a8_gemm_nt_masked
from deepgemm_marlin_quant import weight4bit_nt_kpack2_marlin2_qqq_from_packed  # pyright: ignore[reportMissingImports]

from masked_cases import CASES


def _bench(fn, repeat=9, iters=30):
    values = []
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / iters)

    values.sort()
    return values[0], values[len(values) // 2]


def _bytes_per_case(experts: int, size_m: int, size_n: int, size_k: int, active_m: int) -> int:
    # a: int8 = 1 byte, weight: int4 = 0.5 byte, output: bf16 = 2 bytes
    return experts * (active_m * size_k + size_n * size_k // 2 + active_m * size_n * 2)


def _tflops(experts: int, size_n: int, size_k: int, active_m: int, latency_us: float) -> float:
    flops = 2 * experts * active_m * size_n * size_k
    return flops / (latency_us * 1e-6) / 1e12


def _unpack_int8_to_int4(tensor_int8: torch.Tensor) -> torch.Tensor:
    E, N, K_half = tensor_int8.shape
    high4 = tensor_int8 & 0xF0
    low4 = (tensor_int8 << 4) & 0xF0
    unpacked = torch.empty((E, N, K_half * 2), dtype=torch.int8, device=tensor_int8.device)
    unpacked[:, :, 0::2] = high4.to(torch.int8)
    unpacked[:, :, 1::2] = low4.to(torch.int8)
    return unpacked


def _reference_matmul(q_a, w_signed, a_scale, w_scale, out_dtype):
    A = q_a.float()
    B = w_signed.float()
    C = torch.bmm(A, B.transpose(1, 2))
    C = a_scale * C * w_scale.transpose(1, 2)
    return C.to(out_dtype)


def run_case(name, experts, size_m, size_n, size_k, active_m, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out_dtype = torch.bfloat16

    q_a = torch.randint(-128, 128, (experts, size_m, size_k), device="cuda", dtype=torch.int8).contiguous()
    a_scale = torch.rand((experts, size_m, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)
    weight_packed = torch.randint(-128, 128, (experts, size_n, size_k // 2), device="cuda", dtype=torch.int8)
    w_scale = torch.rand((experts, size_n, 1), device="cuda", dtype=torch.float32).mul_(0.02).add_(0.001)

    w_unpacked = _unpack_int8_to_int4(weight_packed)

    masked_m = torch.full((experts,), active_m, device="cuda", dtype=torch.int32)
    output = torch.zeros(experts, size_m, size_n, device="cuda", dtype=out_dtype)

    ref = _reference_matmul(q_a, w_unpacked, a_scale, w_scale, out_dtype)

    marlin_list = []
    for i in range(weight_packed.shape[0]):
        marlin_list.append(weight4bit_nt_kpack2_marlin2_qqq_from_packed(weight_packed[i]))
    packed_weight = torch.stack(marlin_list, dim=0)

    m_grouped_w4a8_gemm_nt_masked(
        (q_a, a_scale),
        (packed_weight, w_scale),
        output,
        masked_m,
        active_m,
    )
    torch.cuda.synchronize()

    max_err = 0.0
    for expert in range(experts):
        diff = (output[expert, :active_m].float() - ref[expert, :active_m].float()).abs().max().item()
        max_err = max(max_err, diff)
        if not torch.allclose(output[expert, :active_m], ref[expert, :active_m], rtol=1e-2, atol=1e-2):
            raise AssertionError(f"{name} active rows mismatch at expert {expert}: max_err={diff}")

    best_us, median_us = _bench(
        lambda: m_grouped_w4a8_gemm_nt_masked(
            (q_a, a_scale),
            (packed_weight, w_scale),
            output,
            masked_m,
            active_m,
        )
    )
    bytes_moved = _bytes_per_case(experts, size_m, size_n, size_k, active_m)
    tflops = _tflops(experts, size_n, size_k, active_m, best_us)
    bandwidth = bytes_moved / (best_us * 1e-6) / 1e9


    print("------------------------------------------------------")
    print("deepgemm test cases: e: {} m: {} n: {} k: {} masked_m: {}".format(experts, size_m, size_n, size_k, active_m))
    print("deepgemm cost: {:.2f} us, {:.2f} TFlops".format(median_us, tflops))
    print("deepgemm bandwidth: {:.2f} GB/s".format(bandwidth))
    print("------------------------------------------------------")
    return {
        "case": name,
        "E": experts,
        "M": size_m,
        "N": size_n,
        "K": size_k,
        "masked_m": active_m,
        "variant": "small_m" if active_m <= 64 else "prefetch",
        "best_us": f"{best_us:.2f}",
        "median_us": f"{median_us:.2f}",
        "tflops": f"{tflops:.2f}",
        "algo_bw_GBps": f"{bandwidth:.2f}",
        "max_err": f"{max_err:.4f}",
        "status": "PASS",
    }


def main():
    rows = []
    for idx, case in enumerate(CASES):
        row = run_case(*case, seed=20260811 + idx)
        rows.append(row)


    out_path = Path(__file__).parent / "test_deepgemm_masked_w4a8.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
