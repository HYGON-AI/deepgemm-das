from pathlib import Path

import pandas as pd
import torch

import triton

from deepgemm import m_grouped_i8_gemm_nt_masked, marlin_i8_masked_weight
from masked_w8a8_int8_common import (
    check_w8a8_precision,
    generate_m_grouped_masked_w8a8,
    iter_masked_w8a8_int8_cases,
    quantize_activation,
    torch_gemm,
)

SHUFFLE_UNIQUE = 1


def run_masked_w8a8_int8_shuffle_unique():
    out_dtype = torch.bfloat16
    rows = []
    for num_groups, m, expected_m_per_group, n, k in iter_masked_w8a8_int8_cases():
        hidden_states, weight, w_scale, masked_m, d = generate_m_grouped_masked_w8a8(
            m=m,
            n=n,
            k=k,
            num_groups=num_groups,
            expected_m_per_group=expected_m_per_group,
            out_dtype=out_dtype,
        )
        q_a, a_scale = quantize_activation(hidden_states)
        ref_d = torch_gemm(q_a, weight.to(torch.int32), a_scale, w_scale, output_dtype=out_dtype)

        packed_weight = marlin_i8_masked_weight(
            weight,
            shuffle_unique=SHUFFLE_UNIQUE,
        )
        a = [q_a, a_scale]
        b = [packed_weight, w_scale]
        launch_expected_m_per_group = int(masked_m.max().item())

        # m_grouped_i8_gemm_nt_masked(
        #     a,
        #     b,
        #     d,
        #     masked_m,
        #     launch_expected_m_per_group,
        #     shuffle_unique=SHUFFLE_UNIQUE,
        # )
        # torch.cuda.synchronize()

        cost_us = triton.testing.do_bench(
            lambda: m_grouped_i8_gemm_nt_masked(
                a,
                b,
                d,
                masked_m,
                launch_expected_m_per_group,
                shuffle_unique=SHUFFLE_UNIQUE,
            ),
            warmup=20,
            rep=100,
            quantiles=None,
            return_mode="mean",
        ) * 1000

        ok = check_w8a8_precision(ref_d, d, masked_m)
        status = "PASS" if ok else "FAILED"
        print(f"[{status}] W8A8 masked int8 shuffle_unique: M={m}, N={n}, K={k}")
        valid_m = int((masked_m >= 0).sum().item())
        computes = 2 * valid_m * n * k
        tflops = computes / (cost_us / 1e6) / 1e12
        weight_bytes = num_groups * n * k // 2
        data_bytes = valid_m * k + weight_bytes + valid_m * n * d.element_size()
        bandwidth = data_bytes / (cost_us / 1e6) / 1e9
        rows.append({
            "num_groups": num_groups,
            "m": m,
            "expected_m_per_group": expected_m_per_group,
            "launch_expected_m_per_group": launch_expected_m_per_group,
            "valid_m_total": int(masked_m.sum().item()),
            "n": n,
            "k": k,
            "shuffle_unique": SHUFFLE_UNIQUE,
            "tflops": tflops,
            "bandwidth": bandwidth,
            "cost_us": int(cost_us),
            "status": status,
        })
        pd.DataFrame(rows).to_csv(Path(__file__).parent / "test_deepgemm_masked_w8a8_int8_shuffle_unique.csv", index=False)
        if not ok:
            continue
        print(
            f"W8A8 masked int8 shuffle_unique passed: "
            f"M={m}, N={n}, K={k}, shuffle_unique={SHUFFLE_UNIQUE}"
        )
        print("valid_m:", valid_m)
        print("------------------------------------------------------")
        print(f"cost: {cost_us:.2f} us, {tflops:.2f} TFlops")
        print(f"bandwidth: {bandwidth:.2f} GB/s")
        print("------------------------------------------------------")
    if any(row["status"] == "FAILED" for row in rows):
        raise AssertionError("W8A8 masked shuffle_unique has failed cases")


def test_m_grouped_gemm_masked_w8a8_shuffle_unique():
    run_masked_w8a8_int8_shuffle_unique()


if __name__ == "__main__":
    run_masked_w8a8_int8_shuffle_unique()
