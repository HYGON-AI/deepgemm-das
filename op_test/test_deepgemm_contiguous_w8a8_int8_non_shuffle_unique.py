from pathlib import Path

import pandas as pd
import torch

from deepgemm import m_grouped_i8_gemm_nt_contiguous, marlin_i8_contiguous_weight
from contiguous_w8a8_int8_common import (
    check_w8a8_precision,
    generate_m_grouped_contiguous,
    iter_contiguous_w8a8_int8_cases,
    torch_gemm,
)

SHUFFLE_UNIQUE = 0


def run_contiguous_w8a8_int8_non_shuffle_unique():
    rows = []
    for num_groups, expected_m_per_group, n, k in iter_contiguous_w8a8_int8_cases():
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k
        )
        a_int8, a_scale = a
        b_int8, b_scale = b

        ref_d = torch_gemm(
            a_int8, b_int8, ref_d, a_scale, b_scale, 1.0, 0.0, m_index_h=m_indices
        )

        packed_b = marlin_i8_contiguous_weight(
            b_int8,
            shuffle_unique=SHUFFLE_UNIQUE,
            low_memory=True,
        )
        b = [packed_b, b_scale]

        m_grouped_i8_gemm_nt_contiguous(
            a,
            b,
            d,
            m_indices,
            shuffle_unique=SHUFFLE_UNIQUE,
        )
        torch.cuda.synchronize()

        ok = check_w8a8_precision(ref_d, d, m_indices, rtol=1e-2, atol=5e-2)
        status = "PASS" if ok else "FAILED"
        print(f"[{status}] W8A8 contiguous int8 non_shuffle_unique: M={m}, N={n}, K={k}")
        rows.append({
            "num_groups": num_groups,
            "expected_m_per_group": expected_m_per_group,
            "m": m,
            "valid_m": int((m_indices >= 0).sum().item()),
            "n": n,
            "k": k,
            "shuffle_unique": SHUFFLE_UNIQUE,
            "status": status,
        })
        pd.DataFrame(rows).to_csv(Path(__file__).parent / "test_deepgemm_contiguous_w8a8_int8_non_shuffle_unique.csv", index=False)
        if not ok:
            print("精度检查不合格！！！")
            continue
        print("精度检查合格！！！")
        print(
            f"W8A8 contiguous int8 non_shuffle_unique passed: "
            f"M={m}, N={n}, K={k}, shuffle_unique={SHUFFLE_UNIQUE}"
        )
    if any(row["status"] == "FAILED" for row in rows):
        raise AssertionError("W8A8 contiguous non_shuffle_unique has failed cases")


def test_m_grouped_gemm_contiguous_non_shuffle_unique():
    run_contiguous_w8a8_int8_non_shuffle_unique()


if __name__ == "__main__":
    run_contiguous_w8a8_int8_non_shuffle_unique()
