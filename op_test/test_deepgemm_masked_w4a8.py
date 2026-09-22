from pathlib import Path

import torch
import pandas as pd
import random
import triton
from typing import Type
from quant_utils import per_token_quant_int8
from deepgemm import m_grouped_w4a8_gemm_nt_masked
from deepgemm_marlin_quant import weight4bit_nt_kpack2_marlin2_qqq_from_packed   # pyright: ignore[reportMissingImports]

num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def unpack_int8_to_int4(tensor_int8: torch.Tensor) -> torch.Tensor:
    if tensor_int8.dtype != torch.int8:
        raise ValueError("Input tensor must be of type torch.int8")

    E,N, K_half = tensor_int8.shape
    tensor_int8 = tensor_int8.to(torch.int8)
    high4 = tensor_int8 & 0XF0
    low4 = (tensor_int8 << 4) & 0xF0
    unpacked = torch.empty((E, N, K_half * 2), dtype=torch.int8, device=tensor_int8.device)
    unpacked[:, :, 0::2] = high4.to(torch.int8)
    unpacked[:, :, 1::2] = low4.to(torch.int8)
    return unpacked



def native_w8a8_perChannel_batch_matmul(q_a1_all, weight13, qa1_scale_all, w13_scale, output_dtype):

    A = q_a1_all.to(torch.float32)
    B = weight13.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    C = qa1_scale_all * C * w13_scale.transpose(1,2)  # Broadcast per-column scale
    C = C.to(output_dtype)

    return C
           
def deepgemm_masked_impl_w4a8_ep_benchmark(
    hidden_states,
    weight13,
    w13_scale,
    masked_m,
    expected_m_per_group,
    verify_precision=True,
    return_output=False,
):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # w13_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
    
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(weight13.shape[0]):
        w1_marlin_in = weight4bit_nt_kpack2_marlin2_qqq_from_packed(weight13[i])
        w1_marlin_list.append(w1_marlin_in)
    weight13_marlin = torch.stack(w1_marlin_list, dim=0)
    


    w13 = unpack_int8_to_int4(weight13)
    inter_out_gemm1 = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)
    # print("q_a1_all:", q_a1_all.shape, qa1_scale_all.shape)
    # torch.set_printoptions(profile="full")
    # q_a1_all = torch.ones_like(q_a1_all)
    # q_a1_all[:, 128:192] = 0
    # print("q_a1_all:",q_a1_all.shape,q_a1_all)
    # qa1_scale_all = torch.ones_like(qa1_scale_all)

    inter_out_gemm1 = native_w8a8_perChannel_batch_matmul(q_a1_all, w13, qa1_scale_all, w13_scale, output_dtype=hidden_states.dtype)

    if len(inter_out_gemm1) > 0:
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1)

    masked_m = masked_m.to(dtype=torch.int32, device=hidden_states.device)
    torch.cuda.synchronize()

    # warm up
    for _ in range(5):
        m_grouped_w4a8_gemm_nt_masked((q_a1_all, qa1_scale_all),
                                      (weight13_marlin, w13_scale),
                                      gemm1_out_cuda,
                                      masked_m,
                                      expected_m_per_group,
                                      )
    torch.cuda.synchronize()

    # gemm1
    asm_gemm1_cost = triton.testing.do_bench(
        lambda: m_grouped_w4a8_gemm_nt_masked((q_a1_all, qa1_scale_all),
                                              (weight13_marlin, w13_scale),
                                              gemm1_out_cuda,
                                              masked_m,
                                              expected_m_per_group,
                                              ),
        quantiles=None,
        return_mode="mean",
    ) * 1000

    if verify_precision:
        valid = True
        for e in range(masked_m.shape[0]):
            cur_valid_m = int(masked_m[e].item())
            if cur_valid_m == 0:
                continue
            if not torch.allclose(inter_out_gemm1[e, :cur_valid_m, :],
                                  gemm1_out_cuda[e, :cur_valid_m, :],
                                  rtol=1e-2,
                                  atol=1e-2):
                valid = False
                print("run error index: ", e, "cur_valid_m: ", cur_valid_m)
                break
        if valid:
            print("success deepgemm_masked_w4a8 gemm 精度检查合格")
        else:
            print("error deepgemm_masked_w4a8 gemm 精度检查不合格！！！")
            print("rocblas:", inter_out_gemm1)
            print("asm:", gemm1_out_cuda)
            diff_indices = torch.where(inter_out_gemm1 != gemm1_out_cuda)[0]
            print("不同元素的索引:", diff_indices)
        status = "PASS" if valid else "FAILED"
        print(f"[{status}] deepgemm_masked_w4a8")

    if return_output:
        return asm_gemm1_cost, gemm1_out_cuda

    return asm_gemm1_cost, status if verify_precision else None


def deepgemm_masked_impl_w4a8_ep_correctness(hidden_states, weight13, w13_scale, masked_m, expected_m_per_group):
    _, gemm1_out_cuda = deepgemm_masked_impl_w4a8_ep_benchmark(
        hidden_states,
        weight13,
        w13_scale,
        masked_m,
        expected_m_per_group,
        verify_precision=True,
        return_output=True,
    )
    return gemm1_out_cuda

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             expected_m_per_group: int = 256,
                             device: str = "cuda",
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = (torch.randn((E, m, k), device=device) * 5).to(dtype=out_dtype)

    weight13 = to_int8(torch.randn((E, n ,k // 2), device=device) )

    weight13_scale = (torch.randn((E,n,1), device=device, dtype=torch.float32))
    
    # weight13 = torch.ones_like(weight13) * 17
    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group ) # * random.uniform(0.7, 1.3)
        # masked_m[j] = int(256)
    return input,weight13,weight13_scale,masked_m




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    m_list = 2048
    n_list = [4096,6144]
    k_list = [6144,2048]
    E = 16
    expected_m_per_group_list = [1, 2, 4, 8, 16, 32, 64 ,128, 256, 512, 1024, 2048]
    out_dtype=torch.bfloat16
    
    df_E = []
    df_m = []
    df_n = []
    df_k = []
    df_expected_m = []
    df_valid_m = []
    df_per_expert_m_avg = []
    deepgemm_asm_costs = []

    df_tflops = []
    df_bandwidth = []
    df_status = []

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(0, len(k_list), 1):
            print("expected_m_per_group:{} n:{} k:{} ".format(expected_m_per_group, n_list[i], k_list[i]))
            input,weight13,weight13_scale,masked_m=_fused_moe_helper(m=m_list,n=n_list[i],
                    k=k_list[i], E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
            # masked_m = torch.tensor([265, 18, 19, 20, 16, 18, 19, 20, 16, 18, 19, 20, 16, 18, 0, 0, ], device='cuda', dtype=torch.int)
            asm_cost_gemm, status = deepgemm_masked_impl_w4a8_ep_benchmark(input, weight13, weight13_scale, masked_m, expected_m_per_group)

            deepgemm_asm_costs.append(asm_cost_gemm)
            df_E.append(E)
            df_m.append(m_list)
            df_n.append(n_list[i])
            df_k.append(k_list[i])
            df_expected_m.append(expected_m_per_group)

            valid_m = masked_m.sum().item()
            df_valid_m.append(valid_m)
            df_per_expert_m_avg.append(valid_m / E)
            print("valid_m:", valid_m, "masked_m:", masked_m)
            computes = 2 * valid_m * n_list[i] * k_list[i]
            tflops_asm = computes / (asm_cost_gemm / 1e6) / 1e12

            data_bytes = (valid_m * k_list[i] + k_list[i] * n_list[i] * E // 2 + valid_m * n_list[i] * 2)
            bandwidth_asm = data_bytes / (asm_cost_gemm / 1e6) / 1e9

            print("------------------------------------------------------")
            print("deepgemm asm cost: {:.2f} us, {:.2f} TFlops".format(asm_cost_gemm, tflops_asm))
            print("deepgemm asm bandwidth: {:.2f} GB/s".format(bandwidth_asm))
            print("------------------------------------------------------") 

            df_tflops.append(tflops_asm)
            df_bandwidth.append(bandwidth_asm)
            df_status.append(status)

    df = pd.DataFrame({
        'E': df_E,
        'capacity_m': df_m,
        'n': df_n,
        'k': df_k,
        'expected_m': df_expected_m,
        'valid_m': df_valid_m,
        'per_expert_m_avg': df_per_expert_m_avg,
        'asm deepgemm latency(us)': deepgemm_asm_costs,
        'asm deepgemm TFlops': df_tflops,
        'asm deepgemm Bandwidth(GB/s)': df_bandwidth,
        'status': df_status,
    })
    df.to_csv(Path(__file__).parent / "test_deepgemm_masked_w4a8.csv", index=False)
    if any(status == "FAILED" for status in df_status):
        raise AssertionError("W4A8 masked has failed cases")
    excel_name="DeepGEMM_masked_groupgemm_w4a8.xlsx"

    # df.to_excel(excel_name, index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df)

if __name__ == "__main__":
    main()
