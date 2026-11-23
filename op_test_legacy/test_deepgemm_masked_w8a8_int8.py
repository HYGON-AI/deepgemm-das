# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import random
from typing import Type

import pandas as pd
import torch
import triton
from deepgemm.m_group_gemm import m_grouped_w8a8_gemm_nt_masked_ll, pack_int8_weight_enk_to_w6_low_latency
from lmslim.layers.gemm.int8_utils import per_token_quant_int8

from vllm.platforms import current_platform
from generate_masked_m import generate_masked_m
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


seed = 0
torch.manual_seed(seed)

# PyTorch GPU随机种子
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 多GPU情况

# This test uses only the low-latency masked W8A8 path:
# pack_int8_weight_enk_to_w6_low_latency + m_grouped_w8a8_gemm_nt_masked_ll.

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
           

def deepgemm_masked_impl_w8a8_ep_benchmark(hidden_states, weight13, w13_scale, masked_m, expected_m_per_group, verify_precision=True):
    
    # weight13 = torch.ones_like(weight13)
    w13 = weight13.to(torch.int32)
    inter_out_gemm1 = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)

    # q_a1_all = torch.ones_like(q_a1_all)
    # qa1_scale_all = torch.ones_like(qa1_scale_all)
    # w13_scale = torch.ones_like(w13_scale)



    inter_out_gemm1 = native_w8a8_perChannel_batch_matmul(q_a1_all, w13, qa1_scale_all, w13_scale, output_dtype=hidden_states.dtype)

    if len(inter_out_gemm1) > 0:
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1)

    E = len(masked_m)
    M = hidden_states.size(1)
    N1 = weight13.size(1)
    K1 = hidden_states.size(2)//2
    N2 = K1*2
    K2 = N1 // 4

    dtype = hidden_states.dtype

    n_ll, k_ll, e_ll = weight13.size(1), hidden_states.size(2), len(masked_m)
    if not (
        n_ll in (2048, 3072, 4096, 6144, 7168)
        and k_ll in (1024, 1536, 2048, 3072, 6144, 7168)
    ):
        raise ValueError(
            "low-latency masked W8A8: (N,K) must match MOE_LL_*_SWITCH in "
            "csrc/include/low_latency_fp8_masked_utils.h; "
            f"got N={n_ll}, K={k_ll}, E={e_ll}"
        )
    weight13_w6 = pack_int8_weight_enk_to_w6_low_latency(weight13)
    qa_s = (
        qa1_scale_all.squeeze(-1)
        if qa1_scale_all.dim() == 3
        else qa1_scale_all
    ).contiguous()
    w_s = w13_scale.squeeze(-1).contiguous()
    masked_i32 = masked_m.to(torch.int32)

    def _run_gemm():
        m_grouped_w8a8_gemm_nt_masked_ll(
            (q_a1_all, qa_s),
            (weight13_w6, w_s),
            gemm1_out_cuda,
            masked_i32,
            expected_m_per_group,
            block_wise=False,
            cu=128,
        )

    # warm up
    for _ in range(5):
        _run_gemm()
    torch.cuda.synchronize()

    # gemm1
    asm_gemm1_cost = triton.testing.do_bench(
        lambda: _run_gemm(), quantiles=None, return_mode="mean"
    ) * 1000

    if(verify_precision):
        valid = True
        for e in range(masked_m.shape[0]):
            cur_valid_m = int(masked_m[e].item())
            if cur_valid_m == 0:
                continue
            if not torch.allclose(inter_out_gemm1[e, :cur_valid_m, :], gemm1_out_cuda[e, :cur_valid_m, :],rtol=1e-2, atol=1e-2):
                valid = False
                print("run error index: ", e, "cur_valid_m: ", cur_valid_m)
                break
        if valid:
            print("success deepgemm_masked_w8a8 gemm 精度检查合格")
        else:
            print("error deepgemm_masked_w8a8 gemm 精度检查不合格！！！")
            print("rocblas:", inter_out_gemm1)
            print("asm:",gemm1_out_cuda)
            diff_indices = torch.where(inter_out_gemm1 != gemm1_out_cuda)[0]
            print("不同元素的索引:", diff_indices)
        
        # if not (torch.allclose(inter_out_gemm1[:,0:math.floor(expected_m_per_group * 0.7),:], gemm1_out_cuda[:,0:math.floor(expected_m_per_group * 0.7),:], rtol=1e-2, atol=1e-2)):
        #     print("error deepgemm_masked_w8a8 gemm 精度检查不合格！！！")
        #     print("rocblas:", inter_out_gemm1)
        #     # torch.set_printoptions(profile="full") 
        #     print("asm:",gemm1_out_cuda)
        #     diff_indices = torch.where(inter_out_gemm1 != gemm1_out_cuda)[0]
        #     diff_indices = diff_indices.view(-1, inter_out_gemm1.shape[1])
        #     print("不同元素的索引:", diff_indices)
        # else:
        #     print("success deepgemm_masked_w8a8 gemm 精度检查合格")
    
    return asm_gemm1_cost

def _deepgemm_masked_generate(m: int,
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

    weight13 = to_int8(torch.randn((E,n ,k), device=device) )

    weight13_scale = (torch.randn((E,n,1), device=device, dtype=torch.float32))
    
    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))

    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        #masked_m[j] = int(expected_m_per_group) #masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    
    # masked_m_list = [17, 18, 11, 19, 14, 17, 13, 17, 14, 14, 20, 12, 20, 20, 12, 11]
    # masked_m = torch.tensor(masked_m_list, device='cuda', dtype=torch.int)

    # masked_ms = generate_masked_m(E)
    # masked_m = torch.tensor(masked_ms, device='cuda', dtype=torch.int32).reshape(E)

    return input,weight13,weight13_scale,masked_m




def main():
    print(
        "[masked w8a8] GEMM:",
        f"{m_grouped_w8a8_gemm_nt_masked_ll.__module__}.{m_grouped_w8a8_gemm_nt_masked_ll.__qualname__}",
        "| weight reorder:",
        f"{pack_int8_weight_enk_to_w6_low_latency.__module__}.{pack_int8_weight_enk_to_w6_low_latency.__qualname__}",
    )

    # # # deepseekm parameters
    # m_list=4096
    # n_list=[4096, 7168]
    # k_list=[7168, 2048]
    # # k_list=[7296, 2176]


    # # # # # # # # bailing parameters
    # m_list=2048
    # n_list=[4096, 8192]
    # k_list=[8192, 2048]
    # # # k_list=[8320, 2176]


    m_list=2048
    n_list=[4096, 6144]
    k_list=[6144, 2048]
    
    # m_list=[8192]
    E_list=[3, 4, 6, 7, 8, 12, 24, 28]
    expected_m_per_group_list=[8,16,32,64,128,256,512,1024]
    # expected_m_per_group_list=[16]
    #expected_m_per_group_list = [256]
    out_dtype=torch.bfloat16
    
    # n_list_1=n1_list #[256,12]
    # k_list_1=n2_list #[7168,7168]
    
    # n_list_2=n2_list #[7168,7168]
    # k_list_2=k_list # [128,256]
    # block_size=None

    df_E = []
    df_m = []
    df_n = []
    df_k = []
    df_expected_m = []
    deepgemm_asm_costs = []

    df_tflops = []
    df_bandwidth = []

    for E in E_list:
        for expected_m_per_group in expected_m_per_group_list:
            for i in range(0,len(k_list),1):
                print("E:{} expected_m_per_group:{} n:{} k:{} ".format(E, expected_m_per_group, n_list[i],k_list[i]))
                input,weight13,weight13_scale,masked_m=_deepgemm_masked_generate(m=m_list,n=n_list[i],
                        k=k_list[i],E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
                
                asm_cost_gemm = deepgemm_masked_impl_w8a8_ep_benchmark(input, weight13, weight13_scale, masked_m, expected_m_per_group)

                deepgemm_asm_costs.append(asm_cost_gemm)
                df_E.append(E)
                df_m.append(m_list)
                df_n.append(n_list[i])
                df_k.append(k_list[i])
                df_expected_m.append(expected_m_per_group)

                valid_m = masked_m.sum().item()
                print("valid_m:", valid_m, "masked_m:", masked_m)
                computes = 2 * valid_m * n_list[i] * k_list[i]
                tflops_asm = computes / (asm_cost_gemm / 1e6) / 1e12

                data_bytes = (valid_m * k_list[i] + k_list[i] * n_list[i] * E + valid_m * n_list[i] * 2)# bfloat16)   
                bandwidth_asm = data_bytes / (asm_cost_gemm / 1e6) / 1e9

                
                print("------------------------------------------------------")
                print("deepgemm asm cost: {:.2f} us, {:.2f} TFlops".format(asm_cost_gemm, tflops_asm))
                print("deepgemm asm bandwidth: {:.2f} GB/s".format(bandwidth_asm))
                print("------------------------------------------------------") 


                df_tflops.append(tflops_asm)
                df_bandwidth.append(bandwidth_asm)


    df = pd.DataFrame({'E':df_E, 'm':df_m,'n':df_n,'k':df_k,'expected_m':df_expected_m,'asm deepgemm 耗时(us)': deepgemm_asm_costs,'asm deepgemm TFlops':df_tflops,'asm deepgemm Bandwidth(GB/s)':df_bandwidth})
        # 将 DataFrame 写入 Excel 文件
    excel_name="DeepGEMM_masked_groupgemm_w8a8.xlsx"
        
    # df.to_excel(excel_name, index=False)
    print(df)

if __name__ == "__main__":
    main()
