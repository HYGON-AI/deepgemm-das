from pathlib import Path

import torch
import pandas as pd
import random
import math
import torch
import triton
from typing import Type
from deepgemm import m_grouped_bf16_gemm_nt_masked_hipc
from deepgemm.group_gemm_weight_pack import marlin_bf16_masked_weight
from deepgemm_marlin_quant import weight8bit_nt_kpack2_marlin   # pyright: ignore[reportMissingImports]

num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def native_bf16_perChannel_batch_matmul(q_a1_all, weight, output_dtype):

    A = q_a1_all.to(torch.float32)
    B = weight.to(torch.float32)

    print("A shape:", A.shape)
    print("B shape:", B.shape)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    print("C shape after bmm:", C.shape)
    C = C.to(output_dtype)
    return C
           
def deepgemm_masked_impl_bf16_correctness(input, weight, masked_m, expected_m_per_group, verify_precision=True):
    
    
    outputs = []
    start_idx = 0
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    inter_out_gemm = []

    
    inter_out_gemm = native_bf16_perChannel_batch_matmul(input, weight, output_dtype=input.dtype)


    if len(inter_out_gemm) > 0:
        gemm_out_cuda = torch.zeros_like(inter_out_gemm)


    dtype = input.dtype

    torch.cuda.synchronize()

    weight = marlin_bf16_masked_weight(weight)
  
    m_grouped_bf16_gemm_nt_masked_hipc(input, 
                                  weight,
                                    gemm_out_cuda,
                                    masked_m, 
                                    expected_m_per_group,
                                    )
    
    torch.cuda.synchronize()

    for _ in range(10):
        m_grouped_bf16_gemm_nt_masked_hipc(input, 
            weight,
            gemm_out_cuda,
            masked_m, 
            expected_m_per_group,
            )
    # torch.cuda.synchronize()

    # gemm
    hipc_gemm_cost = triton.testing.do_bench(
            lambda: m_grouped_bf16_gemm_nt_masked_hipc(input, 
            weight,
            gemm_out_cuda,
            masked_m, 
            expected_m_per_group,
            ), quantiles=None, return_mode="mean")*1000
    if(verify_precision):
        correct = torch.allclose(inter_out_gemm[:,0:math.floor(expected_m_per_group * 0.7),:], gemm_out_cuda[:,0:math.floor(expected_m_per_group * 0.7),:], rtol=1e-2, atol=1e-2)
        if not correct:
            print("error moe_groupgemm_w16a16 gemm1 精度检查不合格！！！")
            # torch.set_printoptions(profile="full") 
            print("rocblas:", inter_out_gemm)
            print("hipc:", gemm_out_cuda)
            # diff_indices = torch.where(inter_out_gemm != gemm_out_cuda)[0]
            # diff_indices = diff_indices.view(-1, inter_out_gemm.shape[1])
            # print("不同元素的索引:", diff_indices)
            # torch.set_printoptions(profile="full")  
            diff_mask = inter_out_gemm != gemm_out_cuda
            diff_indices = torch.nonzero(diff_mask, as_tuple=False)
            print("存在差异的坐标与值如下：")
            count = 0
            for idx in diff_indices:
                count += 1
                if count > 33:
                    break
                _, i, j = idx.tolist()
                print(f"坐标 ({_}, {i}, {j}): rocblas={inter_out_gemm[_, i, j]}, hipc={gemm_out_cuda[_, i, j]}")
            # fhipc = open('hipc_output.txt', 'w')   
            # print(f'hipc_output: {gemm_out_cuda}', file=fhipc)
            # fhipc.close
            # ftorch = open('torch_output.txt', 'w')   
            # print(f'torch_output: {inter_out_gemm1_out_ref}', file=ftorch)
            # ftorch.close
        else:
            print("success moe_groupgemm_w16a16 gemm1 精度检查合格")
        status = "PASS" if correct else "FAILED"
        print(f"[{status}] deepgemm_masked_w16a16_bf16")
    
    return hipc_gemm_cost, status if verify_precision else None

def _deepgemm_masked_generate(m: int,
                             n: int,
                             k: int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.bfloat16,
                             expected_m_per_group: int = 256,
                             device: str = "cuda",
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = (torch.randn((E, m, k), device=device) * 5).to(dtype=out_dtype)
    weight = torch.randn((E, n, k), device=device).to(dtype=out_dtype)

    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        # masked_m[j] = int(256)
    return input,weight,masked_m




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    m_list=4096
    n_list=[4096, 7168]
    k_list=[7168, 2048]
    E=8 
    expected_m_per_group_list=[32,64,128,256,512,1024]
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
    deepgemm_hipc_costs = []

    df_tflops = []
    df_bandwidth = []
    df_status = []

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(0,len(k_list),1):
            print("expected_m_per_group:{} n:{} k:{} ".format(expected_m_per_group, n_list[i],k_list[i]))
            input,weight,masked_m=_deepgemm_masked_generate(m=m_list,n=n_list[i],
                    k=k_list[i], E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
            
            # weight = torch.ones_like(weight) 

            # rows = torch.arange(-128, 128).unsqueeze(1)  # shape [256, 1]

            # # 扩展成每行重复 128 次，得到 shape [256, 128]
            # tensor = rows.expand(-1, 128).cuda()
            # weight = tensor.to(torch.int8).unsqueeze(0).expand(E, -1, -1)  # shape [1, 256, 128] then expand to [E, 256, 128]
            # print("weight:", weight)

            # input = torch.ones_like(input).to(torch.bfloat16)
            # weight = torch.ones_like(weight).to(torch.bfloat16)

            # row_values = torch.arange(0, 64).reshape(1, -1)
            # # # 广播到所有列
            # input = row_values.expand(input.size(1), input.size(2)).reshape(input.size(0), input.size(1), input.size(2)).to(input.device).to(torch.bfloat16)
            # weight = row_values.expand(weight.size(1), weight.size(2)).reshape(weight.size(0), weight.size(1), weight.size(2)).to(weight.device).to(torch.bfloat16)

            print("masked_m:", masked_m)
            print("input:",input.shape, input.dtype)
            print("weight:",weight.shape,weight.dtype)
            print("masked_m:", masked_m.shape, masked_m.dtype, masked_m)

            
            
            hipc_cost_gemm, status = deepgemm_masked_impl_bf16_correctness(input, weight, masked_m, expected_m_per_group)
            deepgemm_hipc_costs.append(hipc_cost_gemm)
            df_E.append(E)
            df_m.append(m_list)
            df_n.append(n_list[i])
            df_k.append(k_list[i])
            df_expected_m.append(expected_m_per_group)

            valid_m = masked_m.sum().item()
            computes = 2 * valid_m * n_list[i] * k_list[i]
            tflops_hipc = computes / (hipc_cost_gemm / 1e6) / 1e12

            data_bytes = (valid_m * k_list[i] + k_list[i] * n_list[i] * E + valid_m * n_list[i]) * 2 # bfloat16)   
            bandwidth_hipc = data_bytes / (hipc_cost_gemm / 1e6) / 1e9
            print("bandwidth_hipc:", bandwidth_hipc)

            
            print("------------------------------------------------------")
            print("deepgemm hipc cost: {:.2f} us, {:.2f} TFlops, bandwidth: {:.2f} GB/s".format(hipc_cost_gemm, tflops_hipc, bandwidth_hipc))
            print("------------------------------------------------------") 

            df_tflops.append(tflops_hipc)
            df_bandwidth.append(bandwidth_hipc)
            df_status.append(status)


    df = pd.DataFrame({'E':df_E, 'm':df_m,'n':df_n,'k':df_k,'expected_m':df_expected_m,'hipc deepgemm latency(us)': deepgemm_hipc_costs,'hipc deepgemm TFlops':df_tflops,'hipc deepgemm Bandwidth(GB/s)':df_bandwidth, 'status':df_status})
    df.to_csv(Path(__file__).parent / "test_deepgemm_masked_w16a16_bf16.csv", index=False)
    if any(status == "FAILED" for status in df_status):
        raise AssertionError("BF16 masked HIPC has failed cases")
    # 将 DataFrame 写入 Excel 文件
    # excel_name="DeepGEMM_masked_groupgemm_bf16.xlsx"
        
    # df.to_excel(excel_name, index=False)

    # print("表格已保存到 {}文件中。".format(excel_name)) 

if __name__ == "__main__":
    main()
