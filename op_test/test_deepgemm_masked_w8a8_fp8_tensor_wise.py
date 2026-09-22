from pathlib import Path

import torch
import pandas as pd
import random
import math
import torch
import triton
from typing import Type, Optional
from deepgemm import m_grouped_fp8_gemm_nt_masked_tensor_wise, marlin_fp8_masked_weight

num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def native_fp8_perChannel_batch_matmul(input, weight, input_scale, weight_scale, output_dtype):

    A = input.to(torch.float32)
    B = weight.to(torch.float32)

    # print("A shape:", A.shape)
    # print("B shape:", B.shape)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    # print("C shape after bmm:", C.shape)
    # print("input_scale shape:", input_scale.shape)
    # print("weight_scale shape:", weight_scale.shape)
    C = input_scale * C * weight_scale.view(-1, 1, 1)  # Broadcast per-tensor weight scale
    C = C.to(output_dtype)

    return C
           

def deepgemm_masked_impl_fp8_ep_benchmark(input, input_scale, weight, weight_scale, masked_m, expected_m_per_group, out_dtype, enable_overlap, signal, verify_precision=True):
    
    n_ll, k_ll, e_ll = weight.size(1), weight.size(2), weight.size(0)
    if not (
        n_ll in (2048, 4096, 6144)
        and k_ll in (1024, 2048, 6144)
        and e_ll in (1, 64)
    ):
        raise ValueError(
            "m_grouped_fp8_gemm_nt_masked_tensor_wise: (N,K,E) must match MOE_LL_*_SWITCH in "
            "csrc/include/low_latency_fp8_masked_utils.h; "
            f"got N={n_ll}, K={k_ll}, E={e_ll}"
        )
    inter_out_gemm1 = native_fp8_perChannel_batch_matmul(input, weight, input_scale, weight_scale, output_dtype=out_dtype)
    weight_ll = marlin_fp8_masked_weight(weight)
    print("weight shape:", weight.shape)
    print("weight_ll shape:", weight_ll.shape)

    if len(inter_out_gemm1) > 0:
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1)

    torch.cuda.synchronize()

    
    m_grouped_fp8_gemm_nt_masked_tensor_wise((input, input_scale),
                                  (weight_ll, weight_scale),
                                    gemm1_out_cuda,
                                    masked_m,
                                    expected_m_per_group,
                                    enable_overlap,
                                    signal=signal
                                    )
    torch.cuda.synchronize()

    # warm up
    for _ in range(10):
        m_grouped_fp8_gemm_nt_masked_tensor_wise((input, input_scale),
            (weight_ll, weight_scale),
            gemm1_out_cuda,
            masked_m,
            expected_m_per_group,
            enable_overlap,
            signal=signal
            )
    # torch.cuda.synchronize()

    # gemm1
    asm_gemm1_cost = triton.testing.do_bench(
            lambda: m_grouped_fp8_gemm_nt_masked_tensor_wise((input, input_scale),
            (weight_ll, weight_scale),
            gemm1_out_cuda,
            masked_m,
            expected_m_per_group,
            enable_overlap,
            signal=signal
            ), quantiles=None, return_mode="mean")*1000
    
    if(verify_precision):
        correct = True
        for i in range(len(masked_m)):
            valid_rows = masked_m[i]
            if not torch.allclose(inter_out_gemm1[i, 0:valid_rows, :],
                                gemm1_out_cuda[i, 0:valid_rows, :],
                                rtol=1e-2, atol=1e-2):

                correct = False
                print(f"error deepgemm_groupgemm_fp8 gemm  masked[{i}] 精度检查不合格！！！")

                print(f"第{i}组 valid_rows: {valid_rows}")

                ref = inter_out_gemm1[i, 0:valid_rows, :]
                asm = gemm1_out_cuda[i, 0:valid_rows, :]

                # 计算差异
                diff = torch.abs(ref - asm)

                # 构造与 allclose 一致的 mask
                mask = diff > (1e-2 + 1e-2 * torch.abs(asm))

                # 找到不一致坐标
                mismatch_indices = torch.nonzero(mask)

                print(f"不一致元素个数: {mismatch_indices.shape[0]}")

                # 打印前 N 个不一致点
                max_print = 20
                for idx in mismatch_indices[:max_print]:
                    r, c = idx.tolist()
                    print(f"[{r}, {c}] rocblas={ref[r, c].item():.6f}, "
                        f"asm={asm[r, c].item():.6f}, "
                        f"diff={diff[r, c].item():.6f}")

                if mismatch_indices.shape[0] > max_print:
                    print(f"... 还有 {mismatch_indices.shape[0] - max_print} 个未显示")

                # 额外统计信息（强烈建议保留）
                print("max diff:", diff.max().item())
                print("mean diff:", diff.mean().item())

                # 最大误差位置
                max_idx = torch.argmax(diff)
                r = (max_idx // diff.shape[1]).item()
                c = (max_idx % diff.shape[1]).item()
                print(f"最大误差位置: [{r}, {c}] -> "
                    f"rocblas={ref[r, c].item():.6f}, "
                    f"asm={asm[r, c].item():.6f}")
                break
                # # 原始输出（可选，数据大会很长）
                # print("rocblas:", ref)
                # print("asm:", asm)
        if correct:
            print("success deepgemm_groupgemm_fp8 gemm 精度检查合格")
        status = "PASS" if correct else "FAILED"
        print(f"[{status}] deepgemm_masked_w8a8_fp8_tensor_wise")

    return asm_gemm1_cost, status if verify_precision else None


def ceil_div(a, b):
    return (a + b - 1) // b


def _deepgemm_masked_generate(m: int,
                             n: int,
                             k: int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             expected_m_per_group: int = 256,
                             enable_overlap: bool = False,
                             device: str = "cuda",
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = torch.randint(-16, 17, (E, m, k), device=device).to(torch.float8_e4m3fn)

    input_scale = (torch.randn((E,m,1), device=device, dtype=torch.float32))

    weight = torch.randint(-16, 17, (E,n ,k),  device=device).to(torch.float8_e4m3fn)

    weight_scale = (torch.randn((E,1), device=device, dtype=torch.float32))
    
    # input = torch.ones_like(input)
    # input_scale = torch.ones_like(input_scale)
    # weight = torch.ones_like(weight)
    # weight_scale = torch.ones_like(weight_scale)

    # weight = torch.arange(256).unsqueeze(1).expand(-1, 128).reshape(1, 256, 128).to(torch.float8_e4m3fn).to(device)

    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = torch.randint(int(expected_m_per_group*0.7), int(expected_m_per_group*1.3), (1,)).item() #改生成方式能固定maskm的值
        # masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        # masked_m[j] = int(expected_m_per_group)

    max_signal_size = E * ceil_div(m, 64)
    signal: Optional[torch.Tensor] = torch.zeros(max_signal_size, dtype=torch.int32, device='cuda') if enable_overlap else None

    return input,input_scale,weight,weight_scale,masked_m,signal




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 创建解析器对象


    # # # deepseekm parameters
    # m=4096
    # n_list=[4096, 7168]
    # k_list=[7168, 2048]
    # # k_list=[7296, 2176]


    # # # # # # # bailing parameters
    # m=2048
    # n_list=[4096, 8192]
    # k_list=[8192, 2048]
    # # # k_list=[8320, 2176]

    # # # # # # # # glm5 parameters
    m=2048
    n_list=[4096, 6144]
    k_list=[6144, 2048]
    # k_list=[6272, 2176]
    
    E=64
    expected_m_per_group_list=[4, 8, 16, 32,64,128,256,512,1024]
    # expected_m_per_group_list = [16]
    out_dtype=torch.bfloat16
    
    # n_list=[8192]
    # # # n_list=[8448]
    # k_list=[2048]
    # k_list=[2176]
    # block_size=None
    enable_overlap=False

    df_E = []
    df_m = []
    df_n = []
    df_k = []
    df_expected_m = []
    deepgemm_asm_costs = []

    df_tflops = []
    df_bandwidth = []
    df_status = []

    for expected_m_per_group in expected_m_per_group_list:
        for i in range(0,len(k_list),1):
            print("expected_m_per_group:{} n:{} k:{} ".format(expected_m_per_group, n_list[i],k_list[i]))
            input,input_scale,weight,weight_scale,masked_m,signal=_deepgemm_masked_generate(m=m, n=n_list[i],
                    k=k_list[i],E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group, enable_overlap=enable_overlap)
            
            # print("signal:", signal)
            # masked_m = torch.tensor([165, 18, 19, 20, 16, 18, 19, 20, 16, 18, 19, 20, 16, 18, 0, 0, ], device='cuda', dtype=torch.int)
            print("masked_m:", masked_m)
            asm_cost_gemm, status = deepgemm_masked_impl_fp8_ep_benchmark(input, input_scale, weight, weight_scale, masked_m, expected_m_per_group, out_dtype, enable_overlap, signal)
            deepgemm_asm_costs.append(asm_cost_gemm)
            df_E.append(E)
            df_m.append(m)
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
            df_status.append(status)


    df = pd.DataFrame({'E':df_E, 'm':df_m,'n':df_n,'k':df_k,'expected_m':df_expected_m,'asm deepgemm latency(us)': deepgemm_asm_costs,'asm deepgemm TFlops':df_tflops,'asm deepgemm Bandwidth(GB/s)':df_bandwidth, 'status':df_status})
    df.to_csv(Path(__file__).parent / "test_deepgemm_masked_w8a8_fp8_tensor_wise.csv", index=False)
    if any(status == "FAILED" for status in df_status):
        raise AssertionError("FP8 masked tensor-wise has failed cases")
    # 将 DataFrame 写入 Excel 文件
    excel_name="DeepGEMM_masked_groupgemm_fp8.xlsx"
    print(df)
    # df.to_excel(excel_name, index=False)

    # print("表格已保存到 {}文件中。".format(excel_name)) 

if __name__ == "__main__":
    main()
