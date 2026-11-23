import torch
import numpy as np

def weight8bit_nt_kpack2_marlin(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                n_tile=16, ):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
        q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((E, size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
        q = q.permute((0, 1, 3, 2, 4)).contiguous()
        q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q


def weight8bit_nt_kpack2_marlin1(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                k_tile1=4,
                                n_tile=16, 
                                n_tile1=16):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        # q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.permute((0, 3, 1, 4, 2, 5)).contiguous()
        # q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((E, size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        q = q.permute((0, 1, 4, 2, 5, 3, 6)).contiguous()
        # q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q


def weight8bit_nt_kpack2_marlin2(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                k_tile1=4,
                                n_tile=16, 
                                ):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((size_n // (n_tile), n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        q = q.permute((2, 0, 3, 1, 4)).contiguous()
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((E, size_n // (n_tile), n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        q = q.permute((0, 3, 1, 4, 2, 5)).contiguous()
    return q


def _unpack_int8_to_uint4_int8(tensor_int8: torch.Tensor) -> torch.Tensor:
    if tensor_int8.dtype != torch.int8:
        raise ValueError("Input tensor must be of type torch.int8")

    tensor_uint8 = tensor_int8.to(torch.uint8)
    high4 = (tensor_uint8 >> 4) & 0x0F
    low4 = tensor_uint8 & 0x0F

    unpacked_shape = (*tensor_int8.shape[:-1], tensor_int8.shape[-1] * 2)
    unpacked = torch.empty(unpacked_shape, dtype=torch.int8, device=tensor_int8.device)
    unpacked[..., 0::2] = high4.to(torch.int8)
    unpacked[..., 1::2] = low4.to(torch.int8)
    return unpacked


def _pack_uint4_qqq_to_int32(q: torch.Tensor,
                             pack_order=(4, 0, 5, 1, 6, 2, 7, 3)) -> torch.Tensor:
    if q.shape[-1] % 8 != 0:
        raise ValueError("q 的最后一维必须能被 8 整除")

    pack_order = torch.tensor(pack_order, dtype=torch.long, device=q.device)
    if pack_order.numel() != 8:
        raise ValueError("pack_order 必须包含 8 个元素")

    q_shape = q.shape
    q = q.reshape(-1, 8)[:, pack_order].to(torch.int32) & 0x0F

    packed = torch.zeros((q.shape[0],), dtype=torch.int32, device=q.device)
    for i in range(8):
        packed |= q[:, i] << (4 * i)

    return packed.reshape((*q_shape[:-1], q_shape[-1] // 8))


def weight4bit_nt_kpack2_marlin2_qqq_from_packed(weight,
                                                 k_tile=16,
                                                 k_tile1=4,
                                                 n_tile=16,
                                                 pack_order=(0, 4,1,  5, 2, 6, 3, 7),
                                                 ):
    full_weight = _unpack_int8_to_uint4_int8(weight)
    q = weight8bit_nt_kpack2_marlin2(
        full_weight,
        k_tile=k_tile,
        k_tile1=k_tile1,
        n_tile=n_tile,
    )
    return _pack_uint4_qqq_to_int32(q, pack_order=pack_order)

