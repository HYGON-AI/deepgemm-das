#pragma once

#include "moe_marlin/w8a16_linear_device.h"

namespace deepgemm {

static constexpr int LDS_M_PAD = 1;

static inline int smem_bytes(int num_k_warps, int n_per_warp, int num_n_warps, int M_block) {
    int M_padded = M_block + LDS_M_PAD;
    int elem = (num_k_warps > 1)
               ? num_k_warps / 2 * n_per_warp * M_padded
               : n_per_warp * M_padded;
    return elem * num_n_warps * sizeof(float);
}

template <typename Element>
static inline int bm32_smem_bytes(int num_k_warps, int n_per_warp, int num_n_warps,
                                  int M_block, int k_per_warp) {
    const int a_lds_bytes = 32 * k_per_warp * num_k_warps * (int)sizeof(Element);
    const int out_bytes   = smem_bytes(num_k_warps, n_per_warp, num_n_warps, M_block);
    return a_lds_bytes > out_bytes ? a_lds_bytes : out_bytes;
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Main kernel — BlockM=16 (proven, unchanged from original)
// ---------------------------------------------------------------------------
template <typename Element, typename Scale, bool MARLIN_LAYOUT, int NUM_K_WARPS, int NPerWarp, int KPerWarp, int NUM_N_WARPS>
__global__ void __launch_bounds__(NUM_K_WARPS * NUM_N_WARPS * 64, 2)
gemm_w8a16_int8_kernel(
    Element*       __restrict__ d,
    const Element* __restrict__ a,
    const int8_t* __restrict__ b,
    const Scale* __restrict__ scale,
    int M, int N, int K)
{
    constexpr int NSizePerWarp     = 16;
    constexpr int tc_number        = KPerWarp / 16;
    constexpr int NLoop            = NPerWarp / NSizePerWarp;
    constexpr int k_fp16_per_thread = tc_number * 4;
    constexpr int k_i32_per_thread  = k_fp16_per_thread / 4;
    constexpr int kNWarps          = NUM_K_WARPS;
    constexpr int BlockM           = 16;

    const int bidm = blockIdx.x;
    a += (size_t)bidm * BlockM * K;
    d += (size_t)bidm * BlockM * N;
    M = MIN(BlockM, M - bidm * BlockM);

    const int warp_n_idx = threadIdx.y;
    const int bwid       = blockIdx.y * NUM_N_WARPS + warp_n_idx;
    const int tid        = threadIdx.x;
    const int warp_idx   = __builtin_hcu_readfirstlane(tid / 64);
    const int lane       = tid % 64;
    const int row_id     = lane % NSizePerWarp;
    const int col_id     = lane / NSizePerWarp;
    const bool row_valid = row_id < M;

    const int N_len = N - NPerWarp * bwid;
    if (N_len <= 0) return;

    const int block_n = NPerWarp * bwid;
    d += block_n;

    const int lds_M = M + LDS_M_PAD;
    const int num_elem_per_nwarp =
        (NUM_K_WARPS > 1) ? (NUM_K_WARPS / 2 * NPerWarp * lds_M) : (NPerWarp * lds_M);

    using A_VEC = typename w16_vec<Element, k_fp16_per_thread>::type;
    using B_VEC = __attribute__((__vector_size__(k_i32_per_thread  * sizeof(int32_t)))) int32_t;

    A_VEC    a_vec;
    float4_t d_vec[NLoop];
    #pragma unroll
    for (int i = 0; i < NLoop; i++) d_vec[i] = {0.f, 0.f, 0.f, 0.f};

    for (int i = warp_idx * KPerWarp; i < K; i += KPerWarp * kNWarps) {
        const int k_idx = i + col_id * k_fp16_per_thread;

        if (row_valid)
            a_vec = *(A_VEC*)(a + row_id * K + k_idx);

        #pragma unroll
        for (int nn = 0; nn < NLoop; nn++) {
            const int n_idx = NSizePerWarp * nn + row_id;
            auto tb = load_w8a16_weight_vec<B_VEC, MARLIN_LAYOUT>(b, N, K, block_n + n_idx, k_idx);

            #pragma unroll
            for (int ii = 0; ii < tc_number / 2; ii++) {
                w16_vec4_t<Element> bv0;
                w16_vec4_t<Element> bv1;
                dequant_w8a16_to_reg<Element>(static_cast<uint32_t>(tb[ii * 2]),
                                              static_cast<uint32_t>(tb[ii * 2 + 1]),
                                              bv0,
                                              bv1);
                builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec)[ii * 2    ], bv0, d_vec[nn]);
                builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec)[ii * 2 + 1], bv1, d_vec[nn]);
            }
        }
    }

    // K-warp tree reduction via LDS
    extern __shared__ float start_smem[];
    float4_t* out_smem_f32 = reinterpret_cast<float4_t*>(
        start_smem + num_elem_per_nwarp * warp_n_idx);
    Element*   out_smem_w16 = reinterpret_cast<Element*>(start_smem)
                             + num_elem_per_nwarp * warp_n_idx;

    if constexpr (NUM_K_WARPS > 1) {
        const int lds_n_stride = NUM_K_WARPS / 2 * NSizePerWarp * lds_M / 4;
        #pragma unroll
        for (int stride = NUM_K_WARPS; stride > 1; stride /= 2) {
            const int mid = stride / 2;
            if (row_valid) {
                if (warp_idx >= mid && warp_idx < stride) {
                    #pragma unroll
                    for (int nn = 0; nn < NLoop; nn++)
                        out_smem_f32[(warp_idx - mid) * 4 * lds_M
                                     + row_id * 4 + col_id
                                     + lds_n_stride * nn] = d_vec[nn];
                }
            }
            __syncthreads();
            if (warp_idx < mid) {
                #pragma unroll
                for (int nn = 0; nn < NLoop; nn++) {
                    if (row_valid) {
                        float4_t tmp = out_smem_f32[warp_idx * 4 * lds_M
                                                    + row_id * 4 + col_id
                                                    + lds_n_stride * nn];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) d_vec[nn][j] += tmp[j];
                    }
                }
            }
            __syncthreads();
        }
    }

    if constexpr (!MARLIN_LAYOUT && NPerWarp == 32) {
        if (N_len >= NPerWarp) {
            // Target shapes are full N tiles; keep the LDS transpose for
            // coalesced half4 stores and skip tail predicates in the epilogue.
            if (warp_idx == 0 && row_valid) {
                #pragma unroll
                for (int nn = 0; nn < NLoop; nn++) {
                    const int base = nn * NSizePerWarp * lds_M;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const int n_col = block_n + nn * NSizePerWarp + i * 4 + col_id;
                        out_smem_w16[base + row_id * NSizePerWarp + i * 4 + col_id] =
                            float_to_w16<Element>(d_vec[nn][i] * static_cast<float>(scale[n_col]));
                    }
                }
            }
            __syncthreads();

            if (row_valid) {
                for (int nn = warp_idx; nn < NLoop; nn += NUM_K_WARPS) {
                    const int smem_off = nn * NSizePerWarp * lds_M
                                         + row_id * NSizePerWarp + col_id * 4;
                    const int out_off  = row_id * N
                                         + nn * NSizePerWarp + col_id * 4;
                    *reinterpret_cast<w16_vec4_t<Element>*>(d + out_off) =
                        *reinterpret_cast<w16_vec4_t<Element>*>(out_smem_w16 + smem_off);
                }
            }
            return;
        }
    }

    // warp 0 converts fp32 accumulators → fp16 into LDS
    if (warp_idx == 0 && row_valid) {
        #pragma unroll
        for (int nn = 0; nn < NLoop; nn++) {
            const int base = nn * NSizePerWarp * lds_M;
            float sc_vec[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int n_col = block_n + nn * NSizePerWarp + i * 4 + col_id;
                sc_vec[i] = (n_col < N) ? static_cast<float>(scale[n_col]) : 1.0f;
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                out_smem_w16[base + row_id * NSizePerWarp + i * 4 + col_id] =
                    float_to_w16<Element>(d_vec[nn][i] * sc_vec[i]);
        }
    }
    __syncthreads();

    // All warps cooperate writing LDS → global (coalesced half4_t stores)
    if (row_valid) {
        for (int nn = warp_idx; nn < NLoop; nn += NUM_K_WARPS) {
            if (nn * NSizePerWarp + col_id * 4 < N_len) {
                const int smem_off = nn * NSizePerWarp * lds_M
                                     + row_id * NSizePerWarp + col_id * 4;
                const int out_off  = row_id * N
                                     + nn * NSizePerWarp + col_id * 4;
                *reinterpret_cast<w16_vec4_t<Element>*>(d + out_off) =
                    *reinterpret_cast<w16_vec4_t<Element>*>(out_smem_w16 + smem_off);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main kernel — BlockM=32 (prefill-optimized, uses MLoop=2)
// ---------------------------------------------------------------------------
template <typename Element, typename Scale, bool MARLIN_LAYOUT, int NUM_K_WARPS, int NPerWarp, int KPerWarp, int NUM_N_WARPS>
__global__ void __launch_bounds__(NUM_K_WARPS * NUM_N_WARPS * 64, 2)
gemm_w8a16_int8_bm32_kernel(
    Element*       __restrict__ d,
    const Element* __restrict__ a,
    const int8_t* __restrict__ b,
    const Scale* __restrict__ scale,
    int M, int N, int K)
{
    constexpr int NSizePerWarp      = 16;
    constexpr int tc_number         = KPerWarp / 16;
    constexpr int NLoop             = NPerWarp / NSizePerWarp;
    constexpr int k_fp16_per_thread = tc_number * 4;
    constexpr int k_i32_per_thread  = k_fp16_per_thread / 4;
    constexpr int kNWarps           = NUM_K_WARPS;
    constexpr int BLOCK_M           = 32;
    constexpr int MLoop             = BLOCK_M / 16;  // 2
    constexpr int a_lds_half_size   = 16 * NUM_K_WARPS * KPerWarp;

    const int bidm = blockIdx.x;
    const int total_M = M;
    a += (size_t)bidm * BLOCK_M * K;
    d += (size_t)bidm * BLOCK_M * N;
    M = MIN(BLOCK_M, M - bidm * BLOCK_M);

    const int warp_n_idx = threadIdx.y;
    const int bwid       = blockIdx.y * NUM_N_WARPS + warp_n_idx;
    const int tid        = threadIdx.x;
    const int warp_idx   = __builtin_hcu_readfirstlane(tid / 64);
    const int lane       = tid % 64;
    const int row_id     = lane % NSizePerWarp;
    const int col_id     = lane / NSizePerWarp;

    const int N_len = N - NPerWarp * bwid;
    const bool n_active = (N_len > 0);
    const int block_n = NPerWarp * bwid;
    const bool full_n_m17_32_target = (!MARLIN_LAYOUT && total_M <= BLOCK_M && M > 16
                                       && N_len >= NPerWarp);
    const int m_tail = M - 16;

    if (n_active) {
        d += block_n;
    }

    const int lds_M = M + LDS_M_PAD;
    const int num_elem_per_nwarp =
        (NUM_K_WARPS > 1) ? (NUM_K_WARPS / 2 * NPerWarp * lds_M) : (NPerWarp * lds_M);

    using A_VEC = typename w16_vec<Element, k_fp16_per_thread>::type;
    using B_VEC = __attribute__((__vector_size__(k_i32_per_thread  * sizeof(int32_t)))) int32_t;

    A_VEC    a_vec_m[MLoop];
    float4_t d_vec[MLoop][NLoop];
    #pragma unroll
    for (int mm = 0; mm < MLoop; mm++) {
        a_vec_m[mm] = {};
        #pragma unroll
        for (int i = 0; i < NLoop; i++) d_vec[mm][i] = {0.f, 0.f, 0.f, 0.f};
    }
    extern __shared__ __align__(16) char bm32_smem[];
    constexpr int num_load = 16 * KPerWarp / 64 / 8;  // 一个wave通过dowrdx4需要读取的次数
    auto a_addr = prepare_for_buffer_load<Element>(a);
    if (n_active) {
        for (int i = warp_idx * KPerWarp; i < K; i += KPerWarp * kNWarps) {
            int k_cur = i + col_id * k_fp16_per_thread;
            // __shared__ Element a_lds[16 * KPerWarp * kNWarps];
            Element* a_lds = reinterpret_cast<Element*>(bm32_smem);
            if (full_n_m17_32_target) {
                int row_num = 16 / num_load;
                int num_thread_per_row = 64 / row_num;
                const int rid = lane / num_thread_per_row;
                const int cid = lane % num_thread_per_row;
                if (warp_n_idx == 0) {
                    #pragma unroll
                    for(int j = 0; j < num_load; j++) {  // buffer_load A first16 row
                        int lds_offset = warp_idx * 16 * KPerWarp + j * row_num * KPerWarp;
                        int v_offset = (j * row_num + rid) * K + i + cid * 8;
                        inline_buffer_load_dwordx4_lds<Element, 1>(a_lds, a_addr, lds_offset, 0, v_offset);
                    }
                    #pragma unroll
                    for(int j = 0; j < num_load; j++) {  // buffer_load A last16 row
                        int lds_offset = warp_idx * 16 * KPerWarp + j * row_num * KPerWarp;
                        int v_offset = (16 + j * row_num + rid) * K + i + cid * 8;
                        if(16 + j * row_num + rid < M)
                            inline_buffer_load_dwordx4_lds<Element, 1>(a_lds + a_lds_half_size, a_addr, lds_offset, 0, v_offset);
                    }
                }
                wait_buffer_data_arrived<true>(0);
                #pragma unroll
                for(int j = 0; j < num_load; j++) {  // 每个wave都从lds读A_first16
                    int lds_offset = warp_idx * 16 * KPerWarp + row_id * KPerWarp + col_id * k_fp16_per_thread + j * 8;
                    inlineasm_ds_read_b128(reinterpret_cast<size_t>(a_lds + lds_offset), reinterpret_cast<w16_vec8_t<Element>*>(&a_vec_m[0])[j]);
                }
                if(row_id < m_tail) {
                    #pragma unroll
                    for(int j = 0; j < num_load; j++) {  // 每个wave都从lds读A_last16
                        int lds_offset = a_lds_half_size + warp_idx * 16 * KPerWarp + row_id * KPerWarp + col_id * k_fp16_per_thread + j * 8;
                        inlineasm_ds_read_b128(reinterpret_cast<size_t>(a_lds + lds_offset), reinterpret_cast<w16_vec8_t<Element>*>(&a_vec_m[1])[j]);
                    }
                }
                wait_lds_data_arrived<false>(0);
            } else {
                #pragma unroll
                for (int mm = 0; mm < MLoop; mm++) {
                    const int m_base = mm * 16;
                    if (m_base + row_id < M) {
                        a_vec_m[mm] = *(A_VEC*)(a + (m_base + row_id) * K + k_cur);
                    } else {
                        a_vec_m[mm] = {};
                    }
                }
            }
            #pragma unroll
            for (int nn = 0; nn < NLoop; nn++) {
                const int n_idx = NSizePerWarp * nn + row_id;
                auto tb = load_w8a16_weight_vec<B_VEC, MARLIN_LAYOUT>(b, N, K, block_n + n_idx, k_cur);

                #pragma unroll
                for (int ii = 0; ii < tc_number / 2; ii++) {
                    w16_vec4_t<Element> bv0;
                    w16_vec4_t<Element> bv1;
                    dequant_w8a16_to_reg<Element>(static_cast<uint32_t>(tb[ii * 2]),
                                                  static_cast<uint32_t>(tb[ii * 2 + 1]),
                                                  bv0, bv1);
                    #pragma unroll
                    for (int mm = 0; mm < MLoop; mm++) {
                        builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec_m[mm])[ii * 2    ], bv0, d_vec[mm][nn]);
                        builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec_m[mm])[ii * 2 + 1], bv1, d_vec[mm][nn]);
                    }
                }
            }
        }
    }

    // K-warp tree reduction via LDS
    // extern __shared__ float start_smem[];
    float* start_smem = reinterpret_cast<float*>(bm32_smem); 
    float4_t* out_smem_f32 = reinterpret_cast<float4_t*>(
        start_smem + num_elem_per_nwarp * warp_n_idx);
    Element* out_smem_w16 = reinterpret_cast<Element*>(start_smem)
                             + num_elem_per_nwarp * warp_n_idx;

    if constexpr (NUM_K_WARPS > 1) {
        const int lds_n_stride = NUM_K_WARPS / 2 * NSizePerWarp * lds_M / 4;
        if (full_n_m17_32_target) {
            #pragma unroll
            for (int stride = NUM_K_WARPS; stride > 1; stride /= 2) {
                const int mid = stride / 2;
                if (warp_idx >= mid && warp_idx < stride) {
                    #pragma unroll
                    for (int nn = 0; nn < NLoop; nn++)
                        out_smem_f32[(warp_idx - mid) * 4 * lds_M
                                     + row_id * 4 + col_id
                                     + lds_n_stride * nn] = d_vec[0][nn];
                    if (row_id < m_tail) {
                        #pragma unroll
                        for (int nn = 0; nn < NLoop; nn++)
                            out_smem_f32[(warp_idx - mid) * 4 * lds_M
                                         + (16 + row_id) * 4 + col_id
                                         + lds_n_stride * nn] = d_vec[1][nn];
                    }
                }
                __syncthreads();
                if (warp_idx < mid) {
                    #pragma unroll
                    for (int nn = 0; nn < NLoop; nn++) {
                        float4_t tmp = out_smem_f32[warp_idx * 4 * lds_M
                                                    + row_id * 4 + col_id
                                                    + lds_n_stride * nn];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) d_vec[0][nn][j] += tmp[j];
                    }
                    if (row_id < m_tail) {
                        #pragma unroll
                        for (int nn = 0; nn < NLoop; nn++) {
                            float4_t tmp = out_smem_f32[warp_idx * 4 * lds_M
                                                        + (16 + row_id) * 4 + col_id
                                                        + lds_n_stride * nn];
                            #pragma unroll
                            for (int j = 0; j < 4; j++) d_vec[1][nn][j] += tmp[j];
                        }
                    }
                }
                __syncthreads();
            }
        } else {
            #pragma unroll
            for (int stride = NUM_K_WARPS; stride > 1; stride /= 2) {
                const int mid = stride / 2;
                if (warp_idx >= mid && warp_idx < stride) {
                    #pragma unroll
                    for (int mm = 0; mm < MLoop; mm++) {
                        const int m_base = mm * 16;
                        if (m_base + row_id < M) {
                            #pragma unroll
                            for (int nn = 0; nn < NLoop; nn++)
                                out_smem_f32[(warp_idx - mid) * 4 * lds_M
                                             + (m_base + row_id) * 4 + col_id
                                             + lds_n_stride * nn] = d_vec[mm][nn];
                        }
                    }
                }
                __syncthreads();
                if (warp_idx < mid) {
                    #pragma unroll
                    for (int nn = 0; nn < NLoop; nn++) {
                        #pragma unroll
                        for (int mm = 0; mm < MLoop; mm++) {
                            const int m_base = mm * 16;
                            if (m_base + row_id < M) {
                                float4_t tmp = out_smem_f32[warp_idx * 4 * lds_M
                                                            + (m_base + row_id) * 4 + col_id
                                                            + lds_n_stride * nn];
                                #pragma unroll
                                for (int j = 0; j < 4; j++) d_vec[mm][nn][j] += tmp[j];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // Apply per-channel scale after K-loop reduction (matches triton approach)
    if (n_active) {
        if (full_n_m17_32_target) {
            #pragma unroll
            for (int nn = 0; nn < NLoop; nn++) {
                const int n_base = block_n + nn * NSizePerWarp;
                float sc_vec[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    sc_vec[j] = static_cast<float>(scale[n_base + j * 4 + col_id]);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    d_vec[0][nn][j] *= sc_vec[j];
                }
                if (row_id < m_tail) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        d_vec[1][nn][j] *= sc_vec[j];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int nn = 0; nn < NLoop; nn++) {
                const int n_base = block_n + nn * NSizePerWarp;
                float sc_vec[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n_col = n_base + j * 4 + col_id;
                    sc_vec[j] = (n_col < N) ? static_cast<float>(scale[n_col]) : 1.0f;
                }
                #pragma unroll
                for (int mm = 0; mm < MLoop; mm++) {
                    const int m_base = mm * 16;
                    if (m_base + row_id < M) {
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            d_vec[mm][nn][j] *= sc_vec[j];
                        }
                    }
                }
            }
        }
    }

    if (full_n_m17_32_target) {
        if (warp_idx == 0) {
            #pragma unroll
            for (int nn = 0; nn < NLoop; nn++) {
                const int base = nn * NSizePerWarp * lds_M;
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    out_smem_w16[base + row_id * NSizePerWarp + i * 4 + col_id] =
                        float_to_w16<Element>(d_vec[0][nn][i]);
                if (row_id < m_tail) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        out_smem_w16[base + (16 + row_id) * NSizePerWarp + i * 4 + col_id] =
                            float_to_w16<Element>(d_vec[1][nn][i]);
                }
            }
        }
        __syncthreads();

        for (int nn = warp_idx; nn < NLoop; nn += NUM_K_WARPS) {
            const int smem_off0 = nn * NSizePerWarp * lds_M
                                  + row_id * NSizePerWarp + col_id * 4;
            const int out_off0  = row_id * N
                                  + nn * NSizePerWarp + col_id * 4;
            *reinterpret_cast<w16_vec4_t<Element>*>(d + out_off0) =
                *reinterpret_cast<w16_vec4_t<Element>*>(out_smem_w16 + smem_off0);
            if (row_id < m_tail) {
                const int smem_off1 = nn * NSizePerWarp * lds_M
                                      + (16 + row_id) * NSizePerWarp + col_id * 4;
                const int out_off1  = (16 + row_id) * N
                                      + nn * NSizePerWarp + col_id * 4;
                *reinterpret_cast<w16_vec4_t<Element>*>(d + out_off1) =
                    *reinterpret_cast<w16_vec4_t<Element>*>(out_smem_w16 + smem_off1);
            }
        }
        return;
    }

    // warp 0 converts fp32 accumulators → fp16 into LDS
    if (warp_idx == 0) {
        #pragma unroll
        for (int nn = 0; nn < NLoop; nn++) {
            const int base = nn * NSizePerWarp * lds_M;
            #pragma unroll
            for (int mm = 0; mm < MLoop; mm++) {
                const int m_base = mm * 16;
                if (m_base + row_id < M) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        out_smem_w16[base + (m_base + row_id) * NSizePerWarp + i * 4 + col_id] =
                            float_to_w16<Element>(d_vec[mm][nn][i]);
                }
            }
        }
    }
    __syncthreads();

    // All warps cooperate writing LDS → global (coalesced half4_t stores)
    for (int nn = warp_idx; nn < NLoop; nn += NUM_K_WARPS) {
        #pragma unroll
        for (int mm = 0; mm < MLoop; mm++) {
            const int m_base = mm * 16;
            if (m_base + row_id < M) {
                if (nn * NSizePerWarp + col_id * 4 < N_len) {
                    const int smem_off = nn * NSizePerWarp * lds_M
                                         + (m_base + row_id) * NSizePerWarp + col_id * 4;
                    const int out_off  = (m_base + row_id) * N
                                         + nn * NSizePerWarp + col_id * 4;
                    *reinterpret_cast<w16_vec4_t<Element>*>(d + out_off) =
                        *reinterpret_cast<w16_vec4_t<Element>*>(out_smem_w16 + smem_off);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main kernel — BlockM=64 (prefill-optimized, MLoop=4)
// ---------------------------------------------------------------------------
template <typename Element, typename Scale, bool MARLIN_LAYOUT, int NUM_K_WARPS, int NPerWarp, int KPerWarp, int NUM_N_WARPS>
__global__ void __launch_bounds__(NUM_K_WARPS * NUM_N_WARPS * 64, 1)
gemm_w8a16_int8_bm64_kernel(
    Element*       __restrict__ d,
    const Element* __restrict__ a,
    const int8_t* __restrict__ b,
    const Scale* __restrict__ scale,
    int M, int N, int K)
{
    constexpr int NSizePerWarp      = 16;
    constexpr int tc_number         = KPerWarp / 16;
    constexpr int NLoop             = NPerWarp / NSizePerWarp;
    constexpr int k_fp16_per_thread = tc_number * 4;
    constexpr int k_i32_per_thread  = k_fp16_per_thread / 4;
    constexpr int kNWarps           = NUM_K_WARPS;
    constexpr int BLOCK_M           = 64;
    constexpr int MLoop             = BLOCK_M / 16;  // 4

    const int bidm = blockIdx.x;
    a += (size_t)bidm * BLOCK_M * K;
    d += (size_t)bidm * BLOCK_M * N;
    M = MIN(BLOCK_M, M - bidm * BLOCK_M);

    const int warp_n_idx = threadIdx.y;
    const int bwid       = blockIdx.y * NUM_N_WARPS + warp_n_idx;
    const int tid        = threadIdx.x;
    const int warp_idx   = __builtin_hcu_readfirstlane(tid / 64);
    const int lane       = tid % 64;
    const int row_id     = lane % NSizePerWarp;
    const int col_id     = lane / NSizePerWarp;

    const int N_len = N - NPerWarp * bwid;
    const bool n_active = (N_len > 0);
    const int block_n = NPerWarp * bwid;

    if (n_active) {
        d += block_n;
    }

    const int lds_M = M + LDS_M_PAD;
    const int num_elem_per_nwarp =
        (NUM_K_WARPS > 1) ? (NUM_K_WARPS / 2 * NPerWarp * lds_M) : (NPerWarp * lds_M);

    using A_VEC = typename w16_vec<Element, k_fp16_per_thread>::type;
    using B_VEC = __attribute__((__vector_size__(k_i32_per_thread  * sizeof(int32_t)))) int32_t;

    A_VEC    a_vec_m[MLoop];
    float4_t d_vec[MLoop][NLoop];
    #pragma unroll
    for (int mm = 0; mm < MLoop; mm++) {
        a_vec_m[mm] = {};
        #pragma unroll
        for (int i = 0; i < NLoop; i++) d_vec[mm][i] = {0.f, 0.f, 0.f, 0.f};
    }

    if (n_active) {
        for (int i = warp_idx * KPerWarp; i < K; i += KPerWarp * kNWarps) {
            int k_cur = i + col_id * k_fp16_per_thread;
            #pragma unroll
            for (int mm = 0; mm < MLoop; mm++) {
                const int m_base = mm * 16;
                if (m_base + row_id < M) {
                    a_vec_m[mm] = *(A_VEC*)(a + (m_base + row_id) * K + k_cur);
                } else {
                    a_vec_m[mm] = {};
                }
            }
            #pragma unroll
            for (int nn = 0; nn < NLoop; nn++) {
                const int n_idx = NSizePerWarp * nn + row_id;
                auto tb = load_w8a16_weight_vec<B_VEC, MARLIN_LAYOUT>(b, N, K, block_n + n_idx, k_cur);

                #pragma unroll
                for (int ii = 0; ii < tc_number / 2; ii++) {
                    w16_vec4_t<Element> bv0;
                    w16_vec4_t<Element> bv1;
                    dequant_w8a16_to_reg<Element>(static_cast<uint32_t>(tb[ii * 2]),
                                                  static_cast<uint32_t>(tb[ii * 2 + 1]),
                                                  bv0, bv1);
                    #pragma unroll
                    for (int mm = 0; mm < MLoop; mm++) {
                        builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec_m[mm])[ii * 2    ], bv0, d_vec[mm][nn]);
                        builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec_m[mm])[ii * 2 + 1], bv1, d_vec[mm][nn]);
                    }
                }
            }
        }
    }

    // K-warp tree reduction via LDS
    extern __shared__ float start_smem[];
    float4_t* out_smem_f32 = reinterpret_cast<float4_t*>(
        start_smem + num_elem_per_nwarp * warp_n_idx);
    Element*   out_smem_w16 = reinterpret_cast<Element*>(start_smem)
                             + num_elem_per_nwarp * warp_n_idx;

    if constexpr (NUM_K_WARPS > 1) {
        const int lds_n_stride = NUM_K_WARPS / 2 * NSizePerWarp * lds_M / 4;
        #pragma unroll
        for (int stride = NUM_K_WARPS; stride > 1; stride /= 2) {
            const int mid = stride / 2;
            if (warp_idx >= mid && warp_idx < stride) {
                #pragma unroll
                for (int mm = 0; mm < MLoop; mm++) {
                    const int m_base = mm * 16;
                    if (m_base + row_id < M) {
                        #pragma unroll
                        for (int nn = 0; nn < NLoop; nn++)
                            out_smem_f32[(warp_idx - mid) * 4 * lds_M
                                         + (m_base + row_id) * 4 + col_id
                                         + lds_n_stride * nn] = d_vec[mm][nn];
                    }
                }
            }
            __syncthreads();
            if (warp_idx < mid) {
                #pragma unroll
                for (int nn = 0; nn < NLoop; nn++) {
                    #pragma unroll
                    for (int mm = 0; mm < MLoop; mm++) {
                        const int m_base = mm * 16;
                        if (m_base + row_id < M) {
                            float4_t tmp = out_smem_f32[warp_idx * 4 * lds_M
                                                        + (m_base + row_id) * 4 + col_id
                                                        + lds_n_stride * nn];
                            #pragma unroll
                            for (int j = 0; j < 4; j++) d_vec[mm][nn][j] += tmp[j];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Apply per-channel scale after K-loop reduction
    if (n_active) {
        #pragma unroll
        for (int nn = 0; nn < NLoop; nn++) {
            const int n_base = block_n + nn * NSizePerWarp;
            float sc_vec[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n_col = n_base + j * 4 + col_id;
                sc_vec[j] = (n_col < N) ? static_cast<float>(scale[n_col]) : 1.0f;
            }
            #pragma unroll
            for (int mm = 0; mm < MLoop; mm++) {
                const int m_base = mm * 16;
                if (m_base + row_id < M) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        d_vec[mm][nn][j] *= sc_vec[j];
                    }
                }
            }
        }
    }

    // warp 0 converts fp32 accumulators → fp16 into LDS
    if (warp_idx == 0) {
        #pragma unroll
        for (int nn = 0; nn < NLoop; nn++) {
            const int base = nn * NSizePerWarp * lds_M;
            #pragma unroll
            for (int mm = 0; mm < MLoop; mm++) {
                const int m_base = mm * 16;
                if (m_base + row_id < M) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        out_smem_w16[base + (m_base + row_id) * NSizePerWarp + i * 4 + col_id] =
                            float_to_w16<Element>(d_vec[mm][nn][i]);
                }
            }
        }
    }
    __syncthreads();

    // All warps cooperate writing LDS → global (coalesced half4_t stores)
    for (int nn = warp_idx; nn < NLoop; nn += NUM_K_WARPS) {
        #pragma unroll
        for (int mm = 0; mm < MLoop; mm++) {
            const int m_base = mm * 16;
            if (m_base + row_id < M) {
                if (nn * NSizePerWarp + col_id * 4 < N_len) {
                    const int smem_off = nn * NSizePerWarp * lds_M
                                         + (m_base + row_id) * NSizePerWarp + col_id * 4;
                    const int out_off  = (m_base + row_id) * N
                                         + nn * NSizePerWarp + col_id * 4;
                    *reinterpret_cast<w16_vec4_t<Element>*>(d + out_off) =
                        *reinterpret_cast<w16_vec4_t<Element>*>(out_smem_w16 + smem_off);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K kernel — BlockM=16
// ---------------------------------------------------------------------------
template <typename Element, typename Scale, bool MARLIN_LAYOUT, int NUM_K_WARPS, int NPerWarp, int KPerWarp, int NUM_N_WARPS>
__global__ void __launch_bounds__(NUM_K_WARPS * NUM_N_WARPS * 64, 2)
gemm_w8a16_int8_splitk_kernel(
    Element*       __restrict__ d,
    const Element* __restrict__ a,
    const int8_t* __restrict__ b,
    const Scale* __restrict__ scale,
    float*                     splitk_ws,
    int*                       semaphores,
    int M, int N, int K)
{
    constexpr int NSizePerWarp      = 16;
    constexpr int tc_number         = KPerWarp / 16;
    constexpr int NLoop             = NPerWarp / NSizePerWarp;
    constexpr int k_fp16_per_thread = tc_number * 4;
    constexpr int k_i32_per_thread  = k_fp16_per_thread / 4;
    constexpr int kNWarps           = NUM_K_WARPS;
    constexpr int BlockM            = 16;

    const int bidm = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;
    const int SplitKBatch = gridDim.z;

    a         += (size_t)bidm * BlockM * K;
    d         += (size_t)bidm * BlockM * N;
    splitk_ws += (size_t)bidm * BlockM * N;
    semaphores = semaphores + bidm * (gridDim.y * NUM_N_WARPS);
    M = MIN(BlockM, M - bidm * BlockM);

    const int warp_n_idx = threadIdx.y;
    const int bwid       = bidy * NUM_N_WARPS + warp_n_idx;
    const int tid        = threadIdx.x;
    const int warp_idx   = __builtin_hcu_readfirstlane(tid / 64);
    const int lane       = tid % 64;
    const int row_id     = lane % NSizePerWarp;
    const int col_id     = lane / NSizePerWarp;
    const bool row_valid = row_id < M;

    const int N_len = N - NPerWarp * bwid;
    const bool n_active = (N_len > 0);
    const int block_n = NPerWarp * bwid;

    if (n_active) {
        d         += block_n;
        splitk_ws += block_n;
    }

    const int part_k = ((K / KPerWarp - 1) / SplitKBatch + 1) * KPerWarp;
    const int k_start  = bidz * part_k;
    const int k_remaining = K - k_start;
    const int k_length = k_remaining > 0 ? MIN(part_k, k_remaining) : 0;

    a += k_start;

    const int lds_M = M + LDS_M_PAD;
    const int num_elem_per_nwarp =
        (NUM_K_WARPS > 1) ? (NUM_K_WARPS / 2 * NPerWarp * lds_M) : 0;

    using A_VEC = typename w16_vec<Element, k_fp16_per_thread>::type;
    using B_VEC = __attribute__((__vector_size__(k_i32_per_thread  * sizeof(int32_t)))) int32_t;

    A_VEC    a_vec;
    float4_t d_vec[NLoop];
    #pragma unroll
    for (int i = 0; i < NLoop; i++) d_vec[i] = {0.f, 0.f, 0.f, 0.f};

    if (n_active) {
        for (int i = warp_idx * KPerWarp; i < k_length; i += KPerWarp * kNWarps) {
            const int k_idx = i + col_id * k_fp16_per_thread;
            if (row_valid)
                a_vec = *(A_VEC*)(a + row_id * K + k_idx);

            #pragma unroll
            for (int nn = 0; nn < NLoop; nn++) {
                const int n_idx = NSizePerWarp * nn + row_id;
                auto tb = load_w8a16_weight_vec<B_VEC, MARLIN_LAYOUT>(b, N, K, block_n + n_idx, k_start + k_idx);

                #pragma unroll
                for (int ii = 0; ii < tc_number / 2; ii++) {
                    w16_vec4_t<Element> bv0;
                    w16_vec4_t<Element> bv1;
                    dequant_w8a16_to_reg<Element>(static_cast<uint32_t>(tb[ii * 2]),
                                                  static_cast<uint32_t>(tb[ii * 2 + 1]),
                                                  bv0,
                                                  bv1);
                    builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec)[ii * 2    ], bv0, d_vec[nn]);
                    builtin_amdgcn_mmac_w16<Element>(reinterpret_cast<w16_vec4_t<Element>*>(&a_vec)[ii * 2 + 1], bv1, d_vec[nn]);
                }
            }
        }
    }

    // K-warp reduction
    extern __shared__ float start_smem[];
    float4_t* out_smem = reinterpret_cast<float4_t*>(
        start_smem + num_elem_per_nwarp * warp_n_idx);

    if constexpr (NUM_K_WARPS > 1) {
        const int lds_n_stride = NUM_K_WARPS / 2 * NSizePerWarp * lds_M / 4;
        #pragma unroll
        for (int stride = NUM_K_WARPS; stride > 1; stride /= 2) {
            const int mid = stride / 2;
            if (row_valid) {
                if (warp_idx >= mid && warp_idx < stride) {
                    #pragma unroll
                    for (int nn = 0; nn < NLoop; nn++)
                        out_smem[(warp_idx - mid) * 4 * lds_M + row_id * 4 + col_id
                                 + lds_n_stride * nn] = d_vec[nn];
                }
            }
            __syncthreads();
            if (warp_idx < mid) {
                #pragma unroll
                for (int nn = 0; nn < NLoop; nn++) {
                    if (row_valid) {
                        float4_t tmp = out_smem[warp_idx * 4 * lds_M + row_id * 4 + col_id
                                                + lds_n_stride * nn];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) d_vec[nn][j] += tmp[j];
                    }
                }
            }
            __syncthreads();
        }
    }

    if constexpr (!MARLIN_LAYOUT && NPerWarp == 32) {
        if (gridDim.x == 1 && N == 2048 && K == 4096 && SplitKBatch <= 8
            && (size_t)SplitKBatch * BlockM * N * sizeof(float) <= 100000000ull) {
            constexpr size_t split_stride = (size_t)BlockM * 2048;
            float* partial_ws = splitk_ws + (size_t)bidz * split_stride;
            if (warp_idx == 0 && row_valid) {
                #pragma unroll
                for (int nn = 0; nn < NLoop; nn++) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        partial_ws[nn * NSizePerWarp + i * 4 + col_id + row_id * 2048] = d_vec[nn][i];
                    }
                }
            }

            if (warp_idx == 0) __threadfence();

            int order = -1;
            if (warp_idx == 0 && tid == 0)
                order = atomicAdd(semaphores + bwid, 1);
            order = __shfl(order, 0);

            if (order == SplitKBatch - 1) {
                if (tid == 0) semaphores[bwid] = 0;
                const int total = M * 32;
                for (int i = tid * 4; i < total; i += 4 * 64 * NUM_K_WARPS) {
                    const int local_n = i & 31;
                    const int off = (i >> 5) * 2048 + local_n;
                    float4_t fval = {0.f, 0.f, 0.f, 0.f};
                    if (SplitKBatch == 4) {
                        #pragma unroll
                        for (int sk = 0; sk < 4; sk++) {
                            float4_t part = *(float4_t*)(splitk_ws + (size_t)sk * split_stride + off);
                            #pragma unroll
                            for (int ii = 0; ii < 4; ii++) fval[ii] += part[ii];
                        }
                    } else if (SplitKBatch == 8) {
                        #pragma unroll
                        for (int sk = 0; sk < 8; sk++) {
                            float4_t part = *(float4_t*)(splitk_ws + (size_t)sk * split_stride + off);
                            #pragma unroll
                            for (int ii = 0; ii < 4; ii++) fval[ii] += part[ii];
                        }
                    } else {
                        #pragma unroll
                        for (int sk = 0; sk < 8; sk++) {
                            if (sk < SplitKBatch) {
                                float4_t part = *(float4_t*)(splitk_ws + (size_t)sk * split_stride + off);
                                #pragma unroll
                                for (int ii = 0; ii < 4; ii++) fval[ii] += part[ii];
                            }
                        }
                    }
                    w16_vec4_t<Element> hval;
                    Element* hval_elem = reinterpret_cast<Element*>(&hval);
                    #pragma unroll
                    for (int ii = 0; ii < 4; ii++) {
                        const int n_col = block_n + local_n + ii;
                        const float sc = static_cast<float>(scale[n_col]);
                        hval_elem[ii] = float_to_w16<Element>(fval[ii] * sc);
                    }
                    *reinterpret_cast<w16_vec4_t<Element>*>(d + off) = hval;
                }
            }
            return;
        }
    } else if constexpr (!MARLIN_LAYOUT) {
        if (gridDim.x == 1 && N == 2048 && K == 4096 && SplitKBatch <= 8
            && (size_t)SplitKBatch * BlockM * N * sizeof(float) <= 100000000ull) {
            const size_t split_stride = (size_t)BlockM * N;
            float* partial_ws = splitk_ws + (size_t)bidz * split_stride;
            if (n_active && warp_idx == 0 && row_valid) {
                #pragma unroll
                for (int nn = 0; nn < NLoop; nn++) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if (nn * NSizePerWarp + col_id * 4 + i < N_len) {
                            partial_ws[nn * NSizePerWarp + i * 4 + col_id + row_id * N] = d_vec[nn][i];
                        }
                    }
                }
            }

            if (warp_idx == 0) __threadfence();
            __syncthreads();

            if (n_active) {
                int order = -1;
                if (warp_idx == 0 && tid == 0)
                    order = atomicAdd(semaphores + bwid, 1);
                order = __shfl(order, 0);

                if (order == SplitKBatch - 1) {
                    if (tid == 0) semaphores[bwid] = 0;
                    const int total = M * NPerWarp;
                    for (int i = tid * 4; i < total; i += 4 * 64 * NUM_K_WARPS) {
                        if (i % NPerWarp < N_len) {
                            const int off = (i / NPerWarp) * N + (i % NPerWarp);
                            float4_t fval = {0.f, 0.f, 0.f, 0.f};
                            #pragma unroll
                            for (int sk = 0; sk < 8; sk++) {
                                if (sk < SplitKBatch) {
                                    float4_t part = *(float4_t*)(splitk_ws + (size_t)sk * split_stride + off);
                                    #pragma unroll
                                    for (int ii = 0; ii < 4; ii++) fval[ii] += part[ii];
                                }
                            }
                            w16_vec4_t<Element> hval;
                            Element* hval_elem = reinterpret_cast<Element*>(&hval);
                            #pragma unroll
                            for (int ii = 0; ii < 4; ii++) {
                                const int n_col = block_n + (i % NPerWarp) + ii;
                                const float sc = (n_col < N) ? static_cast<float>(scale[n_col]) : 1.0f;
                                hval_elem[ii] = float_to_w16<Element>(fval[ii] * sc);
                            }
                            *reinterpret_cast<w16_vec4_t<Element>*>(d + off) = hval;
                        }
                    }
                }
            }
            return;
        }
    }

    // Atomically accumulate into float workspace
    if (n_active && warp_idx == 0 && row_valid) {
        for (int nn = 0; nn < NLoop; nn++) {
            for (int i = 0; i < 4; i++) {
                if (nn * NSizePerWarp + col_id * 4 + i < N_len) {
                    atomicAdd(splitk_ws
                              + nn * NSizePerWarp + i * 4 + col_id
                              + row_id * N,
                              d_vec[nn][i]);
                }
            }
        }
    }

    __syncthreads();

    // Last block converts workspace → fp16 output
    if (n_active) {
        int order = -1;
        if (warp_idx == 0 && tid == 0)
            order = atomicAdd(semaphores + bwid, 1);
        order = __shfl(order, 0);

        if (order == SplitKBatch - 1) {
            if (tid == 0) semaphores[bwid] = 0;
            const int total = M * NPerWarp;
            for (int i = tid * 4; i < total; i += 4 * 64 * NUM_K_WARPS) {
                if (i % NPerWarp < N_len) {
                    const int off = (i / NPerWarp) * N + (i % NPerWarp);
                    float4_t fval = *(float4_t*)(splitk_ws + off);
                    *(float4_t*)(splitk_ws + off) = {0.f, 0.f, 0.f, 0.f};
                    w16_vec4_t<Element> hval;
                    Element* hval_elem = reinterpret_cast<Element*>(&hval);
                    #pragma unroll
                    for (int ii = 0; ii < 4; ii++) {
                        const int n_col = block_n + (i % NPerWarp) + ii;
                        const float sc = (n_col < N) ? static_cast<float>(scale[n_col]) : 1.0f;
                        hval_elem[ii] = float_to_w16<Element>(fval[ii] * sc);
                    }
                    *reinterpret_cast<w16_vec4_t<Element>*>(d + off) = hval;
                }
            }
        }
    }
}

} // namespace deepgemm
