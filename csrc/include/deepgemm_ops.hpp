#define M_GROUP_GEMM_NT_CONTIGUOUS_PYBIND \
    m.def("m_grouped_w8a8_gemm_nt_nopad_contiguous", &deepgemm::m_grouped_w8a8_gemm_nt_nopad_contiguous, "m_grouped_w8a8_gemm_nt_nopad_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("token_per_expert")); \
    m.def("m_grouped_w8a8_gemm_nt_contiguous", &deepgemm::m_grouped_w8a8_gemm_nt_contiguous, "m_grouped_w8a8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_w16a16_gemm_nt_contiguous", &deepgemm::m_grouped_w16a16_gemm_nt_contiguous, "m_grouped_w16a16_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  \
    m.def("m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc, "m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  \
    m.def("m_grouped_fp8_gemm_nt_contiguous", &deepgemm::m_grouped_fp8_gemm_nt_contiguous, "m_grouped_fp8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  
    
#define M_GROUP_GEMM_NT_MASKED_PYBIND \
    m.def("m_grouped_marlin_w8a8_gemm_nt_compact_masked", &deepgemm::m_grouped_marlin_w8a8_gemm_nt_compact_masked, "m_grouped_marlin_w8a8_gemm_nt_compact_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expert_offsets"), \
        py::arg("expected_m_per_group")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked, "m_grouped_marlin_w4a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked_hipc", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked_hipc, "m_grouped_marlin_w4a8_gemm_nt_masked_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w8a8_gemm_nt_masked", &deepgemm::m_grouped_marlin_w8a8_gemm_nt_masked, "m_grouped_marlin_w8a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("low_latency_grouped_gemm", &deepgemm::group_gemm_fp8_masked, "group_gemm_fp8_masked", \
        py::arg("matrix_a"), \
        py::arg("matrix_b"), \
        py::arg("matrix_a_scale"), \
        py::arg("matrix_b_scale"), \
        py::arg("actual_tokens"), \
        py::arg("matrix_c"),  \
        py::arg("max_tokens"),  \
        py::arg("experts"),  \
        py::arg("cu_s"),  \
        py::arg("block_wise"),  \
        py::arg("b_overlap"),  \
        py::arg("signal"));  \
    m.def("m_grouped_w8a8_gemm_nt_masked_gfx936", &deepgemm::group_gemm_fp8_masked, "group_gemm_fp8_masked", \
        py::arg("matrix_a"), \
        py::arg("matrix_b"), \
        py::arg("matrix_a_scale"), \
        py::arg("matrix_b_scale"), \
        py::arg("actual_tokens"), \
        py::arg("matrix_c"),  \
        py::arg("max_tokens"),  \
        py::arg("experts"),  \
        py::arg("cu_s"),  \
        py::arg("block_wise"),  \
        py::arg("b_overlap"),  \
        py::arg("signal"));  \
    m.def("low_latency_grouped_gemm_tensor_wise", &deepgemm::group_gemm_fp8_masked_tensor_wise, "group_gemm_fp8_masked_tensor_wise", \
        py::arg("matrix_a"), \
        py::arg("matrix_b"), \
        py::arg("matrix_a_scale"), \
        py::arg("matrix_b_scale"), \
        py::arg("actual_tokens"), \
        py::arg("matrix_c"),  \
        py::arg("max_tokens"),  \
        py::arg("experts"),  \
        py::arg("cu_s"),  \
        py::arg("b_overlap"),  \
        py::arg("signal"));  \
    m.def("m_grouped_bf16_gemm_nt_masked", &deepgemm::m_grouped_bf16_gemm_nt_masked, "m_grouped_bf16_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"));  \
    m.def("m_grouped_bf16_gemm_nt_masked_hipc", &deepgemm::m_grouped_bf16_gemm_nt_masked_hipc, "m_grouped_bf16_gemm_nt_masked_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"));

#define M_GROUP_GEMM_NT_MASKED_W4A8_ONLY_PYBIND \
    m.def("m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc, "m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked, "m_grouped_marlin_w4a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked_hipc", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked_hipc, "m_grouped_marlin_w4a8_gemm_nt_masked_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"));
#define GEMM_NT  \
    m.def("fp8_gemm", &deepgemm::fp8_gemm, "fp8_gemm", \
        py::arg("mat_a"), \
        py::arg("mat_b"), \
        py::arg("scale_a"), \
        py::arg("scale_b"), \
        py::arg("output"), \
        py::arg("m"),  \
        py::arg("n"),  \
        py::arg("k"),  \
        py::arg("batch"),  \
        py::arg("transpose_flag"), \
        py::arg("alpha"), \
        py::arg("beta"),  \
        py::arg("bias"));

#define FP8_GEMM_HIPC_PYBIND \
    m.def("fp8_gemm_hipc", &deepgemm::fp8_gemm_hipc, "fp8_gemm_hipc", \
        py::arg("mat_a"), \
        py::arg("mat_b"), \
        py::arg("scale_a"), \
        py::arg("scale_b"), \
        py::arg("output"), \
        py::arg("m"),  \
        py::arg("n"),  \
        py::arg("k"),  \
        py::arg("batch"),  \
        py::arg("transpose_flag"), \
        py::arg("alpha"), \
        py::arg("beta"),  \
        py::arg("bias"));

#define W8A16_LINEAR_PYBIND \
    m.def("w8a16_linear", &deepgemm::w8a16_linear, "w8a16_linear", \
        py::arg("a"), \
        py::arg("b"), \
        py::arg("scale"), \
        py::arg("num_k_warps") = 0, \
        py::arg("n_per_warp") = 0, \
        py::arg("k_per_warp") = 0, \
        py::arg("num_n_warps") = 0, \
        py::arg("splitk") = 0, \
        py::arg("block_m") = 0, \
        py::arg("weight_layout") = 0); \
    m.def("w8a16_linear_bias", &deepgemm::w8a16_linear_bias, "w8a16_linear_bias", \
        py::arg("a"), \
        py::arg("b"), \
        py::arg("scale"), \
        py::arg("bias"), \
        py::arg("num_k_warps") = 0, \
        py::arg("n_per_warp") = 0, \
        py::arg("k_per_warp") = 0, \
        py::arg("num_n_warps") = 0, \
        py::arg("splitk") = 0, \
        py::arg("block_m") = 0, \
        py::arg("weight_layout") = 0); \
    m.def("w8a16_dequant_weight", &deepgemm::w8a16_dequant_weight, "w8a16_dequant_weight", \
        py::arg("b"), \
        py::arg("scale")); \
    m.def("w8a16_dequant_weight_bf16", &deepgemm::w8a16_dequant_weight_bf16, "w8a16_dequant_weight_bf16", \
        py::arg("b"), \
        py::arg("scale"));
#define TF32_HC_PERNORM_GEMM_PYBIND \
    m.def("tf32_hc_pernorm_gemm", &deepgemm::tf32_hc_pernorm_gemm, "tf32_hc_pernorm_gemm", \
        py::arg("A"), \
        py::arg("B"), \
        py::arg("D"), \
        py::arg("sqr_sum"), \
        py::arg("num_splits"));

#define EINSUM_PYBIND \
    m.def("einsum", &deepgemm::einsum::einsum, "einsum", \
        py::arg("expr"), py::arg("a"), py::arg("b"), \
        py::arg("d"), py::arg("c") = py::none()); \
    m.def("fp8_einsum", &deepgemm::einsum::fp8_einsum, "fp8_einsum", \
        py::arg("expr"), py::arg("a"), py::arg("b"), \
        py::arg("d"), py::arg("c") = py::none(), \
        py::arg("recipe") = std::make_tuple(1, 128, 128));
