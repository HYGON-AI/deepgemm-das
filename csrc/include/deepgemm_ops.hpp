#define M_GROUP_GEMM_NT_CONTIGUOUS_PYBIND \
    m.def("m_grouped_w8a8_gemm_nt_nopad_contiguous", &deepgemm::dispatch::m_grouped_w8a8_gemm_nt_nopad_contiguous, "m_grouped_w8a8_gemm_nt_nopad_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("token_per_expert")); \
    m.def("m_grouped_w8a8_gemm_nt_contiguous", &deepgemm::dispatch::m_grouped_w8a8_gemm_nt_contiguous, "m_grouped_w8a8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_w16a16_gemm_nt_contiguous", &deepgemm::dispatch::m_grouped_w16a16_gemm_nt_contiguous, "m_grouped_w16a16_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  \
    m.def("m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc, "m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  \
    m.def("m_grouped_fp8_gemm_nt_contiguous", &deepgemm::dispatch::m_grouped_fp8_gemm_nt_contiguous, "m_grouped_fp8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  
    
#define M_GROUP_GEMM_NT_MASKED_PYBIND \
    m.def("m_grouped_marlin_w8a8_gemm_nt_compact_masked", &deepgemm::dispatch::m_grouped_marlin_w8a8_gemm_nt_compact_masked, "m_grouped_marlin_w8a8_gemm_nt_compact_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expert_offsets"), \
        py::arg("expected_m_per_group")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_masked, "m_grouped_marlin_w4a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked_hipc", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_masked_hipc, "m_grouped_marlin_w4a8_gemm_nt_masked_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"), \
        py::arg("persistent_multiplier") = 0, \
        py::arg("enable_overlap") = false, \
        py::arg("signal") = py::none()); \
    m.def("m_grouped_marlin_w8a8_gemm_nt_masked", &deepgemm::dispatch::m_grouped_marlin_w8a8_gemm_nt_masked, "m_grouped_marlin_w8a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("low_latency_grouped_gemm", &deepgemm::dispatch::group_gemm_fp8_masked, "group_gemm_fp8_masked", \
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
        py::arg("signal"),  \
        py::arg("mode"));  \
    m.def("m_grouped_w8a8_gemm_nt_masked_gfx936", &deepgemm::dispatch::group_gemm_fp8_masked, "group_gemm_fp8_masked", \
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
        py::arg("signal"),  \
        py::arg("mode"));  \
    m.def("m_grouped_bf16_gemm_nt_masked", &deepgemm::dispatch::m_grouped_bf16_gemm_nt_masked, "m_grouped_bf16_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"));  \

#define M_GROUP_GEMM_NT_MASKED_W4A8_ONLY_PYBIND \
    m.def("m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc, "m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_masked, "m_grouped_marlin_w4a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked_hipc", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_masked_hipc, "m_grouped_marlin_w4a8_gemm_nt_masked_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"), \
        py::arg("persistent_multiplier") = 0, \
        py::arg("enable_overlap") = false, \
        py::arg("signal") = py::none());
#define GEMM_NT  \
    m.def("fp8_gemm", &deepgemm::dispatch::fp8_gemm, "fp8_gemm", \
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
    m.def("fp8_gemm_hipc", &deepgemm::dispatch::fp8_gemm_hipc, "fp8_gemm_hipc", \
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
    m.def("w8a16_linear", &deepgemm::dispatch::w8a16_linear, "w8a16_linear", \
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
    m.def("w8a16_linear_bias", &deepgemm::dispatch::w8a16_linear_bias, "w8a16_linear_bias", \
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
    m.def("w8a16_dequant_weight", &deepgemm::dispatch::w8a16_dequant_weight, "w8a16_dequant_weight", \
        py::arg("b"), \
        py::arg("scale")); \
    m.def("w8a16_dequant_weight_bf16", &deepgemm::dispatch::w8a16_dequant_weight_bf16, "w8a16_dequant_weight_bf16", \
        py::arg("b"), \
        py::arg("scale"));
#define TF32_HC_PRENORM_GEMM_PYBIND \
    m.def("tf32_hc_prenorm_gemm", &deepgemm::dispatch::tf32_hc_prenorm_gemm, "tf32_hc_prenorm_gemm", \
        py::arg("A"), \
        py::arg("B"), \
        py::arg("D"), \
        py::arg("sqr_sum"), \
        py::arg("num_splits"));

#define EINSUM_PYBIND \
    m.def("einsum", &deepgemm::dispatch::einsum::einsum, "einsum", \
        py::arg("expr"), py::arg("a"), py::arg("b"), \
        py::arg("d"), py::arg("c") = py::none()); \
    m.def("fp8_einsum", &deepgemm::dispatch::einsum::fp8_einsum, "fp8_einsum", \
        py::arg("expr"), py::arg("a"), py::arg("b"), \
        py::arg("d"), py::arg("c") = py::none(), \
        py::arg("recipe") = std::make_tuple(1, 128, 128));

#define MQA_LOGITS_PYBIND \
    m.def("mqa_logits", &deepgemm::mqa_logits, "Non-paged dense MQA logits", \
          py::arg("Q"), py::arg("K"), py::arg("Weights"), py::arg("KS"), py::arg("KE"), \
          py::arg("seq_len"), py::arg("kv_seq_len"), py::arg("num_heads"), py::arg("head_dim"), \
          py::arg("KV_scale") = torch::nullopt, py::arg("clean_logits") = true, \
          py::arg("D_out") = torch::nullopt); \
    m.def("mqa_logits_compact", &deepgemm::mqa_logits_compact, "Non-paged compact MQA logits", \
          py::arg("Q"), py::arg("K"), py::arg("Weights"), py::arg("KS"), py::arg("KE"), \
          py::arg("seq_len"), py::arg("kv_seq_len"), py::arg("num_heads"), py::arg("head_dim"), \
          py::arg("KV_scale"), py::arg("max_seqlen_k"));

#define PAGED_MQA_LOGITS_PYBIND \
    m.def("paged_mqa_logits", &deepgemm::paged_mqa_logits, "Dense paged MQA logits", \
        py::arg("q"), py::arg("fused_kv_cache"), py::arg("weights"), \
        py::arg("context_lens"), py::arg("block_table"), py::arg("max_model_len"), \
        py::arg("out") = std::nullopt, py::arg("clean_logits") = true, \
        py::arg("implementation") = "auto", py::arg("kernel_id") = -1, py::arg("is_prefill") = false); \
    m.def("page_mqa_logits_sparse_mask", &deepgemm::page_mqa_logits_sparse_mask, "Sparse paged MQA logits", \
        py::arg("q"), py::arg("fused_kv_cache"), py::arg("weights"), \
        py::arg("context_lens"), py::arg("block_table"), py::arg("max_model_len"), \
        py::arg("clean_logits") = true, py::arg("num_warps") = 4, py::arg("persistent_ctas") = 0); \
    m.def("page_mqa_logits_sparse_mask_grouped", &deepgemm::page_mqa_logits_sparse_mask_grouped, \
        "Grouped sparse paged MQA logits", \
        py::arg("q"), py::arg("fused_kv_cache"), py::arg("weights"), \
        py::arg("context_lens"), py::arg("block_table"), py::arg("max_model_len"), \
        py::arg("clean_logits") = true, py::arg("num_warps") = 4, py::arg("persistent_ctas") = 0, \
        py::arg("group_size") = 5); \
    m.def("page_mqa_logits_sparse_mask_grouped_int8", &deepgemm::page_mqa_logits_sparse_mask_grouped_int8, \
        "Grouped sparse paged MQA logits with INT8 KV", \
        py::arg("q"), py::arg("packed_kv_cache"), py::arg("weights"), \
        py::arg("context_lens"), py::arg("block_table"), py::arg("max_model_len"), \
        py::arg("clean_logits") = true, py::arg("num_warps") = 4, py::arg("persistent_ctas") = 0, \
        py::arg("group_size") = 5); \
    m.def("get_paged_mqa_logits_fp4_metadata", &deepgemm::get_paged_mqa_logits_fp4_metadata, \
        "FP4 paged MQA persistent schedule", py::arg("context_lens"), py::arg("block_kv") = 64, \
        py::arg("num_sms") = std::nullopt, py::arg("rows_per_request") = 1); \
    m.def("fp8_fp4_paged_mqa_logits", &deepgemm::fp8_fp4_paged_mqa_logits, \
        "MXFP4 paged MQA logits", py::arg("q"), py::arg("q_sf"), py::arg("kv_cache"), \
        py::arg("weights"), py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), \
        py::arg("max_context_len"), py::arg("table_block_size") = 64, py::arg("out") = std::nullopt, \
        py::arg("rows_per_request") = 1);

#define PAGED_SPARSE_MQA_LOGITS_PYBIND \
    m.def("get_paged_sparse_mqa_logits_metadata", &deepgemm::get_paged_sparse_mqa_logits_metadata, \
        "Sparse paged MQA metadata", py::arg("context_lens"), py::arg("block_table"), py::arg("indices"), \
        py::arg("page_kv"), py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv")); \
    m.def("fp8_fp4_paged_sparse_mqa_logits", &deepgemm::fp8_fp4_paged_sparse_mqa_logits, \
        "MXFP4/MXFP8 sparse paged MQA logits", py::arg("q"), py::arg("q_scales"), \
        py::arg("kv_cache"), py::arg("weights"), py::arg("metadata"), \
        py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"), \
        py::arg("context_lens") = std::nullopt, py::arg("sparse_kv_block_indices") = std::nullopt, \
        py::arg("output") = std::nullopt, py::arg("block_table") = std::nullopt, py::arg("mask_tail") = std::nullopt); \
    m.attr("PAGED_SPARSE_MQA_METADATA_FORMAT") = "hcu_paged_sparse_v2";

// Ops that used to be registered only through torch.ops.deep_gemm. They are
// now bound exclusively through the pybind module (deepgemm.op); declarations
// come from kernel_registry_decls.h.
#define KERNEL_REGISTRY_OPS_PYBIND \
    m.def("m_grouped_fp8_gemm_nt_masked_pair32", \
        &deepgemm::dispatch::m_grouped_fp8_gemm_nt_masked_pair32, \
        "m_grouped_fp8_gemm_nt_masked_pair32", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("output"), \
        py::arg("enable_overlap"), \
        py::arg("signal"), \
        py::arg("mode")); \
    m.def("m_grouped_fp8_gemm_nt_contiguous_wasp_pair32", \
        &deepgemm::dispatch::m_grouped_fp8_gemm_nt_contiguous_wasp_pair32, \
        "m_grouped_fp8_gemm_nt_contiguous_wasp_pair32", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_fp8_gemm_nt_masked_pair32_v2", \
        &deepgemm::dispatch::m_grouped_fp8_gemm_nt_masked_pair32_v2, \
        "m_grouped_fp8_gemm_nt_masked_pair32_v2", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("output"), \
        py::arg("enable_overlap"), \
        py::arg("signal"), \
        py::arg("mode")); \
    m.def("m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2", \
        &deepgemm::dispatch::m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2, \
        "m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_v2", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_int8_gemm_nt_masked_pair32", \
        &deepgemm::dispatch::m_grouped_int8_gemm_nt_masked_pair32, \
        "m_grouped_int8_gemm_nt_masked_pair32", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("output"), \
        py::arg("mode")); \
    m.def("m_grouped_int8_gemm_nt_contiguous_wasp_pair32", \
        &deepgemm::dispatch::m_grouped_int8_gemm_nt_contiguous_wasp_pair32, \
        "m_grouped_int8_gemm_nt_contiguous_wasp_pair32", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_fp8_gemm_nt_contiguous", \
        &deepgemm::dispatch::m_grouped_fp8_gemm_nt_contiguous_registry, \
        "m_grouped_fp8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("block_m"), \
        py::arg("block_n"), \
        py::arg("n_loop_num")); \
    m.def("m_grouped_fp8_gemm_nt_contiguous_marlin", \
        &deepgemm::dispatch::m_grouped_fp8_gemm_nt_contiguous_marlin, \
        "m_grouped_fp8_gemm_nt_contiguous_marlin", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"), \
        py::arg("n_loop_num")); \
    m.def("m_grouped_i8_gemm_nt_contiguous", \
        &deepgemm::dispatch::m_grouped_i8_gemm_nt_contiguous, \
        "m_grouped_i8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("block_m"), \
        py::arg("block_n"), \
        py::arg("n_loop_num")); \
    m.def("moe_w8a8_marlin_prefill_down_fp8", \
        &deepgemm::dispatch::moe_marlin_w8a8_gemm2_prefill_down_fp8, \
        "moe_w8a8_marlin_prefill_down_fp8", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("topk_weights"), \
        py::arg("sorted_token_ids"), \
        py::arg("expert_ids"), \
        py::arg("num_tokens_post_pad"), \
        py::arg("top_k"), \
        py::arg("real_topk")); \
    m.def("m_grouped_wfp4a16_gemm_nt_contiguous", \
        &deepgemm::dispatch::group_gemm_wfp4a16_contiguous, \
        "m_grouped_wfp4a16_gemm_nt_contiguous", \
        py::arg("matrix_a"), \
        py::arg("matrix_b"), \
        py::arg("matrix_c"), \
        py::arg("matrix_b_scale"), \
        py::arg("m_indices")); \
    m.def("low_latency_wfp4a16_gemm", \
        &deepgemm::dispatch::group_gemm_wfp4a16_masked, \
        "low_latency_wfp4a16_gemm", \
        py::arg("matrix_a"), \
        py::arg("matrix_b"), \
        py::arg("matrix_b_scale"), \
        py::arg("actual_tokens"), \
        py::arg("matrix_c"), \
        py::arg("max_tokens"), \
        py::arg("experts")); \
    m.def("low_latency_w4a16_gemm", \
        &deepgemm::dispatch::group_gemm_w4a16_masked, \
        "low_latency_w4a16_gemm", \
        py::arg("matrix_a"), \
        py::arg("matrix_b"), \
        py::arg("matrix_b_scale"), \
        py::arg("actual_tokens"), \
        py::arg("matrix_c"), \
        py::arg("max_tokens"), \
        py::arg("experts")); \
    m.def("m_grouped_w8a8_gemm_nt_contiguous_gfx936", \
        &deepgemm::dispatch::i8_contig_ll, \
        "m_grouped_w8a8_gemm_nt_contiguous_gfx936", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("block_m"), \
        py::arg("block_n"), \
        py::arg("n_loop_num"));

#define GROUPED_GEMM_GFX92A_PYBIND \
    m.def("low_latency_grouped_gemm", &deepgemm::dispatch::group_gemm_fp8_masked, "group_gemm_fp8_masked", \
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
        py::arg("signal"),  \
        py::arg("mode"));  \
    m.def("m_grouped_bf16_gemm_nt_masked", &deepgemm::dispatch::m_grouped_bf16_gemm_nt_masked, "m_grouped_bf16_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"));  \
    m.def("m_grouped_w16a16_gemm_nt_contiguous", &deepgemm::dispatch::m_grouped_w16a16_gemm_nt_contiguous, "m_grouped_w16a16_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  \
    m.def("m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc, "m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  \
m.def("m_grouped_marlin_w4a8_gemm_nt_masked_hipc", &deepgemm::dispatch::m_grouped_marlin_w4a8_gemm_nt_masked_hipc, "m_grouped_marlin_w4a8_gemm_nt_masked_hipc", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"), \
        py::arg("persistent_multiplier") = 0, \
        py::arg("enable_overlap") = false, \
        py::arg("signal") = py::none());