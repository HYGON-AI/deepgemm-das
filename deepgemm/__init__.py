import os
import subprocess
import importlib
from pathlib import Path
from torch.utils.cpp_extension import ROCM_HOME
ROOT_DIR = Path(__file__).parent.resolve()

def _run_cmd(cmd, shell=False):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, stderr=subprocess.DEVNULL, shell=shell).decode("ascii").strip()
    except Exception:
        return None
    
def get_dtk_version(ROCM_HOME):
    dtk_version = _run_cmd(["cat", os.path.join(ROCM_HOME, '.info/rocm_version')])
    dtk_version = ''.join(dtk_version.split('.'))
    dtk_version = int(dtk_version)
    return dtk_version

dtk_version = get_dtk_version(ROCM_HOME)
op = importlib.import_module('.op', __name__)
from .print_size import install_op_size_printer
install_op_size_printer(op)

from .utils import gfx, DEEPGEMM_ROOT_DIR, DEEPGEMM_ASM_DIR
from .m_group_gemm_nt_contiguous import m_grouped_i8_gemm_nt_contiguous, m_grouped_i8_gemm_nt_contiguous_nopad, \
      m_grouped_bf16_gemm_nt_contiguous, m_grouped_fp8_gemm_nt_contiguous, m_grouped_fp8_gemm_nt_contiguous_tensor_wise, m_grouped_wfp4a16_gemm_nt_contiguous, m_grouped_w4a8_gemm_nt_contiguous_hipc, get_m_grouped_fp8_contiguous_config
from .group_gemm_weight_pack import (
    marlin_fp8_contiguous_weight,
    marlin_fp8_masked_weight,
    marlin_i8_contiguous_weight,
    marlin_i8_masked_weight,
    pack_i8_contiguous_weight_gfx936_asm_inplace,
    pack_i8_masked_weight_gfx936_asm_inplace,
    pack_fp8_contiguous_weight,
    pack_fp8_masked_weight,
    pack_i8_contiguous_weight,
    pack_i8_masked_weight,
    pack_int8_weight_enk_to_w6_gfx936_pack_permute,
    pack_w8a8_gfx936_weight_to_w6,
    pack_w4a8_moe_hipc_weight,
    pack_w4a8_moe_hipc_weight_inplace,
    view_w4a8_moe_hipc_weight_n32_layout,
    pack_bf16_masked_weight,
)
from .moe_marlin_prefill import (
    assert_moe_prefill_down_fp8_shapes,
    moe_w8a8_marlin_prefill_down_fp8,
)
from .m_group_gemm_nt_masked import m_grouped_w4a8_gemm_nt_masked, m_grouped_w4a8_gemm_nt_masked_hipc, \
      m_grouped_w8a8_gemm_nt_masked, m_grouped_i8_gemm_nt_masked, m_grouped_w8a8_gemm_nt_masked_gfx936, m_grouped_w8a8_gemm_nt_masked_gfx936_pack_permute, m_grouped_bf16_gemm_nt_masked, m_grouped_bf16_gemm_nt_masked_hipc, m_grouped_w8a8_gemm_nt_masked_impl, m_grouped_w8a8_gemm_nt_compact_masked
from .gemm import fp8_gemm, fp8_gemm_hipc
from .m_group_gemm import m_grouped_fp8_gemm_nt_masked, m_grouped_fp8_gemm_nt_masked_tensor_wise
if hasattr(op, "einsum") and hasattr(op, "fp8_einsum"):
    from .op import einsum, fp8_einsum
from .tf32_hc_pernorm_gemm import tf32_hc_pernorm_gemm
if os.environ.get("DEEPGEMM_BUILD_ONLY_W8A16") == "1" or not hasattr(op, "fp8_einsum"):
    from .w8a16_linear import (
        clear_w8a16_dequant_cache,
        get_w8a16_linear_config,
        w8a16_linear,
        w8a16_marlin_weight,
    )
    from .m_group_gemm_nt_masked import (
        m_grouped_w4a8_gemm_nt_masked,
        m_grouped_w4a8_gemm_nt_masked_hipc,
        m_grouped_w8a8_gemm_nt_masked,
        m_grouped_i8_gemm_nt_masked,
        m_grouped_w8a8_gemm_nt_masked_impl,
        m_grouped_w8a8_gemm_nt_masked_gfx936,
    )
    from .m_group_gemm_nt_contiguous import (
        get_m_grouped_fp8_contiguous_config,
        m_grouped_bf16_gemm_nt_contiguous,
        m_grouped_fp8_gemm_nt_contiguous,
        m_grouped_wfp4a16_gemm_nt_contiguous,
        m_grouped_w4a8_gemm_nt_contiguous_hipc,
        m_grouped_i8_gemm_nt_contiguous,
        m_grouped_i8_gemm_nt_contiguous_nopad,
    )
else:

    from .utils import gfx, DEEPGEMM_ROOT_DIR, DEEPGEMM_ASM_DIR
    from .m_group_gemm_nt_contiguous import m_grouped_i8_gemm_nt_contiguous, m_grouped_i8_gemm_nt_contiguous_nopad, \
          m_grouped_bf16_gemm_nt_contiguous, m_grouped_fp8_gemm_nt_contiguous, m_grouped_wfp4a16_gemm_nt_contiguous, m_grouped_w4a8_gemm_nt_contiguous_hipc, get_m_grouped_fp8_contiguous_config
    from .moe_marlin_prefill import (
        assert_moe_prefill_down_fp8_shapes,
        moe_w8a8_marlin_prefill_down_fp8,
    )
    from .m_group_gemm_nt_masked import m_grouped_w4a8_gemm_nt_masked, m_grouped_w4a8_gemm_nt_masked_hipc, \
          m_grouped_w8a8_gemm_nt_masked, m_grouped_i8_gemm_nt_masked, m_grouped_w8a8_gemm_nt_masked_gfx936, m_grouped_w8a8_gemm_nt_masked_gfx936_pack_permute, m_grouped_bf16_gemm_nt_masked, m_grouped_w8a8_gemm_nt_masked_impl, m_grouped_w8a8_gemm_nt_compact_masked
    from .gemm import fp8_gemm, fp8_gemm_hipc
    from .m_group_gemm import m_grouped_fp8_gemm_nt_masked, m_grouped_fp8_gemm_nt_masked_tensor_wise
    from .w8a16_linear import (
        clear_w8a16_dequant_cache,
        get_w8a16_linear_config,
        w8a16_linear,
        w8a16_marlin_weight,
    )
    from .op import einsum, fp8_einsum
    from .tf32_hc_pernorm_gemm import tf32_hc_pernorm_gemm

from .legacy_compat import *
from .m_group_gemm import (
    grouped_gemm_w4a16_nt_masked_entry,
    grouped_gemm_wfp4a16_nt_masked_entry,
    grouped_gemm_w4a16_nt_masked_triton_ref_entry,
    moe_w4a16_groupwise_triton,
    w4a16_marlin_weight,
)
