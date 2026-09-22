import torch
from typing import Optional, Dict, Any, List, NamedTuple
import functools
import os
import math
from . import op


def tf32_hc_pernorm_gemm(a: torch.Tensor, 
                         b: torch.Tensor, 
                        d: torch.Tensor, 
                        sqr_sum: torch.Tensor, 
                        num_splits: Optional[int] = None):
    op.tf32_hc_pernorm_gemm(a, b, d, sqr_sum, num_splits)