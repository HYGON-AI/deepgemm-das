# Third-party notices

This release preserves original copyright and license declarations. The root
MIT license does not replace a third-party or file-level Apache-2.0 declaration.
The project owner has authorized publication of the internal MIT contributions
and precompiled kernels. Repository-level contribution attribution is recorded
in [COPYRIGHT.md](COPYRIGHT.md).

## Fixed source and public baseline

| Item | Reference |
| --- | --- |
| Distributed source snapshot | `795b1b20d61b1716eb9cc2e3efd04c03a57d9b41` |
| Existing public release used for synchronization | `3910d3a89dcde925639e9997cb930370f00d2066` |
| DeepGEMM upstream repository | https://github.com/deepseek-ai/DeepGEMM.git |
| Recorded DeepGEMM upstream baseline | `main`, `559d79fb6994a58b8a15b4b93bf13ccc16edf247` |
| DeepGEMM license | MIT; `Copyright (c) 2025 DeepSeek`; full text in `LICENSE` |

The recorded DeepGEMM baseline is inherited from the existing public release.
This synchronization retains the adapted operator implementations and does not
replace them with a new DeepSeek upstream checkout.

## Bundled components

| Component and local path | Version / fixed content | License and attribution |
| --- | --- | --- |
| hipBLASLt, `3rdparty/hipblaslt-install-dtk2504.tar.gz` | DTK 25.04 distribution; embedded hipBLASLt `0.10.0`; SHA-256 `6a4cb978b21b1fddfc0f1fca085ce5b87f02805c2bbce7b57c6b12625a7cad8a` | Original MIT notice, `Copyright (C) 2024 Advanced Micro Devices, Inc.`; inside the archive at `hipblaslt-install/share/doc/hipblaslt/LICENSE.md`, also reproduced unchanged in `LICENSES/hipBLASLt-MIT`. |
| hipBLASLt, `3rdparty/hipblaslt-install-dtk2604.tar.gz` | DTK 26.04 distribution; embedded hipBLASLt `0.10.0`; SHA-256 `8ee233fa82b2c885bec29e00a48be6de94beda0be28cd11acc67acf8513a05b8` | Original MIT notice retained as above. |
| OPUS, `csrc/include/opus/opus.hpp` | Vendored file pinned by the distributed source snapshot above | Original `MIT License` and `Copyright (C) 2025-2026 carlus.huang@amd.com` retained. Standard MIT permission text is included in `LICENSES/OPUS-MIT`. |
| AMD-derived helpers, `csrc/include/hip_common.h`, `csrc/common/deepgemm_ops.cpp` | Vendored/adapted files pinned by the distributed source snapshot | Original MIT SPDX and `Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.` retained. HYGON modifications are acknowledged at repository level. |
| Apache-2.0 operator implementations, including MQA and selected GEMM files | Original file-level declarations retained in the distributed snapshot | Apache-2.0, full text in `LICENSES/Apache-2.0`. `csrc/arch/wrapper/mqa_logits_dispatch.cu` retains its attribution to `tensor_scores/csrc/tensor_ops/backends/score_api.cu`. HYGON copyright and existing adaptation statements are retained where present. |
| Precompiled kernels, `deepgemm/hsa/**/*.co` | Original binaries retained from the fixed source snapshot | Publication of the internal kernels is authorized by the project owner under MIT, subject to any component-specific original declaration. No binary contents were changed for this release. |
| Generated kernel includes and tuning configuration | Original generated content retained from the fixed source snapshot | No generated file was mechanically assigned a new copyright header. Repository-level internal contribution authorization applies while existing original declarations remain applicable. |

Both hipBLASLt archives are unchanged from the existing public release. Exact
external source commits for the DTK distributions, OPUS, AMD-derived helpers,
and the referenced tensor_scores base were not included in the supplied source
materials. The fixed distributed snapshot and archive hashes identify the
actual content shipped here; they are not represented as verified external
upstream commit mappings.

## External environment and optional dependencies

`setup.py` declares the release runtime dependency `torch==2.13.0`, as confirmed
by the project owner. Use the approved HIP-enabled distribution matching the
target HCU/DTK environment. PyTorch and platform distributions retain their own
licenses and notices; they are not relicensed by this repository.

The source additionally references the following external packages/tools. They
are not bundled by this source synchronization, except for the hipBLASLt archives
listed above. This list does not add new mandatory runtime dependencies.

| Scope | Components |
| --- | --- |
| Platform build/runtime environment | DTK/HIP, rocBLAS, hipBLAS |
| Optional/reference MegaMoE paths | `deep_ep`, `lightop`, `rocshmem` |
| Operator/test paths | `aiter`, `lmslim`, `vllm`, `triton` |
| Python support/testing | `numpy`, `pandas`, `pytest`, PyYAML (`yaml`) |
| Build/repair/test tooling | `setuptools`, `packaging`, `wheel`, `pybind11`, `cmake`, `ninja`, `auditwheel`, `patchelf`, `openpyxl`, `scipy` |

Build and test tool versions follow the selected validation environment and the
existing build scripts. This source package does not include an independently
validated lockfile for every optional tool or external HCU distribution.
