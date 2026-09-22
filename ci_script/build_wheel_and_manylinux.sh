#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Building deepgemm wheel in ${repo_root}"
cd "${repo_root}"

PYTORCH_ROCM_ARCH="gfx936;gfx938" python setup.py bdist_wheel

wheel_count="$(find dist -maxdepth 1 -type f -name '*.whl' | wc -l)"
if [[ "${wheel_count}" -ne 1 ]]; then
  echo "Expected exactly one wheel in dist/ after build_wheel_and_manylinux.sh, found ${wheel_count}" >&2
  exit 1
fi

shopt -s nullglob
wheels=(dist/deepgemm*.whl)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one deepgemm wheel in dist/, found ${#wheels[@]}" >&2
  exit 1
fi

torch_path="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")"
if [[ -z "${torch_path}" || ! -d "${torch_path}/lib" ]]; then
  echo "Unable to locate the PyTorch library directory" >&2
  exit 1
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${torch_path}/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${torch_path}/lib"
fi

platform="manylinux_$(ldd --version | awk 'NR==1 {print $NF}' | tr '.' '_')_x86_64"
echo "Repairing ${wheels[0]} for ${platform}"
auditwheel show "${wheels[0]}"

excludes=(
  --exclude libgalaxyhip.so.5
  --exclude libMIOpen.so.1
  --exclude librccl.so.1
  --exclude libhipblas.so.0
  --exclude libhipfft.so
  --exclude libhiprand.so.1
  --exclude libhipsolver.so.0
  --exclude libhipsparse.so.0
  --exclude libhipnn.so
  --exclude librocblas.so.0
  --exclude librocsolver.so.0
  --exclude librocfft.so.0
  --exclude librocrand.so.1
  --exclude librocsparse.so.0
  --exclude librocm_smi64.so.2
  --exclude librocfft-device-0.so.0
  --exclude librocfft-device-1.so.0
  --exclude librocfft-device-2.so.0
  --exclude librocfft-device-3.so.0
  --exclude libc10.so
  --exclude libc10_hip.so
  --exclude libtorch_cpu.so
  --exclude libtorch_hip.so
  --exclude libtorch_python.so
  --exclude libtorch.so
  --exclude libtorchaudio_ffmpeg4.so
  --exclude libtorchaudio_ffmpeg5.so
  --exclude libtorchaudio_ffmpeg6.so
  --exclude libtorchaudio.so
  --exclude libtorchaudio_sox.so
  --exclude _torchaudio_ffmpeg4.so
  --exclude _torchaudio_ffmpeg5.so
  --exclude _torchaudio_ffmpeg6.so
  --exclude _torchaudio.so
  --exclude _torchaudio_sox.so
  --exclude libavcodec.so.58
  --exclude libavcodec.so.59
  --exclude libavcodec.so.60
  --exclude libavdevice.so.58
  --exclude libavdevice.so.59
  --exclude libavdevice.so.60
  --exclude libavfilter.so.7
  --exclude libavfilter.so.8
  --exclude libavfilter.so.9
  --exclude libavformat.so.58
  --exclude libavformat.so.59
  --exclude libavformat.so.60
  --exclude libavutil.so.56
  --exclude libavutil.so.57
  --exclude libavutil.so.58
  --exclude libsox.so
  --exclude libomp.so
  --exclude libhipblaslt.so.0
  --exclude libhipblas.so.2
  --exclude libhipfft.so.0
  --exclude libhipsparse.so.1
  --exclude librocblas.so.4
  --exclude librocsparse.so.1
)

auditwheel repair --plat "${platform}" --strip "${excludes[@]}" -w dist "${wheels[0]}"

repaired=(dist/*manylinux*.whl)
if [[ "${#repaired[@]}" -ne 1 ]]; then
  echo "Expected exactly one repaired manylinux deepgemm wheel in dist/, found ${#repaired[@]}" >&2
  exit 1
fi
