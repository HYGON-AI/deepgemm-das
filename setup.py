import torch
from typing import Dict, List, Optional, Sequence, Union
import glob
import os
import shutil
import shlex
import subprocess
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME
from torch.utils.hipify import hipify_python
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from pkg_resources import packaging  # type: ignore[attr-defined]
from get_version import get_version, get_dtk_version

hipify_python.HIPIFY_HEADER_FILE = ""

dtk_version = get_dtk_version(ROCM_HOME)
COMMON_SOURCES = [
    './csrc/common/deepgemm_ops.cpp',
    './csrc/common/000_deep_gemm_torch_library_root.cu',
]

GFX936_GFX938_SHARED_SOURCES = [
    './csrc/arch/gfx936_gfx938_shared/asm_m_grouped_w16a16_gemm_nt_contiguous.cu',
    './csrc/arch/gfx936_gfx938_shared/asm_m_grouped_w8a8_gemm_nt_contiguous.cu',
    './csrc/arch/gfx936_gfx938_shared/low_latency_grouped_gemm.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_fp8_gemm_nt_maksed_low_latency.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_bf16_gemm_nt_masked.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_bf16_gemm_nt_masked_hipc.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_w4a16_nt_masked.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_contiguous_hipc.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_masked.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_masked_hipc.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_w8a8_gemm_nt_masked.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_w8a8_gemm_nt_contiguous_gfx936.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_wfp4a16_nt_contiguous.cu',
    './csrc/arch/gfx936_gfx938_shared/m_grouped_wfp4a16_nt_masked.cu',
    './csrc/arch/gfx936_gfx938_shared/w8a16_linear.cu',

]

GFX92a_SOURCES = [
    './csrc/arch/gfx92a/wfp4a16_contiguous_launch.cu',
    './csrc/arch/gfx92a/wfp4a16_masked_launch.cu',
    './csrc/arch/gfx92a/m_grouped_fp8_gemm_nt_contiguous.cu',
    # # 未完成
    # './csrc/arch/gfx92a/m_grouped_w4a8_gemm_nt_contiguous.cu',
    # # 测试
    # './csrc/arch/gfx936_gfx938_shared/m_grouped_w8a8_gemm_nt_masked_hipc_persistent_kernel.cu',
]


GFX936_SOURCES = [
    './csrc/arch/gfx936/wfp4a16_contiguous_launch.cu',
    './csrc/arch/gfx936/wfp4a16_masked_launch.cu',
]


GFX938_SOURCES = [
    './csrc/arch/gfx938/asm_m_grouped_fp8_gemm_nt_contiguous.cu',
    './csrc/arch/gfx938/einsum.cu',
    './csrc/arch/gfx938/fp8_gemm.cu',
    './csrc/arch/gfx938/fp8_gemm_hipc.cu',
    './csrc/arch/gfx938/m_grouped_fp8_gemm_nt_contiguous.cu',
    './csrc/arch/gfx938/m_grouped_fp8_gemm_nt_contiguous_optimized.cu',
    './csrc/arch/gfx938/m_grouped_i8_gemm_nt_contiguous_optimized.cu',
    './csrc/arch/gfx938/moe_w8a8_marlin_prefill_down_fp8.cu',
    './csrc/arch/gfx938/tf32_hc_pernorm_gemm.cu',
    './csrc/arch/gfx938/wfp4a16_contiguous_launch.cu',
    './csrc/arch/gfx938/wfp4a16_masked_launch.cu',
]


SOURCE_SCAN_PATTERNS = [
    './csrc/*.cpp',
    './csrc/common/*.cpp',
    './csrc/common/*.cu',
    './csrc/arch/**/*.cpp',
    './csrc/arch/**/*.cu',
]

SOURCE_GROUP_ORDER = [
    'common',
    'gfx936_gfx938_shared',
    'gfx92a',
    'gfx936',
    'gfx938',
]

SOURCE_GROUPS: Dict[str, List[str]] = {
    'common': COMMON_SOURCES,
    'gfx936_gfx938_shared': GFX936_GFX938_SHARED_SOURCES,
    'gfx936': GFX936_SOURCES,
    'gfx938': GFX938_SOURCES,
    'gfx92a': GFX92a_SOURCES,
}

SOURCE_GROUP_ARCHS: Dict[str, Optional[Sequence[str]]] = {
    'common': None,
    'gfx936_gfx938_shared': ('gfx92a', 'gfx936', 'gfx938'),
    'gfx936': ('gfx936',),
    'gfx938': ('gfx938',),
    'gfx92a': ('gfx92a',),
}

_SOURCE_ARCHS_BY_PATH: Dict[str, List[str]] = {}
_SOURCE_GROUP_BY_PATH: Dict[str, str] = {}


def _find_rocm_home() -> Optional[str]:
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        try:
            pipe_hipcc = subprocess.Popen(
                ["which hipcc | xargs readlink -f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            hipcc, _ = pipe_hipcc.communicate()
            rocm_home = os.path.dirname(os.path.dirname(hipcc.decode().rstrip('\r\n')))
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        except Exception:
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'")
    return rocm_home

def _normalize_rocm_arch(arch: Optional[str]) -> Optional[str]:
    if not arch:
        return None
    arch = arch.split(':', 1)[0]
    return arch if arch.startswith('gfx') else None


def _current_rocm_arch() -> Optional[str]:
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return _normalize_rocm_arch(getattr(props, 'gcnArchName', None))
    except Exception:
        pass
    return None


def _default_rocm_archs() -> List[str]:
    dtk_ver = int(str(dtk_version)[:4])
    supported_archs = ['gfx936']
    if dtk_ver >= 2604:  # dtk>=2604 can support gfx938
        supported_archs.append('gfx938')
    current_arch = _current_rocm_arch()
    # if current_arch == 'gfx92a':
    #     print('Current ROCm arch is gfx92a; building DeepGEMM default target archs '
    #           f'{supported_archs} instead.')
    #     return supported_archs
    if current_arch:
        return [current_arch]
    return supported_archs

def _arch_probe_compiler() -> Optional[str]:
    rocm_home = ROCM_HOME or _find_rocm_home()
    candidates = []
    if rocm_home:
        candidates.append(os.path.join(rocm_home, 'bin', 'aicc'))
        candidates.append(os.path.join(rocm_home, 'bin', 'hipcc'))
    candidates += ['aicc', 'hipcc']
    for compiler in candidates:
        resolved = compiler if os.path.isabs(compiler) else shutil.which(compiler)
        if resolved and os.path.exists(resolved):
            return resolved
    return None


_SUPPORTED_ROCM_ARCH_CACHE = {}


def _can_compile_rocm_arch(arch: str) -> bool:
    if arch in _SUPPORTED_ROCM_ARCH_CACHE:
        return _SUPPORTED_ROCM_ARCH_CACHE[arch]
    compiler = _arch_probe_compiler()
    if compiler is None:
        return True
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, 'probe.hip')
        obj = os.path.join(tmpdir, 'probe.o')
        with open(src, 'w', encoding='utf-8') as f:
            f.write('#include <hip/hip_runtime.h>\n__global__ void probe() {}\n')
        cmd = [compiler, f'--offload-arch={arch}', '-x', 'hip', '-c', src, '-o', obj]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(f"Skipping default ROCm arch {arch}: current toolchain cannot compile it.")
        _SUPPORTED_ROCM_ARCH_CACHE[arch] = False
        return False
    _SUPPORTED_ROCM_ARCH_CACHE[arch] = True
    return True


def _resolve_rocm_archs() -> List[str]:
    env_archs = os.environ.get('PYTORCH_ROCM_ARCH')
    if env_archs:
        return [arch for arch in (_normalize_rocm_arch(arch) for arch in env_archs.split(';')) if arch]

    archs = [arch for arch in _default_rocm_archs() if _can_compile_rocm_arch(arch)]
    if not archs:
        archs = ['gfx936']
    return archs


def _normalize_source_path(path: str) -> str:
    return os.path.normpath(path).replace(os.sep, '/')


def _source_path_variants(path: str) -> List[str]:
    abs_path = os.path.abspath(path)
    variants = [_normalize_source_path(abs_path)]
    root, ext = os.path.splitext(abs_path)
    if ext == '.cu':
        variants.append(_normalize_source_path(root + '.hip'))
    elif ext == '.hip':
        variants.append(_normalize_source_path(root + '.cu'))
    return variants


def _append_unique(target: List[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _validate_source_groups(
    build_only_w8a16: bool = False,
    build_only_w4a8_masked: bool = False,
) -> None:
    seen: Dict[str, str] = {}
    duplicates = []
    expected_sources = set()
    for group in SOURCE_GROUP_ORDER:
        for source in SOURCE_GROUPS[group]:
            norm_source = _normalize_source_path(source)
            if norm_source in seen:
                duplicates.append(f'{norm_source} in {seen[norm_source]} and {group}')
            seen[norm_source] = group
            expected_sources.add(norm_source)

    actual_sources = set(
        _normalize_source_path(source)
        for pattern in SOURCE_SCAN_PATTERNS
        for source in glob.glob(pattern, recursive=True)
    )

    if build_only_w8a16:
        expected_sources = {
            _normalize_source_path('./csrc/common/deepgemm_ops.cpp'),
            _normalize_source_path('./csrc/arch/gfx936_gfx938_shared/w8a16_linear.cu'),
        }
        actual_sources = expected_sources
    elif build_only_w4a8_masked:
        expected_sources = {
            _normalize_source_path('./csrc/common/deepgemm_ops.cpp'),
            _normalize_source_path('./csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_masked.cu'),
            _normalize_source_path('./csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_masked_hipc.cu'),
            _normalize_source_path('./csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_contiguous_hipc.cu'),
        }
        actual_sources = expected_sources

    missing_sources = sorted(actual_sources - expected_sources)
    stale_sources = sorted(expected_sources - actual_sources)
    if duplicates or missing_sources or stale_sources:
        details = []
        if duplicates:
            details.append('duplicate sources: ' + ', '.join(duplicates))
        if missing_sources:
            details.append('unclassified sources: ' + ', '.join(missing_sources))
        if stale_sources:
            details.append('missing source files: ' + ', '.join(stale_sources))
        raise RuntimeError('DeepGEMM source groups are out of sync; ' + '; '.join(details))


def _active_kernel_archs() -> set:
    active_archs = set()
    for group in SOURCE_GROUP_ORDER:
        if group == 'common' or not SOURCE_GROUPS[group]:
            continue
        allowed_archs = SOURCE_GROUP_ARCHS[group]
        if allowed_archs is not None:
            active_archs.update(allowed_archs)
    return active_archs


def _validate_requested_archs(target_archs: List[str]) -> None:
    active_archs = _active_kernel_archs()
    if not any(arch in active_archs for arch in target_archs):
        raise RuntimeError(
            f'DeepGEMM has no active kernel source group for target archs {target_archs}.'
        )


def _resolve_group_archs(group: str, target_archs: List[str]) -> List[str]:
    allowed_archs = SOURCE_GROUP_ARCHS[group]
    if allowed_archs is None:
        return list(target_archs)
    return [arch for arch in target_archs if arch in allowed_archs]


def _remember_source_group(source: str, group: str, archs: List[str]) -> None:
    for variant in _source_path_variants(source):
        _SOURCE_ARCHS_BY_PATH[variant] = list(archs)
        _SOURCE_GROUP_BY_PATH[variant] = group


def _build_op_sources_and_archs(
    build_only_w8a16: bool,
    build_only_w4a8_masked: bool,
    target_archs: List[str],
) -> List[str]:
    global _SOURCE_ARCHS_BY_PATH, _SOURCE_GROUP_BY_PATH
    _validate_source_groups(build_only_w8a16, build_only_w4a8_masked)
    _SOURCE_ARCHS_BY_PATH = {}
    _SOURCE_GROUP_BY_PATH = {}

    if build_only_w8a16:
        op_files = [
            './csrc/arch/gfx936_gfx938_shared/w8a16_linear.cu',
            './csrc/common/deepgemm_ops.cpp',
        ]
        for source in op_files:
            _remember_source_group(source, 'build_only_w8a16', list(target_archs))
        return op_files

    if build_only_w4a8_masked:
        op_files = [
            './csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_masked.cu',
            './csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_masked_hipc.cu',
            './csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_contiguous_hipc.cu',
            './csrc/common/deepgemm_ops.cpp',
        ]
        for source in op_files:
            _remember_source_group(source, 'build_only_w4a8_masked', list(target_archs))
        return op_files

    _validate_requested_archs(target_archs)
    op_files: List[str] = []
    for group in SOURCE_GROUP_ORDER:
        group_sources = SOURCE_GROUPS[group]
        if not group_sources:
            continue
        group_archs = _resolve_group_archs(group, target_archs)
        if not group_archs:
            continue
        _append_unique(op_files, group_sources)
        for source in group_sources:
            _remember_source_group(source, group, group_archs)
    return op_files


def _get_source_archs(path: str) -> List[str]:
    for variant in _source_path_variants(path):
        if variant in _SOURCE_ARCHS_BY_PATH:
            return _SOURCE_ARCHS_BY_PATH[variant]
    return _resolve_rocm_archs()


def _get_source_group(path: str) -> str:
    for variant in _source_path_variants(path):
        if variant in _SOURCE_GROUP_BY_PATH:
            return _SOURCE_GROUP_BY_PATH[variant]
    return 'default'


def _get_rocm_arch_flags_for_archs(archs: Sequence[str], cflags: Optional[List[str]] = None) -> List[str]:
    if cflags is not None:
        for flag in cflags:
            if 'amdgpu-target' in flag or 'offload-arch' in flag:
                return ['-fno-gpu-rdc', '--gpu-max-threads-per-block=1024']
    flags = ['--offload-arch=%s' % arch for arch in archs]
    flags += ['-fno-gpu-rdc']
    flags += ['--gpu-max-threads-per-block=1024']
    return flags


def _get_rocm_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    return _get_rocm_arch_flags_for_archs(_resolve_rocm_archs(), cflags)

def _get_rocm_arch_flags_marlin(cflags: Optional[List[str]] = None) -> List[str]:
    flags = ['--offload-arch=%s' % arch for arch in _resolve_rocm_archs()]
    flags +=['--gpu-max-threads-per-block=512']
    flags +=['-mllvm']
    flags +=['-support-768-vgprs=true']
    # flags +=['--save-temps']
    return flags

ROCM_HOME = _find_rocm_home()
IS_HIP_EXTENSION = True if ((ROCM_HOME is not None) and (torch.version.hip is not None)) else False
COMMON_HIP_FLAGS = [
    '-fPIC',
    '-D__HIP_PLATFORM_HCC__=1',
    '-DUSE_ROCM=1',
]

COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
]

def is_ninja_available():
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability():
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions")


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext


def _join_rocm_home(*paths) -> str:
    if ROCM_HOME is None:
        raise EnvironmentError('ROCM_HOME environment variable is not set. ')
    return os.path.join(ROCM_HOME, *paths)

def _find_hip_compiler() -> str:
    for compiler in [
        os.environ.get('AICC'),
        shutil.which('aicc'),
        '/opt/dtk/bin/aicc',
        '/opt/dtk/aillvm/bin/aicc',
        os.environ.get('HIPCC'),
        shutil.which('hipcc'),
    ]:
        if compiler and os.path.exists(compiler):
            return compiler
    return _join_rocm_home('bin', 'hipcc')

def _write_ninja_file(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags, sources,
                      objects, ldflags, library_target, with_cuda,
                      cuda_arch_flags_by_source: Optional[Dict[str, List[str]]] = None) -> None:
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_arch_flags_by_source = {
        _normalize_source_path(source): sanitize_flags(flags)
        for source, flags in (cuda_arch_flags_by_source or {}).items()
    }
    ldflags = sanitize_flags(ldflags)
    assert len(sources) == len(objects)
    assert len(sources) > 0
    compiler = os.environ.get('CXX', 'c++')
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda:
        if IS_HIP_EXTENSION:
            nvcc = _find_hip_compiler()
        config.append(f'nvcc = {nvcc}')
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')
    sources = [os.path.abspath(file) for file in sources]
    compile_rule = ['rule compile']
    compile_rule.append('  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
    compile_rule.append('  depfile = $out.d')
    compile_rule.append('  deps = gcc')
    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        required_cuda_version = packaging.version.parse('10.2')
        has_cuda_version = torch.version.cuda is not None
        if has_cuda_version and packaging.version.parse(torch.version.cuda) >= required_cuda_version:
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
        elif IS_HIP_EXTENSION:
            # hipcc/aicc 都基于 clang，支持 gcc 风格的 -MMD -MF
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            nvcc_gendeps = '-MMD -MF $out.d'
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags $cuda_arch_flags')
    build = []
    for source_file, object_file in zip(sources, objects):
        source_file_abs = source_file
        is_cuda_source = _is_cuda_file(source_file_abs) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        source_file = source_file_abs.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')
        if is_cuda_source:
            norm_source = _normalize_source_path(source_file_abs)
            arch_flags = cuda_arch_flags_by_source.get(norm_source, [])
            build.append(f'  cuda_arch_flags = {" ".join(arch_flags)}')
            build.append(f'  source_group = {_get_source_group(source_file_abs)}')
    if library_target is not None:
        link_rule = ['rule link']
        link_rule.append('  command = $cxx $in $ldflags -o $out')
        link = [f'build {library_target}: link {" ".join(objects)}']
        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write(f'{lines}\n\n')

def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...')
        return int(max_jobs)
    if verbose:
        print('Allowing ninja to set a default number of workers... ')
    return None

def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fileno = 1
        subprocess.run(command, stdout=stdout_fileno if verbose else subprocess.PIPE, stderr=subprocess.STDOUT,
                       cwd=build_directory, check=True, env=env)
    except subprocess.CalledProcessError as e:
        _, error, _ = sys.exc_info()
        message = error_prefix
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e

def _write_ninja_file_and_compile_objects(sources: List[str], objects, cflags, post_cflags, cuda_cflags,
                                          cuda_post_cflags, build_directory: str, verbose: bool,
                                          with_cuda: Optional[bool],
                                          cuda_arch_flags_by_source: Optional[Dict[str, List[str]]] = None) -> None:
    verify_ninja_availability()
    compiler = os.environ.get('CXX', 'c++')
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...')
    _write_ninja_file(path=build_file_path, cflags=cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags,
                      cuda_post_cflags=cuda_post_cflags, sources=sources, objects=objects, ldflags=None,
                      library_target=None, with_cuda=with_cuda,
                      cuda_arch_flags_by_source=cuda_arch_flags_by_source)
    if verbose:
        print('Compiling objects...')
    _run_ninja_build(
        build_directory,
        verbose,
        error_prefix='Error compiling objects for extension')

class BuildReleaseExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super(BuildReleaseExtension, self).__init__(*args, **kwargs)

    def _add_gnu_cpp_abi_flag(self, extension) -> None:
        """添加与当前 PyTorch 一致的 _GLIBCXX_USE_CXX11_ABI。PyTorch 2.9+ 可能已从 BuildExtension 移除此方法，故在子类中实现。"""
        abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
        if abi is not None:
            self._add_compile_flag(extension, "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(abi)))

    def build_extensions(self) -> None:
        self._check_abi()
        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)
        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []
            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            # PyTorch 2.9+ (pybind11 v3) 不再暴露 _PYBIND11_*，用 getattr 默认值兼容
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}", None)
                if val is not None:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']

        def append_std17_if_no_std_present(cflags) -> None:
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++17'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def convert_to_absolute_paths_inplace(paths):
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0,
                                    extra_preargs=None, extra_postargs=None, depends=None):
            output_dir = os.path.abspath(output_dir)
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)
            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            if (debug):
                print("debug mode")
            else:
                if('-g' in extra_cc_cflags):
                    extra_cc_cflags.remove('-g')
                if('-Wall' in extra_cc_cflags):
                    extra_cc_cflags.remove('-Wall')
                print("release mode")

            other_sources = []
            for s in sources:
                abs_path = os.path.abspath(s)
                other_sources.append(s)
            
            other_objects = [obj for src, obj in zip(sources, objects) if src in other_sources]
            
            with_cuda = any(map(_is_cuda_file, sources))
            
            if other_sources:
                if isinstance(extra_postargs, dict):
                    post_cflags = extra_postargs['cxx']
                else:
                    post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    post_cflags = COMMON_HIP_FLAGS + post_cflags
                append_std17_if_no_std_present(post_cflags)
                
                cuda_post_cflags = None
                cuda_arch_flags_by_source = None
                if with_cuda:
                    if isinstance(extra_postargs, dict):
                        cuda_post_cflags = extra_postargs['nvcc']
                    else:
                        cuda_post_cflags = list(extra_postargs)
                    if IS_HIP_EXTENSION:
                        cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                    append_std17_if_no_std_present(cuda_post_cflags)
                    if IS_HIP_EXTENSION:
                        cuda_arch_flags_by_source = {}
                        for source in other_sources:
                            if not _is_cuda_file(source):
                                continue
                            source_archs = _get_source_archs(source)
                            arch_flags = [
                                shlex.quote(flag)
                                for flag in _get_rocm_arch_flags_for_archs(source_archs, cuda_post_cflags)
                            ]
                            for variant in _source_path_variants(source):
                                cuda_arch_flags_by_source[variant] = arch_flags
                    cuda_cflags = [shlex.quote(f) for f in common_cflags]
                    cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]
                
                _write_ninja_file_and_compile_objects(
                    sources=other_sources, 
                    objects=other_objects,
                    cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                    post_cflags=[shlex.quote(f) for f in post_cflags],
                    cuda_cflags=cuda_cflags if with_cuda else None,
                    cuda_post_cflags=cuda_post_cflags if with_cuda else None,
                    build_directory=output_dir,
                    verbose=True,
                    with_cuda=with_cuda,
                    cuda_arch_flags_by_source=cuda_arch_flags_by_source
                )
            
            # 返回所有目标文件
            return other_objects

        self.compiler.compile = unix_wrap_ninja_compile
        build_ext.build_extensions(self)

def torch_version_over110():
    version=torch.__version__.split('.')
    return int(version[0])*100+int(version[1])>110

def get_extensions():
    extensions = []
    include_dirs = []
    library_dirs = []
    define_macros = []
    # 显式设置 TORCH_EXTENSION_NAME 宏，确保编译时正确设置
    define_macros.append(('TORCH_EXTENSION_NAME', 'op'))
    build_only_w8a16 = os.environ.get("DEEPGEMM_BUILD_ONLY_W8A16") == "1"
    build_only_w4a8_masked = os.environ.get("DEEPGEMM_BUILD_ONLY_W4A8_MASKED") == "1"
    if build_only_w8a16 and build_only_w4a8_masked:
        raise RuntimeError(
            "DEEPGEMM_BUILD_ONLY_W8A16 and "
            "DEEPGEMM_BUILD_ONLY_W4A8_MASKED are mutually exclusive"
        )
    if build_only_w8a16:
        define_macros.append(("DEEPGEMM_BUILD_ONLY_W8A16", "1"))
    if build_only_w4a8_masked:
        define_macros.append(("DEEPGEMM_BUILD_ONLY_W4A8_MASKED", "1"))

    target_archs = _resolve_rocm_archs()
    enabled_archs = set(target_archs) if build_only_w8a16 else set()
    if not build_only_w8a16:
        for group in SOURCE_GROUP_ORDER:
            if SOURCE_GROUPS[group]:
                enabled_archs.update(_resolve_group_archs(group, target_archs))
    if "gfx92a" in enabled_archs:
        define_macros.append(("DEEPGEMM_ENABLE_GFX92A", "1"))
    if "gfx936" in enabled_archs:
        define_macros.append(("DEEPGEMM_ENABLE_GFX936", "1"))
    if "gfx938" in enabled_archs:
        define_macros.append(("DEEPGEMM_ENABLE_GFX938", "1"))
    # 全局默认编译选项（移除链接选项，链接选项应该在 extra_link_args 中）
    extra_compile_args = {'cxx': ['-O3', '-w'], 'nvcc': ['-O3', '-w', '-mllvm', '-support-768-vgprs=true', '-DHIP_ENABLE_WARP_SYNC_BUILTINS']}
    if torch_version_over110():
        extra_compile_args['nvcc'].append('-DTORCH_VERSION_OVER_110')
    
    # 链接选项
    extra_link_args = ['-ldl', '-lrocblas', '-lhipblaslt', '-lhipblas']
    
    op_files = _build_op_sources_and_archs(
        build_only_w8a16, build_only_w4a8_masked, target_archs)
    print(f'DeepGEMM target ROCm archs: {target_archs}')
    if build_only_w8a16:
        print(f'DeepGEMM source group build_only_w8a16: archs={target_archs}, sources={len(op_files)}')
    elif build_only_w4a8_masked:
        print(f'DeepGEMM source group build_only_w4a8_masked: archs={target_archs}, sources={len(op_files)}')
    else:
        for group in SOURCE_GROUP_ORDER:
            group_sources = SOURCE_GROUPS[group]
            if not group_sources:
                continue
            group_archs = _resolve_group_archs(group, target_archs)
            if not group_archs:
                print(f'DeepGEMM source group {group}: skipped, sources={len(group_sources)}')
                continue
            print(
                f'DeepGEMM source group {group}: '
                f'archs={group_archs}, sources={len(group_sources)}'
            )
    extension = CUDAExtension
    rocm_home=_find_rocm_home()
    include_dirs.append(rocm_home+'/hiprand/include')
    include_dirs.append(rocm_home+'/hiprand/include')
    include_dirs.append(rocm_home+'/rocrand/include')
    include_dirs.append(os.path.abspath('./3rdparty/hipblaslt-install/include'))
    include_dirs.append(rocm_home+'/hipblas/include')
    include_dirs.append(os.path.abspath('./csrc/include/'))
    include_dirs.append(os.path.abspath('./csrc/arch/gfx936_gfx938_shared/include/'))
    
    # 添加库目录
    library_dirs.append(os.path.abspath('./3rdparty/hipblaslt-install/include'))
    if rocm_home:
        library_dirs.append(rocm_home+'/lib')
    
    ext_ops = extension(
        name="deepgemm.op",
        sources=op_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

    extensions.append(ext_ops)
    return extensions

def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"{os.environ['PYTORCH_VERSION']}"
    return torch.__version__

setup(
    name='deepgemm',
    version=get_version(ROCM_HOME),
    description='DeepGEMM Library',
    keywords='GEMM',
    packages=find_packages(),
    include_package_data=False,
    package_data={
        'deepgemm': [
            "hsa/*",
            "hsa/*/*",
            "hsa/*/*/*",
            "hsa/*/*/*/*",
            "hsa/*/*/*/*/*",
            "configs/*.json",
            "configs/*/*/*.json",
        ]
    },
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildReleaseExtension
    },
    zip_safe=False,
    install_requires=[
         "torch==2.10.0",
     ],
)
