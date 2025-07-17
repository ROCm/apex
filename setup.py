import sys
import warnings
import os
import glob
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
        BuildExtension, 
        CppExtension, 
        CUDAExtension, 
        CUDA_HOME, 
        ROCM_HOME,
        load,
     )

import typing
import shlex

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from op_builder import get_default_compute_capabilities, OpBuilder
from op_builder.all_ops import ALL_OPS, accelerator_name
from op_builder.builder import installed_cuda_version

from accelerator import get_accelerator

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
torch_dir = torch.__path__[0]



# https://github.com/pytorch/pytorch/pull/71881
# For the extensions which have rocblas_gemm_flags_fp16_alt_impl we need to make sure if at::BackwardPassGuard exists.
# It helps the extensions be backward compatible with old PyTorch versions.
# The check and ROCM_BACKWARD_PASS_GUARD in nvcc/hipcc args can be retired once the PR is merged into PyTorch upstream.

context_file = os.path.join(torch_dir, "include", "ATen", "Context.h")
if os.path.exists(context_file):
    lines = open(context_file, 'r').readlines()
    found_Backward_Pass_Guard = False
    found_ROCmBackward_Pass_Guard = False
    for line in lines:
        if "BackwardPassGuard" in line:
            # BackwardPassGuard has been renamed to ROCmBackwardPassGuard
            # https://github.com/pytorch/pytorch/pull/71881/commits/4b82f5a67a35406ffb5691c69e6b4c9086316a43
            if "ROCmBackwardPassGuard" in line:
                found_ROCmBackward_Pass_Guard = True
            else:
                found_Backward_Pass_Guard = True
            break

found_aten_atomic_header = False
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "Atomic.cuh")):
    found_aten_atomic_header = True

def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None or ROCM_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor

def get_rocm_bare_metal_version(rocm_dir):
    raw_output = subprocess.check_output([rocm_dir + "/bin/hipcc", "--version"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("version:") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )

def check_rocm_torch_binary_vs_bare_metal(rocm_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_rocm_bare_metal_version(rocm_dir)
    torch_binary_major = torch.version.hip.split(".")[0]
    torch_binary_minor = torch.version.hip.split(".")[1]

    print("\nCompiling rocm extensions with")
    print(raw_output + "from " + rocm_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )

def raise_if_home_none(global_option: str) -> None:
    if CUDA_HOME is not None or ROCM_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

def get_apex_version():
    cwd = os.path.dirname(os.path.abspath(__file__))
    apex_version_file = os.path.join(cwd, "version.txt")
    if os.path.exists(apex_version_file):
        with open(apex_version_file) as f:
            apex_version = f.read().strip()
    else:
        raise RuntimeError("version.txt file is missing")
    if os.getenv("BUILD_VERSION"):
        apex_version = os.getenv("BUILD_VERSION")
    if os.getenv("DESIRED_CUDA"):
        apex_version += "+" + os.getenv("DESIRED_CUDA")
        if os.getenv("APEX_COMMIT"):
            apex_version += ".git"+os.getenv("APEX_COMMIT")[:8]
    return apex_version

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


def check_cudnn_version_and_warn(global_option: str, required_cudnn_version: int) -> bool:
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    if not (cudnn_available and (cudnn_version >= required_cudnn_version)):
        warnings.warn(
            f"Skip `{global_option}` as it requires cuDNN {required_cudnn_version} or later, "
            f"but {'cuDNN is not available' if not cudnn_available else cudnn_version}"
        )
        return False
    return True

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

print("\n\ntorch.version.hip  = {}\n\n".format(torch.version.hip))
ROCM_MAJOR = int(torch.version.hip.split('.')[0])
ROCM_MINOR = int(torch.version.hip.split('.')[1])

def check_if_rocm_pytorch():
    is_rocm_pytorch = False
    if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    return is_rocm_pytorch

IS_ROCM_PYTORCH = check_if_rocm_pytorch()

if not torch.cuda.is_available() and not IS_ROCM_PYTORCH:
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
            if int(bare_metal_minor) > 0:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
elif not torch.cuda.is_available() and IS_ROCM_PYTORCH:
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, Apex will cross-compile for the same gfx targets\n'
          'used by default in ROCm PyTorch\n')

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
    raise RuntimeError(
        "Apex requires Pytorch 0.4 or newer.\nThe latest stable release can be obtained from https://pytorch.org/"
    )

# cmdclass = {}
ext_modules = []

extras = {}

# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ["-DVERSION_GE_1_1"]
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ["-DVERSION_GE_1_3"]
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ["-DVERSION_GE_1_5"]
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

if not IS_ROCM_PYTORCH:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
else:
    _, bare_metal_version, bare_metal_minor  = get_rocm_bare_metal_version(ROCM_HOME)

if IS_ROCM_PYTORCH and (ROCM_MAJOR >= 6):
    version_dependent_macros += ["-DHIPBLAS_V2"] 


if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    if TORCH_MAJOR == 0:
        raise RuntimeError("--cpp_ext requires Pytorch 1.0 or later, "
                           "found torch.__version__ = {}".format(torch.__version__)
                           )


# ***************************** Op builder **********************

def get_env_if_set(key, default: typing.Any = ""):
    """
    Returns an environment variable if it is set and not "",
    otherwise returns a default value. In contrast, the fallback
    parameter of os.environ.get() is skipped if the variable is set to "".
    """
    return os.environ.get(key, None) or default

def command_exists(cmd):
    if sys.platform == "win32":
        safe_cmd = shlex.split(f'{cmd}')
        result = subprocess.Popen(safe_cmd, stdout=subprocess.PIPE)
        return result.wait() == 1
    else:
        safe_cmd = shlex.split(f"bash -c type {cmd}")
        result = subprocess.Popen(safe_cmd, stdout=subprocess.PIPE)
        return result.wait() == 0


BUILD_OP_PLATFORM = 1 if sys.platform == "win32" else 0
BUILD_OP_DEFAULT = int(get_env_if_set('APEX_BUILD_OPS', BUILD_OP_PLATFORM))
print(f"APEX_BUILD_OPS={BUILD_OP_DEFAULT}")

ext_modules2 = []

def is_env_set(key):
    """
    Checks if an environment variable is set and not "".
    """
    return bool(os.environ.get(key, None))

def op_envvar(op_name):
    assert hasattr(ALL_OPS[op_name], 'BUILD_VAR'), \
        f"{op_name} is missing BUILD_VAR field"
    return ALL_OPS[op_name].BUILD_VAR


def op_enabled(op_name):
    env_var = op_envvar(op_name)
    return int(get_env_if_set(env_var, BUILD_OP_DEFAULT))

install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()

    # If op is requested but not available, throw an error.
    if op_enabled(op_name) and not op_compatible:
        env_var = op_envvar(op_name)
        if not is_env_set(env_var):
            builder.warning(f"Skip pre-compile of incompatible {op_name}; One can disable {op_name} with {env_var}=0")
        continue

    # If op is compatible but install is not enabled (JIT mode).
    if IS_ROCM_PYTORCH and op_compatible and not op_enabled(op_name):
        builder.hipify_extension()

    # If op install enabled, add builder to extensions.
    if op_enabled(op_name) and op_compatible:
        install_ops[op_name] = op_enabled(op_name)
        ext_modules2.append(builder.builder())

print(f'Install Ops={install_ops}')
    
if "--cuda_ext" in sys.argv:
    raise_if_home_none("--cuda_ext")
    
    if not IS_ROCM_PYTORCH:
        check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)
    else:
        check_rocm_torch_binary_vs_bare_metal(ROCM_HOME)

# Write out version/git info.
git_hash_cmd = shlex.split("bash -c \"git rev-parse --short HEAD\"")
git_branch_cmd = shlex.split("bash -c \"git rev-parse --abbrev-ref HEAD\"")
if command_exists('git') and not is_env_set('APEX_BUILD_STRING'):
    try:
        result = subprocess.check_output(git_hash_cmd)
        git_hash = result.decode('utf-8').strip()
        result = subprocess.check_output(git_branch_cmd)
        git_branch = result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"

# Parse the apex version string from version.txt.
version_str = get_apex_version()

# Build specifiers like .devX can be added at install time. Otherwise, add the git hash.
# Example: `APEX_BUILD_STRING=".dev20201022" python -m build --no-isolation`.

# Building wheel for distribution, update version file.
if is_env_set('APEX_BUILD_STRING'):
    # Build string env specified, probably building for distribution.
    with open('build.txt', 'w') as fd:
        fd.write(os.environ['APEX_BUILD_STRING'])
    version_str += os.environ['APEX_BUILD_STRING']
elif os.path.isfile('build.txt'):
    # build.txt exists, probably installing from distribution.
    with open('build.txt', 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source.
    version_str += f'+{git_hash}'

torch_version = ".".join([str(TORCH_MAJOR), str(TORCH_MINOR)])
bf16_support = False
# Set cuda_version to 0.0 if cpu-only.
cuda_version = "0.0"
nccl_version = "0.0"
# Set hip_version to 0.0 if cpu-only.
hip_version = "0.0"
if torch.version.cuda is not None:
    cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    if sys.platform != "win32":
        if isinstance(torch.cuda.nccl.version(), int):
            # This will break if minor version > 9.
            nccl_version = ".".join(str(torch.cuda.nccl.version())[:2])
        else:
            nccl_version = ".".join(map(str, torch.cuda.nccl.version()[:2]))
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_available():
        bf16_support = torch.cuda.is_bf16_supported()
if hasattr(torch.version, 'hip') and torch.version.hip is not None:
    hip_version = ".".join(torch.version.hip.split('.')[:2])
torch_info = {
    "version": torch_version,
    "bf16_support": bf16_support,
    "cuda_version": cuda_version,
    "nccl_version": nccl_version,
    "hip_version": hip_version
}

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")
with open('apex/git_version_info_installed.py', 'w') as fd:
    fd.write(f"version='{version_str}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")
    fd.write(f"installed_ops={install_ops}\n")
    fd.write(f"accelerator_name='{accelerator_name}'\n")
    fd.write(f"torch_info={torch_info}\n")



with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="apex",
    version=get_apex_version(),
    packages=find_packages(
        exclude=("build", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info", "op_builder", "accelerator")
    ),
    description="PyTorch Extensions written by NVIDIA",
    ext_modules=ext_modules2,
    cmdclass={'build_ext': BuildExtension} if ext_modules2 else {},
    extras_require=extras,
    install_requires=required,
    include_package_data=True
)

