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

from op_builder.all_ops import ALL_OPS
import shutil

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


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
extras = {}

if not IS_ROCM_PYTORCH:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
else:
    _, bare_metal_version, bare_metal_minor  = get_rocm_bare_metal_version(ROCM_HOME)


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

BUILD_OP_DEFAULT = 0
BUILD_CPP_OPS = int(get_env_if_set('APEX_BUILD_CPP_OPS', BUILD_OP_DEFAULT))
BUILD_CUDA_OPS = int(get_env_if_set('APEX_BUILD_CUDA_OPS', BUILD_OP_DEFAULT))
build_flags = {
    "APEX_BUILD_CPP_OPS" : BUILD_CPP_OPS,
    "APEX_BUILD_CUDA_OPS" : BUILD_CUDA_OPS,
    }

if BUILD_CPP_OPS or BUILD_CUDA_OPS:
    if TORCH_MAJOR == 0:
        raise RuntimeError("--cpp_ext requires Pytorch 1.0 or later, "
                           "found torch.__version__ = {}".format(torch.__version__)
                           )

def is_env_set(key):
    """
    Checks if an environment variable is set and not "".
    """
    return bool(os.environ.get(key, None))

def get_op_build_env_name(op_name):
    assert hasattr(ALL_OPS[op_name], 'BUILD_VAR'), \
        f"{op_name} is missing BUILD_VAR field"
    return ALL_OPS[op_name].BUILD_VAR


def op_build_enabled(op_name):
    env_var = get_op_build_env_name(op_name)
    return int(get_env_if_set(env_var, BUILD_OP_DEFAULT))

def is_op_build_included(op_name):
    #check if operation has BUILD_FLAG defined
    assert hasattr(ALL_OPS[op_name], 'INCLUDE_FLAG'), \
        f"{op_name} is missing INCLUDE_FLAG field"
    include_flag = ALL_OPS[op_name].INCLUDE_FLAG
    return get_env_if_set(include_flag, False)

ext_modules = []
install_ops = dict.fromkeys(ALL_OPS.keys(), False)

for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()
    build_enabled = op_build_enabled(op_name) or is_op_build_included(op_name)

    # If op is requested but not available, throw an error.
    if build_enabled and not op_compatible:
        env_var = get_op_build_env_name(op_name)
        builder.warning(f"Skip pre-compile of incompatible {op_name}; One can disable {op_name} with {env_var}=0")
        continue

    # If op is compatible but install is not build enabled (JIT mode).
    if IS_ROCM_PYTORCH and op_compatible and not build_enabled:
        builder.hipify_extension()

    # If op build enabled, add builder to extensions.
    # Also check if corresponding flags are checked
    if build_enabled and op_compatible:
        install_ops[op_name] = True
        ext_modules.append(builder.builder())

print(f'Install Ops={install_ops}')  

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
    fd.write(f"build_flags={build_flags}\n")
    fd.write(f"torch_info={torch_info}\n")

if "--cpp_ext" in sys.argv:
    sys.argv.remove("--cpp_ext")

if "--cuda_ext" in sys.argv:
    sys.argv.remove("--cuda_ext")

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Find python files in compatibility folder
compatibility_dir = os.path.join(this_dir, 'compatibility')
py_modules = []

if os.path.exists(compatibility_dir):
    for file in os.listdir(compatibility_dir):
        if file.endswith('.py') and file != '__init__.py':
            module_name = f"{file[:-3]}"
            py_modules.append(module_name)

            #copy outside temporarily
            src_file = os.path.join(compatibility_dir, file)
            dst_file = os.path.join(this_dir, file)
            shutil.copy2(src_file, dst_file)
else:
    print("Warning: compatibility folder not found")

print ("-----py_modules--------", py_modules)

setup(
    name="apex",
    version=get_apex_version(),
    packages=find_packages(
        exclude=("build", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info", "op_builder")
    ),
    description="PyTorch Extensions written by NVIDIA",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    extras_require=extras,
    install_requires=required,
    include_package_data=True,
    py_modules=py_modules
)

#delete the temporarily copied compatibility files
for py_module in py_modules:
    path = dst_file = os.path.join(this_dir, py_module + ".py")
    if os.path.exists(path):
        os.remove(path)