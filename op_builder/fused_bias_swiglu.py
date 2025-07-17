from .builder import CUDAOpBuilder
import sys
import os

class FusedBiasSwiGLUBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FUSED_BIAS_SWIGLU'
    NAME = "fused_bias_swiglu"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return [
            "csrc/megatron/fused_bias_swiglu.cpp",
            "csrc/megatron/fused_bias_swiglu_cuda.cu"
        ]

    def include_paths(self):
        return ['csrc']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__'
            ] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags.extend(
                [
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda'
                ])
        else:
            # Handle ROCm arch flags
            amdgpu_targets = os.environ.get('PYTORCH_ROCM_ARCH', '')
            if not amdgpu_targets:
                print("Warning: PYTORCH_ROCM_ARCH environment variable is empty.")
                print("Using default architecture. Set this variable for specific GPU targets.")
                print("Example: export PYTORCH_ROCM_ARCH=gfx906")
                amdgpu_targets = "gfx906"
            try:
                for amdgpu_target in amdgpu_targets.split(';'):
                    if amdgpu_target:
                        nvcc_flags += [f'--offload-arch={amdgpu_target}']
            except Exception as e:
                print(f"Warning: Error processing PYTORCH_ROCM_ARCH: {e}")
                print("Falling back to default architecture gfx906")
                nvcc_flags += ['--offload-arch=gfx906']
        return nvcc_flags
