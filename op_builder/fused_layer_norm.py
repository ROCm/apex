from .builder import CUDAOpBuilder

import sys


class FusedLayerNormBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FUSED_LAYER_NORM'
    INCLUDE_FLAG = "APEX_CUDA_OPS"
    NAME = "fused_layer_norm_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['csrc/layer_norm_cuda.cpp', 'csrc/layer_norm_cuda_kernel.cu']

    def include_paths(self):
        return ['csrc']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags.extend(['--use_fast_math', '-maxrregcount=50'])
        return nvcc_flags