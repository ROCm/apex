from .builder import CUDAOpBuilder

import sys


class FusedIndexMul2dBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FUSED_INDEX_MUL_2D'
    NAME = "fused_index_mul_2d"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['contrib/csrc/index_mul_2d/index_mul_2d_cuda.cpp',
                'contrib/csrc/index_mul_2d/index_mul_2d_cuda_kernel.cu']

    def include_paths(self):
        return ['contrib/csrc/']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags += ['--use_fast_math', '--ftz=false']
        else:
            nvcc_flags += self.aten_atomic_args()
        return nvcc_flags