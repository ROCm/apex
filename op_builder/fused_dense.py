from .builder import CUDAOpBuilder

import sys


class FusedDenseBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FUSED_DENSE'
    NAME = "fused_dense_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['csrc/fused_dense_base.cpp', 'csrc/fused_dense_cuda.cu']

    def include_paths(self):
        return ['csrc']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        return ['-O3'] + self.version_dependent_macros()