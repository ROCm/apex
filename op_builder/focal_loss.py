from .builder import CUDAOpBuilder

import sys


class FocalLossBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FOCAL_LOSS'
    NAME = "focal_loss_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['contrib/csrc/focal_loss/focal_loss_cuda.cpp',
                'contrib/csrc/focal_loss/focal_loss_cuda_kernel.cu']

    def include_paths(self):
        return ['contrib/csrc/' ]
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        if self.is_rocm_pytorch():
            nvcc_flags = ['-O3'] + self.version_dependent_macros()
        else:
            nvcc_flags = ['-O3', '--ftz=false', '--use_fast_math']
        return nvcc_flags