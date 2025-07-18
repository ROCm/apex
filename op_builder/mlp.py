from .builder import CUDAOpBuilder

import sys


class MlpBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_MLP'
    NAME = "mlp_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['csrc/mlp.cpp',
                'csrc/mlp_cuda.cu']

    def include_paths(self):
        return ['csrc']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if self.is_rocm_pytorch():
            nvcc_flags.extend(self.backward_pass_guard_args())
        return nvcc_flags 