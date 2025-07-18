from .builder import CUDAOpBuilder

import sys


class DistributedAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_DISTRIBUTED_ADAM'
    NAME = "distributed_adam_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['contrib/csrc/optimizers/multi_tensor_distopt_adam.cpp',
                'contrib/csrc/optimizers/multi_tensor_distopt_adam_kernel.cu']

    def include_paths(self):
        return ['contrib/csrc/',
                'csrc']
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags += ['--use_fast_math']
        return nvcc_flags