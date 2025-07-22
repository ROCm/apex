from .builder import CUDAOpBuilder

import sys


class BnpBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_BNP'
    INCLUDE_FLAG = "APEX_CUDA_OPS"
    NAME = "bnp"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['contrib/csrc/groupbn/batch_norm.cu',
                'contrib/csrc/groupbn/ipc.cu',
                'contrib/csrc/groupbn/interface.cpp',
                'contrib/csrc/groupbn/batch_norm_add_relu.cu']

    def include_paths(self):
        return ['contrib/csrc', 'csrc']

    def cxx_args(self):
        return self.version_dependent_macros()

    def nvcc_args(self):
        return ['-DCUDA_HAS_FP16=1',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__'] + self.version_dependent_macros()