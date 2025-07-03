from .builder import CUDAOpBuilder

import sys


class XentropyBuilder(CUDAOpBuilder):
    BUILD_VAR = 'DS_BUILD_XENTROPY'
    NAME = "xentropy_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['contrib/csrc/xentropy/interface.cpp',
                'contrib/csrc/xentropy/xentropy_kernel.cu']

    def include_paths(self):
        return ['csrc', 'contrib/csrc/' ]

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros() 
        return nvcc_flags