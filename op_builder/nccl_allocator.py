from .builder import CUDAOpBuilder

import sys


class NCCLAllocatorBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_NCCL_ALLOCATOR'
    NAME = "_apex_nccl_allocator"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ["contrib/csrc/nccl_allocator/NCCLAllocator.cpp"]

    def include_paths(self):
        return ['contrib/csrc/']
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros() + self.generator_args()

    def nvcc_args(self):
        return self.nccl_args()

    def is_compatible(self):
        available_nccl_version = self.nccl_version()
        if available_nccl_version >= (2, 19):
            return True
        else:
            return False