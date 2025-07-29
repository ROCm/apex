from .builder import CUDAOpBuilder

import sys


class PeerMemoryBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_PEER_MEMORY'
    INCLUDE_FLAG = "APEX_BUILD_CUDA_OPS"
    NAME = "peer_memory_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ["contrib/csrc/peer_memory/peer_memory_cuda.cu",
                "contrib/csrc/peer_memory/peer_memory.cpp"]

    def include_paths(self):
        return ['contrib/csrc/']
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros() + self.generator_args()