from .builder import CUDAOpBuilder

import sys


class NCCLP2PBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_NCCL_P2P'
    INCLUDE_FLAG = "APEX_CUDA_OPS"
    NAME = "nccl_p2p_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ["contrib/csrc/nccl_p2p/nccl_p2p_cuda.cu",
                "contrib/csrc/nccl_p2p/nccl_p2p.cpp"]

    def include_paths(self):
        return ['contrib/csrc/']
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros() + self.generator_args()