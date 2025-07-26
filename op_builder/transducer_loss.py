from .builder import CUDAOpBuilder
import sys


class TransducerLossBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_TRANSDUCER_LOSS'
    INCLUDE_FLAG = "APEX_CUDA_OPS"
    NAME = "transducer_loss_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ["contrib/csrc/transducer/transducer_loss.cpp",
                "contrib/csrc/transducer/transducer_loss_kernel.cu"]

    def include_paths(self):
        return ['contrib/csrc/' ]
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros() 
        if not self.is_rocm_pytorch():
            nvcc_flags += self.nvcc_threads_args()
        return nvcc_flags