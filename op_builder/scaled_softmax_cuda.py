from .builder import CUDAOpBuilder

import sys

class ScaledSoftmaxCudaBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_SCALED_SOFTMAX_CUDA'
    NAME = "scaled_softmax_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return [
            "csrc/megatron/scaled_softmax_cpu.cpp",
            "csrc/megatron/scaled_softmax_cuda.cu"
        ]

    def include_paths(self):
        return ['csrc']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__'
            ] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags.extend(
                [
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda'
                ])
        return nvcc_flags

