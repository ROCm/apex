from .builder import CUDAOpBuilder

import sys


class FusedRopeBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FUSED_ROPE'
    INCLUDE_FLAG = "APEX_BUILD_CUDA_OPS"
    NAME = "fused_rotary_positional_embedding"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ["csrc/megatron/fused_rotary_positional_embedding.cpp",
                "csrc/megatron/fused_rotary_positional_embedding_cuda.cu"]

    def include_paths(self):
        return ['csrc', 'csrc/megatron']

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