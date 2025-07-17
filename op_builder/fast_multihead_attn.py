from .builder import CUDAOpBuilder

import sys


class FastMultiheadAttnBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FAST_MULTIHEAD_ATTN'
    NAME = "fast_multihead_attn"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['contrib/csrc/multihead_attn/multihead_attn_frontend.cpp',
                    'contrib/csrc/multihead_attn/additive_masked_softmax_dropout_cuda.cu',
                    "contrib/csrc/multihead_attn/masked_softmax_dropout_cuda.cu",
                    "contrib/csrc/multihead_attn/encdec_multihead_attn_cuda.cu",
                    "contrib/csrc/multihead_attn/encdec_multihead_attn_norm_add_cuda.cu",
                    "contrib/csrc/multihead_attn/self_multihead_attn_cuda.cu",
                    "contrib/csrc/multihead_attn/self_multihead_attn_bias_additive_mask_cuda.cu",
                    "contrib/csrc/multihead_attn/self_multihead_attn_bias_cuda.cu",
                    "contrib/csrc/multihead_attn/self_multihead_attn_norm_add_cuda.cu"]

    def include_paths(self):
        return ['csrc/',
                'contrib/csrc/',
                'contrib/csrc/multihead_attn']
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros() + self.generator_args()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros() + self.generator_args() 
        if not self.is_rocm_pytorch():
            nvcc_flags += ['-U__CUDA_NO_HALF_OPERATORS__',
                         '-U__CUDA_NO_HALF_CONVERSIONS__',
                         '--expt-relaxed-constexpr',
                         '--expt-extended-lambda',
                         '--use_fast_math'] + self.compute_capability_args()
        else:
            nvcc_flags += ['-I/opt/rocm/include/hiprand',
                          '-I/opt/rocm/include/rocrand',
                          '-U__HIP_NO_HALF_OPERATORS__',
                          '-U__HIP_NO_HALF_CONVERSIONS__'] + self.backward_pass_guard_args()
        return nvcc_flags