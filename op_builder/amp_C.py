from .builder import CUDAOpBuilder

import sys


class AmpCBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_AMP_C'
    NAME = "amp_C"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['csrc/amp_C_frontend.cpp',
                'csrc/multi_tensor_sgd_kernel.cu',
                'csrc/multi_tensor_scale_kernel.cu',
                'csrc/multi_tensor_axpby_kernel.cu',
                'csrc/multi_tensor_l2norm_kernel.cu',
                'csrc/multi_tensor_l2norm_kernel_mp.cu',
                'csrc/multi_tensor_l2norm_scale_kernel.cu',
                'csrc/multi_tensor_lamb_stage_1.cu',
                'csrc/multi_tensor_lamb_stage_2.cu',
                'csrc/multi_tensor_adam.cu',
                'csrc/multi_tensor_adagrad.cu',
                'csrc/multi_tensor_novograd.cu',
                'csrc/multi_tensor_lars.cu',
                'csrc/multi_tensor_lamb.cu',
                'csrc/multi_tensor_lamb_mp.cu']

    def include_paths(self):
        return ['csrc/']
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags += ['-lineinfo', '--use_fast_math']
        return nvcc_flags