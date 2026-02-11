from .builder import CUDAOpBuilder
import sys


class FusedConvBiasReluBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FUSED_CONV_BIAS_RELU'
    INCLUDE_FLAG = "APEX_BUILD_CUDA_OPS"
    NAME = "fused_conv_bias_relu"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        if self.is_rocm_pytorch():
            return ["contrib/csrc/conv_bias_relu/conv_bias_relu_rocm.cpp"]
        else:
            return ["contrib/csrc/conv_bias_relu/conv_bias_relu.cpp"]

    def include_paths(self):
        paths = ['contrib/csrc/'] 
        if not self.is_rocm_pytorch():
            paths.append("apex/contrib/csrc/cudnn-frontend/include")
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.generator_args() + self.version_dependent_macros()

    def libraries_args(self):
        if self.is_rocm_pytorch():
            return super().libraries_args() + ['MIOpen']
        else:
            return super().libraries_args()