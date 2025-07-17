from .builder import CUDAOpBuilder
import sys


class TransducerJointBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_TRANSDUCER_JOINT'
    NAME = "transducer_joint_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ["contrib/csrc/transducer/transducer_joint.cpp",
                "contrib/csrc/transducer/transducer_joint_kernel.cu"]

    def include_paths(self):
        return ['contrib/csrc/',
                #it uses philox.cuh from contrib/csrc/multihead_attn
                'contrib/csrc/multihead_attn'] 
        
    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros() + self.generator_args() 

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros() + self.generator_args() 
        if not self.is_rocm_pytorch():
            nvcc_flags += self.nvcc_threads_args()
        return nvcc_flags