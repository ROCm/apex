from .builder import CUDAOpBuilder

class ScaledMaskedSoftmaxCudaBuilder(CUDAOpBuilder):
    BUILD_VAR = 'DS_BUILD_SCALED_MASKED_SOFTMAX_CUDA'
    NAME = "scaled_masked_softmax_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return [
            "csrc/megatron/scaled_masked_softmax_cpu.cpp",
            "csrc/megatron/scaled_masked_softmax_cuda.cu"
        ]

    def include_paths(self):
        # Both csrc and csrc/megatron are included in the original extension
        return ['csrc', 'csrc/megatron']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        if self.is_rocm_pytorch():
            return [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__'
            ] + self.version_dependent_macros()
        else:
            return [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda'
            ] + self.version_dependent_macros()
