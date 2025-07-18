from .builder import CUDAOpBuilder

class FusedWeightGradientMlpCudaBuilder(CUDAOpBuilder):
    BUILD_VAR = 'APEX_BUILD_FUSED_WEIGHT_GRADIENT_MLP'
    NAME = "fused_weight_gradient_mlp_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return [
            "csrc/megatron/fused_weight_gradient_dense.cpp",
            "csrc/megatron/fused_weight_gradient_dense_cuda.cu",
            "csrc/megatron/fused_weight_gradient_dense_16bit_prec_cuda.cu",
        ]

    def include_paths(self):
        # Both csrc and csrc/megatron are included in the original extension
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
                    '--expt-extended-lambda',
                    "--use_fast_math"
                ]) + self.compute_capability_args()
        return nvcc_flags

