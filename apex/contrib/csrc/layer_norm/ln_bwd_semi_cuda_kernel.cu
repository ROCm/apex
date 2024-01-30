#include "ln.h"
#include "ln_utils.cuh"
#include "ln_kernel_traits.h"
#include "ln_bwd_kernels.cuh"

using namespace layer_norm;

template<
    typename weight_t,
    typename input_t,
    typename output_t,
    typename compute_t,
    typename index_t,
    int HIDDEN_SIZE, 
    int CTAS_PER_ROW, 
    int WARPS_M, 
    int WARPS_N, 
    int BYTES_PER_LDG_MAIN,
    int BYTES_PER_LDG_FINAL
>
void launch_(LaunchParams<BwdParams> &launch_params, const bool configure_params){

    using Kernel_traits = Kernel_traits<weight_t,
                                        input_t,
                                        output_t,
                                        compute_t,
                                        index_t,
                                        HIDDEN_SIZE,
                                        CTAS_PER_ROW,
                                        WARPS_M,
                                        WARPS_N,
                                        BYTES_PER_LDG_MAIN
                                        >;
    auto kernel = &ln_bwd_kernel<Kernel_traits>;

    if( configure_params ) {
        int ctas_per_sm;
#ifdef USE_ROCM
        hipError_t status_ = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES);
#else
        cudaError_t status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES);
#endif
        launch_params.params.ctas_per_col = launch_params.props->multiProcessorCount * ctas_per_sm / Kernel_traits::CTAS_PER_ROW;
        launch_params.barrier_size = 0;
        launch_params.workspace_bytes = 0;
        if(Kernel_traits::CTAS_PER_ROW > 1) {
            launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
            launch_params.workspace_bytes = launch_params.params.ctas_per_col 
                                          * Kernel_traits::WARPS_M  
                                          * Kernel_traits::CTAS_PER_ROW 
                                          * sizeof(typename Kernel_traits::reduce_t)
                                          * 2;
        }
        return;
    }

    if( Kernel_traits::SMEM_BYTES >= 48 * 1024 ) {
        // hipify missing cudaFuncSetAttribute, cudaFuncAttributeMaxDynamicSharedMemorySize
#ifdef USE_ROCM
        CHECK_CUDA(hipFuncSetAttribute((const void *)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::SMEM_BYTES));
#else
        CHECK_CUDA(cudaFuncSetAttribute((const void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::SMEM_BYTES));
#endif
    }
    auto stream = launch_params.stream;
    auto ctas_per_col = launch_params.params.ctas_per_col;

    if( Kernel_traits::CTAS_PER_ROW == 1 ) {
        kernel<<<ctas_per_col, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES, stream>>>(launch_params.params);
    } else {
	    dim3 grid(Kernel_traits::CTAS_PER_ROW * ctas_per_col);
        dim3 block(Kernel_traits::THREADS_PER_CTA);
        void *params_ = (void *)&launch_params.params;
#ifdef USE_ROCM
        hipLaunchCooperativeKernel((void *)kernel, grid, block, (void **)&params_, Kernel_traits::SMEM_BYTES, stream);
#else
        cudaLaunchCooperativeKernel((void *)kernel, grid, block, (void **)&params_, Kernel_traits::SMEM_BYTES, stream);
#endif
    }

    using Kernel_traits_f = layer_norm::Kernel_traits_finalize<HIDDEN_SIZE,
                                                               weight_t,
                                                               input_t,
                                                               output_t,
                                                               compute_t,
                                                               index_t,
                                                               32 * 32,  // THREADS_PER_CTA
                                                               BYTES_PER_LDG_FINAL>;

    auto kernel_f = &layer_norm::ln_bwd_finalize_kernel<Kernel_traits_f>;
    kernel_f<<<Kernel_traits_f::CTAS, Kernel_traits_f::THREADS_PER_CTA, 0, stream>>>(launch_params.params);
}

#ifdef USE_ROCM
constexpr bool is_rocm = true;
#else
constexpr bool is_rocm = false;
#endif

// Create backward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_BWD_LAUNCHER(  768, fp32, fp32, fp32, fp32, 1, 4, 3, 8, 8);
REGISTER_BWD_LAUNCHER(  768, fp16, fp16, fp16, fp32, 1, 4, 3, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(  768, fp16, fp32, fp16, fp32, 1, 4, 3, 8, 8);
REGISTER_BWD_LAUNCHER(  768, bf16, bf16, bf16, fp32, 1, 4, 3, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(  768, bf16, fp32, bf16, fp32, 1, 4, 3, 8, 8);

REGISTER_BWD_LAUNCHER( 1024, fp32, fp32, fp32, fp32, 1, 4, 4, 8, 4);
REGISTER_BWD_LAUNCHER( 1024, fp16, fp16, fp16, fp32, 1, 4, 4, 4, 4);
REGISTER_BWD_LAUNCHER( 1024, fp16, fp32, fp16, fp32, 1, 4, 4, 8, 4);
REGISTER_BWD_LAUNCHER( 1024, bf16, bf16, bf16, fp32, 1, 4, 4, 4, 4);
REGISTER_BWD_LAUNCHER( 1024, bf16, fp32, bf16, fp32, 1, 4, 4, 8, 4);

REGISTER_BWD_LAUNCHER( 1536, fp32, fp32, fp32, fp32, 1, 4, 3, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER( 1536, fp16, fp16, fp16, fp32, 1, 4, 3, is_rocm ? 4 :  8, 4);
REGISTER_BWD_LAUNCHER( 1536, fp16, fp32, fp16, fp32, 1, 4, 3, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER( 1536, bf16, bf16, bf16, fp32, 1, 4, 3, is_rocm ? 4 :  8, 4);
REGISTER_BWD_LAUNCHER( 1536, bf16, fp32, bf16, fp32, 1, 4, 3, is_rocm ? 8 : 16, 4);

REGISTER_BWD_LAUNCHER( 2048, fp32, fp32, fp32, fp32, 1, 4, 4, 8, 4);
REGISTER_BWD_LAUNCHER( 2048, fp16, fp16, fp16, fp32, 1, 4, 4, 4, 4);
REGISTER_BWD_LAUNCHER( 2048, fp16, fp32, fp16, fp32, 1, 4, 4, 8, 4);
REGISTER_BWD_LAUNCHER( 2048, bf16, bf16, bf16, fp32, 1, 4, 4, 4, 4);
REGISTER_BWD_LAUNCHER( 2048, bf16, fp32, bf16, fp32, 1, 4, 4, 8, 4);

REGISTER_BWD_LAUNCHER( 2304, fp32, fp32, fp32, fp32, 1, 1, 4, is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER( 2304, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER( 2304, fp16, fp32, fp16, fp32, 1, 1, 4, is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER( 2304, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER( 2304, bf16, fp32, bf16, fp32, 1, 1, 4, is_rocm ? 4 : 8, 4);

REGISTER_BWD_LAUNCHER( 3072, fp32, fp32, fp32, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER( 3072, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER( 3072, fp16, fp32, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER( 3072, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER( 3072, bf16, fp32, bf16, fp32, 1, 1, 4, 8, 4);

REGISTER_BWD_LAUNCHER( 3840, fp32, fp32, fp32, fp32, 1, 1, 4, is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER( 3840, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER( 3840, fp16, fp32, fp16, fp32, 1, 1, 4, is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER( 3840, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER( 3840, bf16, fp32, bf16, fp32, 1, 1, 4, is_rocm ? 4 : 8, 4);

REGISTER_BWD_LAUNCHER( 4096, fp32, fp32, fp32, fp32, 1, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER( 4096, fp16, fp16, fp16, fp32, 1, 1, 8, 4, 4);
REGISTER_BWD_LAUNCHER( 4096, fp16, fp32, fp16, fp32, 1, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER( 4096, bf16, bf16, bf16, fp32, 1, 1, 8, 4, 4);
REGISTER_BWD_LAUNCHER( 4096, bf16, fp32, bf16, fp32, 1, 1, 8, 8, 4);

REGISTER_BWD_LAUNCHER( 5120, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 5120, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER( 5120, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 5120, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER( 5120, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER( 6144, fp32, fp32, fp32, fp32, 1, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER( 6144, fp16, fp16, fp16, fp32, 1, 1, 8, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER( 6144, fp16, fp32, fp16, fp32, 1, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER( 6144, bf16, bf16, bf16, fp32, 1, 1, 8, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER( 6144, bf16, fp32, bf16, fp32, 1, 1, 8, 8, 4);

REGISTER_BWD_LAUNCHER( 8192, fp32, fp32, fp32, fp32, 1, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER( 8192, fp16, fp16, fp16, fp32, 1, 1, 8, 4, 4);
REGISTER_BWD_LAUNCHER( 8192, fp16, fp32, fp16, fp32, 1, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER( 8192, bf16, bf16, bf16, fp32, 1, 1, 8, 4, 4);
REGISTER_BWD_LAUNCHER( 8192, bf16, fp32, bf16, fp32, 1, 1, 8, 8, 4);

REGISTER_BWD_LAUNCHER(10240, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(10240, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(10240, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(12288, fp32, fp32, fp32, fp32, 1, 1, 16, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(12288, fp16, fp16, fp16, fp32, 1, 1, 16, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER(12288, fp16, fp32, fp16, fp32, 1, 1, 16, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(12288, bf16, bf16, bf16, fp32, 1, 1, 16, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER(12288, bf16, fp32, bf16, fp32, 1, 1, 16, 8, 4);

REGISTER_BWD_LAUNCHER(12800, fp32, fp32, fp32, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(12800, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 4 :  8, 4);
REGISTER_BWD_LAUNCHER(12800, fp16, fp32, fp16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(12800, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 4 :  8, 4);
REGISTER_BWD_LAUNCHER(12800, bf16, fp32, bf16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);

REGISTER_BWD_LAUNCHER(14336, fp32, fp32, fp32, fp32, 1, 1, 4,  8, 4);
REGISTER_BWD_LAUNCHER(14336, fp16, fp16, fp16, fp32, 1, 1, 4,  is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER(14336, fp16, fp32, fp16, fp32, 1, 1, 4,  8, 4);
REGISTER_BWD_LAUNCHER(14336, bf16, bf16, bf16, fp32, 1, 1, 4,  is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER(14336, bf16, fp32, bf16, fp32, 1, 1, 4,  8, 4);

REGISTER_BWD_LAUNCHER(15360, fp32, fp32, fp32, fp32, 1, 1, 4,  is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER(15360, fp16, fp16, fp16, fp32, 1, 1, 4,  is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER(15360, fp16, fp32, fp16, fp32, 1, 1, 4,  is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER(15360, bf16, bf16, bf16, fp32, 1, 1, 4,  is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER(15360, bf16, fp32, bf16, fp32, 1, 1, 4,  is_rocm ? 4 : 8, 4);

REGISTER_BWD_LAUNCHER(16384, fp32, fp32, fp32, fp32, 1, 1, 16, 8, 4);
REGISTER_BWD_LAUNCHER(16384, fp16, fp16, fp16, fp32, 1, 1, 16, 4, 4);
REGISTER_BWD_LAUNCHER(16384, fp16, fp32, fp16, fp32, 1, 1, 16, 8, 4);
REGISTER_BWD_LAUNCHER(16384, bf16, bf16, bf16, fp32, 1, 1, 16, 4, 4);
REGISTER_BWD_LAUNCHER(16384, bf16, fp32, bf16, fp32, 1, 1, 16, 8, 4);

REGISTER_BWD_LAUNCHER(18432, fp32, fp32, fp32, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(18432, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 4 :  8, 4);
REGISTER_BWD_LAUNCHER(18432, fp16, fp32, fp16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(18432, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 4 :  8, 4);
REGISTER_BWD_LAUNCHER(18432, bf16, fp32, bf16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);

REGISTER_BWD_LAUNCHER(20480, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(20480, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(20480, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(20480, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(20480, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(24576, fp32, fp32, fp32, fp32, 1, 1, 8, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(24576, fp16, fp16, fp16, fp32, 1, 1, 8, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER(24576, fp16, fp32, fp16, fp32, 1, 1, 8, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(24576, bf16, bf16, bf16, fp32, 1, 1, 8, is_rocm ? 4 : 16, 4);
REGISTER_BWD_LAUNCHER(24576, bf16, fp32, bf16, fp32, 1, 1, 8, 8, 4);

REGISTER_BWD_LAUNCHER(25600, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(25600, fp16, fp16, fp16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(25600, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(25600, bf16, bf16, bf16, fp32, 1, 1, 4, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(25600, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(30720, fp32, fp32, fp32, fp32, 1, 1, 8, is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER(30720, fp16, fp16, fp16, fp32, 1, 1, 8, is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER(30720, fp16, fp32, fp16, fp32, 1, 1, 8, is_rocm ? 4 : 8, 4);
REGISTER_BWD_LAUNCHER(30720, bf16, bf16, bf16, fp32, 1, 1, 8, is_rocm ? 2 : 4, 4);
REGISTER_BWD_LAUNCHER(30720, bf16, fp32, bf16, fp32, 1, 1, 8, is_rocm ? 4 : 8, 4);

REGISTER_BWD_LAUNCHER(32768, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, fp16, fp16, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, fp16, fp32, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, bf16, bf16, bf16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, bf16, fp32, bf16, fp32, 1, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(40960, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(40960, fp16, fp16, fp16, fp32, 1, 1, 8, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(40960, fp16, fp32, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(40960, bf16, bf16, bf16, fp32, 1, 1, 8, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(40960, bf16, fp32, bf16, fp32, 1, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(49152, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(49152, fp16, fp16, fp16, fp32, 1, 1, 8, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(49152, fp16, fp32, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(49152, bf16, bf16, bf16, fp32, 1, 1, 8, is_rocm ? 8 : 16, 4);
REGISTER_BWD_LAUNCHER(49152, bf16, fp32, bf16, fp32, 1, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(65536, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(65536, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(65536, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(65536, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(65536, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);