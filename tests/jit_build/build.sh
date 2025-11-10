#parse the arguments
JIT_CONDITION="$2"
echo "JIT_CONDITION $JIT_CONDITION"

cd ../..

git checkout Refactor_build
git submodule update --init --recursive 

# uninstall apex 
pip uninstall apex 
make clean

#install apex for different conditions 
if [ "$JIT_CONDITION" = "1" ]; then
    pip install . --no-build-isolation 
    LOG_FILE=results_jit_unit_test1.log
    LOG_FILE2=results_jit_unit_test1c.log
elif [ "$JIT_CONDITION" = "2" ]; then
    APEX_BUILD_CPP_OPS=1 pip install . --no-build-isolation
    LOG_FILE=results_jit_unit_test2.log
    LOG_FILE2=results_jit_unit_test2c.log
elif [ "$JIT_CONDITION" = "3" ]; then
    APEX_BUILD_CUDA_OPS=1 pip install . --no-build-isolation
    LOG_FILE=results_jit_unit_test3.log
    LOG_FILE2=results_jit_unit_test3c.log
elif [ "$JIT_CONDITION" = "4" ]; then
    APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1 pip install . --no-build-isolation
    LOG_FILE=results_jit_unit_test4.log
    LOG_FILE2=results_jit_unit_test4c.log
elif [ "$JIT_CONDITION" = "5" ]; then
    APEX_BUILD_FUSED_DENSE=1 pip install . --no-build-isolation
    LOG_FILE=results_jit_unit_test5.log
    LOG_FILE2=results_jit_unit_test5c.log
elif [ "$JIT_CONDITION" = "6" ]; then
    python setup.py install --cpp_ext --cuda_ext
    LOG_FILE=results_jit_unit_test6.log
    LOG_FILE2=results_jit_unit_test6c.log
elif [ "$JIT_CONDITION" = "7" ]; then
    APEX_BUILD_AMP_C=1 APEX_BUILD_APEX_C=1 APEX_BUILD_BNP=1 \
    APEX_BUILD_DISTRIBUTED_ADAM=1 APEX_BUILD_DISTRIBUTED_LAMB=1 APEX_BUILD_FAST_MULTIHEAD_ATTN=1 \
    APEX_BUILD_FOCAL_LOSS=1 APEX_BUILD_FUSED_ADAM=1 APEX_BUILD_FUSED_BIAS_SWIGLU=1 \
    APEX_BUILD_FUSED_DENSE=1 APEX_BUILD_FUSED_INDEX_MUL_2D=1 APEX_BUILD_FUSED_LAMB=1 \
    APEX_BUILD_FUSED_LAYER_NORM=1 APEX_BUILD_FUSED_ROPE=1 APEX_BUILD_FUSED_WEIGHT_GRADIENT_MLP=1 \
    APEX_BUILD_GENERIC_SCALED_MASKED_SOFTMAX_CUDA=1 APEX_BUILD_MLP=1 APEX_BUILD_NCCL_ALLOCATOR=1 \
    APEX_BUILD_NCCL_P2P=1 APEX_BUILD_PEER_MEMORY=1 APEX_BUILD_SCALED_MASKED_SOFTMAX_CUDA=1 \
    APEX_BUILD_SCALED_SOFTMAX_CUDA=1 APEX_BUILD_SCALED_UPPER_TRIANG_MASKED_SOFTMAX_CUDA=1 APEX_BUILD_SYNCBN=1 \
    APEX_BUILD_TRANSDUCER_JOINT=1 APEX_BUILD_TRANSDUCER_LOSS=1 APEX_BUILD_XENTROPY=1 pip install . --no-build-isolation
    LOG_FILE=results_jit_unit_test7.log
    LOG_FILE2=results_jit_unit_test7c.log
elif [ "$JIT_CONDITION" = "8" ]; then
    python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
    LOG_FILE=results_jit_unit_test8.log
    LOG_FILE2=results_jit_unit_test8c.log
elif [ "$JIT_CONDITION" = "9" ]; then
    APEX_BUILD_CPP_OPS=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
    LOG_FILE=results_jit_unit_test9.log
    LOG_FILE2=results_jit_unit_test9c.log
elif [ "$JIT_CONDITION" = "10" ]; then
    APEX_BUILD_CUDA_OPS=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
    LOG_FILE=results_jit_unit_test10.log
    LOG_FILE2=results_jit_unit_test10c.log
elif [ "$JIT_CONDITION" = "11" ]; then
    APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
    LOG_FILE=results_jit_unit_test11.log
    LOG_FILE2=results_jit_unit_test11c.log
elif [ "$JIT_CONDITION" = "12" ]; then
    APEX_BUILD_FUSED_DENSE=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
    LOG_FILE=results_jit_unit_test12.log
    LOG_FILE2=results_jit_unit_test12c.log
elif [ "$JIT_CONDITION" = "13" ]; then
    APEX_BUILD_AMP_C=1 APEX_BUILD_APEX_C=1 APEX_BUILD_BNP=1 \
    APEX_BUILD_DISTRIBUTED_ADAM=1 APEX_BUILD_DISTRIBUTED_LAMB=1 APEX_BUILD_FAST_MULTIHEAD_ATTN=1 \
    APEX_BUILD_FOCAL_LOSS=1 APEX_BUILD_FUSED_ADAM=1 APEX_BUILD_FUSED_BIAS_SWIGLU=1 \
    APEX_BUILD_FUSED_DENSE=1 APEX_BUILD_FUSED_INDEX_MUL_2D=1 APEX_BUILD_FUSED_LAMB=1 \
    APEX_BUILD_FUSED_LAYER_NORM=1 APEX_BUILD_FUSED_ROPE=1 APEX_BUILD_FUSED_WEIGHT_GRADIENT_MLP=1 \
    APEX_BUILD_GENERIC_SCALED_MASKED_SOFTMAX_CUDA=1 APEX_BUILD_MLP=1 APEX_BUILD_NCCL_ALLOCATOR=1 \
    APEX_BUILD_NCCL_P2P=1 APEX_BUILD_PEER_MEMORY=1 APEX_BUILD_SCALED_MASKED_SOFTMAX_CUDA=1 \
    APEX_BUILD_SCALED_SOFTMAX_CUDA=1 APEX_BUILD_SCALED_UPPER_TRIANG_MASKED_SOFTMAX_CUDA=1 APEX_BUILD_SYNCBN=1 \
    APEX_BUILD_TRANSDUCER_JOINT=1 APEX_BUILD_TRANSDUCER_LOSS=1 APEX_BUILD_XENTROPY=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
    LOG_FILE=results_jit_unit_test13.log
    LOG_FILE2=results_jit_unit_test13c.log
fi 