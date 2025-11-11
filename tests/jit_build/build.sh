#parse the arguments
JIT_CONDITION="$2"
echo "JIT_CONDITION $JIT_CONDITION"

echo $(pwd)

git checkout Refactor_build
git submodule update --init --recursive 

# uninstall apex 
pip uninstall apex -y
make clean

#install apex for different conditions 
if [ "$JIT_CONDITION" = "1" ]; then
    pip install . --no-build-isolation 
elif [ "$JIT_CONDITION" = "2" ]; then
    APEX_BUILD_CPP_OPS=1 pip install . --no-build-isolation
elif [ "$JIT_CONDITION" = "3" ]; then
    APEX_BUILD_CUDA_OPS=1 pip install . --no-build-isolation
elif [ "$JIT_CONDITION" = "4" ]; then
    APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1 pip install . --no-build-isolation
elif [ "$JIT_CONDITION" = "5" ]; then
    APEX_BUILD_FUSED_DENSE=1 pip install . --no-build-isolation
elif [ "$JIT_CONDITION" = "6" ]; then
    python setup.py install --cpp_ext --cuda_ext
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
elif [ "$JIT_CONDITION" = "8" ]; then
    python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
elif [ "$JIT_CONDITION" = "9" ]; then
    APEX_BUILD_CPP_OPS=1 python -m build --wheel --no-isolation .
elif [ "$JIT_CONDITION" = "10" ]; then
    APEX_BUILD_CUDA_OPS=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
elif [ "$JIT_CONDITION" = "11" ]; then
    APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
elif [ "$JIT_CONDITION" = "12" ]; then
    APEX_BUILD_FUSED_DENSE=1 python -m build --wheel --no-isolation .
    pip install dist/apex-*.whl 
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
fi 