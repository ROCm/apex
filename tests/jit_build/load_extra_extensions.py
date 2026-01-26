from apex.op_builder.fused_lamb import FusedLambBuilder
from apex.op_builder.generic_scaled_masked_softmax_cuda import GenericScaledMaskedSoftmaxCudaBuilder
from apex.op_builder.scaled_softmax_cuda import ScaledSoftmaxCudaBuilder
from apex.op_builder.nccl_p2p import NCCLP2PBuilder

'''
generic_scaled_masked_softmax_cuda
scaled_softmax_cuda
fused_lamb_cuda
nccl_p2p_cuda
'''

FusedLambBuilder().load()
GenericScaledMaskedSoftmaxCudaBuilder().load()
ScaledSoftmaxCudaBuilder().load()
NCCLP2PBuilder().load()