try:
    import torch
    from apex.op_builder import XentropyBuilder
    xentropy_cuda = XentropyBuilder().load()
    
    from .softmax_xentropy import SoftmaxCrossEntropyLoss
    del torch
    del xentropy_cuda
    del softmax_xentropy
except ImportError as err:
    print("apex was installed without --xentropy flag, contrib.xentropy is not available")
