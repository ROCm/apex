try:
    import torch
    from apex.op_builder import BnpBuilder
    bnp = BnpBuilder().load()
    from .batch_norm import BatchNorm2d_NHWC
    del torch
    del bnp
    del batch_norm
except ImportError as err:
    print("apex was installed without --bnp flag, contrib.groupbn is not available")
