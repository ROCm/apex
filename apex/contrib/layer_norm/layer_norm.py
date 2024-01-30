import importlib
import numbers

import torch
from torch.nn import init
from torch.nn import functional as F

from apex._autocast_utils import _cast_if_autocast_enabled
import fast_layer_norm
import fused_layer_norm_cuda

#global fused_layer_norm_cuda
#fused_layer_norm_cuda = None


class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, normalized_shape, epsilon):
        x = x.contiguous()
        gamma_ = gamma.contiguous()
        beta_ = beta.contiguous()
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm.ln_fwd(xmat, gamma, beta, epsilon)
        ctx.normalized_shape = normalized_shape
        ctx.eps = epsilon
        ctx.save_for_backward(x, gamma_, beta_, mu, rsigma)
        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        # assert dy.is_contiguous()
        dy_ = dy.contiguous()  # this happens!
        x, gamma, beta, mu, rsigma= ctx.saved_tensors
        #x, gamma, mu, rsigma= ctx.saved_tensors
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        dymat = dy_.view(xmat.shape)
        dxmat = dgamma = dbeta = None
        if hidden_size > 12288:
            #dxmat = fused_layer_norm_cuda.backward(dy_, mu, rsigma, xmat, ctx.normalized_shape, ctx.eps)
            dxmat, dgamma, dbeta = fused_layer_norm_cuda.backward_affine(
                dymat, mu, rsigma, xmat, ctx.normalized_shape, gamma, beta, ctx.eps
            )
            dx = dxmat.view(x.shape)
        else:
            dxmat, dgamma, dbeta, _, _ = fast_layer_norm.ln_bwd(dymat, xmat, mu, rsigma, gamma)
            dx = dxmat.view(x.shape)
        return dx, dgamma, dbeta, None, None


def _fast_layer_norm(x, weight, bias, normalized_shape, epsilon):
    args = _cast_if_autocast_enabled(x, weight, bias, normalized_shape, epsilon)
    with torch.cuda.amp.autocast(enabled=False):
        return FastLayerNormFN.apply(*args)


class FastLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.epsilon = eps
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = torch.nn.Parameter(torch.Tensor(*normalized_shape))
        self.bias = torch.nn.Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if not x.is_cuda:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.epsilon)
        else:
            return _fast_layer_norm(x, self.weight, self.bias, self.normalized_shape, self.epsilon)
