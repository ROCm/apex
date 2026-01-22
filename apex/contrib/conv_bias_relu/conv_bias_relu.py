import torch
from torch.autograd import gradcheck

try:
   import fused_conv_bias_relu
except ImportError:
   fused_conv_bias_relu = None


class ConvBiasReLU_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, padding, stride):
        ctx.bias_shape = bias.shape if bias is not None else None
        if bias is not None and bias.dim() != 1:
            bias = bias.view(-1)
        outputs = fused_conv_bias_relu.forward([x, weight, bias], padding, stride)
        ctx.save_for_backward(x, weight, outputs[0])
        ctx.padding = padding
        ctx.stride = stride

        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward(bwd_args, padding, stride)

        grad_bias = grads[2]
        if grad_bias is not None and ctx.bias_shape is not None and grad_bias.shape != ctx.bias_shape:
            grad_bias = grad_bias.view(ctx.bias_shape)

        return grads[0], grads[1], grad_bias, None, None


class ConvBiasMaskReLU_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, mask, padding, stride):
        ctx.bias_shape = bias.shape if bias is not None else None
        if bias is not None and bias.dim() != 1:
            bias = bias.view(-1)
        outputs = fused_conv_bias_relu.forward_mask([x, weight, bias, mask], padding, stride)
        ctx.save_for_backward(x, weight, outputs[0])
        ctx.padding = padding
        ctx.stride = stride

        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward(bwd_args, padding, stride)

        grad_bias = grads[2]
        if grad_bias is not None and ctx.bias_shape is not None and grad_bias.shape != ctx.bias_shape:
            grad_bias = grad_bias.view(ctx.bias_shape)

        return grads[0], grads[1], grad_bias, None, None, None


class ConvBias_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, padding, stride):
        ctx.bias_shape = bias.shape if bias is not None else None
        if bias is not None and bias.dim() != 1:
            bias = bias.view(-1)
        outputs = fused_conv_bias_relu.forward_no_relu([x, weight, bias], padding, stride)
        ctx.save_for_backward(x, weight)
        ctx.padding = padding
        ctx.stride = stride

        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward(bwd_args, padding, stride)

        grad_bias = grads[2]
        if grad_bias is not None and ctx.bias_shape is not None and grad_bias.shape != ctx.bias_shape:
            grad_bias = grad_bias.view(ctx.bias_shape)

        return grads[0], grads[1], grad_bias, None, None


ConvBiasReLU = ConvBiasReLU_.apply
ConvBiasMaskReLU = ConvBiasMaskReLU_.apply
ConvBias = ConvBias_.apply
