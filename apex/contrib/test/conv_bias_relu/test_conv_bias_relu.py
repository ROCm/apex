import copy
import math
import random
import unittest

import torch
import torch.nn.functional as F

HAS_CONV_BIAS_RELU = None
try:
    from apex.contrib.conv_bias_relu import ConvBiasReLU, ConvBias, ConvBiasMaskReLU
except ImportError as e:
    HAS_CONV_BIAS_RELU = False
else:
    HAS_CONV_BIAS_RELU = True


@unittest.skipIf(not HAS_CONV_BIAS_RELU, "`apex.contrib.conv_bias_relu` is not found.")
class FusedDenseTest(unittest.TestCase):
    def setUp(self, seed=0):
        torch.manual_seed(seed)
        
        self.batch_size = random.randint(1, 64)
        self.in_channels = random.randint(1, 64) * 8
        self.out_channels = random.randint(1, 64) * 8
        self.in_height = self.in_width = random.randint(5, 100)
        self.conv_kernel_size = random.randint(1, 5)
        self.conv_pad = random.randint(0, int(self.conv_kernel_size / 2))
        self.conv_stride = random.randint(1, 5)
        self.conv_dilation = 1
        self.out_height = self.out_width = \
            math.floor((self.in_height + 2 * self.conv_pad - \
                        self.conv_dilation * (self.conv_kernel_size - 1) - 1) / self.conv_stride + 1)

        self.x = torch.randint(low=-16, high=16,
                               size=[self.batch_size, self.in_channels, self.in_height, self.in_width]) \
                               .cuda().to(memory_format=torch.channels_last).float()
        self.x_ = self.x.clone()
        self.x.requires_grad_()
        self.x_.requires_grad_()

        self.mask = torch.randn([self.batch_size, self.out_channels, self.out_height, self.out_width]).cuda().to(memory_format=torch.channels_last)
        self.mask = (self.mask > 0).to(torch.int8)
        self.mask_ = self.mask.clone()

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels, self.conv_kernel_size,
                                     stride=self.conv_stride, padding=self.conv_pad).cuda().to(memory_format=torch.channels_last)
        self.conv1_ = copy.deepcopy(self.conv1)

        print()
        print('> input=[{}, {}, {}, {}]'.format(self.batch_size, self.in_channels, self.in_height, self.in_width))
        print('> kernel=[{}, {}, {}, {}], stride={}, pad={}'.format(self.out_channels, self.in_channels,
                                                                    self.conv_kernel_size, self.conv_kernel_size,
								    self.conv_stride, self.conv_pad))

    def test_conv_bias_relu(self):
        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out = ConvBiasReLU(self.x, self.conv1.weight, self.conv1.bias.reshape(1, -1, 1, 1), self.conv_pad, self.conv_stride)
            loss = (out.float()**2).sum() / out.numel()
        loss.backward()
        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out_ = F.relu(self.conv1_(self.x_))
            loss_ = (out_**2).sum() / out_.numel()
        loss_.backward()

        self.assertTrue(torch.allclose(self.x_, self.x, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.conv1_.bias.grad, self.conv1.bias.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.conv1_.weight.grad, self.conv1.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.x_.grad, self.x.grad, atol=1e-3, rtol=1e-3, equal_nan=True))

    def test_conv_bias(self):
        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out = ConvBias(self.x, self.conv1.weight, self.conv1.bias.reshape(1, -1, 1, 1), self.conv_pad, self.conv_stride)
            loss = (out.float()**2).sum() / out.numel()
        loss.backward()

        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out_ = self.conv1_(self.x_)
            loss_ = (out_**2).sum() / out_.numel()
        loss_.backward()

        self.assertTrue(torch.allclose(self.x_, self.x, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.conv1_.bias.grad, self.conv1.bias.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.conv1_.weight.grad, self.conv1.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.x_.grad, self.x.grad, atol=1e-3, rtol=1e-3, equal_nan=True))

    def test_conv_bias_mask_relu(self):
        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out = ConvBiasMaskReLU(self.x, self.conv1.weight, self.conv1.bias.reshape(1, -1, 1, 1), self.mask, self.conv_pad, self.conv_stride)
            loss = (out.float()**2).sum() / out.numel()
        loss.backward()
        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out_ = F.relu(self.conv1_(self.x_) * self.mask_)
            loss_ = (out_**2).sum() / out_.numel()
        loss_.backward()

        self.assertTrue(torch.allclose(self.x_, self.x, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.conv1_.bias.grad, self.conv1.bias.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.conv1_.weight.grad, self.conv1.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.x_.grad, self.x.grad, atol=1e-3, rtol=1e-3, equal_nan=True))


    def test_conv_bias_retinanet(self):
        # RetinaNet configuration
        batch_size = 32
        in_channels = 256
        out_channels = 2376
        h, w = 100, 100
        
        # Input in NHWC format with HALF precision
        x = torch.randn(batch_size, in_channels, h, w).cuda()\
            .to(memory_format=torch.channels_last).half()
        x_ = x.clone()
        x.requires_grad_()
        x_.requires_grad_()
        
        # Conv layer
        conv = torch.nn.Conv2d(in_channels, out_channels, 3, 
                            stride=1, padding=1).cuda()\
                            .to(memory_format=torch.channels_last)
        conv_ = copy.deepcopy(conv)
        
        # Test with FP16
        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out = ConvBias(x, conv.weight, conv.bias.reshape(1, -1, 1, 1), 1, 1)
            loss = (out.float()**2).sum() / out.numel()
        loss.backward()
        
        # Reference with FP16
        with torch.amp.autocast(device_type="cuda", dtype=torch.half):
            out_ = conv_(x_)
            loss_ = (out_**2).sum() / out_.numel()
        loss_.backward()

        self.assertTrue(torch.allclose(out, out_, atol=1e-2, rtol=1e-2))


if __name__ == '__main__':
    unittest.main()

