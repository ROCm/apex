#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <tuple>

// ROCm implementation of fused_conv_bias_relu using ATen operations.
// While not a single fused kernel call via MIOpen Fusion Plan yet,
// this allows the build to succeed and uses ROCm-optimized kernels.

std::vector<at::Tensor> conv_bias_relu_forward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto bias = inputs[2];

    auto out = at::convolution(x, weight, bias, {stride, stride}, {padding, padding}, {1, 1}, false, {0, 0}, 1);
    out = at::relu(out);

    return {out};
}

std::vector<at::Tensor> conv_bias_relu_backward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto out = inputs[2];
    auto grad_output = inputs[3];

    // ReLU backward: grad_output * (out > 0)
    auto grad_relu = grad_output * (out > 0).to(grad_output.dtype());

    int64_t bias_size = weight.size(0);
    std::vector<int64_t> bias_sizes = {bias_size};

    auto grads = at::convolution_backward(grad_relu, x, weight, 
                                         bias_sizes,
                                         {stride, stride}, {padding, padding}, {1, 1}, 
                                         false, {0, 0}, 1,
                                         {true, true, true});

    return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
}

std::vector<at::Tensor> conv_bias_forward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto bias = inputs[2];

    auto out = at::convolution(x, weight, bias, {stride, stride}, {padding, padding}, {1, 1}, false, {0, 0}, 1);

    return {out};
}

std::vector<at::Tensor> conv_bias_backward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto grad_output = inputs[2];

    int64_t bias_size = weight.size(0);
    std::vector<int64_t> bias_sizes = {bias_size};

    auto grads = at::convolution_backward(grad_output, x, weight, 
                                         bias_sizes,
                                         {stride, stride}, {padding, padding}, {1, 1}, 
                                         false, {0, 0}, 1,
                                         {true, true, true});

    return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
}

std::vector<at::Tensor> conv_bias_mask_relu_forward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto bias = inputs[2];
    auto mask = inputs[3];

    auto out = at::convolution(x, weight, bias, {stride, stride}, {padding, padding}, {1, 1}, false, {0, 0}, 1);
    out = out * mask.to(out.dtype());
    out = at::relu(out);

    return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_bias_relu_forward, "Fused Conv-Bias-ReLU forward (ROCm)");
  m.def("backward", &conv_bias_relu_backward, "Fused Conv-Bias-ReLU backward (ROCm)");
  m.def("forward_no_relu", &conv_bias_forward, "Fused Conv-Bias forward (ROCm)");
  m.def("backward_no_relu", &conv_bias_backward, "Fused Conv-Bias backward (ROCm)");
  m.def("forward_mask", &conv_bias_mask_relu_forward, "Fused Conv-Bias-Mask-ReLU forward (ROCm)");
}
