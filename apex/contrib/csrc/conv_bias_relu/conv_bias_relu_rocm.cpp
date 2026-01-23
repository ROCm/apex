#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>
#include <miopen/miopen.h>

// Try to include PyTorch's MIOpen handle helper
#if defined(USE_ROCM)
#include <ATen/miopen/Handle.h>
#endif

#define MIOPEN_CHECK(status)                                                               \
    if (status != miopenStatusSuccess) {                                                   \
        std::string err = "MIOpen error: " + std::to_string(status);                       \
        throw std::runtime_error(err);                                                     \
    }

// Plan Cache for MIOpen Fusion
struct FusionPlanEntry {
    miopenFusionPlan_t plan;
    miopenFusionOpDescriptor_t conv_op;
    miopenFusionOpDescriptor_t bias_op;
    miopenFusionOpDescriptor_t activ_op;
};

static std::unordered_map<std::string, FusionPlanEntry> plan_cache;

std::string get_cache_key(const at::Tensor& x, const at::Tensor& w, int64_t padding, int64_t stride) {
    return std::to_string(x.size(0)) + "_" + std::to_string(x.size(1)) + "_" + 
           std::to_string(x.size(2)) + "_" + std::to_string(x.size(3)) + "_" +
           std::to_string(w.size(0)) + "_" + std::to_string(w.size(1)) + "_" +
           std::to_string(w.size(2)) + "_" + std::to_string(w.size(3)) + "_" +
           std::to_string(padding) + "_" + std::to_string(stride) + "_" +
           (x.is_contiguous(at::MemoryFormat::ChannelsLast) ? "NHWC" : "NCHW");
}

std::vector<at::Tensor> conv_bias_relu_forward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto bias = inputs[2];

    // Fallback if not CUDA/HIP
    if (!x.is_cuda()) {
        auto out = at::convolution(x, weight, bias, {stride, stride}, {padding, padding}, {1, 1}, false, {0, 0}, 1);
        return {at::relu(out)};
    }

    miopenHandle_t handle = at::native::getMiopenHandle();
    std::string key = get_cache_key(x, weight, padding, stride);
    
    bool is_nhwc = x.is_contiguous(at::MemoryFormat::ChannelsLast);
    miopenDataType_t dtype = (x.scalar_type() == at::kHalf) ? miopenHalf : miopenFloat;

    // Check cache
    if (plan_cache.find(key) == plan_cache.end()) {
        miopenFusionPlan_t plan;
        miopenTensorDescriptor_t input_desc;
        miopenCreateTensorDescriptor(&input_desc);
        
        if (is_nhwc) {
            std::vector<int> dims = {(int)x.size(0), (int)x.size(1), (int)x.size(2), (int)x.size(3)};
            std::vector<int> strides = {(int)(x.size(1) * x.size(2) * x.size(3)), 1, (int)(x.size(1) * x.size(3)), (int)x.size(1)};
            MIOPEN_CHECK(miopenSetTensorDescriptor(input_desc, dtype, 4, dims.data(), strides.data()));
        } else {
            MIOPEN_CHECK(miopenSet4dTensorDescriptor(input_desc, dtype, x.size(0), x.size(1), x.size(2), x.size(3)));
        }

        MIOPEN_CHECK(miopenCreateFusionPlan(&plan, miopenVerticalFusion, input_desc));

        // 1. Conv Op
        miopenFusionOpDescriptor_t conv_op;
        miopenConvolutionDescriptor_t conv_desc;
        miopenCreateConvolutionDescriptor(&conv_desc);
        MIOPEN_CHECK(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution, 
            padding, padding, stride, stride, 1, 1));
        
        miopenTensorDescriptor_t weight_desc;
        miopenCreateTensorDescriptor(&weight_desc);
        if (is_nhwc) {
            std::vector<int> w_dims = {(int)weight.size(0), (int)weight.size(1), (int)weight.size(2), (int)weight.size(3)};
            std::vector<int> w_strides = {(int)(weight.size(1) * weight.size(2) * weight.size(3)), 1, (int)(weight.size(1) * weight.size(3)), (int)weight.size(1)};
            MIOPEN_CHECK(miopenSetTensorDescriptor(weight_desc, dtype, 4, w_dims.data(), w_strides.data()));
        } else {
            MIOPEN_CHECK(miopenSet4dTensorDescriptor(weight_desc, dtype, weight.size(0), weight.size(1), weight.size(2), weight.size(3)));
        }

        MIOPEN_CHECK(miopenCreateOpConvForward(plan, &conv_op, conv_desc, weight_desc));

        // 2. Bias Op
        miopenFusionOpDescriptor_t bias_op;
        miopenTensorDescriptor_t bias_desc;
        miopenCreateTensorDescriptor(&bias_desc);
        MIOPEN_CHECK(miopenSet4dTensorDescriptor(bias_desc, dtype, 1, x.size(1), 1, 1));
        MIOPEN_CHECK(miopenCreateOpBiasForward(plan, &bias_op, bias_desc));

        // 3. Activation Op
        miopenFusionOpDescriptor_t activ_op;
        MIOPEN_CHECK(miopenCreateOpActivationForward(plan, &activ_op, miopenActivationRELU));

        // Compile
        MIOPEN_CHECK(miopenCompileFusionPlan(handle, plan));
        
        plan_cache[key] = {plan, conv_op, bias_op, activ_op};
        
        miopenDestroyTensorDescriptor(input_desc);
        miopenDestroyTensorDescriptor(weight_desc);
        miopenDestroyTensorDescriptor(bias_desc);
        miopenDestroyConvolutionDescriptor(conv_desc);
    }

    auto& entry = plan_cache[key];

    // Calculate output dimensions
    int64_t n = x.size(0);
    int64_t oc = weight.size(0);
    int64_t h = (x.size(2) + 2 * padding - weight.size(2)) / stride + 1;
    int64_t w = (x.size(3) + 2 * padding - weight.size(3)) / stride + 1;
    std::vector<int64_t> out_shape = {n, oc, h, w};
    
    auto out = at::empty(out_shape, x.options().memory_format(is_nhwc ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous));

    miopenTensorDescriptor_t input_desc, output_desc;
    miopenCreateTensorDescriptor(&input_desc);
    miopenCreateTensorDescriptor(&output_desc);

    if (is_nhwc) {
        std::vector<int> x_dims = {(int)x.size(0), (int)x.size(1), (int)x.size(2), (int)x.size(3)};
        std::vector<int> x_strides = {(int)(x.size(1) * x.size(2) * x.size(3)), 1, (int)(x.size(1) * x.size(3)), (int)x.size(1)};
        miopenSetTensorDescriptor(input_desc, dtype, 4, x_dims.data(), x_strides.data());

        std::vector<int> y_dims = {(int)out.size(0), (int)out.size(1), (int)out.size(2), (int)out.size(3)};
        std::vector<int> y_strides = {(int)(out.size(1) * out.size(2) * out.size(3)), 1, (int)(out.size(1) * out.size(3)), (int)out.size(1)};
        miopenSetTensorDescriptor(output_desc, dtype, 4, y_dims.data(), y_strides.data());
    } else {
        miopenSet4dTensorDescriptor(input_desc, dtype, x.size(0), x.size(1), x.size(2), x.size(3));
        miopenSet4dTensorDescriptor(output_desc, dtype, out.size(0), out.size(1), out.size(2), out.size(3));
    }

    miopenOperatorArgs_t args;
    miopenCreateOperatorArgs(&args);
    
    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenSetOpArgsConvForward(args, entry.conv_op, &alpha, &beta, weight.data_ptr()));
    MIOPEN_CHECK(miopenSetOpArgsBiasForward(args, entry.bias_op, &alpha, &beta, bias.data_ptr()));
    MIOPEN_CHECK(miopenSetOpArgsActivationForward(args, entry.activ_op, &alpha, &beta, 0, 0, 0));

    MIOPEN_CHECK(miopenExecuteFusionPlan(handle, entry.plan, 
        input_desc, x.data_ptr(),
        output_desc, out.data_ptr(),
        args));

    miopenDestroyOperatorArgs(args);
    miopenDestroyTensorDescriptor(input_desc);
    miopenDestroyTensorDescriptor(output_desc);

    return {out};
}

std::vector<at::Tensor> conv_bias_relu_backward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto out = inputs[2];
    auto grad_output = inputs[3];
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
    return {at::convolution(x, weight, bias, {stride, stride}, {padding, padding}, {1, 1}, false, {0, 0}, 1)};
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
    return {at::relu(out)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_bias_relu_forward, "Fused Conv-Bias-ReLU forward (ROCm MIOpen Fusion)");
  m.def("backward", &conv_bias_relu_backward, "Fused Conv-Bias-ReLU backward (ROCm)");
  m.def("forward_no_relu", &conv_bias_forward, "Fused Conv-Bias forward (ROCm)");
  m.def("backward_no_relu", &conv_bias_backward, "Fused Conv-Bias backward (ROCm)");
  m.def("forward_mask", &conv_bias_mask_relu_forward, "Fused Conv-Bias-Mask-ReLU forward (ROCm)");
}
