#include <torch/extension.h>
#include <vector>
#include <cassert>
#include "compat.h"

namespace {
void compute_n1_n2(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    int& n1,
    int& n2)
{
    int idiff = input.ndimension() - normalized_shape.size();
    n2 = 1;
    for (int i = 0;  i < (int)normalized_shape.size();  ++i) {
	    assert( input.sizes()[i+idiff] == normalized_shape[i] );
	    n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0;  i < idiff;  ++i) {
	    n1 *= input.sizes()[i];
    }
}

void check_args(
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
    TORCH_CHECK(!beta.defined() || beta.sizes().equals(normalized_shape));
}

void check_args(
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
}


void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    int& n1,
    int& n2
    )
{
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input,normalized_shape,n1,n2);
}

void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma,beta);
}

void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma);
}
}

void cuda_layer_norm(
    at::Tensor* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> layer_norm(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon) {
  CHECK_INPUT(input);
  int n1,n2;
  check_args(input,normalized_shape,n1,n2);
  at::Tensor output = at::empty_like(input);
  at::Tensor mean = at::empty({n1}, input.options().dtype(input.scalar_type()==at::ScalarType::Half || input.scalar_type()==at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  at::Tensor invvar = at::empty_like(mean);
  cuda_layer_norm(&output,&mean,&invvar,&input,n1,n2,
      normalized_shape,NULL,NULL,epsilon);
  return {output, mean, invvar};
}

std::vector<at::Tensor> layer_norm_affine(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1,n2;
  check_args(input,normalized_shape,gamma,beta,n1,n2);
  at::Tensor output = at::empty_like(input);
  const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
  at::Tensor mean = at::empty({n1}, input.options().dtype(stats_dtype));
  at::Tensor invvar = at::empty_like(mean);
  cuda_layer_norm(&output,&mean,&invvar,&input,n1,n2,
      normalized_shape,&gamma,&beta,epsilon);
  return {output, mean, invvar};
}

std::vector<at::Tensor> layer_norm_affine_mixed_dtypes(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(input);
  int n1, n2;
  check_args(input, normalized_shape, n1, n2);
  at::Tensor output = at::empty_like(input, gamma.options().dtype(gamma.scalar_type()));
  at::Tensor mean = at::empty({n1}, input.options().dtype(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  at::Tensor invvar = at::empty_like(mean);
   cuda_layer_norm(&output, &mean, &invvar, &input, n1, n2,
      normalized_shape, &gamma, &beta, epsilon);
  return {output, mean, invvar};
}

void cuda_layer_norm_gradient(
    at::Tensor* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta,
    bool memory_efficient
    );

at::Tensor layer_norm_gradient(
    at::Tensor dout,
    c10::optional<at::Tensor> mean_,
    at::Tensor invvar,
    at::Tensor input_or_output,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon,
    bool memory_efficient) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input_or_output);
  int n1,n2;
  check_args(input_or_output,normalized_shape,n1,n2);
  at::Tensor grad_input = at::empty_like(input_or_output);
  if (mean_.has_value()) {
    cuda_layer_norm_gradient(&dout,&mean_.value(),&invvar,&input_or_output,n1,n2,
        normalized_shape,NULL,NULL,epsilon,
        &grad_input,NULL,NULL,memory_efficient);
  } else {
    cuda_layer_norm_gradient(&dout,NULL,&invvar,&input_or_output,n1,n2,
        normalized_shape,NULL,NULL,epsilon,
        &grad_input,NULL,NULL,memory_efficient);
  }
  return grad_input;
}

std::vector<at::Tensor> layer_norm_gradient_affine(
    at::Tensor dout,
    c10::optional<at::Tensor> mean_,
    at::Tensor invvar,
    at::Tensor input_or_output,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon,
    bool memory_efficient) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input_or_output);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1,n2;
  check_args(input_or_output,normalized_shape,gamma,beta,n1,n2);
  at::Tensor grad_input = at::empty_like(input_or_output);
  at::Tensor grad_gamma = at::empty_like(gamma);
  at::Tensor grad_beta = at::empty_like(beta);
//   at::Tensor *mean = mean_.has_value() ? &mean_.value() : NULL;
  if (mean_.has_value()) {
    cuda_layer_norm_gradient(&dout,&mean_.value(),&invvar,&input_or_output,n1,n2,
        normalized_shape,&gamma,&beta,epsilon,
        &grad_input,&grad_gamma,&grad_beta,memory_efficient);
  } else {
    cuda_layer_norm_gradient(&dout,NULL,&invvar,&input_or_output,n1,n2,
        normalized_shape,&gamma,&beta,epsilon,
        &grad_input,&grad_gamma,&grad_beta,memory_efficient);
  }
  return {grad_input, grad_gamma, grad_beta};
}

void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> rms_norm(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon) {
  CHECK_INPUT(input);
  int n1,n2;
  check_args(input,normalized_shape,n1,n2);
  at::Tensor output = at::empty_like(input);
  at::Tensor invvar = at::empty({n1}, input.options().dtype(input.scalar_type()==at::ScalarType::Half || input.scalar_type()==at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  cuda_rms_norm(&output,&invvar,&input,n1,n2,
      normalized_shape,NULL,epsilon);
  return {output, invvar};
}

std::vector<at::Tensor> rms_norm_affine(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    double epsilon) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  int n1,n2;
  check_args(input,normalized_shape,gamma,n1,n2);
  at::Tensor output = at::empty_like(input);
  const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
  at::Tensor invvar = at::empty({n1}, input.options().dtype(stats_dtype));
  cuda_rms_norm(&output,&invvar,&input,n1,n2,
      normalized_shape,&gamma,epsilon);
  return {output, invvar};
}

std::vector<at::Tensor> rms_norm_affine_mixed_dtypes(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    double epsilon) {
  CHECK_INPUT(input);
  int n1, n2;
  check_args(input, normalized_shape, n1, n2);
  at::Tensor output = at::empty_like(input, gamma.options().dtype(gamma.scalar_type()));
  at::Tensor invvar = at::empty({n1}, input.options().dtype(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));

   cuda_rms_norm(&output,&invvar, &input, n1, n2,
      normalized_shape, &gamma,epsilon);
  return {output,invvar};
}

void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    bool memory_efficient);

at::Tensor rms_norm_gradient(
    at::Tensor dout,
    at::Tensor invvar,
    at::Tensor input_or_output,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon,
    bool memory_efficient) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input_or_output);
  int n1,n2;
  check_args(input_or_output,normalized_shape,n1,n2);
  at::Tensor grad_input = at::empty_like(input_or_output);
  cuda_rms_norm_gradient(&dout,&invvar,&input_or_output,n1,n2,
      normalized_shape,NULL,epsilon,
      &grad_input,NULL,memory_efficient);
  return grad_input;
}

std::vector<at::Tensor> rms_norm_gradient_affine(
    at::Tensor dout,
    at::Tensor invvar,
    at::Tensor input_or_output,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    double epsilon,
    bool memory_efficient) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input_or_output);
  CHECK_INPUT(gamma);
  int n1,n2;
  check_args(input_or_output,normalized_shape,gamma,n1,n2);
  at::Tensor grad_input = at::empty_like(input_or_output);
  at::Tensor grad_gamma = at::empty_like(gamma);
  cuda_rms_norm_gradient(&dout,&invvar,&input_or_output,n1,n2,
      normalized_shape,&gamma,epsilon,
      &grad_input,&grad_gamma,memory_efficient);
  return {grad_input, grad_gamma};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_affine", &layer_norm_affine, "LayerNorm forward (CUDA)", py::call_guard<py::gil_scoped_release>());
  m.def("forward", &layer_norm, "LayerNorm forward (CUDA)", py::call_guard<py::gil_scoped_release>());
  m.def("backward_affine", &layer_norm_gradient_affine, "LayerNorm backward (CUDA)", py::call_guard<py::gil_scoped_release>());
  m.def("backward", &layer_norm_gradient, "LayerNorm backward (CUDA)", py::call_guard<py::gil_scoped_release>());

  m.def("forward_affine_mixed_dtypes", &layer_norm_affine_mixed_dtypes, "LayerNorm forward with mixed dtypes (CUDA) compatible with Megatron's implementation", py::call_guard<py::gil_scoped_release>());

  m.def("rms_forward_affine", &rms_norm_affine, "RMSNorm forward (CUDA)", py::call_guard<py::gil_scoped_release>());
  m.def("rms_forward", &rms_norm, "RMSNorm forward (CUDA)", py::call_guard<py::gil_scoped_release>());
  m.def("rms_backward_affine", &rms_norm_gradient_affine, "RMSNorm backward (CUDA)", py::call_guard<py::gil_scoped_release>());
  m.def("rms_backward", &rms_norm_gradient, "RMSNorm backward (CUDA)", py::call_guard<py::gil_scoped_release>());

  m.def("rms_forward_affine_mixed_dtypes", &rms_norm_affine_mixed_dtypes, "RMSNorm forward with mixed dtypes (CUDA) compatible with Megatron's implementation", py::call_guard<py::gil_scoped_release>());
}