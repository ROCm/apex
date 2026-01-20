#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>


#define ASSERT_UINT4_ALIGNED(PTR)                                              \
  TORCH_INTERNAL_ASSERT(is_aligned<uint4>(PTR), "Tensor " #PTR " is not uint4 aligned")

template <class T> bool is_aligned(const void *ptr) noexcept {
  auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
  return !(iptr % alignof(T));
}

template <bool SMOOTHING, int ILP, typename scalar_t, typename labelscalar_t,
          typename accscalar_t, typename outscalar_t>
__global__ void focal_loss_forward_cuda_kernel(
    outscalar_t *loss, scalar_t *partial_grad,
    const scalar_t *__restrict__ cls_output,
    const labelscalar_t *__restrict__ cls_targets_at_level,
    const float *__restrict__ num_positives_sum, const int64_t num_examples,
    const int64_t num_classes, const int64_t num_real_classes,
    const float alpha, const float gamma, const float smoothing_factor) {
  extern __shared__ unsigned char shm[];
  accscalar_t *loss_shm = reinterpret_cast<accscalar_t *>(shm);
  labelscalar_t *labels_shm = reinterpret_cast<labelscalar_t *>(shm + blockDim.x * sizeof(accscalar_t));
  
  loss_shm[threadIdx.x] = 0;
  accscalar_t loss_acc = 0;

  accscalar_t one = accscalar_t(1.0);
  accscalar_t K = accscalar_t(2.0);
  
  accscalar_t nn_norm, np_norm, pn_norm, pp_norm;

  // *_norm is used for label smoothing only
  if (SMOOTHING) {
    nn_norm = one - smoothing_factor / K;
    np_norm = smoothing_factor / K;
    pn_norm = smoothing_factor - smoothing_factor / K;
    pp_norm = one - smoothing_factor + smoothing_factor / K;
  }

  uint4 p_vec, grad_vec;

  // Accumulate loss on each thread
  int64_t stride = (int64_t)gridDim.x * blockDim.x * ILP;
  int64_t i_base = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x) * ILP;
  int64_t idy = i_base / num_classes;
  int64_t base_yid = i_base % num_classes;

  int64_t stride_div = stride / num_classes;
  int64_t stride_rem = stride % num_classes;

  for (int64_t loop_offset = (int64_t)blockIdx.x * blockDim.x * ILP;
       loop_offset < num_examples * num_classes;
       loop_offset += stride) {
    
    // 1. Cooperatively load labels into LDS
    int64_t idy_block_start = loop_offset / num_classes;
    int64_t idy_block_end = (loop_offset + (int64_t)blockDim.x * ILP - 1) / num_classes;
    int64_t num_labels_to_load = idy_block_end - idy_block_start + 1;

    for (int l = threadIdx.x; l < num_labels_to_load; l += blockDim.x) {
      labels_shm[l] = cls_targets_at_level[idy_block_start + l];
    }
    __syncthreads();

    // 2. Process elements
    int64_t i = loop_offset + threadIdx.x * ILP;
    if (i < num_examples * num_classes) {
      labelscalar_t y = labels_shm[idy - idy_block_start];
      int64_t pos_idx = idy * num_classes + y;
      p_vec = *(uint4 *)&cls_output[i];

      // Skip ignored matches
      if (y == -2) {
#pragma unroll
        for (int j = 0; j < ILP; j++) {
          *((scalar_t *)(&grad_vec) + j) = 0;
        }
        *(uint4 *)&partial_grad[i] = grad_vec;
      } else {
#pragma unroll
        for (int j = 0; j < ILP; j++) {
          // Skip the pad classes
          if (base_yid + j >= num_real_classes) {
            *((scalar_t *)(&grad_vec) + j) = 0;
            continue;
          }

          accscalar_t p = static_cast<accscalar_t>(*((scalar_t *)(&p_vec) + j));

          // Optimized Transcendental: Single exp and stable log1p
          accscalar_t abs_p = (p >= 0) ? p : -p;
          accscalar_t exp_neg_abs = ::exp(-abs_p);
          accscalar_t sigma = (p >= 0) ? (one / (one + exp_neg_abs)) : (exp_neg_abs / (one + exp_neg_abs));
          accscalar_t off_a = ::log1p(exp_neg_abs) + ((p < 0) ? abs_p : 0);

          // Negative matches
          accscalar_t base = SMOOTHING ? nn_norm * p : p;
          accscalar_t off_b = (SMOOTHING ? np_norm : 0) - sigma;
          accscalar_t coeff_f1 = one - alpha;
          accscalar_t coeff_f2 = sigma;
          accscalar_t coeff_b1 = gamma;
          accscalar_t coeff_b2 = one - sigma;

          // Positive matches
          if (y >= 0 && (i + j == pos_idx)) {
            base = SMOOTHING ? pn_norm * p : 0;
            off_b = (SMOOTHING ? pp_norm : one) - sigma;
            coeff_f1 = alpha;
            coeff_f2 = one - sigma;
            coeff_b1 = -gamma;
            coeff_b2 = sigma;
          }

          accscalar_t coeff_f = coeff_f1 * ::pow(coeff_f2, gamma);
          accscalar_t coeff_b = coeff_b1 * coeff_b2;

          accscalar_t loss_t = coeff_f * (base + off_a);
          accscalar_t grad = coeff_f * (coeff_b * (base + off_a) - off_b);

          loss_acc += loss_t;
          *((scalar_t *)(&grad_vec) + j) = static_cast<scalar_t>(grad);
        }
        *(uint4 *)&partial_grad[i] = grad_vec;
      }
    }

    // Update indices for next iteration (Division-free)
    idy += stride_div;
    base_yid += stride_rem;
    if (base_yid >= num_classes) {
      idy++;
      base_yid -= num_classes;
    }
    __syncthreads(); // Ensure LDS is ready for next iteration
  }
  loss_shm[threadIdx.x] = loss_acc;

  // Intra-CTA reduction
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      loss_shm[threadIdx.x] += loss_shm[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Inter-CTA reduction
  if (threadIdx.x == 0) {
    accscalar_t normalizer = accscalar_t(1.0) / static_cast<accscalar_t>(num_positives_sum[0]);
    loss_acc = loss_shm[0] * normalizer;
    atomicAdd(loss, loss_acc);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t,
          typename outscalar_t>
__global__ void focal_loss_backward_cuda_kernel(
    scalar_t *partial_grad, const outscalar_t *__restrict__ grad_output,
    const float *__restrict__ num_positives_sum, const uint64_t numel) {
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ILP;

  accscalar_t normalizer = static_cast<accscalar_t>(grad_output[0]) /
                           static_cast<accscalar_t>(num_positives_sum[0]);

  // The input is enforced to pad to use vector load, thus there's no need to
  // check whether the last element of ILP can out of bound.
  if (idx >= numel)
    return;

  uint4 grad_vec;
  grad_vec = *(uint4 *)&partial_grad[idx];
#pragma unroll(ILP)
  for (int i = 0; i < ILP; i++) {
    auto grad = static_cast<accscalar_t>(*((scalar_t *)(&grad_vec) + i));
    grad *= normalizer;
    *((scalar_t *)(&grad_vec) + i) = static_cast<scalar_t>(grad);
  }
  *(uint4 *)&partial_grad[idx] = grad_vec;
}

std::vector<at::Tensor> focal_loss_forward_cuda(
    const at::Tensor &cls_output, const at::Tensor &cls_targets_at_level,
    const at::Tensor &num_positives_sum, const int64_t num_real_classes,
    const float alpha, const float gamma, const float smoothing_factor) {
  // Checks required for correctness
  TORCH_INTERNAL_ASSERT(cls_output.size(-1) >= num_real_classes,
             "Incorrect number of real classes.");
  TORCH_INTERNAL_ASSERT(cls_targets_at_level.scalar_type() == at::kLong,
             "Invalid label type.");
  TORCH_INTERNAL_ASSERT(
      (num_positives_sum.numel() == 1) &&
          (num_positives_sum.scalar_type() == at::kFloat),
      "Expect num_positives_sum to be a float32 tensor with only one element.");
  TORCH_INTERNAL_ASSERT(cls_output.dim() == cls_targets_at_level.dim() + 1,
             "Mis-matched dimensions between class output and label.");
  for (int64_t i = 0; i < cls_targets_at_level.dim(); i++)
    TORCH_INTERNAL_ASSERT(cls_output.size(i) == cls_targets_at_level.size(i),
               "Mis-matched shape between class output and label.");

  // Checks required for better performance
  const int ILP = sizeof(uint4) / cls_output.element_size();
  ASSERT_UINT4_ALIGNED(cls_output.data_ptr());
  TORCH_INTERNAL_ASSERT(cls_output.size(-1) % ILP == 0,
             "Pad number of classes first to take advantage of 128 bit load.");
  TORCH_INTERNAL_ASSERT(num_real_classes >= ILP, "Too few classes.");

  int64_t num_classes = cls_output.size(-1);
  int64_t num_examples = cls_output.numel() / num_classes;
  at::Tensor loss = at::zeros({}, cls_output.options().dtype(at::kFloat));

  // Compute the incompelete gradient during fprop since most of the heavy
  // functions of bprop are the same as fprop, thus trade memory for compute
  // helps with focal loss.
  at::Tensor partial_grad = at::empty_like(cls_output);

  // The grid contains 2 CTA per SM, each CTA loop on input with stride till the
  // last item.
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, at::cuda::current_device());
  dim3 block(512);
  dim3 grid(2 * props.multiProcessorCount);

  // Specialize on label smoothing or not to reduce redundant operations
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (smoothing_factor == 0.0f) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_fprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          const int ILP = sizeof(uint4) / sizeof(scalar_t);
          // Allocate enough SHM for reduction AND label cache (max labels ~514)
          size_t shm_size = block.x * sizeof(accscalar_t) + (block.x * ILP / ILP + 2) * sizeof(labelscalar_t);
          focal_loss_forward_cuda_kernel<false, ILP, scalar_t, labelscalar_t,
                                         accscalar_t, outscalar_t>
              <<<grid, block, shm_size, stream>>>(
                  loss.data_ptr<outscalar_t>(),
                  partial_grad.data_ptr<scalar_t>(),
                  cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum.data_ptr<float>(), num_examples,
                  num_classes, num_real_classes, alpha, gamma,
                  smoothing_factor);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_fprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          const int ILP = sizeof(uint4) / sizeof(scalar_t);
          size_t shm_size = block.x * sizeof(accscalar_t) + (block.x * ILP / ILP + 2) * sizeof(labelscalar_t);
          focal_loss_forward_cuda_kernel<true, ILP, scalar_t, labelscalar_t,
                                         accscalar_t, outscalar_t>
              <<<grid, block, shm_size, stream>>>(
                  loss.data_ptr<outscalar_t>(),
                  partial_grad.data_ptr<scalar_t>(),
                  cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum.data_ptr<float>(), num_examples,
                  num_classes, num_real_classes, alpha, gamma,
                  smoothing_factor);
        });
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return {loss, partial_grad};
}

at::Tensor focal_loss_backward_cuda(const at::Tensor &grad_output,
                                    const at::Tensor &partial_grad,
                                    const at::Tensor &num_positives_sum) {
  // Each thread process ILP elements
  const int ILP = sizeof(uint4) / partial_grad.element_size();
  dim3 block(512);
  dim3 grid((partial_grad.numel() + block.x * ILP - 1) / (block.x * ILP));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      partial_grad.scalar_type(), "focal_loss_bprop", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        using outscalar_t = float;
        const int ILP = sizeof(uint4) / sizeof(scalar_t);
        focal_loss_backward_cuda_kernel<ILP, scalar_t, accscalar_t, outscalar_t>
            <<<grid, block, 0, stream>>>(partial_grad.data_ptr<scalar_t>(),
                                         grad_output.data_ptr<outscalar_t>(),
                                         num_positives_sum.data_ptr<float>(),
                                         partial_grad.numel());
      });

  AT_CUDA_CHECK(cudaGetLastError());
  return partial_grad;
}
