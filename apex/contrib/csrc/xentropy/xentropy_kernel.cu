/**
 * From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * From Caffe2:
 *
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 *
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 *
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 *
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 *
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 *
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>

#include "type_shim.h"
#include "compat.h"

#define ALIGN_BYTES 16

#ifdef USE_ROCM
#define SYNCWARP(mask)
#else
#define SYNCWARP(mask) __syncwarp(mask)
#endif

using Tensor = at::Tensor;
using TensorList = at::TensorList;
using ScalarType = at::ScalarType;
using at::acc_type;

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : logsum(max_input + std::log(sum)) {}

  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_log_sum_exp)
    : logsum(max_log_sum_exp) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(input - logsum);
  }

  const AccumT logsum;
};

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - std::exp(static_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};



const int max_threads = 1024;

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));
  while (block_size < (max_block_size/2)) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<uint64_t>(at::cuda::warp_size()));
  return dim3(block_size);
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};


////////////////////////////////////////////////////////////////////////////////
// Regular kernel (fast when dim_size is large; requires inner_size == 1)
////////////////////////////////////////////////////////////////////////////////


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + std::exp(v - max_k);
  }

  const AccumT max_k;
};

template <template<typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT>& r,
            AccumT defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  uint32_t mask = (((uint64_t)1) << (blockDim.x / C10_WARP_SIZE)) - 1;
  if (threadIdx.x < C10_WARP_SIZE) {
    int lane = threadIdx.x % C10_WARP_SIZE;
    if (lane < blockDim.x / C10_WARP_SIZE) {
#pragma unroll
      for (int i = 0; i < C10_WARP_SIZE; ++i) {
        warpVal = r(warpVal, smem[lane * C10_WARP_SIZE + i]);
      }
      SYNCWARP(mask);
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <template<typename> class Reduction1, template<typename> class Reduction2, typename AccumT>
__device__ __forceinline__ void
blockReduce(AccumT* smem,
            AccumT* reducVal1,
            AccumT val1,
            const Reduction1<AccumT>& r1,
            AccumT defaultVal1,
            AccumT* reducVal2,
            AccumT val2,
            const Reduction2<AccumT>& r2,
            AccumT defaultVal2)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val1;
  smem[blockDim.x + threadIdx.x] = val2;

  __syncthreads();

  AccumT warpVal1 = defaultVal1;
  AccumT warpVal2 = defaultVal2;

  // First warp will perform per-warp reductions for the remaining warps
  uint32_t mask = (((uint64_t)1) << (blockDim.x / C10_WARP_SIZE)) - 1;
  if (threadIdx.x < C10_WARP_SIZE) {
    int lane = threadIdx.x % C10_WARP_SIZE;
    if (lane < blockDim.x / C10_WARP_SIZE) {
#pragma unroll
      for (int i = 0; i < C10_WARP_SIZE; ++i) {
        warpVal1 = r1(warpVal1, smem[lane * C10_WARP_SIZE + i]);
        warpVal2 = r2(warpVal2, smem[lane * C10_WARP_SIZE + i + blockDim.x]);
      }
      SYNCWARP(mask);
      smem[lane] = warpVal1;
      smem[lane + blockDim.x] = warpVal2;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal1 = defaultVal1;
  AccumT blockVal2 = defaultVal2;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
      blockVal1 = r1(blockVal1, smem[i]);
      blockVal2 = r2(blockVal2, smem[i + blockDim.x]);
    }
    smem[0] = blockVal1;
    smem[blockDim.x] = blockVal2;
  }

  // Sync and broadcast
  __syncthreads();
  *reducVal1 = smem[0];
  *reducVal2 = smem[blockDim.x];
  __syncthreads();
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(int shift,
          T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LoadT;
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;

  // shift and do 1
  if(shift > 0){
    data -= shift;
    size += shift;
    if(threadIdx.x >= shift){
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  int last = size % (ILP * blockDim.x);

  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];

    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }

  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

template <template<typename, typename> class Reduction1, template<typename, typename> class Reduction2, int ILP, typename T, typename AccumT>
__device__ __forceinline__ void
ilpReduce(int shift,
          T* data,
          int size,
          AccumT* reducVal1,
          const Reduction1<T, AccumT>& r1,
          AccumT defaultVal1,
          AccumT* reducVal2,
          const Reduction2<T, AccumT>& r2,
          AccumT defaultVal2)
{
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LoadT;

  AccumT threadVal1 = defaultVal1;
  AccumT threadVal2 = defaultVal2;
  int offset = threadIdx.x;

  // shift and do 1
  if(shift > 0){
    data -= shift;
    size += shift;
    if(threadIdx.x >= shift){
      threadVal1 = r1(threadVal1, data[offset]);
      threadVal2 = r2(threadVal2, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  int last = size % (ILP * blockDim.x);

  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];

    for (int j = 0; j < ILP; ++j) {
      threadVal1 = r1(threadVal1, v[j]);
      threadVal2 = r2(threadVal2, v[j]);
    }
  }

  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x) {
    threadVal1 = r1(threadVal1, data[offset]);
    threadVal2 = r2(threadVal2, data[offset]);
  }

  *reducVal1 = threadVal1;
  *reducVal2 = threadVal2;
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxXEntropyForward(
    accscalar_t *losses,
    outscalar_t *max_log_sum_exp,
    scalar_t *input,
    int64_t *labels,
    int64_t classes,
    const float smoothing)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  //output += blockIdx.x * classes;
  const int shift = ((uint64_t)input) % ALIGN_BYTES / sizeof(scalar_t);

  int64_t label = labels[blockIdx.x];

  // find the max and sum
  accscalar_t threadMax, threadSum, max_k, sum_k;
  ilpReduce<MaxFloat, AddFloat, ILP, scalar_t, accscalar_t>(
    shift, input, classes,
    &threadMax, MaxFloat<scalar_t, accscalar_t>(),
    -at::numeric_limits<accscalar_t>::max(),
    &threadSum, AddFloat<scalar_t, accscalar_t>(),
    static_cast<accscalar_t>(0));

  blockReduce<Max, Add, accscalar_t>(
      sdata,
      &max_k, threadMax, Max<accscalar_t>(),
      -at::numeric_limits<accscalar_t>::max(),
      &sum_k, threadSum, Add<accscalar_t>(),
      static_cast<accscalar_t>(0));

  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);

  // calculate per element loss with label smoothing
  // reserve max + log_sum_exp for bprop
  if (threadIdx.x == 0) {
    accscalar_t log_prob = epilogue(static_cast<accscalar_t>(input[label]));
    losses[blockIdx.x] = (max_k + std::log(sumAll) - sum_k / classes) \
      * smoothing - log_prob * (1 - smoothing);
    max_log_sum_exp[blockIdx.x] = max_k + std::log(sumAll);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t>
__device__ __forceinline__ void
apply(scalar_t *gradInput,
      scalar_t *logits,
      outscalar_t *max_log_sum_exp,
      outscalar_t *gradOutput,
      int64_t *labels,
      const float smoothing,
      int classes)
{
  accscalar_t smooth_positives = 1.0 - smoothing;
  accscalar_t smooth_negatives = smoothing / classes;
  accscalar_t tmpGradOutput = gradOutput[blockIdx.x];
  int64_t label = labels[blockIdx.x];
  accscalar_t coeff = max_log_sum_exp[blockIdx.x];

  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);

  for (; offset < classes - last; offset += blockDim.x * ILP) {
    accscalar_t tmpLogits[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmpLogits[j] = static_cast<accscalar_t>(logits[offset + j * blockDim.x]);
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      gradInput[offset + j * blockDim.x] = tmpGradOutput * (
        std::exp(tmpLogits[j] - coeff) - static_cast<accscalar_t>(
          (offset + j * blockDim.x == label) ? 1 : 0) *
        smooth_positives - smooth_negatives);
  }

  for (; offset < classes; offset += blockDim.x)
    gradInput[offset] = tmpGradOutput * (std::exp(
        static_cast<accscalar_t>(logits[offset]) - coeff) -
        static_cast<accscalar_t>((offset == label) ? 1 : 0) *
        smooth_positives - smooth_negatives);
}


template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t>
__device__ __forceinline__ void
aligned_apply(int shift,
              scalar_t *gradInput,
              scalar_t *logits,
              outscalar_t *max_log_sum_exp,
              outscalar_t *gradOutput,
              int64_t *labels,
              const float smoothing,
              int classes)
{
  accscalar_t smooth_positives = 1.0 - smoothing;
  accscalar_t smooth_negatives = smoothing / classes;
  accscalar_t tmpGradOutput = gradOutput[blockIdx.x];
  int64_t label = labels[blockIdx.x];
  accscalar_t coeff = max_log_sum_exp[blockIdx.x];

  int offset = threadIdx.x;

  // shift and do 1
  if(shift > 0){
    logits -= shift;
    gradInput -= shift;
    classes += shift;
    if(threadIdx.x >= shift){
      gradInput[offset] = tmpGradOutput * (std::exp(
        static_cast<accscalar_t>(logits[offset]) - coeff) -
        static_cast<accscalar_t>(((offset - shift) == label) ? 1 : 0) *
        smooth_positives - smooth_negatives);
    }
    classes -= blockDim.x;
    gradInput += blockDim.x;
    logits += blockDim.x;
    shift -= blockDim.x;
  }

  int last = classes % (ILP * blockDim.x);

  typedef typename std::aligned_storage<ILP*sizeof(scalar_t), ILP*alignof(scalar_t)>::type LoadT;
  // input
  scalar_t v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);
  // output
  scalar_t r[ILP];
  LoadT* result = reinterpret_cast<LoadT*>(&r);

  for (; offset * ILP < (classes - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(logits)[offset];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      r[j] = tmpGradOutput * (std::exp(
          static_cast<accscalar_t>(v[j]) - coeff) -
          static_cast<accscalar_t>(((ILP * offset + j - shift) == label) ? 1 : 0) *
          smooth_positives - smooth_negatives);
    }
    reinterpret_cast<LoadT*>(gradInput)[offset] = *result;
  }

  offset = classes - last + threadIdx.x;
  for (; offset < classes; offset += blockDim.x)
    gradInput[offset] = tmpGradOutput * (std::exp(
        static_cast<accscalar_t>(logits[offset]) - coeff) -
        static_cast<accscalar_t>(((offset - shift) == label) ? 1 : 0) *
        smooth_positives - smooth_negatives);

}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxXEntropyBackward(
    scalar_t *gradInput,
    scalar_t *logits,
    outscalar_t *max_log_sum_exp,
    outscalar_t *gradOutput,
    int64_t *labels,
    const float smoothing,
    int classes)
{
  gradInput += blockIdx.x * classes;
  logits += blockIdx.x * classes;

  // Do vectorized load/store when input/output have same alignment
  const int shift = ((uint64_t)logits) % ALIGN_BYTES / sizeof(scalar_t);
  const int shift_ = ((uint64_t)gradInput) % ALIGN_BYTES / sizeof(scalar_t);
  if (shift == shift_){
    aligned_apply<ILP, scalar_t, accscalar_t, outscalar_t>(shift, gradInput, logits, max_log_sum_exp, gradOutput, labels, smoothing, classes);
  }
  else {
    apply<ILP, scalar_t, accscalar_t, outscalar_t>(gradInput, logits, max_log_sum_exp, gradOutput, labels, smoothing, classes);
  }

}

template<template<typename, typename, typename> class Epilogue>
std::vector<Tensor> host_softmax_xentropy(
        const Tensor & input_,
        const Tensor & labels_,
        const float smoothing,
        const bool half_to_float){
  if (half_to_float) AT_ASSERTM(input_.type().scalarType() == ScalarType::Half || input_.type().scalarType() == ScalarType::BFloat16,"conversion is supported for Half and BFloat16 type only");
  AT_ASSERTM(labels_.type().scalarType() == ScalarType::Long,"Label type should be CUDA Long");

  auto input = input_.contiguous();
  Tensor max_log_sum_exp = at::empty_like(labels_, half_to_float ? input.options().dtype(ScalarType::Float) : input.options());
  Tensor losses = at::empty_like(labels_, input_.options().dtype(ScalarType::Float));

  static_assert(std::is_same<acc_type<at::Half, true>, float>::value ||
    std::is_same<acc_type<at::Half, true>, double>::value,
    "accscalar_t for half should be float or double");
  AT_ASSERTM(input.dim() == 2, "Currently only 2 dim input supported");
  AT_ASSERTM(labels_.dim() == 1, "Labels should be 1 dimensional");
  AT_ASSERTM(input.size(0) == labels_.size(0), "Input and label should have same number of examples");
  AT_ASSERTM(input.numel() > 0, "Number of classes in input should not be 0");

  const int64_t dim = 1;
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);
  int64_t inner_size = 1;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input.size(i);
  for (int64_t i = dim + 1; i < input.dim(); ++i)
    inner_size *= input.size(i);
  // This kernel spawns a block per each element in the batch.
  // XXX: it assumes that inner_size == 1
  TORCH_CHECK(inner_size == 1, "Currently only inner size 1 supported");

  dim3 grid(outer_size);

  using namespace at;
  DISPATCH_FLOAT_AND_HALF_AND_BFLOAT16(input.scalar_type(), 0, "host_softmax_xentropy",
    using accscalar_t = at::acc_type<scalar_t_0, true>;
    const int ILP = sizeof(float4)/sizeof(scalar_t_0);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
    if (!half_to_float) {
      cunn_SoftMaxXEntropyForward<ILP, scalar_t_0, accscalar_t, scalar_t_0, Epilogue>
        <<<grid, block, 2 * block.x * sizeof(accscalar_t), stream>>>(
          losses.DATA_PTR<accscalar_t>(), max_log_sum_exp.DATA_PTR<scalar_t_0>(),
          input.DATA_PTR<scalar_t_0>(), labels_.DATA_PTR<int64_t>(),
          dim_size, smoothing
      );
    } else {
      cunn_SoftMaxXEntropyForward<ILP, scalar_t_0, accscalar_t, accscalar_t, Epilogue>
        <<<grid, block, 2 * block.x * sizeof(accscalar_t), stream>>>(
          losses.DATA_PTR<accscalar_t>(), max_log_sum_exp.DATA_PTR<accscalar_t>(),
          input.DATA_PTR<scalar_t_0>(), labels_.DATA_PTR<int64_t>(),
          dim_size, smoothing
      );
    }
  );

  C10_CUDA_CHECK(cudaGetLastError());

  std::vector<at::Tensor> ret = {losses, max_log_sum_exp};
  return ret;
}

template<template<typename, typename, typename> class Epilogue>
Tensor host_softmax_xentropy_backward(
    const at::Tensor &grad_loss,
    const at::Tensor &logits_,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing,
    bool half_to_float) {
  const int64_t dim = 1;
  Tensor gI = at::empty_like(logits_);
  if (grad_loss.numel() == 0) {
    return gI;
  }

  auto grad = grad_loss.contiguous();
  auto logits = logits_.contiguous();

  static_assert(std::is_same<acc_type<at::Half, true>, float>::value ||
    std::is_same<acc_type<at::Half, true>, double>::value,
    "accscalar_t for half should be float or double");
  if (grad.dim() == 0) grad = grad.view(1);

  AT_ASSERTM(logits_.dim() == 2, "Currently only 2 dim input supported");
  AT_ASSERTM(labels.dim() == 1, "Labels should be 1 dimensional");
  AT_ASSERTM(logits_.numel() > 0, "Number of classes in input should not be 0");
  AT_ASSERTM(logits_.size(0) == labels.size(0), "Input and label should have same number of examples");
  AT_ASSERTM(labels.size(0) == grad.size(0), "Label and loss should have same number of examples");

  int64_t outer_size = 1;
  int64_t dim_size = logits.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= logits.size(i);
  for (int64_t i = dim + 1; i < logits.dim(); ++i)
    inner_size *= logits.size(i);
  // See descriptions of kernels above.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(inner_size == 1, "Currently only inner size 1 supported");

  dim3 grid(outer_size);

  DISPATCH_FLOAT_AND_HALF_AND_BFLOAT16(gI.scalar_type(), 0, "host_softmax_xentropy_backward",
    using accscalar_t = acc_type<scalar_t_0, true>;
    const int ILP = sizeof(float4)/sizeof(scalar_t_0);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
    if (!half_to_float) {
      cunn_SoftMaxXEntropyBackward<ILP, scalar_t_0, accscalar_t, scalar_t_0, Epilogue>
       <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
          gI.DATA_PTR<scalar_t_0>(), logits.DATA_PTR<scalar_t_0>(),
          max_log_sum_exp.DATA_PTR<scalar_t_0>(),
          grad.DATA_PTR<scalar_t_0>(), labels.DATA_PTR<int64_t>(),
          smoothing, dim_size
      );
    } else {
      cunn_SoftMaxXEntropyBackward<ILP, scalar_t_0, accscalar_t, accscalar_t, Epilogue>
       <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
          gI.DATA_PTR<scalar_t_0>(), logits.DATA_PTR<scalar_t_0>(),
          max_log_sum_exp.DATA_PTR<accscalar_t>(),
          grad.DATA_PTR<accscalar_t>(), labels.DATA_PTR<int64_t>(),
          smoothing, dim_size
      );
    }
  );

  C10_CUDA_CHECK(cudaGetLastError());
  return gI;
}

std::vector<Tensor> softmax_xentropy_cuda(const Tensor &input, const Tensor &labels, const float smoothing, const bool half_to_float){
  return host_softmax_xentropy<LogSoftMaxForwardEpilogue>(input, labels, smoothing, half_to_float);
}

at::Tensor softmax_xentropy_backward_cuda(
    const at::Tensor &grad_loss,
    const at::Tensor &logits,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing) {
  bool half_to_float = grad_loss.type().scalarType() != logits.type().scalarType();
  if (half_to_float) {
     AT_ASSERTM((grad_loss.type().scalarType() == ScalarType::Float && (logits.type().scalarType() == ScalarType::Half || logits.type().scalarType() == ScalarType::BFloat16)), "expected input and grad types to match, or input to be at::Half or at::Bfloat16 and grad to be at::Float");
  }
  return host_softmax_xentropy_backward<LogSoftMaxBackwardEpilogue>(grad_loss, logits, max_log_sum_exp, labels, smoothing, half_to_float);
}
