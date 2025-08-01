/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file nhwc_batch_norm.h
 * \brief CUDA NHWC Batch Normalization code
 * \author Shankara Rao Thejaswi Nanditale, Dick Carter, Evgeni Krimer
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_H_
#define MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_H_

#include "dnn.h"

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include "nhwc_batch_norm_kernel.h"
#include "cuda_utils.h"
#include "c10/macros/Macros.h"
#include <ATen/cuda/CUDAContext.h>

#define VERBOSE_DEFAULT false

class NhwcBatchNorm {
 public:
  NhwcBatchNorm() {
    name_ = "nhwc_batchnorm";
    createTensorDescriptor(&X_tensor_desc_);
    createTensorDescriptor(&Y_tensor_desc_);
  }

  ~NhwcBatchNorm() {
    destroyTensorDescriptor(X_tensor_desc_);
    destroyTensorDescriptor(Y_tensor_desc_);
  }

  void die() {
    std::cerr << "batchnorm not initialized" << std::endl;
    exit(-1);
  }

  void fwd(cudaStream_t stream, bool use_relu, void* my_data, void* pair_data, void* pair_data2, void* pair_data3, const int bn_group, const int magic, const int occupancy, const int grid_dim_x, const bool coop);
  void dgrad(cudaStream_t stream, bool use_relu, void* my_data, void* pair_data, void* pair_data2, void* pair_data3, const int bn_group, const int magic, const int occupancy, const int grid_dim_x, const bool coop);
  void fwdInference(cudaStream_t stream, bool use_relu);
  dim3 calc_fwd_grid(int *loop, const int grid_dim_x);
  dim3 calc_bwd_grid(int *loop, const int grid_dim_x);

  void setInputDescriptor(const dnnTensorFormat_t format,
                                  const dnnDataType_t     data_type,
                                  int n, int c, int h, int w, int bn_group) {
    m_ = n * h * w;
    int m_bn_adjusted = m_ * bn_group;
    c_ = c;
    // factor to scale sum of squared errors to get saved variance.  Must be 1/nhw.
    svar_inv_count_ = 1.f / m_bn_adjusted;
    // factor to scale sum of squared errors to get running variance. Should be 1/(nhw-1).
    int divisor = m_bn_adjusted - 1;
    // nhw == 1 is unlikely, but by setting the rvar_inv_count_ == 1.f, we avoid running var infs.
    rvar_inv_count_ = divisor == 0 ? 1.f : 1.f / divisor;
    setTensorDescriptor(X_tensor_desc_, format, data_type, n, c, h, w);
  }

  void setOutputDescriptor(const dnnTensorFormat_t format,
                                   const dnnDataType_t     data_type,
                                   int n, int c, int h, int w) {
    setTensorDescriptor(Y_tensor_desc_, format, data_type, n, c, h, w);
  }

  const std::vector<size_t> numWorkspaceBytes() const;

  void setWorkspacePointers(
      const std::vector<void*>&  workspace,
      const std::vector<size_t>& num_workspace_bytes);

  void setInputOutputPointers(void* X, void* dX, void* Y, void *dY) {
    X_ = X;
    dX_  = dX;
    Y_   = Y;
    dY_  = dY;
  }

  // Sets the pointers for the scale and weight (in that order) data and derivative buffers.
  void setWeightPointers(const std::vector<void*>& weight_pointers,
                                 const std::vector<void*>& deriv_pointers) {
    assert(weight_pointers.size() == 2);
    assert(deriv_pointers.size()  == 2);
    scale_  = static_cast<float*>(weight_pointers[0]);
    bias_   = static_cast<float*>(weight_pointers[1]);
    dscale_ = static_cast<float*>(deriv_pointers[0]);
    dbias_  = static_cast<float*>(deriv_pointers[1]);
  }

  // Sets the pointers for the population mean and variance buffers, in that order.
  void setParameterPointers(const std::vector<void*>& param_pointers) {
    assert(param_pointers.size() == 2);
    population_mean_     = static_cast<float*>(param_pointers[0]);
    population_variance_ = static_cast<float*>(param_pointers[1]);
  }

  void setConstants(const double exp_avg_factor, const double eps) {
    exp_avg_factor_ = exp_avg_factor;
    eps_ = eps;
  }

  void processCudnnStatus(const dnnStatus_t& status,
                          const std::string& string = std::string(),
                          bool verbose = VERBOSE_DEFAULT) {
#ifdef USE_ROCM
    if (status != DNN_STATUS_SUCCESS)
      LOG(FATAL) << string << " " << miopenGetErrorString(status);
    else if (verbose)
      LOG(INFO) << string << " " << miopenGetErrorString(status);
#else
    if (status != DNN_STATUS_SUCCESS)
      LOG(FATAL) << string << " " << cudnnGetErrorString(status);
    else if (verbose)
      LOG(INFO) << string << " " << cudnnGetErrorString(status);
#endif
  }

  void checkCudaStatus(const std::string& string = std::string(),
                       bool verbose = VERBOSE_DEFAULT) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
      LOG(FATAL) << string << " " << cudaGetErrorString(status);
    else if (verbose)
      LOG(INFO) << string << " " << cudaGetErrorString(status);
  }

  size_t size_retired_ctas(int grid_y) const {
    // Note that the value of max_grid_y to handle known GPUs is about 160.
    const int max_grid_y = 1024;
    if (grid_y > max_grid_y)
      LOG(INFO) << "GPU capabilities exceeds assumptions.";
    const int retired_cta_bytes = max_grid_y * 2 * sizeof(int);
    // Since the region will be initialized once and used for many kernels,
    // the idea is to return an ample size that will cover all uses.
    return retired_cta_bytes;
  }

  dnnTensorDescriptor_t  X_tensor_desc_ = nullptr;
  dnnTensorDescriptor_t  Y_tensor_desc_ = nullptr;

  void*  X_ = nullptr;
  void* dX_ = nullptr;
  void*  Y_ = nullptr;
  void* dY_ = nullptr;

  // Learned scale and bias weights.
  float* scale_  = nullptr;
  float* dscale_ = nullptr;
  float* bias_   = nullptr;
  float* dbias_  = nullptr;

  // Computed population mean and variance parameters.
  float* population_mean_     = nullptr;
  float* population_variance_ = nullptr;

  // Workspace buffers for minibatch mean and variance (computed in fwd, needed by bwd).
  float* minibatch_mean_     = nullptr;
  float* minibatch_variance_ = nullptr;

  int m_ = 0;  // Number of values per channel that BN is normalizing.
  int c_ = 0;  // Number of channels over which BN is normalizing.

  float svar_inv_count_ = 0.f;  // factor to scale sum of squared errors to get saved variance
  float rvar_inv_count_ = 0.f;  // factor to scale sum of squared errors to get running variance

  double exp_avg_factor_ = 0.;
  double eps_            = 0.;
  std::string name_;

 private:
  void setTensorDescriptor(dnnTensorDescriptor_t descriptor,
                           dnnTensorFormat_t format,
                           dnnDataType_t     data_type,
                           int n, int c, int h, int w) {
    dnnStatus_t status = DNN_STATUS_SUCCESS;
#ifdef USE_ROCM
    status = miopenSet4dTensorDescriptor(descriptor, data_type, n, c, h, w);
#else
    status = cudnnSetTensor4dDescriptor(descriptor, format, data_type, n, c, h, w);
#endif
    processCudnnStatus(status, "set tensor descriptor");
  }

  void createTensorDescriptor(dnnTensorDescriptor_t *descriptor) {
    dnnStatus_t status = DNN_STATUS_SUCCESS;
#ifdef USE_ROCM
    status = miopenCreateTensorDescriptor(descriptor);
#else
    status = cudnnCreateTensorDescriptor(descriptor);
#endif
    processCudnnStatus(status, "create tensor_descriptor");
  }

  void destroyTensorDescriptor(dnnTensorDescriptor_t descriptor) {
    dnnStatus_t status = DNN_STATUS_SUCCESS;
#ifdef USE_ROCM
    status = miopenDestroyTensorDescriptor(descriptor);
#else
    status = cudnnDestroyTensorDescriptor(descriptor);
#endif
    processCudnnStatus(status, "destroy tensor_descriptor");
  }

 protected:
  float *partial_sums_ = nullptr;
  int *partial_counts_ = nullptr;
  int *retired_ctas_   = nullptr;

  void _setFwdParams(NhwcBatchNormFwdParams *params) const;
  void _setFwdInferenceParams(NhwcBatchNormFwdInferenceParams *params) const;
  void _setBwdParams(NhwcBatchNormBwdParams *params) const;

  // @todo: ability to configure these?
  // Kernel params
  static const int USE_ONLINE_APPROACH = 1;
  static const int THREADS_PER_CTA = 512;
  static const int THREADS_PER_PIXEL = 32;
  static const int C_ELEMENTS_PER_CTA = 128;
  static const int ELEMENTS_PER_LDG = C_ELEMENTS_PER_CTA / THREADS_PER_PIXEL;
  static const int MAX_SMEM_WITHOUT_OPT_IN = 48 * 1024;

  typedef uint16_t StorageType;
  //typedef float StorageType;
  // increasing this to 6 causes spills in fwd kernel!
  static const int PIXELS_PER_THREAD_IN_REGISTERS_FWD = 1;
  static const int PIXELS_PER_THREAD_IN_REGISTERS_BWD = 1;
  static const int PIXELS_PER_THREAD_IN_SMEM_FWD = 0;
  static const int PIXELS_PER_THREAD_IN_SMEM_BWD = 0;

  static const int PIXELS_PER_THREAD_FWD = PIXELS_PER_THREAD_IN_REGISTERS_FWD + \
      PIXELS_PER_THREAD_IN_SMEM_FWD;
  static const int PIXELS_PER_THREAD_BWD = PIXELS_PER_THREAD_IN_REGISTERS_BWD + \
      PIXELS_PER_THREAD_IN_SMEM_BWD;
  static const int PIXELS_PER_THREAD_FWD_INFERENCE = 4;

  // Derived params
  static const size_t SMEM_SIZE_FWD = PIXELS_PER_THREAD_IN_SMEM_FWD*THREADS_PER_CTA*\
      ELEMENTS_PER_LDG*sizeof(StorageType);
  static const size_t SMEM_SIZE_BWD = PIXELS_PER_THREAD_IN_SMEM_BWD*THREADS_PER_CTA*\
      ELEMENTS_PER_LDG*2*sizeof(StorageType);
  static const int PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
  static const int PIXELS_PER_CTA_FWD = THREADS_PER_CTA/THREADS_PER_PIXEL * \
      PIXELS_PER_THREAD_FWD;
  static const int PIXELS_PER_CTA_BWD = THREADS_PER_CTA/THREADS_PER_PIXEL * \
      PIXELS_PER_THREAD_BWD;
  static const int PIXELS_PER_CTA_FWD_INFERENCE = THREADS_PER_CTA/THREADS_PER_PIXEL * \
      PIXELS_PER_THREAD_FWD_INFERENCE;

  // max grid.y in case of group bn is limited by exchange buffer size
  static const int MAX_GBN_BLOCK_Y = 256;

  // Helper function to launch the forward kernel.

  // We calculate (based on smem usage) the achievable occupancy and make sure we run a kernel
  // version that was compiled with that occupancy in its launch bounds.  This way, we avoid
  // needless register spills.
  void _fwdKernelLauncher(cudaStream_t stream, NhwcBatchNormFwdParams params,
                                dim3 grid_dim, int outer_loops, bool use_relu, const int occupancy, const bool coop) {

#ifdef USE_ROCM
#define LAUNCH_FWD_KERNEL(OUTER_LOOPS, USE_RELU, USE_ADD_RELU, COMPILED_FOR_OCCUPANCY, COOP) \
    do { \
        CHECK(SMEM_SIZE_FWD <= MAX_SMEM_WITHOUT_OPT_IN) << "Nhwc batchnorm kernel smem too big."; \
        auto fwd_func = nhwc_batch_norm_fwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_FWD, \
                        PIXELS_PER_THREAD_IN_SMEM_FWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        USE_RELU, \
                        USE_ADD_RELU, \
                        COMPILED_FOR_OCCUPANCY>; \
        if (COMPILED_FOR_OCCUPANCY > 1) { \
            hipFuncSetAttribute((void *) fwd_func, hipFuncAttributePreferredSharedMemoryCarveout, 100); \
            checkCudaStatus(name_ + " fwd ser coop kernel (cudaFuncSetAttribute carveout)"); \
        } \
        void *params_ptr = static_cast<void*>(&params); \
        using FWD_FUNC = decltype(nhwc_batch_norm_fwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_FWD, \
                        PIXELS_PER_THREAD_IN_SMEM_FWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        USE_RELU, \
                        USE_ADD_RELU, \
                        COMPILED_FOR_OCCUPANCY>); \
        if (COOP) { \
            hipLaunchCooperativeKernel<FWD_FUNC>(fwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_FWD, \
                stream); \
        } else { \
            hipLaunchKernel((void *) fwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_FWD, \
                stream); \
        } \
        checkCudaStatus(name_ + " fwd ser coop kernel"); \
    } while (0)
#else
#define LAUNCH_FWD_KERNEL(OUTER_LOOPS, USE_RELU, USE_ADD_RELU, COMPILED_FOR_OCCUPANCY, COOP) \
    do { \
        CHECK(SMEM_SIZE_FWD <= MAX_SMEM_WITHOUT_OPT_IN) << "Nhwc batchnorm kernel smem too big."; \
        auto fwd_func = nhwc_batch_norm_fwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_FWD, \
                        PIXELS_PER_THREAD_IN_SMEM_FWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        USE_RELU, \
                        USE_ADD_RELU, \
                        COMPILED_FOR_OCCUPANCY>; \
        if (COMPILED_FOR_OCCUPANCY > 1) { \
            cudaFuncSetAttribute(fwd_func, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
            checkCudaStatus(name_ + " fwd ser coop kernel (cudaFuncSetAttribute carveout)"); \
        } \
        void *params_ptr = static_cast<void*>(&params); \
        using FWD_FUNC = decltype(nhwc_batch_norm_fwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_FWD, \
                        PIXELS_PER_THREAD_IN_SMEM_FWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        USE_RELU, \
                        USE_ADD_RELU, \
                        COMPILED_FOR_OCCUPANCY>); \
        if (COOP) { \
            cudaLaunchCooperativeKernel<FWD_FUNC>(fwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_FWD, \
                stream); \
        } else { \
            cudaLaunchKernel<FWD_FUNC>(fwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_FWD, \
                stream); \
        } \
        checkCudaStatus(name_ + " fwd ser coop kernel"); \
    } while (0)
#endif

    // Don't try for an occupancy > 2 as this will squeeze register use and create spills.
    if (outer_loops == 1 && use_relu) {
      if (occupancy >= 2)
        LAUNCH_FWD_KERNEL(1, true, false, 2, coop);
      else
        LAUNCH_FWD_KERNEL(1, true, false, 1, coop);
    } else if (outer_loops == 1 && !use_relu) {
      if (occupancy >= 2)
        LAUNCH_FWD_KERNEL(1, false, false, 2, coop);
      else
        LAUNCH_FWD_KERNEL(1, false, false, 1, coop);
    } else if (use_relu) {
      if (occupancy >= 2)
        LAUNCH_FWD_KERNEL(0, true, false, 2, coop);
      else
        LAUNCH_FWD_KERNEL(0, true, false, 1, coop);
    } else {
      if (occupancy >= 2)
        LAUNCH_FWD_KERNEL(0, false, false, 2, coop);
      else
        LAUNCH_FWD_KERNEL(0, false, false, 1, coop);
    }
#undef LAUNCH_FWD_KERNEL
  }

  // Helper function to launch the backward kernel.

  void _bwdKernelLauncher(cudaStream_t stream, NhwcBatchNormBwdParams params,
                                dim3 grid_dim, int outer_loops, bool use_relu, const int occupancy, const bool coop) {
#ifdef USE_ROCM
#define LAUNCH_BWD_KERNEL(OUTER_LOOPS, COMPILED_FOR_OCCUPANCY, COOP) \
    do { \
        CHECK(SMEM_SIZE_BWD <= MAX_SMEM_WITHOUT_OPT_IN) << "Nhwc batchnorm kernel smem too big."; \
        auto bwd_func = nhwc_batch_norm_bwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>; \
        if (COMPILED_FOR_OCCUPANCY > 1) { \
            hipFuncSetAttribute((void *) bwd_func, hipFuncAttributePreferredSharedMemoryCarveout, 100); \
            checkCudaStatus(name_ + " bwd coop serial kernel (cudaFuncSetAttribute carveout)"); \
        } \
        void *params_ptr = static_cast<void*>(&params); \
        using BWD_FUNC = decltype(nhwc_batch_norm_bwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>); \
        if (COOP) { \
            hipLaunchCooperativeKernel<BWD_FUNC>(bwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } else { \
            hipLaunchKernel((void *) bwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } \
        checkCudaStatus(name_ + " bwd coop serial kernel"); \
    } while (0)

#define LAUNCH_BWD_RELU_KERNEL(OUTER_LOOPS, COMPILED_FOR_OCCUPANCY, COOP) \
    do { \
        CHECK(SMEM_SIZE_BWD <= MAX_SMEM_WITHOUT_OPT_IN) << "Nhwc batchnorm kernel smem too big."; \
        auto bwd_relu_func = nhwc_batch_norm_bwd_relu< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>; \
        if (COMPILED_FOR_OCCUPANCY > 1) { \
            hipFuncSetAttribute((void *) bwd_relu_func, hipFuncAttributePreferredSharedMemoryCarveout, 100); \
            checkCudaStatus(name_ + " bwd-relu coop serial kernel (cudaFuncSetAttribute carveout)"); \
        } \
        void *params_ptr = static_cast<void*>(&params); \
        using BWD_RELU_FUNC = decltype(nhwc_batch_norm_bwd_relu< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>); \
        if (COOP) { \
            hipLaunchCooperativeKernel<BWD_RELU_FUNC>(bwd_relu_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } else { \
            hipLaunchKernel((void *) bwd_relu_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } \
        checkCudaStatus(name_ + " bwd-relu coop serial kernel"); \
    } while (0)
#else
#define LAUNCH_BWD_KERNEL(OUTER_LOOPS, COMPILED_FOR_OCCUPANCY, COOP) \
    do { \
        CHECK(SMEM_SIZE_BWD <= MAX_SMEM_WITHOUT_OPT_IN) << "Nhwc batchnorm kernel smem too big."; \
        auto bwd_func = nhwc_batch_norm_bwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>; \
        if (COMPILED_FOR_OCCUPANCY > 1) { \
            cudaFuncSetAttribute(bwd_func, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
            checkCudaStatus(name_ + " bwd coop serial kernel (cudaFuncSetAttribute carveout)"); \
        } \
        void *params_ptr = static_cast<void*>(&params); \
        using BWD_FUNC = decltype(nhwc_batch_norm_bwd< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>); \
        if (COOP) { \
            cudaLaunchCooperativeKernel<BWD_FUNC>(bwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } else { \
            cudaLaunchKernel<BWD_FUNC>(bwd_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } \
        checkCudaStatus(name_ + " bwd coop serial kernel"); \
    } while (0)

#define LAUNCH_BWD_RELU_KERNEL(OUTER_LOOPS, COMPILED_FOR_OCCUPANCY, COOP) \
    do { \
        CHECK(SMEM_SIZE_BWD <= MAX_SMEM_WITHOUT_OPT_IN) << "Nhwc batchnorm kernel smem too big."; \
        auto bwd_relu_func = nhwc_batch_norm_bwd_relu< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>; \
        if (COMPILED_FOR_OCCUPANCY > 1) { \
            cudaFuncSetAttribute(bwd_relu_func, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
            checkCudaStatus(name_ + " bwd-relu coop serial kernel (cudaFuncSetAttribute carveout)"); \
        } \
        void *params_ptr = static_cast<void*>(&params); \
        using BWD_RELU_FUNC = decltype(nhwc_batch_norm_bwd_relu< \
                        StorageType, \
                        THREADS_PER_CTA, \
                        THREADS_PER_PIXEL, \
                        PIXELS_PER_THREAD_IN_REGISTERS_BWD, \
                        PIXELS_PER_THREAD_IN_SMEM_BWD, \
                        ELEMENTS_PER_LDG, \
                        USE_ONLINE_APPROACH, \
                        OUTER_LOOPS, \
                        COMPILED_FOR_OCCUPANCY>); \
        if (COOP) { \
            cudaLaunchCooperativeKernel<BWD_RELU_FUNC>(bwd_relu_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } else { \
            cudaLaunchKernel<BWD_RELU_FUNC>(bwd_relu_func, \
                grid_dim, \
                THREADS_PER_CTA, \
                &params_ptr, \
                SMEM_SIZE_BWD, \
                stream); \
        } \
        checkCudaStatus(name_ + " bwd-relu coop serial kernel"); \
    } while (0)
#endif

    // Don't try for an occupancy > 2 as this will squeeze register use and create spills.
    if (outer_loops == 1 && use_relu) {
      if (occupancy >= 2)
        LAUNCH_BWD_RELU_KERNEL(1, 2, coop);
      else
        LAUNCH_BWD_RELU_KERNEL(1, 1, coop);
    } else if (outer_loops == 1 && !use_relu) {
      if (occupancy >= 2)
        LAUNCH_BWD_KERNEL(1, 2, coop);
      else
        LAUNCH_BWD_KERNEL(1, 1, coop);
    } else if (use_relu) {
      if (occupancy >= 2)
        LAUNCH_BWD_RELU_KERNEL(0, 2, coop);
      else
        LAUNCH_BWD_RELU_KERNEL(0, 1, coop);
    } else {
      if (occupancy >= 2)
        LAUNCH_BWD_KERNEL(0, 2, coop);
      else
        LAUNCH_BWD_KERNEL(0, 1, coop);
    }
#undef LAUNCH_BWD_KERNEL
  }

 public:

  // Calculate the expected fwd kernel occupancy, as dictated by shared memory usage.
  static int smem_driven_fwd_occupancy(int device_id, const int max_cta_per_sm) {
    using namespace at::cuda::utils;
    int fwd_reduction_bytes = THREADS_PER_PIXEL*(THREADS_PER_CTA/at::cuda::warp_size())*ELEMENTS_PER_LDG*sizeof(float);
    int fwd_smem_bytes = SMEM_SIZE_FWD + fwd_reduction_bytes;
    int occupancy = MaxSharedMemoryPerMultiprocessor(device_id) / fwd_smem_bytes;
    return std::min(max_cta_per_sm, occupancy);
  }

  // Calculate the expected bwd kernel occupancy, as dictated by shared memory usage.
  static int smem_driven_bwd_occupancy(int device_id, const int max_cta_per_sm) {
    using namespace at::cuda::utils;
    int bwd_reduction_bytes = THREADS_PER_PIXEL*(THREADS_PER_CTA/at::cuda::warp_size())*ELEMENTS_PER_LDG*sizeof(float);
    int bwd_smem_bytes = SMEM_SIZE_BWD + bwd_reduction_bytes;
    int occupancy = MaxSharedMemoryPerMultiprocessor(device_id) / bwd_smem_bytes;
    return std::min(max_cta_per_sm, occupancy);
  }
};

const std::vector<size_t> NhwcBatchNorm::numWorkspaceBytes() const {
  assert(c_ > 0);

  // choose the max memory required between fwd/bwd passes
  int grid_x_fwd = div_up(m_, PIXELS_PER_CTA_FWD);
  int grid_x_bwd = div_up(m_, PIXELS_PER_CTA_BWD);
  int grid_x = max(grid_x_fwd, grid_x_bwd);
  int grid_y = div_up(c_, C_ELEMENTS_PER_CTA);

  const size_t num_mean_bytes     = c_ * sizeof(float);
  const size_t num_variance_bytes = num_mean_bytes;
  const size_t size_sums          = grid_y*grid_x*THREADS_PER_PIXEL*\
      ELEMENTS_PER_LDG*2*sizeof(float);
  const size_t size_counts        = grid_y*grid_x*sizeof(int);

  return {num_mean_bytes, num_variance_bytes,
          size_retired_ctas(grid_y), size_sums, size_counts};
}

void NhwcBatchNorm::setWorkspacePointers(
      const std::vector<void*>& workspace,
      const std::vector<size_t>& num_workspace_bytes) {
  assert(workspace.size() == 5);
  assert(num_workspace_bytes.size() == 5);

  minibatch_mean_     = static_cast<float*>(workspace[0]);
  minibatch_variance_ = static_cast<float*>(workspace[1]);
  retired_ctas_       = static_cast<int*>(workspace[2]);
  partial_sums_       = static_cast<float*>(workspace[3]);
  partial_counts_     = static_cast<int*>(workspace[4]);
}

void NhwcBatchNorm::_setFwdParams(NhwcBatchNormFwdParams *params) const {
  params->gmem_src          = static_cast<uint16_t*>(X_);
  params->gmem_dst          = static_cast<uint16_t*>(Y_);
  params->gmem_src1         = nullptr;
  params->gmem_bias         = bias_;
  params->gmem_scale        = scale_;
  params->gmem_running_mean = population_mean_;
  params->gmem_running_var  = population_variance_;
  params->gmem_saved_mean   = minibatch_mean_;
  params->gmem_saved_var    = minibatch_variance_;
  params->gmem_relu_bitmask = nullptr;
  params->nhw               = m_;
  params->c                 = c_;
  params->svar_inv_count    = svar_inv_count_;
  params->rvar_inv_count    = rvar_inv_count_;
  params->gmem_sums         = partial_sums_;
  params->gmem_counts       = partial_counts_;
  params->gmem_retired_ctas = retired_ctas_;
  params->var_eps           = eps_;
  params->outer_loops       = 0;
  params->exp_avg_factor    = static_cast<float>(exp_avg_factor_);
  params->c_blks            = div_up(c_, C_ELEMENTS_PER_CTA);
}

void NhwcBatchNorm::_setFwdInferenceParams(NhwcBatchNormFwdInferenceParams
                                                        *params) const {
  params->gmem_src   = static_cast<uint16_t*>(X_);
  params->gmem_dst   = static_cast<uint16_t*>(Y_);
  params->gmem_src1  = nullptr;
  params->gmem_bias  = bias_;
  params->gmem_scale = scale_;
  params->gmem_mean  = population_mean_;
  params->gmem_var   = population_variance_;
  params->nhw        = m_;
  params->c          = c_;
  params->var_eps    = eps_;
}

void NhwcBatchNorm::_setBwdParams(NhwcBatchNormBwdParams *params) const {
  params->gmem_src          = static_cast<uint16_t*>(X_);
  params->gmem_dy           = static_cast<uint16_t*>(dY_);
  params->gmem_dst          = static_cast<uint16_t*>(dX_);
  params->gmem_dst1         = nullptr;
  params->gmem_relu_bitmask = nullptr;
  params->gmem_dscale       = dscale_;
  params->gmem_dbias        = dbias_;
  params->gmem_scale        = scale_;
  params->gmem_bias         = bias_;
  params->gmem_saved_mean   = minibatch_mean_;
  params->gmem_saved_var    = minibatch_variance_;
  params->nhw               = m_;
  params->c                 = c_;
  params->svar_inv_count    = svar_inv_count_;
  params->gmem_sums         = partial_sums_;
  params->gmem_retired_ctas = retired_ctas_;
  params->outer_loops       = 0;
  params->c_blks            = div_up(c_, C_ELEMENTS_PER_CTA);
}

void NhwcBatchNorm::fwdInference(cudaStream_t stream, bool use_relu) {
  bool ptrs_are_set =
      X_tensor_desc_ != nullptr
      && Y_tensor_desc_ != nullptr
      && scale_ != nullptr
      && bias_ != nullptr
      //      && minibatch_mean_ != nullptr
      //      && minibatch_variance_ != nullptr
      && population_mean_ != nullptr
      && population_variance_ != nullptr
      && X_ != nullptr
      //      && dX_ != nullptr
      && Y_ != nullptr
      //      && dY_ != nullptr
      //      && dscale_ != nullptr
      //      && dbias_ != nullptr
      && partial_sums_   != nullptr
      && partial_counts_ != nullptr;

  if (!ptrs_are_set)
    die();

  dim3 grid_dim;
  grid_dim.x = div_up(m_, PIXELS_PER_CTA_FWD_INFERENCE);
  grid_dim.y = div_up(c_, C_ELEMENTS_PER_CTA);

  // @todo: maybe just move this inside initialize routine?
  NhwcBatchNormFwdInferenceParams params;
  _setFwdInferenceParams(&params);

  if (use_relu) {
    nhwc_batch_norm_fwd_inference
      <StorageType, THREADS_PER_CTA, THREADS_PER_PIXEL, ELEMENTS_PER_LDG, true, false>
    <<<grid_dim, THREADS_PER_CTA, 0, stream>>>(params);
    checkCudaStatus(name_ + " fwd_inference-relu kernel");
  } else {
    nhwc_batch_norm_fwd_inference
      <StorageType, THREADS_PER_CTA, THREADS_PER_PIXEL, ELEMENTS_PER_LDG, false, false>
    <<<grid_dim, THREADS_PER_CTA, 0, stream>>>(params);
    checkCudaStatus(name_ + " fwd_inference kernel");
  }
}

dim3 NhwcBatchNorm::calc_fwd_grid(int *loop, const int grid_dim_x) {
  dim3 grid_dim;
  grid_dim.x = div_up(m_, PIXELS_PER_CTA_FWD);
  int c_blks = div_up(c_, C_ELEMENTS_PER_CTA);
  unsigned int max_grid_x = grid_dim_x;
  if (grid_dim.x <= max_grid_x) {
    *loop = 1;
    if (max_grid_x / grid_dim.x > 1) {
      grid_dim.y = std::min(c_blks, static_cast<int>(max_grid_x / grid_dim.x));
      assert(grid_dim.y<MAX_GBN_BLOCK_Y); //FIXME: turn into a loop
    } else {
      grid_dim.y = 1;
    }
  } else {
    grid_dim.x = max_grid_x;
    grid_dim.y = 1;
    int nhw_in_regs = m_ - PIXELS_PER_THREAD_IN_SMEM_FWD*PIXELS_PER_LDG*grid_dim.x;
    int pixels_per_iteration = PIXELS_PER_THREAD_IN_REGISTERS_FWD*PIXELS_PER_LDG*grid_dim.x;
    *loop = div_up(nhw_in_regs, pixels_per_iteration);
  }
  return grid_dim;
}

dim3 NhwcBatchNorm::calc_bwd_grid(int *loop, const int grid_dim_x) {
  dim3 grid_dim;
  grid_dim.x = div_up(m_, PIXELS_PER_CTA_BWD);
  int c_blks = div_up(c_, C_ELEMENTS_PER_CTA);
  unsigned int max_grid_x = grid_dim_x;
  if (grid_dim.x <= max_grid_x) {
    *loop = 1;
    if (max_grid_x / grid_dim.x > 1) {
      grid_dim.y = std::min(c_blks, static_cast<int>(max_grid_x / grid_dim.x));
      assert(grid_dim.y<MAX_GBN_BLOCK_Y); //FIXME: turn into a loop
    } else {
      grid_dim.y = 1;
    }
  } else {
    grid_dim.x = max_grid_x;
    grid_dim.y = 1;
    int nhw_in_regs = m_ - PIXELS_PER_THREAD_IN_SMEM_BWD*PIXELS_PER_LDG*grid_dim.x;
    int pixels_per_iteration = PIXELS_PER_THREAD_IN_REGISTERS_BWD*PIXELS_PER_LDG*grid_dim.x;
    *loop = div_up(nhw_in_regs, pixels_per_iteration);
  }
  return grid_dim;
}

void NhwcBatchNorm::fwd(cudaStream_t stream, bool use_relu, void* my_data, void* pair_data, void* pair_data2, void* pair_data3,
                        const int bn_group, const int magic, const int occupancy, const int grid_dim_x, const bool coop) {
  bool ptrs_are_set =
      X_tensor_desc_ != nullptr
      && Y_tensor_desc_ != nullptr
      && scale_ != nullptr
      && bias_ != nullptr
      && minibatch_mean_ != nullptr
      && minibatch_variance_ != nullptr
      && population_mean_ != nullptr
      && population_variance_ != nullptr
      && X_ != nullptr
      //      && dX_ != nullptr
      && Y_ != nullptr
      //      && dY_ != nullptr
      //      && dscale_ != nullptr
      //      && dbias_ != nullptr
      && partial_sums_   != nullptr
      && partial_counts_ != nullptr
      && retired_ctas_   != nullptr;

  if (!ptrs_are_set)
    die();

  // reset of retired_cta_count no longer needed

  NhwcBatchNormFwdParams params;
  _setFwdParams(&params);
  params.my_data = my_data;
  params.pair_datas[0] = pair_data;
  params.pair_datas[1] = pair_data2;
  params.pair_datas[2] = pair_data3;
  params.magic = magic;
  params.sync_iters = (bn_group==8)?3:(bn_group >> 1);

  dim3 grid_dim = calc_fwd_grid(&params.outer_loops, grid_dim_x);
  _fwdKernelLauncher(stream, params, grid_dim, params.outer_loops, use_relu, occupancy, coop);
}

void NhwcBatchNorm::dgrad(cudaStream_t stream, bool use_relu, void* my_data, void* pair_data, void* pair_data2, void* pair_data3, 
                          const int bn_group, const int magic, const int occupancy, const int grid_dim_x, const bool coop) {
  bool ptrs_are_set =
      X_tensor_desc_ != nullptr
      && Y_tensor_desc_ != nullptr
      && scale_ != nullptr
      && (bias_ != nullptr || !use_relu)
      && minibatch_mean_ != nullptr
      && minibatch_variance_ != nullptr
      //      && population_mean_ != nullptr
      //      && population_variance_ != nullptr
      && X_ != nullptr
      && dX_ != nullptr
      //      && Y_ != nullptr
      && dY_ != nullptr
      && dscale_ != nullptr
      && dbias_ != nullptr;

  if (!ptrs_are_set)
    die();

  // reset of retired_cta_count no longer needed

  NhwcBatchNormBwdParams params;
  _setBwdParams(&params);
  params.my_data = my_data;
  params.pair_datas[0] = pair_data;
  params.pair_datas[1] = pair_data2;
  params.pair_datas[2] = pair_data3;
  params.magic = magic;
  params.sync_iters = (bn_group==8)?3:(bn_group >> 1);
  params.wgrad_coeff = 1.0 / bn_group;

  dim3 grid_dim = calc_bwd_grid(&params.outer_loops, grid_dim_x);
  _bwdKernelLauncher(stream, params, grid_dim, params.outer_loops, use_relu, occupancy, coop);
}

#endif  // MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_H_
