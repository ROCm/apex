# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder

import sys


class FusedDenseBuilder(CUDAOpBuilder):
    BUILD_VAR = 'DS_BUILD_FUSED_DENSE'
    NAME = "fused_dense_cuda"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ['csrc/fused_dense_base.cpp', 'csrc/fused_dense_cuda.cu']

    def include_paths(self):
        return []

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if self.is_rocm_pytorch():
            ROCM_MAJOR, ROCM_MINOR = self.installed_rocm_version()
            nvcc_flags += ['-DROCM_VERSION_MAJOR=%s' % ROCM_MAJOR, '-DROCM_VERSION_MINOR=%s' % ROCM_MINOR]
        else:
            nvcc_flags.extend(
                ['-allow-unsupported-compiler' if sys.platform == "win32" else '', '-lineinfo', '--use_fast_math'] +
                self.compute_capability_args())
        return nvcc_flags