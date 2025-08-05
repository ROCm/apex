# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CPUOpBuilder


class NotImplementedBuilder(CPUOpBuilder):
    BUILD_VAR = "APEX_BUILD_NOT_IMPLEMENTED"
    NAME = "apex_not_implemented"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'apex.{self.NAME}_op'

    def load(self, verbose=True):
        raise ValueError("This op had not been implemented on CPU backend.")

    def sources(self):
        return []