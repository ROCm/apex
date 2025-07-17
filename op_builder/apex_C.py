from .builder import TorchCPUOpBuilder

import sys


class ApexCBuilder(TorchCPUOpBuilder):
    BUILD_VAR = 'APEX_BUILD_C'
    NAME = "apex_C"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'apex.{self.NAME}'

    def sources(self):
        return ["csrc/flatten_unflatten.cpp"]

    def include_paths(self):
        return ['csrc/' ]
        
    def libraries_args(self):
        args = super().libraries_args()
        return args