# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib

class BuilderUtils:
    def op_builder_dir(self):
        try:
            # is op_builder from apex or a 3p version? this should only succeed if it's apex
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __apex__  # noqa: F401 # type: ignore
            return "op_builder"
        except ImportError:
            return "apex.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        else:
            self.class_dict = {}
            # begin initialize for create_op_builder()
            # put all valid class name <--> class type mapping into class_dict
            op_builder_dir = self.op_builder_dir()
            op_builder_module = importlib.import_module(op_builder_dir)
            op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
            for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
                # avoid self references,
                # skip sub_directories which contains ops for other backend(cpu, npu, etc.).
                if module_name != 'all_ops' and module_name != 'builder' and not os.path.isdir(
                        os.path.join(op_builder_absolute_path, module_name)):
                    module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                    for member_name in module.__dir__():
                        if member_name.endswith(
                                'Builder'
                        ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
                            if not member_name in self.class_dict:
                                self.class_dict[member_name] = getattr(module, member_name)
            # end initialize for create_op_builder()

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return None

# List of all available ops

# append all builder names into __op_builders__
builder_utils = BuilderUtils()
op_builder_dir = builder_utils.op_builder_dir()
op_builder_module = importlib.import_module(op_builder_dir)
__op_builders__ = []

for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(op_builder_module.__file__)]):
    # avoid self references
    if module_name != 'all_ops' and module_name != 'builder':
        module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
        for member_name in module.__dir__():
            if member_name.endswith('Builder'):
                # append builder to __op_builders__ list
                builder = builder_utils.create_op_builder(member_name)
                __op_builders__.append(builder)
                
ALL_OPS = {op.name: op for op in __op_builders__ if op is not None}