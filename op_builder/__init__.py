# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import os
import pkgutil
import importlib

from .builder import get_default_compute_capabilities, OpBuilder

__apex__ = True

# List of all available op builders from apex op_builder
try:
    import apex.op_builder  # noqa: F401 # type: ignore
    op_builder_dir = "apex.op_builder"
except ImportError:
    op_builder_dir = "op_builder"

__op_builders__ = []

this_module = sys.modules[__name__]


def builder_closure(member_name):
    if op_builder_dir == "op_builder":
        # during installation time cannot get builder due to torch not installed,
        # return closure instead
        def _builder():
            from apex.op_builder.all_ops import BuilderUtils
            builder = BuilderUtils().create_op_builder(member_name)
            return builder

        return _builder
    else:
        # during runtime, return op builder class directly
        from apex.op_builder.all_ops import BuilderUtils
        builder = BuilderUtils().get_op_builder(member_name)
        return builder

# this is for the import statement such as 'from apex.op_builder import FusedLayerNormBuilder' to work
# reflect builder names and add builder closure, such as 'apex.op_builder.FusedLayerNormBuilder()' creates op builder 
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(this_module.__file__)]):
    if module_name != 'all_ops' and module_name != 'builder':
        module = importlib.import_module(f".{module_name}", package=op_builder_dir)
        for member_name in module.__dir__():
            if member_name.endswith('Builder') and member_name != "OpBuilder" and member_name != "CUDAOpBuilder":
                # assign builder name to variable with same name
                # the following is equivalent to i.e. TransformerBuilder = "TransformerBuilder"
                this_module.__dict__[member_name] = builder_closure(member_name)