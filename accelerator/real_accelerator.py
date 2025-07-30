# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

try:
    # Importing logger currently requires that torch is installed, hence the try...except
    # TODO: Remove logger dependency on torch.
    from apex.accelerator.logging import logger as accel_logger
except ImportError as e:
    accel_logger = None

try:
    from accelerator.abstract_accelerator import ApexAccelerator as dsa1
except ImportError as e:
    dsa1 = None
try:
    from apex.accelerator.abstract_accelerator import ApexAccelerator as dsa2
except ImportError as e:
    dsa2 = None

SUPPORTED_ACCELERATOR_LIST = ['cuda', 'cpu']

ds_accelerator = None


def _validate_accelerator(accel_obj):
    # because abstract_accelerator has different path during
    # build time (accelerator.abstract_accelerator)
    # and run time (apex.accelerator.abstract_accelerator)
    # and extension would import the
    # run time abstract_accelerator/apex as its base
    # class, so we need to compare accel_obj with both base class.
    # if accel_obj is instance of ApexAccelerator in one of
    # accelerator.abstractor_accelerator
    # or apex.accelerator.abstract_accelerator, consider accel_obj
    # is a conforming object
    if not ((dsa1 is not None and isinstance(accel_obj, dsa1)) or (dsa2 is not None and isinstance(accel_obj, dsa2))):
        raise AssertionError(f"{accel_obj.__class__.__name__} accelerator is not subclass of ApexAccelerator")

    # TODO: turn off is_available test since this breaks tests
    # assert accel_obj.is_available(), \
    #    f'{accel_obj.__class__.__name__} accelerator fails is_available() test'


def is_current_accelerator_supported():
    return get_accelerator().device_name() in SUPPORTED_ACCELERATOR_LIST


def get_accelerator():
    global ds_accelerator
    if ds_accelerator is not None:
        return ds_accelerator

    accelerator_name = None
    ds_set_method = None
    # 1. Detect whether there is override of apex accelerators from environment variable.
    if "APEX_ACCELERATOR" in os.environ.keys():
        accelerator_name = os.environ["APEX_ACCELERATOR"]
        if accelerator_name == "cpu":
            pass 
        elif accelerator_name not in SUPPORTED_ACCELERATOR_LIST:
            raise ValueError(f'APEX_ACCELERATOR must be one of {SUPPORTED_ACCELERATOR_LIST}. '
                             f'Value "{accelerator_name}" is not supported')
        ds_set_method = "override"

    # 2. If no override, detect which accelerator to use automatically
    if accelerator_name is None:
        # We need a way to choose among different accelerator types.
        # Currently we detect which accelerator extension is installed
        # in the environment and use it if the installing answer is True.
        # An alternative might be detect whether CUDA device is installed on
        # the system but this comes with two pitfalls:
        # 1. the system may not have torch pre-installed, so
        #    get_accelerator().is_available() may not work.
        # 2. Some scenario like install on login node (without CUDA device)
        #    and run on compute node (with CUDA device) may cause mismatch
        #    between installation time and runtime.

        if accelerator_name is None:
            try:
                import torch

                # Determine if we are on a GPU or x86 CPU with torch.
                # "torch.cuda.is_available()" provides a stronger guarantee,     #ignore-cuda
                # ensuring that we are free from CUDA initialization errors.
                # While "torch.cuda.device_count() > 0" check ensures that       #ignore-cuda
                # we won't try to do any CUDA calls when no device is available
                # For reference: https://github.com/deepspeedai/DeepSpeed/pull/6810
                if torch.cuda.device_count() > 0 and torch.cuda.is_available():  #ignore-cuda
                    accelerator_name = "cuda"
            except (RuntimeError, ImportError) as e:
                # TODO need a more decent way to detect which accelerator to use, consider using nvidia-smi command for detection
                pass
        if accelerator_name is None:
            # borrow this log from PR#5084
            if accel_logger is not None:
                accel_logger.warning(
                    "Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.")
            # cpu added as catch-all when accelerator detection fails
            accelerator_name = "cpu"

        ds_set_method = "auto detect"

    # 3. Set ds_accelerator accordingly
    if accelerator_name == "cuda":
        from .cuda_accelerator import CUDA_Accelerator

        ds_accelerator = CUDA_Accelerator()
    elif accelerator_name == "cpu":
        from .cpu_accelerator import CPU_Accelerator

        ds_accelerator = CPU_Accelerator()
    _validate_accelerator(ds_accelerator)
    if accel_logger is not None:
        accel_logger.info(f"Setting apex_accelerator to {ds_accelerator._name} ({ds_set_method})")
    return ds_accelerator


def set_accelerator(accel_obj):
    global ds_accelerator
    _validate_accelerator(accel_obj)
    if accel_logger is not None:
        accel_logger.info(f"Setting apex_accelerator to {accel_obj._name} (model specified)")
    ds_accelerator = accel_obj
