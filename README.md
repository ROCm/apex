# Introduction

This repository holds ROCm variant of Nvidia's Apex: https://github.com/NVIDIA/apex. 
The aim of Apex repository is to streamline mixed precision and distributed training in Pytorch.
Some of the code here will be included in upstream Pytorch eventually.
The intent of Apex is to make up-to-date utilities available to users as quickly as possible.

## Full API Documentation: [https://nvidia.github.io/apex](https://nvidia.github.io/apex)

## [GTC 2019](https://github.com/mcarilli/mixed_precision_references/tree/master/GTC_2019) and [Pytorch DevCon 2019](https://github.com/mcarilli/mixed_precision_references/tree/master/Pytorch_Devcon_2019) Slides

# Contents

## 1. Amp:  Automatic Mixed Precision

`apex.amp` is a tool to enable mixed precision training by changing only 3 lines of your script.
Users can easily experiment with different pure and mixed precision training modes by supplying
different flags to `amp.initialize`.

[Webinar introducing Amp](https://info.nvidia.com/webinar-mixed-precision-with-pytorch-reg-page.html)
(The flag `cast_batchnorm` has been renamed to `keep_batchnorm_fp32`).

[API Documentation](https://nvidia.github.io/apex/amp.html)

[Comprehensive Imagenet example](https://github.com/rocm/apex/tree/master/examples/imagenet)

[DCGAN example coming soon...](https://github.com/rocm/apex/tree/master/examples/dcgan)

[Moving to the new Amp API](https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users) (for users of the deprecated "Amp" and "FP16_Optimizer" APIs)

## 2. Distributed Training

`apex.parallel.DistributedDataParallel` is a module wrapper, similar to
`torch.nn.parallel.DistributedDataParallel`.  It enables convenient multiprocess distributed training,
optimized for NVIDIA's NCCL communication library.

[API Documentation](https://nvidia.github.io/apex/parallel.html)

[Python Source](https://github.com/rocm/apex/tree/master/apex/parallel)

[Example/Walkthrough](https://github.com/rocm/apex/tree/master/examples/simple/distributed)

The [Imagenet example](https://github.com/rocm/apex/tree/master/examples/imagenet)
shows use of `apex.parallel.DistributedDataParallel` along with `apex.amp`.

### Synchronized Batch Normalization

`apex.parallel.SyncBatchNorm` extends `torch.nn.modules.batchnorm._BatchNorm` to
support synchronized BN.
It allreduces stats across processes during multiprocess (DistributedDataParallel) training.
Synchronous BN has been used in cases where only a small
local minibatch can fit on each GPU.
Allreduced stats increase the effective batch size for the BN layer to the
global batch size across all processes (which, technically, is the correct
formulation).
Synchronous BN has been observed to improve converged accuracy in some of our research models.

### Checkpointing

To properly save and load your `amp` training, we introduce the `amp.state_dict()`, which contains all `loss_scalers` and their corresponding unskipped steps,
as well as `amp.load_state_dict()` to restore these attributes.

In order to get bitwise accuracy, we recommend the following workflow:
```python
# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
torch.save(checkpoint, 'amp_checkpoint.pt')
...

# Restore
model = ...
optimizer = ...
checkpoint = torch.load('amp_checkpoint.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])

# Continue training
...
```

Note that we recommend restoring the model using the same `opt_level`. Also note that we recommend calling the `load_state_dict` methods after `amp.initialize`.

# Installation

## Containers
ROCm pytorch containers contain apex package and these are available from https://hub.docker.com/r/rocm/pytorch.

## From Source

Torch must be installed before installing apex. We recommend using the nightly Pytorch obtainable from https://github.com/rocm/pytorch. The latest stable release obtainable from https://pytorch.org should also work.

Apex on ROCm supports both python only build and extension build.
Note: Pytorch version recommended is >=1.5 for extension build.

### The following command will install all the extensions, which will be built and linked at runtime using [PyTorch's JIT (just-in-time) loader](https://pytorch.org/docs/stable/cpp_extension.html):
This requires ninja to be installed
```
pip install . --no-build-isolation
```

### Supported Versions
| ``APEX Version`` | ``APEX branch`` | ``Torch Version`` |
|------------------|-----------------|-------------------|
| ``1.9.0``        | release/1.9.0   | ``2.9``           | 
| ``1.8.0``        | release/1.8.0   | ``2.8``           | 
| ``1.7.0``        | release/1.7.0   | ``2.7``           | 
| ``1.6.0``        | release/1.6.0   | ``2.6``           | 
| ``1.5.0``        | release/1.5.0   | ``2.5``           | 
| ``1.4.0``        | release/1.4.0   | ``2.4``           | 
| ``1.3.0``        | release/1.3.0   | ``2.3``           | 
| ``1.2.0``        | release/1.2.0   | ``2.2``           | 
| ``1.1.0``        | release/1.1.0   | ``2.1``           |
| ``1.0.0``        | release/1.0.0   | ``2.0`` and older |


The relation between APEX and ROCm PyTorch is maintained in file `related_commits` in [ROCm PyTorch release branches](https://github.com/ROCm/pytorch/branches/all?query=release) in the following format. 

```
ubuntu|pytorch|apex|release/1.0.0|06c33eee43f7a22f3ed7d9c3e5be0ddd757dc345|https://github.com/ROCmSoftwarePlatform/apex
centos|pytorch|apex|release/1.0.0|06c33eee43f7a22f3ed7d9c3e5be0ddd757dc345|https://github.com/ROCmSoftwarePlatform/apex
```

### To pre-build and install all the supported extensions while installing apex, use the following command in apex folder:
```
APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1 pip install . --no-build-isolation
```

It is also possible to pre-build and install specific extensions by using the following command in apex folder:
```
APEX_BUILD_<OP_NAME>=1 pip install . --no-build-isolation
```
The following extensions are supported:
| extension | environment to build specific extension | install option |
|-----------|-----------|-----------|
| amp_C | APEX_BUILD_AMP_C=1 | APEX_BUILD_CUDA_OPS=1 |
| apex_C | APEX_BUILD_APEX_C=1 | APEX_BUILD_CPP_OPS=1 |
| bnp | APEX_BUILD_BNP=1 | APEX_BUILD_CUDA_OPS=1 |
| distributed_adam_cuda | APEX_BUILD_DISTRIBUTED_ADAM=1 | APEX_BUILD_CUDA_OPS=1 |
| distributed_lamb_cuda | APEX_BUILD_DISTRIBUTED_LAMB=1 | APEX_BUILD_CUDA_OPS=1 |
| fast_multihead_attn | APEX_BUILD_FAST_MULTIHEAD_ATTN=1 | APEX_BUILD_CUDA_OPS=1 |
| focal_loss_cuda | APEX_BUILD_FOCAL_LOSS=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_adam_cuda | APEX_BUILD_FUSED_ADAM=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_bias_swiglu | APEX_BUILD_FUSED_BIAS_SWIGLU=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_conv_bias_relu | APEX_BUILD_FUSED_CONV_BIAS_RELU=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_dense_cuda | APEX_BUILD_FUSED_DENSE=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_index_mul_2d | APEX_BUILD_FUSED_INDEX_MUL_2D=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_lamb_cuda | APEX_BUILD_FUSED_LAMB=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_layer_norm_cuda | APEX_BUILD_FUSED_LAYER_NORM=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_rotary_positional_embedding | APEX_BUILD_FUSED_ROPE=1 | APEX_BUILD_CUDA_OPS=1 |
| fused_weight_gradient_mlp_cuda | APEX_BUILD_FUSED_WEIGHT_GRADIENT_MLP=1 | APEX_BUILD_CUDA_OPS=1 |
| generic_scaled_masked_softmax_cuda | APEX_BUILD_GENERIC_SCALED_MASKED_SOFTMAX_CUDA=1 | APEX_BUILD_CUDA_OPS=1 |
| mlp_cuda | APEX_BUILD_MLP=1 | APEX_BUILD_CUDA_OPS=1 |
| _apex_nccl_allocator | APEX_BUILD_NCCL_ALLOCATOR=1 | APEX_BUILD_CUDA_OPS=1 |
| nccl_p2p_cuda | APEX_BUILD_NCCL_P2P=1 | APEX_BUILD_CUDA_OPS=1 |
| peer_memory_cuda | APEX_BUILD_PEER_MEMORY=1 | APEX_BUILD_CUDA_OPS=1 |
| scaled_masked_softmax_cuda | APEX_BUILD_SCALED_MASKED_SOFTMAX_CUDA=1 | APEX_BUILD_CUDA_OPS=1 |
| scaled_softmax_cuda | APEX_BUILD_SCALED_SOFTMAX_CUDA=1 | APEX_BUILD_CUDA_OPS=1 |
| scaled_upper_triang_masked_softmax_cuda | APEX_BUILD_SCALED_UPPER_TRIANG_MASKED_SOFTMAX_CUDA=1 | APEX_BUILD_CUDA_OPS=1 |
| syncbn | APEX_BUILD_SYNCBN=1 | APEX_BUILD_CUDA_OPS=1 |
| transducer_joint_cuda | APEX_BUILD_TRANSDUCER_JOINT=1 | APEX_BUILD_CUDA_OPS=1 |
| transducer_loss_cuda | APEX_BUILD_TRANSDUCER_LOSS=1 | APEX_BUILD_CUDA_OPS=1 |
| xentropy_cuda | APEX_BUILD_XENTROPY=1 | APEX_BUILD_CUDA_OPS=1 |

For example, to build FUSED_DENSE​ you can use the following command: 
```
APEX_BUILD_FUSED_DENSE​=1 pip install . --no-build-isolation
```
This will pre-build and install FUSED_DENSE​ module and rest of the modules are installed to be JIT built and loaded at runtime. 

Aiter backend can be built and used for fused rope. To install aiter:
```
make aiter
```

To use aiter in fused rope, you can use the flag ```USE_ROCM_AITER_ROPE_BACKEND=1```.

### To add a new module into jit loader

What is JIT (just-in-time) load? Just-in-time load helps to build the specific modules that are used without needing to build all modules during installation time. This helps to significantly reduce installation time. Without JIT load, it would take roughtly 30 minutes to install apex. With JIT load, it takes less than 1 minute to install apex. 

A python script is provided to ease the process of adding a new module to JIT load. 
For this, the user must create C++/CUDA source code for a new apex module in either csrc or apex/contrib/csrc folder. 
This script helps to create a builder and a loader for the apex module.
The builder creates the .so file for the apex module (during installation or jit load time) and the loader loads the .so file when the module is imported.

To run the script:

```
python scripts/jit_module.py <apex_module_name>
``` 

The user should provide the name used to import the module i.e. import fused_bias_swiglu.
If the user does not provide the module name, the script will ask for the module name
```
What is the name of the module?
``` 

The script is interactive and asks two questions 
1. Is this a CUDA module? (Y/n)
2. Enter the sources (comma separated) Press Enter to skip 

If the user answers yes to cuda module, it builds with CUDAOpBuilder otherwise it builds as a cpu operation with CPUOpBuilder. The default is cuda operation.
The user must mention the list of .cpp, .h, .cu files used to compile the module as a comma separated list.
This argument is used to define the return value of sources() method in the builder module.
This will be used to also find the list of directories (include_paths() method) i.e. -I flag in g++ compiler. 
The user can decide to skip the list of sources and add it manually to the builder file created by the script. 

e.g.  
```
python scripts/jit_module.py fused_bias_swiglu
1. Is this a CUDA module? (Y/n) y
2. Enter the sources (comma separated) Press Enter to skip csrc/megatron/fused_bias_swiglu.cpp,csrc/megatron/fused_bias_swiglu_cuda.cu
```

**Directory structure (fused_bias_swiglu example):**

The above example creates a builder - [op_builder/fused_bias_swiglu.py](op_builder/fused_bias_swiglu.py) and a loader - [compatibility/fused_bias_swiglu.py](compatibility/fused_bias_swiglu.py).

```
apex/                                    # repo root
├── csrc/                                # C++/CUDA source (or apex/contrib/csrc)
│   └── megatron/
│       ├── fused_bias_swiglu.cpp        # PyBind11 module defs, declarations
│       └── fused_bias_swiglu_cuda.cu    # CUDA kernels / implementation
├── op_builder/                          # Builder: compiles sources → .so
│   └── fused_bias_swiglu.py             # FusedBiasSwiGLUBuilder (NAME = "fused_bias_swiglu", sources(), etc.)
├── compatibility/                       # Loader: JIT-loads .so when module is imported
│   └── fused_bias_swiglu.py             # _FusedBiasSwiGLUModule (loads via apex.op_builder.FusedBiasSwiGLUBuilder)
└── apex/                                # Python package
    └── fused_bias_swiglu/               # User-facing API (import apex.fused_bias_swiglu)
        ├── __init__.py
        └── fused_bias_swiglu.py         # imports fused_bias_swiglu, wraps forward/backward, etc.
```


The user must not edit the loader code. 

The script creates an initial builder code and the users can edit the methods in the module.

The builder module is created in op_builder folder and must override either CPUOpBuilder or CUDAOpBuilder class and define the following attributes and methods:

| Attribute | Purpose |
|-----------|-----------|
| BUILD_VAR | The environment variable to indicate prebuilding the module when installing apex e.g. APEX_BUILD_FUSED_BIAS_SWIGLU for fused_bias_swiglu|
| INCLUDE_FLAG | Either APEX_BUILD_CUDA_OPS or APEX_BUILD_CPU_OPS to indicate whether the module will be built for gpu or cpu |
| NAME | name of module e.g. fused_bias_swiglu |

| Method | Purpose | Necessary to override | 
|-----------|-----------|-----------|
| absolute_name | return the namespace where the module will be installed | Yes |
| sources | list of C++/CUDA source files for the module | Yes |
| include_paths | list of folders where the included headers mentioned in the source files are placed | No |
| cxx_args | return a list of extra compiler flags for the C++ compiler when building C++ sources (e.g. optimization level, preprocessor macros) | No |
| nvcc_args | return a list of extra compiler flags for nvcc when building CUDA sources (e.g. -O3, architecture flags, preprocessor macros) | No |
| is_compatible | can this module be installed and loaded considering the environment e.g.minimum torch version supported | No |
| libraries_args  | list of libraries to compile against e.g. MIOpen | No |





### To create a wheel and then install apex using the wheel, use the following command in apex folder:
```
python -m build --wheel --no-isolation (can use the same environment variables to build specific extensions, cpp extensions and cuda extensions)
pip install dist/apex-*.whl​
```

### To uninstall apex and its extensions, use the following command in apex folder:
```
pip uninstall apex
make clean
```

### Enable hipblasLT on ROCm
hipblasLT is supported only on mi300 (gfx942) only.  
python setup.py automatically builds apex with hipblasLT support only if GPU device id is gfx942  
To verify if hipblasLT support is enabled, check the build logs  
INFO: IS_HIPBLASLT_SUPPORTED value is True  ==> indicates apex is built with hipblasLT support  
INFO: IS_HIPBLASLT_SUPPORTED value is False  

### Linux
For performance and full functionality, we recommend installing Apex with
CUDA and C++ extensions via
```bash
git clone https://github.com/rocm/apex
cd apex
pip install . --no-build-isolation
```


### [Experimental] Windows
`pip install . --no-build-isolation` may work if you were able to build Pytorch from source
on your system. A Python-only build via `pip install --no-build-isolation -v --no-cache-dir .` is more likely to work.  
If you installed Pytorch in a Conda environment, make sure to install Apex in that same environment.

# Release notes

## release/1.10.0

Build and installation related
- Support JIT (just-in-time) load cpp and CUDA extensions

## release/1.9.0

- No new features were added in this release cycle.

## release/1.8.0

Unit test related
- Fix transformer unit tests
- Fix fused dense gelu dense unit tests

## release/1.7.0

Build and installation related
- Support use of BUILD_VERSION environment to override version.txt when creating apex wheels
- Disable aiter installation by default. make aiter command is used to build apex

Unit test related
- Include running transformer tests in L0/run_test.py
- Fix transformer unit tests
- Fix batch norm unit tests
- Fix fused dense gelu dense unit tests

## release/1.6.0

Upgraded extensions
- Support unscale_grads in transformer Grad scaler
- Support amp function in fused dense, mlp
- Support blas backend flag in fused dense 
- Support not destroying process group for distributed tests
- Upgrade fused adam to support parameters - capturable, master weights, grad scaler
- Upgrade distributed fused adam to support bias_correction, adam_w_mode, overlap_param_sync, store_params, store_param_remainders, with_scaled_states, nccl_ub
- Upgrade distributed fused lamb to support parameters fused_norm, full_ar, set_param_views_to_flat_buffer, skip_allgather, fuse_scale, param_order, nccl_allgather_channels

Unit test related
- Fix fused dense, fused rope, mlp unit tests
- Add test fused adam unit test
- Include running fused dense tests in L0/run_test.py


## release/1.5.0

Added extensions
- fused bias swiglu
- fused gradient accumulator
- fused rope
  
Upgraded extensions
- Support blaslt backend in fused weight gradient dense module



