import torch.utils.cpp_extension
import shutil
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from op_builder.all_ops import ALL_OPS

torch_ext_directory = torch.utils.cpp_extension._get_build_directory("", False)

install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    path = os.path.join(torch_ext_directory, op_name)
    if os.path.exists(path):
        print ("removing torch extension", op_name, "at", torch_ext_directory)
        shutil.rmtree(path)