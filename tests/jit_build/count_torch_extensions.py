import os

import torch.utils.cpp_extension

torch_ext_directory = torch.utils.cpp_extension._get_build_directory("", False)
#count the number of folders
folders = [f for f in os.listdir(torch_ext_directory) if os.path.isdir(os.path.join(torch_ext_directory, f))]
count = len(folders)
print(count)