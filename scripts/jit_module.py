"""
Script to test JIT module for Apex.
"""

import sys
import os

class JitModule:

    def __init__(self):
        self.op_builder_folder = "op_builder"
        self.compatability_folder = "compatibility"

    def get_module_name(self, builder_file_name):
        #open builder file and read the NAME attribute  
        with open(os.path.join(self.op_builder_folder, f"{builder_file_name}.py"), "r") as f:
            contents = f.read()
        for line in contents.split("\n"):
            if "NAME = " in line:
                return line.split("NAME = ")[1].strip()[1:-1]
        return None


    def create_loader_class_name(self, module_name):
        parts = module_name.split("_")
        new_name = ""
        for part in parts:
            new_name += part.capitalize()
        return f"_{new_name}Module"


    def create_builder_class_name(self, module_name):
        parts = module_name.split("_")
        new_name = ""
        for part in parts:
            new_name += part.capitalize()
        return f"{new_name}Builder"

    
    def create_build_var(self, module_name):
        return f"APEX_BUILD_{module_name.upper()}"


    def check_if_builder_module_exists(self, module_name):
        if os.path.exists(os.path.join(self.op_builder_folder, f"{module_name}.py")):
            return True
        else:
            return False

    def check_if_loader_module_exists(self, module_name):
        if os.path.exists(os.path.join(self.compatability_folder, f"{module_name}.py")):
            return True
        else:
            return False


    def findBuilderClassName(self, builder_name):
        #read file contents of op_builder/builder_name.py
        with open(os.path.join(self.op_builder_folder, f"{builder_name}.py"), "r") as f:
            contents = f.read()
        #find the class name that inherits from CPUOpBuilder or CUDAOpBuilder
        for line in contents.split("\n"):
            if "class" in line:
                return line.split("class")[1].split("(")[0].strip()
        return None


    def create_loader(self, builder_name):
        module_name = self.get_module_name(builder_name) or builder_name
        #check if a loader module in compatability folder
        is_loader_exists = self.check_if_loader_module_exists(module_name)
        if is_loader_exists:
            print(f"Loader module {module_name} exists")
            return

        #create loader class name to use in loader module
        loader_class_name = self.create_loader_class_name(module_name)

        #find builder class name to use in the loader
        builder_class_name = self.findBuilderClassName(builder_name)

        #create a loader module in compatability folder
        with open(os.path.join(self.compatability_folder, f"{module_name}.py"), "w") as f:
            f.write(f"import sys\n")
            f.write(f"import importlib\n")
            f.write(f"\n")
            f.write(f"class {loader_class_name}:\n")
            f.write(f"    def __init__(self):\n")
            f.write(f"        self._loaded_module = None\n")
            f.write(f"        self._loading = False\n")
            f.write(f"\n")
            f.write(f"    def _load_module(self):\n")
            f.write(f"        if self._loaded_module is None and not self._loading:\n")
            f.write(f"            self._loading = True\n")
            f.write(f"            try:\n")
            f.write(f"                apex_op_builder = importlib.import_module('apex.op_builder')\n")
            f.write(f"                builder = getattr(apex_op_builder, '{builder_class_name}')\n")
            f.write(f"                self._loaded_module = builder().load()\n")
            f.write(f"            except Exception as e:\n")
            f.write(f"                self._loading = False\n")
            f.write(f"                raise ImportError('Failed to load " + builder_name + " :' + str(e))\n")
            f.write(f"            finally:\n")
            f.write(f"                self._loading = False\n")
            f.write(f"        return self._loaded_module\n")
            f.write(f"\n")
            f.write(f"    def __getattr__(self, name):\n")
            f.write(f"        if name.startswith('_'):\n")
            f.write(f"            raise AttributeError(f'module {module_name} has no attribute ' + name)\n")
            f.write(f"        return getattr(self._load_module(), name)\n") #dynamic loading of the module
            f.write(f"\n")
            f.write(f"    def __dir__(self):\n")
            f.write(f"        try:\n")
            f.write(f"            return dir(self._load_module())\n")
            f.write(f"        except:\n")
            f.write(f"            return []\n")
            f.write(f"\n")
            f.write(f"    def __repr__(self):\n")
            f.write(f"        return '<module {module_name}>'\n")
            f.write(f"\n")
            f.write(f"sys.modules[__name__] = {loader_class_name}()\n")

        print(f"Loader module {module_name} created")


    def create_builder(self, module_name):
        #Interactively prompt for builder details and create the builder module.
        if_cuda_module = input("Is this a CUDA module? (Y/n) ").strip() or "y"
        sources = input("Enter the sources (comma separated). Press Enter to skip ").strip()


        if if_cuda_module == "y":
            class_name = "CUDAOpBuilder"
            include_flag = "APEX_BUILD_CUDA_OPS"
        else:
            class_name = "CPUOpBuilder"
            include_flag = "APEX_BUILD_CPU_OPS"

        builder_class_name = self.create_builder_class_name(module_name)
        build_var = self.create_build_var(module_name)
        
        if len(sources) == 0:
            sources_list = []
            sources_list_string = "[]"
        else:
            sources_list = sources.split(",")
            sources_list_string = "[" + ",".join(["'" + source.strip() + "'" for source in sources_list]) + "]"
        print(f"sources_list_string: {sources_list_string}")

        include_paths = []
        for source in sources_list:
            if "csrc" in source and "csrc" not in include_paths:
                include_paths.append("csrc")
            elif "contrib/csrc" in source and "contrib/csrc" not in include_paths:
                include_paths.append("contrib/csrc")
        include_paths_string = "[" + ",".join(["'" + path.strip() + "'" for path in include_paths]) + "]"

        with open(os.path.join(self.op_builder_folder, f"{module_name}.py"), "w") as f:
            if if_cuda_module == "y":
                f.write(f"from .builder import CUDAOpBuilder\n")
            else:
                f.write(f"from .builder import CPUOpBuilder\n")
            f.write(f"\n")
            f.write(f"class {builder_class_name}({class_name}):\n")
            f.write(f"    BUILD_VAR = \"{build_var}\"\n")
            f.write(f"    INCLUDE_FLAG = \"{include_flag}\"\n")
            f.write(f"    NAME = \"{module_name}\"\n")
            f.write(f"\n")
            f.write(f"    def __init__(self):\n")
            f.write(f"        super().__init__(name=self.NAME)\n")
            f.write(f"\n")
            f.write(f"    def absolute_name(self):\n")
            f.write(f"        return f'apex.{{self.NAME}}'\n")
            f.write(f"\n")  
            f.write(f"    def sources(self):\n")
            if len(sources) == 0:
                f.write(f"        #This method returns the list of source files to be compiled\n")
                f.write(f"        #Please mention the full path of the source files\n")
                f.write(f"        #e.g. ['csrc/fused_dense_base.cpp', 'csrc/fused_dense_cuda.cu']\n")
            f.write(f"        return {sources_list_string}\n")  
            f.write(f"\n")
            f.write(f"    def include_paths(self):\n")
            if len(sources) == 0:
                f.write(f"        #This method returns the list of include directories\n")
                f.write(f"        #Please mention the full path of the include directories\n")
                f.write(f"        #e.g. ['csrc', 'contrib/csrc']\n")
            f.write(f"        return {include_paths_string}\n")
            f.write(f"\n")
            f.write(f"    def cxx_args(self):\n")
            f.write(f"        return super().cxx_args() + self.generator_args() + self.version_dependent_macros()\n")
            f.write(f"\n")
            f.write(f"    def nvcc_args(self):\n")  
            f.write(f"        return super().nvcc_args() + ['-O3'] + self.version_dependent_macros()\n")
            f.write(f"\n")
            f.write(f"    def is_compatible(self, verbose=False):\n")
            f.write(f"        return True\n")
            f.write(f"\n")
            f.write(f"    def libraries_args(self):\n")
            f.write(f"        return self.libraries_args()\n")

        print(f"Builder module {builder_name} created")


    def add_jit_module(self, builder_name):
        #check if builder module exists
        is_builder_exists = self.check_if_builder_module_exists(builder_name)
        if not is_builder_exists:
            self.create_builder(builder_name)
        else:
            print(f"Builder module {builder_name} already exists")

        #get module name from builder name
        module_name = self.get_module_name(builder_name)
        if module_name is None:
            print(f"Module name for builder {builder_name} not found")
            return

        #if the loader module does not exist, create it
        if not self.check_if_loader_module_exists(builder_name):
            self.create_loader(builder_name)


def main():
    jit_module = JitModule()
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
    else:
        module_name = input("What is the name of the module? ").strip()
    if not module_name:
        print("No module name provided.")
        sys.exit(1)
    success = jit_module.add_jit_module(module_name)
    if success:
        print("JIT module added")

if __name__ == "__main__":
    main()