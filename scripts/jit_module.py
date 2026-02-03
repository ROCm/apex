"""
Script to test JIT module for Apex.
"""

import sys
import os

class JitModule:

    def __init__(self):
        self.op_builder_folder = "op_builder"
        self.compatability_folder = "compatibility"


    def create_loader_class_name(self, module_name):
        parts = module_name.split("_")
        new_name = ""
        for part in parts:
            new_name += part.capitalize()
        return f"_{new_name}Module"


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


    def add_jit_module(self, module_name, builder_name):

        is_builder_exists = self.check_if_builder_module_exists(builder_name)
        if not is_builder_exists:
            print(f"Builder module {builder_name} does not exist")
            return

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
            f.write(f"                raise ImportError(Failed to load " + builder_name + " : + str(e) + )\n")
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
        return True


def main():
    jit_module = JitModule()
    module_name = sys.argv[1]
    builder_name = sys.argv[2]
    success = jit_module.add_jit_module(module_name, builder_name)
    if success:
        print("JIT module added")

if __name__ == "__main__":
    main()

        
        
