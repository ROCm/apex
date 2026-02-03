"""
Script to test JIT module for Apex.
"""

import sys
import os

class JitModule:

    def __init__(self):
        self.op_builder_folder = "op_builder"
        self.compatability_folder = "compatibility"


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

    def add_jit_module(self, module_name):

        if "cuda" in module_name:
            builder_module_name = module_name[:-5]
        else:
            builder_module_name = f"{module_name}"

        print(f"Builder module name: {builder_module_name}")

        is_builder_exists = self.check_if_builder_module_exists(builder_module_name)
        if not is_builder_exists:
            print(f"Builder module {builder_module_name} does not exist")
            return

        #check if a loader module in compatability folder
        is_loader_exists = self.check_if_loader_module_exists(module_name)
        if is_loader_exists:
            print(f"Loader module {module_name} exists")
            return

        #create a loader module in compatability folder
        with open(os.path.join(self.compatability_folder, f"{module_name}.py"), "w") as f:
            f.write(f"import sys\n")
            f.write(f"import importlib\n")
            f.write(f"class _{module_name}Module:\n")
            f.write(f"    def __init__(self):\n")
            f.write(f"        self._loaded_module = None\n")
            f.write(f"        self._loading = False\n")
            f.write(f"    def _load_module(self):\n")
            f.write(f"        if self._loaded_module is None and not self._loading:\n")
            f.write(f"            self._loading = True\n")
            f.write(f"            try:\n")
            f.write(f"                apex_op_builder = importlib.import_module('apex.op_builder')\n")
            f.write(f"                builder = getattr(apex_op_builder, '{module_name}Builder')\n")
            f.write(f"                self._loaded_module = builder().load()\n")
            f.write(f"            except Exception as e:\n")
            f.write(f"                self._loading = False\n")
            f.write(f"                raise e\n")
            f.write(f"            finally:\n")
            f.write(f"                self._loading = False\n")
            f.write(f"        return self._loaded_module\n")
            f.write(f"    def __getattr__(self, name):\n")
            f.write(f"        if name.startswith('_'):\n")
            f.write(f"            raise AttributeError(f'module {module_name} has no attribute ' + name)\n")
            f.write(f"        return getattr(self._loaded_module, name)\n") #dynamic loading of the module

            f.write(f"    def __dir__(self):\n")
            f.write(f"        try:\n")
            f.write(f"            return dir(self._loaded_module)\n")
            f.write(f"        except:\n")
            f.write(f"            return []\n")
            f.write(f"    def __repr__(self):\n")
            f.write(f"        return f'<module {module_name}>'\n")

            f.write(f"sys.modules[__name__] = _{module_name}Module()\n")

        print(f"Loader module {module_name} created")
        return True


def main():
    jit_module = JitModule()
    module_name = sys.argv[1]
    success = jit_module.add_jit_module(module_name)
    if success:
        print("JIT module added")

if __name__ == "__main__":
    main()

        
        
