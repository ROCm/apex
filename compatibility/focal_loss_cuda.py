import sys
import importlib

class _FocalLossCudaModule:
    def __init__(self):
        self._loaded_module = None
        self._loading = False

    def _load_module(self):
        if self._loaded_module is None and not self._loading:
            self._loading = True
            try:
                apex_op_builder = importlib.import_module('apex.op_builder')
                builder = getattr(apex_op_builder, 'FocalLossBuilder')
                self._loaded_module = builder().load()
            except Exception as e:
                self._loading = False
                raise ImportError(f"Failed to load focal_loss_cuda : {e}")
            finally:
                self._loading = False
        return self._loaded_module

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"module focal_loss_cuda has no attribute '{name}'")
        return getattr(self._load_module(), name)

    def __dir__(self):
        try:
            return dir(self._load_module())
        except:
            return []

    def __repr__(self):
        return "<module 'focal_loss_cuda'>"

sys.modules[__name__] = _FocalLossCudaModule()