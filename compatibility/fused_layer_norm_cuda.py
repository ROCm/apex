import sys
import importlib

class _FusedLayerCudaModule:
    def __init__(self):
        self._loaded_module = None
        self._loading = False

    def _load_module(self):
        if self._loaded_module is None and not self._loading:
            self._loading = True
            try:
                #import the builder
                apex_op_builder = importlib.import_module('apex.op_builder')
                mlp_builder = getattr(apex_op_builder, 'FusedLayerNormBuilder')

                #load the module
                self._loaded_module = mlp_builder().load()
            except Exception as e:
                self._loading = False
                raise ImportError(f"Failed to load fused_layer_norm_cuda : {e}")
            finally:
                self._loading = False
        return self._loaded_module
    
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"module fused_layer_norm_cuda has no attribute '{name}'")
        
        module = self._load_module()
        return getattr(module, name)

    def __dir__(self):
        try:
            module = self._load_module()
            return dir(module)
        except:
            return []
        
    def __repr__(self):
        return "<module 'fused_layer_norm_cuda'>"
    
#replace module with lazy loader
sys.modules[__name__] = _FusedLayerCudaModule()