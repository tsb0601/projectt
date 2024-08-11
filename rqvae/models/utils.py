import importlib
from safetensors.torch import load_model, save_model
from header import *
def load_model_from_ckpt(model:nn.Module, ckpt_path:str, strict:bool = True) -> nn.Module:
    if ckpt_path.endswith('.pt'):
        ckpt = torch.load(ckpt_path)
        keys = model.load_state_dict(ckpt, strict = strict)    
    elif ckpt_path.endswith('.safetensors'):
        keys = load_model(model, ckpt_path, strict = strict)
    else:
        raise ValueError(f'[!]ERROR: ckpt_path should end with .pt or .safetensors, but got {ckpt_path}')
    return model, keys
"""partially modified from https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py"""
def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


