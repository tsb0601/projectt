from _collections_abc import dict_items
from model import VisionTransformer
from configs import get_base_config
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from jax._src import compilation_cache
cache_path = '/home/bytetriper/.cache/jax/compilation_cache/test'
import os
os.makedirs(cache_path, exist_ok=True)
print(compilation_cache._is_cache_enabled())
compilation_cache.set_cache_dir(path=cache_path)
print(compilation_cache._is_cache_enabled())
    
def rename_and_nest_keys(flat_dict: dict) -> dict:
    """
    Converts a flat dictionary (or FrozenDict) with dot-separated keys
    into a nested dictionary structure.
    
    Args:
        flat_dict (FrozenDict or dict): The input flat dictionary with dot-separated keys.
    
    Returns:
        Nested dictionary or FrozenDict.
    """
    # Convert FrozenDict to mutable dict if necessary
    if isinstance(flat_dict, frozen_dict.FrozenDict):
        flat_dict = flat_dict.unfreeze()
    param_dict = flat_dict['params']
    def nest_dict(flat_dict):
        nested = {}
        for flat_key, value in flat_dict.items():
            # Split the key by '.'
            keys = flat_key.split('.')
            current_level = nested
            for key in keys[:-1]:
                # Create nested levels if they don't exist
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
            # Set the final key
            current_level[keys[-1]] = value
        return nested
    
    # Build nested dict
    nested = nest_dict(param_dict)
    
    flat_dict['params'] = nested
    # Optionally convert back to FrozenDict
    return flat_dict
def print_tree(params, indent=0, max_d = 2):
    """Recursively print model parameters as a tree."""
    for key, value in params.items():
        if isinstance(value, dict):  # If nested dictionary, recurse
            print("  " * indent + f"{key}", end=": \n" if indent < max_d else "\n")
            if indent < max_d:
                print_tree(value, indent + 1, max_d)
        else:
            # Print leaf nodes with their shapes
            print("  " * indent + f"{key}: shape={value.shape}")


config = get_base_config()
config.image_size = (256, 256)
model = VisionTransformer(**config, num_classes=1000)

#rng = jax.random.PRNGKey(0)
init_rng = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
example_input = jnp.ones((1, 256, 256, 3))
labels = None
params = model.init(
    init_rng, imgs = example_input, train=False
)  # Initialize parameters

# print_tree(params)
#print(type(params))
#params: dict
#print_tree(params)
#print("+="*40)
# Convert the flat dictionary to a nested dictionary
#params = rename_and_nest_keys(params)
#print_tree(params, max_d=10000)
# lets do a forward pass
output = model.apply(
    params, imgs = example_input, train=True, rngs = {"dropout": jax.random.PRNGKey(1)}
)
print(output[0], output[1].shape)  # (1, 1000) (1, 16, 16, 768)
