from model import VisionTransformer
from configs import get_base_config
import jax
import jax.numpy as jnp
from flax.core import frozen_dict

def print_tree(params, indent=0):
    """Recursively print model parameters as a tree."""
    for key, value in params.items():
        if isinstance(value, dict):  # If nested dictionary, recurse
            print("  " * indent + f"{key}:")
            print_tree(value, indent + 1)
        else:
            # Print leaf nodes with their shapes
            print("  " * indent + f"{key}: shape={value.shape}")

config = get_base_config()
model = VisionTransformer(**config, num_classes=1000)

rng = jax.random.PRNGKey(0)
example_input = jnp.ones((1,256, 256, 3))
labels = None
params = model.init(rng, inputs = {
    'image': example_input,
    'label': labels
    }, train= True)  # Initialize parameters

print_tree(params)