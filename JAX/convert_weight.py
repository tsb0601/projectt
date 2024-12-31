from flax.core import frozen_dict
from model import VisionTransformer
from configs import get_base_config
import jax
import jax.numpy as jnp
import torch
import numpy as np
from jax._src import compilation_cache
cache_dir = '/home/bytetriper/.cache/jax/compilation_cache/convert'
import os
import pickle
from PIL import Image
os.makedirs(cache_dir, exist_ok=True)
compilation_cache.set_cache_dir(path=cache_dir)
def flatten_nested_dict(nested_dict, parent_key='', sep='__SEP__'):
    """
    Flatten a nested dictionary into a single-level dictionary with keys as paths.

    Args:
        nested_dict (dict): Nested dictionary to flatten.
        parent_key (str): Base key string for recursion.
        sep (str): Separator between keys.

    Returns:
        tuple:
            - dict: Flattened dictionary.
            - dict: Mapping of original keys to flattened keys.
    """
    items = []
    mapping = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            sub_items, sub_mapping = flatten_nested_dict(v, new_key, sep=sep)
            items.extend(sub_items.items())
            mapping.update(sub_mapping)
        else:
            items.append((new_key, v))
        mapping[new_key] = new_key.replace(sep, '.')
    return dict(items), mapping

def convert_sep_to_dot(flat_dict, sep='__SEP__'):
    """
    Convert the separator in flattened keys to dots for compatibility.

    Args:
        flat_dict (dict): Flattened dictionary.
        sep (str): Separator used in keys.

    Returns:
        tuple:
            - dict: Dictionary with keys using dots as separators.
            - dict: Mapping of original keys to dot-separated keys.
    """
    converted_dict = {}
    mapping = {}
    for key, value in flat_dict.items():
        new_key = key.replace(sep, '.')
        converted_dict[new_key] = value
        mapping[new_key] = key  # Keep track of conversion for reversal
    return converted_dict, mapping
def convert_dot_to_sep(flat_dict, mapping):
    """
    Convert the separator in flattened keys back to the original separator.

    Args:
        flat_dict (dict): Flattened dictionary.
        sep (str): Separator used in keys.

    Returns:
        dict: Dictionary with keys using the original separator.
    """
    converted_dict = {}
    for key, value in flat_dict.items():
        converted_dict[mapping[key]] = value
    return converted_dict
def unflatten_dict(flat_dict, sep='__SEP__'):
    """
    Convert a flattened dictionary back into a nested dictionary.

    Args:
        flat_dict (dict): Flattened dictionary to unflatten.
        sep (str): Separator between keys.

    Returns:
        dict: Nested dictionary.
    """
    nested_dict = {}
    for flat_key, value in flat_dict.items():
        keys = flat_key.split(sep)
        current_level = nested_dict
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
        current_level[keys[-1]] = value
    return nested_dict
def convert_conv2d_weight(weight: torch.Tensor, config:dict = None) -> jnp.ndarray:
    assert len(weight.shape) == 4, "Conv2D weight should be 4D"
    #print('converting conv2d weight of shape:', weight.shape)
    weight =  weight.numpy()
    weight = weight.transpose((2, 3, 1, 0))
    weight = jnp.array(weight)
    return weight
def convert_linear_weight(weight: torch.Tensor, config:dict = None) -> jnp.ndarray:
    assert len(weight.shape) == 2 , "Linear weight should be 2D, got shape: " + str(weight.shape)
    weight = weight.T
    weight =  weight.numpy()
    weight = jnp.array(weight)
    return weight
def convert_qkv_weight(weight: torch.Tensor, config:dict = None) -> jnp.ndarray:
    assert len(weight.shape) == 2 , "qkv weight should be 2D, got shape: " + str(weight.shape)
    num_heads = config.transformer.num_heads
    weight = weight.T
    #print('converting qkv weight of shape:', weight.shape)
    weight = weight.view(weight.shape[0], num_heads, -1)
    weight =  weight.numpy()
    weight = jnp.array(weight)  
    #print('converted qkv weight of shape:', weight.shape)
    return weight
def convert_qkv_bias(bias: torch.Tensor, config:dict = None) -> jnp.ndarray:
    assert len(bias.shape) == 1 , "qkv bias should be 1D, got shape: " + str(bias.shape)
    num_heads = config.transformer.num_heads
    bias = bias.view(num_heads, -1)
    bias =  bias.numpy()
    bias = jnp.array(bias)
    return bias
def convert_output_weight(weight: torch.Tensor, config:dict = None) -> jnp.ndarray:
    assert len(weight.shape) == 2 , "Output weight should be 2D, got shape: " + str(weight.shape)
    num_heads = config.transformer.num_heads
    head_dim = weight.shape[1] // num_heads
    weight = weight.T
    weight = weight.view(num_heads, head_dim, -1)
    weight =  weight.numpy()
    weight = jnp.array(weight)
    return weight
def convert_id_restore(id_restore: torch.Tensor, config:dict = None) -> jnp.ndarray:
    id_restore =  id_restore.numpy()
    id_restore = jnp.array(id_restore).astype(jnp.float32)
    return id_restore
def convert_basic(weight: torch.Tensor, config:dict = None) -> jnp.ndarray:
    weight =  weight.numpy()
    weight = jnp.array(weight)
    return weight
def dispatcher(name: str ) -> callable:
    if 'patch_embeddings.projection.weight' in name:
        return convert_conv2d_weight
    elif 'norm' in name:
        return convert_basic
    elif 'query.weight' in name or 'key.weight' in name or 'value.weight' in name:
        return convert_qkv_weight
    elif 'attention.output.dense.weight' in name:
        return convert_output_weight
    elif 'query.bias' in name or 'key.bias' in name or 'value.bias' in name:
        return convert_qkv_bias
    elif 'default_id_restore' in name:
        return convert_id_restore
    elif 'weight' in name: # Linear weight
        return convert_linear_weight
    
    else:
        return convert_basic
def torch_to_jax_renaming(torch_dict: dict, jax_dict: dict, config: dict) -> dict:
    """
    state_dict() to jax weight
    """
    jax_rename = {
        'scale': 'weight',
        'kernel': 'weight',
        'model.decoder_pos_embed.position_embeddings': 'model.decoder.decoder_pos_embed',
        'attention.value': 'attention.attention.value',
        'attention.key': 'attention.attention.key',
        'attention.query': 'attention.attention.query',
    }
    jax_name_mapping = {}
    for key in jax_dict.keys():
        jax_name_mapping[key] = key
        cp_key = key

        for k, v in jax_rename.items():
            if k in key:
                cp_key = cp_key.replace(k, v)
                jax_name_mapping[key] = cp_key
                #print(f"Renaming {key} to {jax_name_mapping[key]}")
                #break
    matched_keys = set(jax_name_mapping.values()).intersection(set(torch_dict.keys()))
    unmatched_keys = set(jax_name_mapping.values()).difference(set(torch_dict.keys()))
    print('unmatched keys:', unmatched_keys)
    assert len(unmatched_keys) == 0, f"Unmatched keys: {unmatched_keys}"
    for key, torch_key in jax_name_mapping.items():
        if torch_key not in torch_dict.keys():
            print(f"Key {key} | {torch_key} not in torch_dict")
        torch_weight = torch_dict[torch_key]
        converter = dispatcher(torch_key)
        #print(f'converting {torch_key} with function {converter}')
        if 'decoder' in key:
            cfg = config.decoder
        else:
            cfg = config
        jax_dict[key] = converter(torch_weight, cfg)
    return jax_dict   
def init_model():
    config = get_base_config()
    config.image_size = (256, 256)
    #config.transformer.num_layers = 2
    #config.decoder.transformer.num_layers = 12
    model = VisionTransformer(**config, num_classes=1000)

    #rng = jax.random.PRNGKey(0)
    init_rng = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
    example_input = jnp.ones((1, 256, 256, 3))
    labels = None
    params = model.init(
        init_rng, imgs = example_input, train=False
    )  # Initialize parameters
    return params, model, init_rng, config
import sys
def convert_torch_to_jax(torch_weight_path: str, save_path: str):
    model_params, model, init_rng, config = init_model()
    model_params = model_params['params']
    # Flatten the model parameters
    flat_params, mapping = flatten_nested_dict(model_params)
    # Convert the separator in keys to dots
    flat_params, mapping = convert_sep_to_dot(flat_params)
    # Load the PyTorch weights
    torch_weights = torch.load(torch_weight_path)
    # Convert the PyTorch weights to JAX format
    jax_weights = torch_to_jax_renaming(torch_weights, flat_params, config)
    # Convert the flattened dictionary back to nested
    undotted_params = convert_dot_to_sep(jax_weights, mapping)
    params = unflatten_dict(undotted_params)
    params = frozen_dict.freeze({'params': params})
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)
    return params
def get_model_from_jax_weight(jax_weight_path: str):
    with open(jax_weight_path, 'rb') as f:
        params = pickle.load(f)
    model_params, model, init_rng, config = init_model()
    #model_params = model_params['params']
    #model_params.update(params['params'])
    #model = VisionTransformer(**config, num_classes=1000)
    return model, params
def get_model_from_torch_weight(torch_weight_path: str):
    model_params, model, init_rng, config = init_model()
    model_params = model_params['params']
    # Flatten the model parameters
    flat_params, mapping = flatten_nested_dict(model_params)
    # Convert the separator in keys to dots
    flat_params, mapping = convert_sep_to_dot(flat_params)
    # Load the PyTorch weights
    torch_weights = torch.load(torch_weight_path)
    # Convert the PyTorch weights to JAX format
    jax_weights = torch_to_jax_renaming(torch_weights, flat_params, config)
    # Convert the flattened dictionary back to nested
    undotted_params = convert_dot_to_sep(jax_weights, mapping)
    params = unflatten_dict(undotted_params)
    params = frozen_dict.freeze({'params': params})
    model = VisionTransformer(**config, num_classes=1000)
    return model, params
jax_model_encode = VisionTransformer.apply_encoder
jax_model_decode = VisionTransformer.apply_decoder
def main():
    torch_weight_path = sys.argv[1]
    model_params, model, init_rng, config = init_model()
    model_params = model_params['params']
    # Flatten the model parameters
    flat_params, mapping = flatten_nested_dict(model_params)
    # Convert the separator in keys to dots
    flat_params, mapping = convert_sep_to_dot(flat_params)
    #print('TEST shape:',flat_params['model.vit.encoder.layers.0.output.dense.kernel'].shape)
    #for key in flat_params.keys():
    #    print(key, ':', flat_params[key].shape)
    # Load the PyTorch weights
    torch_weights = torch.load(torch_weight_path)
    # Convert the PyTorch weights to JAX format
    jax_weights = torch_to_jax_renaming(torch_weights, flat_params, config)
    #for key in jax_weights.keys():
    #    print(key, ':', jax_weights[key].shape)
    # Convert the flattened dictionary back to nested
    undotted_params = convert_dot_to_sep(jax_weights, mapping)
    params = unflatten_dict(undotted_params)
    #print(params.keys())
    params = frozen_dict.freeze({'params': params})
    import pickle
    with open(os.path.join(os.path.dirname(torch_weight_path), 'mae_jax.pkl'), 'wb') as f:
        pickle.dump(params, f)
    # do a forward pass
    #example_input = jnp.ones((1, 256, 256, 3)) # (B, H, W, C)
    # random input
    #example_input = jnp.array(np.random.uniform(0, 1, (1, 256, 256, 3)))
    image_path = '../visuals/npz_image.png'
    from PIL import Image
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = img / 255.
    img = jnp.array(img)
    example_input = img[None, ...]
    example_input = jnp.array(example_input)
    example_input = jnp.tile(example_input, (4, 1, 1, 1))
    print('example_input:', example_input.shape)
    #loss, preds = model.apply(params, example_input, train=False, rngs = {"dropout": jax.random.PRNGKey(1)})
    latent, mask, ids_restore = model.apply(params, example_input, train=False, rngs = {"dropout": jax.random.PRNGKey(1)}, method= VisionTransformer.apply_encoder)
    patched_preds, preds = model.apply(params, latent, None, train=False, rngs = {"dropout": jax.random.PRNGKey(1)}, method= VisionTransformer.apply_decoder)
    loss = model.compute_loss(example_input, patched_preds, mask)
    print(loss, preds.shape, preds.min(), preds.max())
    l1_loss = jnp.abs(example_input - preds).mean()
    print('L1 loss:', l1_loss)
    # save as image
    preds = preds.clip(0, 1)
    img = np.array(preds[0]) * 255.
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save('preds.png')
if __name__ == "__main__":
    main()