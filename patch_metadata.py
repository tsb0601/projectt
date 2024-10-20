from functools import partial
from torch_xla._internal import tpu
import os
import pickle

def load_metadata(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def _get_metadata_reload(metadata_name, env_metadata, network_endpoint_metadata):
    if metadata_name == 'tpu-env':
        return env_metadata
    elif metadata_name == 'worker-network-endpoints':
        return network_endpoint_metadata
    else:
        raise ValueError(f"Unknown metadata '{metadata_name}'.")
    
def patch_metadata():
    env_metadata = load_metadata("/home/bytetriper/tpu-env.txt")
    network_endpoint_metadata = load_metadata("/home/bytetriper/worker-network-endpoints.txt")
    print(type(env_metadata))
    print("--------------------")
    print(type(network_endpoint_metadata))  
    tpu._get_metadata = partial(_get_metadata_reload, env_metadata=env_metadata, network_endpoint_metadata=network_endpoint_metadata)
    print("--------------------")
