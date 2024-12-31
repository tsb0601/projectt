import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import torch
from rqvae.models.utils import instantiate_from_config
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData
from rqvae.models import create_model
from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from rqvae.models.connectors import base_connector
from rqvae.models.interfaces import *
import sys
import os
from omegaconf import OmegaConf
from check_utils import *
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
def print_tree(params, indent=0):
    """Recursively print model parameters as a tree."""
    for key, value in params.items():
        if isinstance(value, dict):  # If nested dictionary, recurse
            print("  " * indent + f"{key}:")
            print_tree(value, indent + 1)
        else:
            # Print leaf nodes with their shapes
            print("  " * indent + f"{key}: shape={value.shape}")
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage1_model_wrapper, _  = create_model(config, is_master=True) # load ckpt if available
    stage1_model_wrapper:Stage1ModelWrapper
    print('stage1 model:',stage1_model_wrapper)
    stage1_model = stage1_model_wrapper.stage_1_model
    connector = stage1_model_wrapper.connector
    print_tree(stage1_model.state_dict())