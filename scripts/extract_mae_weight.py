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
config_path = sys.argv[1]
save_path = sys.argv[2]
print('config_path:',config_path)
print('save_path:',save_path)
os.makedirs(save_path,exist_ok=True)
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage1_model_wrapper, _  = create_model(config, is_master=True) # load ckpt if available
    stage1_model_wrapper:Stage1ModelWrapper
    print('stage1 model:',stage1_model_wrapper)
    stage1_model = stage1_model_wrapper.stage_1_model
    connector = stage1_model_wrapper.connector
    state_dict = stage1_model.state_dict()
    print('state_dict:',state_dict.keys())
    # save the state_dict
    torch.save(state_dict,os.path.join(save_path,'mae.pt'))