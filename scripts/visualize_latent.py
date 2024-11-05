import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # add the root directory to the path
from PIL import Image
import torch
from rqvae.models.utils import instantiate_from_config
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets import create_dataset
from rqvae.models.interfaces import *
from rqvae.models import create_model
from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from rqvae.models.connectors import base_connector

from omegaconf import OmegaConf
import numpy as np
import argparse


def get_stage1_model(config:dict)->Stage1ModelWrapper:
    stage1_model_wrapper, _  = create_model(config, is_master=True)
    stage1_model_wrapper:Stage1ModelWrapper
    return stage1_model_wrapper

def visualize_latent(model:Stage1ModelWrapper, latent_path:str, use_connector:bool = False)-> Image.Image:
    latent = np.load(latent_path)['latent'] 
    print('latent shape:', latent.shape)
    encodings = Stage1Encodings(zs=torch.tensor(latent).unsqueeze(0)) # 1, latent_shape
    encodings = model.connector.reverse(encodings) if use_connector else encodings
    recon_output = model.decode(encodings)
    recon = recon_output.xs_recon
    recon, _ = model.get_recon_imgs(recon, recon)
    print('recon shape:', recon.shape)
    # recon should be in [0,1]
    recon = recon.squeeze(0).detach().cpu().permute(1,2,0).numpy()
    recon = (recon * 255).astype(np.uint8)
    recon = Image.fromarray(recon)
    return recon
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize latent representation of the dataset')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--latent_path', type=str, help='Path to the latent representation file')
    parser.add_argument('--use_connector', action='store_true', help='Use connector')
    return parser.parse_args()
def main(args):
    config = OmegaConf.load(args.config_path)
    stage1_model_wrapper = get_stage1_model(config.arch)
    recon = visualize_latent(stage1_model_wrapper, args.latent_path, args.use_connector)
    recon.save('./visuals/latent_visualization.png')
    print('recon saved at ./visuals/latent_visualization.png')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)