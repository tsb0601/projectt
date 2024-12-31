import os
import sys
from tkinter import NO
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
from rqvae.models.DiT.models.DiT import DiT
from rqvae.models.DiT import DiT_Stage2
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
class Id(nn.Module):
    """
    hack to replace DiT
    """
    def __init__(self):
        super().__init__()
    def forward(self, x, t, y):
        return torch.zeros_like(x)
def single_round_noise_sampling(x: torch.Tensor, diffusion, model, t, noise = None):
    """
    sample noise from the diffusion model
    """
    # sample noise
    if noise is None:
        noise = torch.randn_like(x)
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs = {'y': None}, noise= noise)
    mse, weighted_mse = loss_dict['mse'], loss_dict['loss']
    return mse, weighted_mse, noise
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
im_size = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (256, 256)
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage2_model_wrapper, _  = create_model(config, is_master=True) # load ckpt if available
    stage2_model_wrapper:Stage2ModelWrapper
    dit = stage2_model_wrapper.stage_2_model
    dit: DiT_Stage2
    diffusion = dit.diffusion
    # calculate loss
    model = Id()
    sample_rounds = 10000
    triples = []
    x = torch.zeros(1, 3, *im_size) - 1 # dummy input
    for _ in tqdm(range(sample_rounds), desc='sampling noise'):
        t = torch.rand(x.size(0), device=x.device) # sample t from uniform distribution
        mse, weighted_mse, noise = single_round_noise_sampling(x, diffusion, model, t)
        triples.append((mse, weighted_mse, noise, t))
    # save the triples as a npz
    triples = torch.stack([torch.stack([mse, weighted_mse, t]) for mse, weighted_mse, _, t in triples])
    triples = triples.cpu().numpy()
    np.savez('./ckpt/diffusion_loss.npz', triples=triples)
    print('Saved diffusion loss to ./ckpt/diffusion_loss.npz')
    # draw the plot
    t = triples[:, 2]
    mse = triples[:, 0]
    weighted_mse = triples[:, 1]
    weight = weighted_mse / mse
    avg_weight = weight.mean()
    avg_loss = mse.mean()
    avg_weighted_loss = weighted_mse.mean()
    print(f'Average mse: {avg_loss}, average weighted mse: {avg_weighted_loss}, average weight: {avg_weight}')
    weight = weight / weight.max() # normalize weight
    # make dot smallest possible as their are many dots
    plt.scatter(t, mse, label='mse', s=0.1)
    plt.scatter(t, weighted_mse, label='weighted_mse', s=0.1)
    plt.scatter(t, weight, label='weight (normalized)', s=0.1)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('loss')
    plt.savefig('./visuals/diffusion_loss.png')
    print('Saved plot to ./visuals/diffusion_loss.png')