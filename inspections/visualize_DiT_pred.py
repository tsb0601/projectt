from PIL import Image
import torch
from torch import nn
import os
import sys
import numpy as np
#set working directory to be the parent directory of the current file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.models import create_model
from utils import get_example_data
from rqvae.models.DiT import DiT_Stage2
from rqvae.models.DiT.diffusion import gaussian_diffusion
from rqvae.models.interfaces import *
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
def decode_as_sample(t: torch.Tensor, model: Stage2ModelWrapper) -> torch.Tensor:
    _data = Stage1Encodings(
        zs=t,
        additional_attr={
            
        }
    )
    return model.decode(_data).xs_recon
config_path = sys.argv[1]

ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None

im_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256

assert os.path.isfile(config_path), f'Invalid config path {config_path}'
from omegaconf import OmegaConf
config = OmegaConf.load(config_path)
model, _ = create_model(config.arch, is_master=True)
model: Stage2ModelWrapper
if ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

assert isinstance(model, Stage2ModelWrapper), f'Invalid model type {type(model)}'

stage2_model: DiT_Stage2 = model.stage_2_model

assert isinstance(stage2_model, DiT_Stage2), f'Invalid model type {type(stage2_model)}'

data = get_example_data(im_size)

diffusion = stage2_model.infer_diffusion # use infer diffusion rather than training

assert isinstance(diffusion, gaussian_diffusion.GaussianDiffusion), f'Invalid diffusion type {type(diffusion)}'

# visualize the diffusion process
# we should have a figure of 4 rows,
# 1st row: original image * 10
# 2nd row: noised image * 10
# 3rd row: noise * 10
# 4rd row: predicted noise * 10


originals = []
noised_images = []
noises = []
preds = []
with torch.no_grad():
    latent = model.encode(data)
    # visualize the latent space
    # latent should be shape (B, C, H, W)
    print('latent shape:', latent.zs.shape)
    pixel_latent = model.decode(latent).xs_recon[0]
    timesteps = torch.linspace(0, diffusion.sqrt_alphas_cumprod.shape[0] - 1, 10).long()
    originals.extend([pixel_latent] * 10)
    for i in range(10):
        zs = latent.zs
        t = timesteps[i].repeat(zs.shape[0])
        training_loss = stage2_model.compute_loss(
            latent,
            None,
            data
        )['loss_total']
        print('t:', t, 'zs:', zs.shape, 'timesteps:', timesteps[i])
        noise = torch.randn_like(zs)
        q_sample = diffusion.q_sample(zs, t, noise = noise)
        decoded_noise = decode_as_sample(noise, model)[0] * 0.5 + 0.5 # noise is in [-1, 1]
        decoded_q_sample = decode_as_sample(q_sample, model)[0]
        # visual
        noises.append(decoded_noise.clamp(0, 1))
        noised_images.append(decoded_q_sample.clamp(0, 1))
        labels = data.condition
        if labels is None:
            # we set null labels to num_classes
            labels = torch.tensor(
                [stage2_model.num_classes] * zs.shape[0], device=zs.device
            ).long()
        model_output = stage2_model.model(q_sample, t, labels)
        C = zs.shape[1]
        if model_output.shape[1] > C:
            model_output = model_output[:, :C]
        noise_mse_loss = nn.functional.mse_loss(model_output, noise)
        print('noise_mse_loss:', noise_mse_loss, 'training_loss:', training_loss)
        pred_x0 = diffusion._predict_xstart_from_eps(q_sample, t, model_output)
        decoded_output = decode_as_sample(pred_x0, model)[0]
        preds.append(decoded_output.clamp(0, 1))
# visualize the diffusion process
originals = torch.stack(originals)
noised_images = torch.stack(noised_images)
noises = torch.stack(noises)
preds = torch.stack(preds)
all_images = torch.cat([originals, noised_images, noises, preds], dim=0)
# make grid
grid = make_grid(all_images, nrow=10)
print('grid shape:', grid.shape) 
# save the figure in './visuals/diffusion.png'

im = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
im.save('./visuals/diffusion.png')
