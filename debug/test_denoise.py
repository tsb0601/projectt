import os
import sys
# add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn import base
import torch
from rqvae.models.utils import instantiate_from_config
from rqvae.models.interfaces import *
from rqvae.models import create_model
import rqvae.models.DiT as DiT  
from omegaconf import OmegaConf
from rqvae.img_datasets.interfaces import LabeledImageData
from PIL import Image
import numpy as np
import torch_xla.core.xla_model as xm
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid
import torch_xla.runtime as xrt
xrt.initialize_cache('/home/bytetriper/.cacahe/xla_compile/tmp', readonly=False)
torch.manual_seed(11451)
config_root = '/home/bytetriper/VAE-enhanced/ckpt_gcs/aW_DiT_B_1_b256_lr1e-4_ep120/DiTwmae/23072024_084126/'
#config_root = '/home/bytetriper/VAE-enhanced/ckpt_gcs/aW_DiT_B_2_b4096_blr1e-4_ep120/DiTwklvae/21072024_214020/'
config_path = os.path.join(config_root, 'config.yaml')
@torch.no_grad()
def main():
    config = OmegaConf.load(config_path)
    model, _  = create_model(config.arch,stage=2)
    model: Stage2ModelWrapper
    ckpt_path = 'ep_81-checkpoint/0-model.pt'
    ckpt_path = os.path.join(config_root, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    device = xm.xla_device()
    model.to(device)
    model.eval()
    #print(model)
    img_path = './test.png'
    pil_image = Image.open(img_path).resize((256,256))
    bsz = 20
    x = ToTensor()(pil_image).unsqueeze(0).repeat(bsz,1,1,1)
    #x = torch.randn(4,3,256,256)
    #y = torch.randint(0,1000,(4,))
    y = torch.ones(bsz) * 1000 # unconditional
    y = y.long().to(device)
    data = LabeledImageData(
        img=x,
        condition=y,
        additional_attr= {}
    )
    data._to(device)
    encodings = model.stage_1_model.encode(data)
    encodings = model.connector.forward(encodings)
    zs = encodings.zs
    print(zs.shape)
    latent_dim = zs.shape[1]
    stage1_model = model.stage_1_model
    dit: DiT.DiT = model.stage_2_model.model
    diffusion: DiT.diffusion.SpacedDiffusion = model.stage_2_model.diffusion
    connector: base_connector = model.connector
    # set t to unifromly in [0,1000]
    t = torch.linspace(0, diffusion.num_timesteps - 1, bsz , device=zs.device)
    print('num_timesteps:', diffusion.num_timesteps)
    #t = 700 * torch.ones(bsz, device=zs.device)
    t = t.long()
    print(t.shape, t) 
    
    # zs: bsz, latent_dim, patch_num, patch_num
    noise = torch.randn_like(zs[0]).repeat(bsz, 1, 1, 1)
    z_t = diffusion.q_sample(zs, t, noise=noise)
    model_output = dit.forward(z_t, t, y)
    print(model_output.shape)
    if model_output.shape[1] == 2 * latent_dim:
        model_output, model_var_values = torch.split(model_output, latent_dim, dim=1)
    mse_loss = ((model_output - noise)**2).mean()
    print(mse_loss)
    zs_hat = diffusion._predict_xstart_from_eps(z_t, t, model_output)
    #zs_hat = diffusion.sdedit(
    #    dit.forward, 
    #    z_t.shape,
    #    zs,
    #    t[0],
    #    clip_denoised=False,
    #    model_kwargs=dict(y=y),
    #    device = device,
    #    progress=True
    #)
    stage2_model_output = Stage2ModelOutput(
        zs_pred = zs_hat,
        zs_degraded = zs,
        additional_attr = {}
    )
    stage1_decode_input = connector.reverse(stage2_model_output)
    stage1_output = stage1_model.decode(stage1_decode_input)
    x_hat = stage1_output.xs_recon
    print(x_hat.shape)
    x_hat = x_hat.clamp(0,1)
    x_hat = make_grid(x_hat, nrow=min(4,bsz))
    #x_hat = ToPILImage()(x_hat)
    #x_hat.save('./visuals/x_hat.png')
    x_t_decode_output = Stage2ModelOutput(
        zs_pred = z_t,
        zs_degraded = zs,
        additional_attr = {}
    )
    x_t_decode_input = connector.reverse(x_t_decode_output)
    x_t_output = stage1_model.decode(x_t_decode_input)
    x_t = x_t_output.xs_recon
    x_t = x_t.clamp(0,1)
    x_t = make_grid(x_t, nrow=min(4,bsz)) # shape: 3, 256, 256 * bsz
    # concat x_t and x_hat
    x_t = torch.cat([x_t, x_hat], dim=1) # shape: 3, 256 * 2, 256 * bsz
    x_t = ToPILImage()(x_t)
    x_t.save('./visuals/x_contrast.png')
if __name__ == '__main__':
    main()