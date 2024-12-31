import os
import sys
# add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from rqvae.models.utils import instantiate_from_config
from rqvae.models.interfaces import *
from rqvae.models import create_model
from omegaconf import OmegaConf
from rqvae.img_datasets.interfaces import LabeledImageData
from PIL import Image
import numpy as np
import torch_xla.core.xla_model as xm
from torchvision.transforms import ToTensor, ToPILImage
config_path = '/home/bytetriper/VAE-enhanced/configs/imagenet256/stage2/DiTwmae.yaml'
def main():
    config = OmegaConf.load(config_path)
    model, _  = create_model(config.arch,stage=2)
    model: Stage2ModelWrapper
    ckpt_path = '/home/bytetriper/VAE-enhanced/ckpt_gcs/aW_DiT_S_2_b4096_blr1e-4_ep120/DiTwmae/22072024_074430/ep_last-checkpoint/0-model.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    device = xm.xla_device()
    model.to(device)
    model.eval()
    #print(model)
    img_path = './test.png'
    pil_image = Image.open(img_path).resize((256,256))
    x = ToTensor()(pil_image).unsqueeze(0).repeat(4,1,1,1)
    #x = torch.randn(4,3,256,256)
    #y = torch.randint(0,1000,(4,))
    y = torch.Tensor([1000,1000,1000,1000]).long().to(device)
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
    #zs = zs.permute(0,2,3,1).reshape(4, -1 ,zs.shape[1])
    #print(zs.shape)
    t = torch.randint(0, model.stage_2_model.diffusion.num_timesteps, (4,), device=device)
    random_noise = torch.randn_like(zs) 
    print(t)
    z_t = model.stage_2_model.diffusion.q_sample(zs, t, noise=random_noise)
    eps_pred = model.stage_2_model.model(z_t, t, y=y)
    #print(eps_pred.shape, eps_pred.mean(),eps_pred.std(),eps_pred.dtype)
    mean_pred = eps_pred[:,:768,...]
    var_pred = eps_pred[:,768:,...]
    print(mean_pred.shape, mean_pred.mean(),mean_pred.std(),mean_pred.dtype)
    print(var_pred.shape, var_pred.mean(),var_pred.std(),var_pred.dtype)
    print(random_noise.shape, random_noise.mean(),random_noise.std(),random_noise.dtype)
    mse_loss = ((eps_pred[:,:768,...] - random_noise)**2).mean()
    print(mse_loss)
    #for i in range(zs.shape[1]):
    #    print('i:',zs[0,i,].norm(p=2))
    #print('zs:',zs[0].norm(p=2, dim=-1).mean())
    #outputs = model.infer(data)
    #print(outputs.xs_recon.shape)
    #gen_image = ToPILImage()(outputs.xs_recon[0].clamp(0,1))
    #gen_image.save('./visuals/dit_gen.png')
if __name__ == '__main__':
    main()