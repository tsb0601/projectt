import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.models.MAE import Stage1MAE,ViTImageProcessor
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData
from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from rqvae.models.connectors import MAE_Diffusion_connector
ckpt_path = './ckpt_gcs/model_zoo/mae_base_256_ft_r'
with torch.no_grad():
    mae = Stage1MAE(ckpt_path)
    connector = MAE_Diffusion_connector()
    mae.eval()
    processor = ViTImageProcessor.from_pretrained(ckpt_path)
    image_path = '/home/bytetriper/VAE-enhanced/test.png'
    second_im_path = '/home/bytetriper/VAE-enhanced/visuals/test_imagenet_orig.png'
    image_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    print(f'Stage1MAE model loaded with mean {image_mean} and std {image_std}, mask_ratio {mae.model.config.mask_ratio}')
    image = Image.open(image_path).resize((mae.model.config.image_size, mae.model.config.image_size)).convert('RGB')
    second_image = Image.open(second_im_path).resize((mae.model.config.image_size, mae.model.config.image_size)).convert('RGB')
    #repeat 20 times
    #image = image.repeat(1, 1, 1, 1)
    image = ToTensor()(image).unsqueeze(0)
    second_image = ToTensor()(second_image).unsqueeze(0)
    # set second image to random noise
    second_image = torch.rand_like(second_image)
    # normalize and clip to [0, 1]
    second_image = (second_image + 1) / 2
    second_image = torch.clamp(second_image, 0, 1)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=torch.cat([image, second_image], dim=0), condition=torch.tensor([0, 1]))
    print(data.img.shape, data.condition)
    latent_output = mae.encode(data)
    latent_output = connector.forward(latent_output)
    zs = latent_output.zs
    scale = (.0, .1, .2, .3, .4, .5, .6, .7, .8, .9,.92,.94,.96,.98,.99,.992,.994,.996,.998, 1.)
    zs_origin = zs[0]
    zs_add = zs[1]
    interpolated = []
    random_noise = torch.rand_like(zs_origin)
    from math import sqrt
    for s in scale:
        zs_inter = sqrt(s) * zs_origin + sqrt(1 - s) * random_noise
        interpolated.append(zs_inter)
    interpolated = torch.stack(interpolated)
    latent_output.zs = interpolated
    reverse_output = connector.reverse(latent_output)
    recon_output = mae.decode(reverse_output)
    recon = recon_output.xs_recon
    # recon : len(scale), 3, 256, 256
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    recon = make_grid(torch.cat([recon], dim=0), nrow=len(scale)//2)
    recon = ToPILImage()(recon)
    recon.save('./visuals/interpolant_latent_noise.png')