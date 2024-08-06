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
    mae = Stage1MAE(ckpt_path,no_cls=True)
    connector = MAE_Diffusion_connector()
    mae.eval()
    processor = ViTImageProcessor.from_pretrained(ckpt_path)
    image_path = '/home/bytetriper/VAE-enhanced/test.png'
    image_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    print(f'Stage1MAE model loaded with mean {image_mean} and std {image_std}, mask_ratio {mae.model.config.mask_ratio}')
    image = Image.open(image_path).resize((mae.model.config.image_size, mae.model.config.image_size)).convert('RGB')
    #repeat 20 times
    #image = image.repeat(1, 1, 1, 1)
    image = ToTensor()(image).unsqueeze(0)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    latent_output = mae.encode(data)
    print(latent_output.zs.std(dim=-1).mean(), latent_output.zs.mean(dim=-1).mean())
    #latent_output = connector.forward(latent_output)
    #reverse_output = connector.reverse(latent_output)
    #reverse_output.zs = reverse_output.zs
    recon_output = mae.decode(latent_output)
    recon = recon_output.xs_recon
    loss = mae.compute_loss(recon_output, data)['loss_total']
    l1_loss = (recon.clamp(0,1) - image.clamp(0,1)).abs().mean()
    print(recon.shape, recon.min(), recon.max(), loss, l1_loss)
    recon_image = ToPILImage()(recon[0].clamp(0., 1.))
    recon_image.save('./visuals/mae_recon.png')