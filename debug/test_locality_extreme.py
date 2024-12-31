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
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=torch.cat([image, second_image], dim=0), condition=torch.tensor([0, 1]))
    print(data.img.shape, data.condition)
    latent_output = mae.encode(data)
    latent_output = connector.forward(latent_output)
    reverse_output = connector.reverse(latent_output)
    zs = reverse_output.zs
    scale = (.0, .25, .5, .75, 1.)
    zs_origin = zs[0]
    zs_add = zs[1]
    interpolated = []
    rand_token_idx = torch.randint(0, zs_origin.shape[0] - 1, (len(scale),))
    print(rand_token_idx, zs_origin.shape)
    for s in range(len(scale)):
        # randomly select only ONE token to be zs_add, the rest zs_origin
        zs_inter = zs_origin.clone()
        zs_inter[rand_token_idx[s] + 1] = zs_add[rand_token_idx[s] + 1] # take cls token into account
        interpolated.append(zs_inter)
    interpolated = torch.stack(interpolated)
    reverse_output.zs = interpolated
    recon_output = mae.decode(reverse_output)
    recon = recon_output.xs_recon
    image_interpolated = []
    for s in range(len(scale)):
        image_inter = image.clone().squeeze(0) # 3, 256, 256
        patch_size = image.shape[2] // int(zs_origin.shape[0] ** 0.5) # 16
        patch_num_per_ax = int(zs_origin.shape[0] ** 0.5) # 16
        token_idx = int(rand_token_idx[s])
        x_idx = token_idx % patch_num_per_ax
        y_idx = token_idx // patch_num_per_ax
        x_st, x_ed = x_idx * patch_size, (x_idx + 1) * patch_size
        y_st, y_ed = y_idx * patch_size, (y_idx + 1) * patch_size
        print(x_st, x_ed, y_st, y_ed)
        image_inter[:, y_st:y_ed, x_st:x_ed] = second_image.squeeze(0)[:, y_st:y_ed, x_st:x_ed]
        image_interpolated.append(image_inter)
    image_interpolated = torch.stack(image_interpolated)
    interpolated_data = LabeledImageData(img=image_interpolated)
    interpolated_latent_output = mae.encode(interpolated_data)
    interpolated_latent_output = connector.forward(interpolated_latent_output)
    interpolated_reverse_output = connector.reverse(interpolated_latent_output)
    interpolated_recon_output = mae.decode(interpolated_reverse_output)
    interpolated_recon = interpolated_recon_output.xs_recon
    # recon : len(scale), 3, 256, 256
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    recon = make_grid(torch.cat([image_interpolated, recon, interpolated_recon], dim=0), nrow=len(scale))
    recon = ToPILImage()(recon)
    recon.save('./visuals/interpolant_extreme.png')