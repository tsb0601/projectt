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
    #try to visualize the latent
    zs = latent_output.zs
    print(zs.shape)
    zs = zs[0].squeeze()
    # zs: 768, 16, 16
    zs = zs.permute(1, 2, 0)
    zs = zs.mean(dim=-1,keepdim=True)
    # normailze
    zs =( zs - zs.min()) / (zs.max() - zs.min())
    zs = zs.cpu().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(zs)
    plt.savefig('./visuals/latent.png')