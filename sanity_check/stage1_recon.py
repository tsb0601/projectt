import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import torch
from rqvae.models.utils import instantiate_from_config
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData
from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from rqvae.models.connectors import base_connector
from rqvae.models.interfaces import *
import sys
import os
from omegaconf import OmegaConf
config_path = sys.argv[1]
im_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    if config.get('stage', 1) == 2:
        stage1_model_config = config.stage_1
    else:
        stage1_model_config = config
    stage1_model:Stage1Model = instantiate_from_config(stage1_model_config)
    print('stage1 model:',stage1_model)
    image_path = '/home/bytetriper/VAE-enhanced/test.png'
    image = Image.open(image_path).resize((im_size, im_size)).convert('RGB')
    #repeat 2 times to asssure model works with batch size > 1
    image = ToTensor()(image).unsqueeze(0).repeat(2,1,1,1)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    latent_output = stage1_model.encode(data)
    print('latent std, mean w.r.t last dimension:',latent_output.zs.std(dim=-1).mean(), latent_output.zs.mean(dim=-1).mean())
    print('latent shape:',latent_output.zs.shape)
    #latent_output = connector.forward(latent_output)
    #reverse_output = connector.reverse(latent_output)
    #reverse_output.zs = reverse_output.zs
    recon_output = stage1_model.decode(latent_output)
    recon = recon_output.xs_recon
    
    print('encode-decode image shape, max min:',recon.shape, recon.min(), recon.max())
    print('reconstruction image')
    recon_output = stage1_model(data) 
    loss = stage1_model.compute_loss(recon_output, data)['loss_total']
    l1_loss = (recon.clamp(0,1) - image.clamp(0,1)).abs().mean()
    print('loss & L1 loss:',loss, l1_loss)
    recon_image = ToPILImage()(recon[0].clamp(0., 1.))
    recon_image.save('./visuals/sanity_recon.png')