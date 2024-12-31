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
from rqvae.models import create_model
from omegaconf import OmegaConf
from check_utils import *
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
im_size = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (256, 256)
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage1_model_wrapper, _  = create_model(config, is_master=True)
    stage1_model_wrapper:Stage1ModelWrapper
    stage1_model = stage1_model_wrapper.stage_1_model
    print('stage1 model:',stage1_model)
    image = get_default_image(im_size)
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