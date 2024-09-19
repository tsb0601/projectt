from operator import is_
import os
import sys
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
config_path = sys.argv[1]
im_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage1_model_wrapper, _  = create_model(config, is_master=True)
    stage1_model_wrapper:Stage1ModelWrapper
    print('stage1 model:',stage1_model_wrapper)
    stage1_model = stage1_model_wrapper.stage_1_model
    connector = stage1_model_wrapper.connector
    image_path = '/home/bytetriper/VAE-enhanced/test.png'
    image = Image.open(image_path).resize((im_size, im_size)).convert('RGB')
    #repeat 2 times to asssure model works with batch size > 1
    image = ToTensor()(image).unsqueeze(0).repeat(2,1,1,1)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    print("=" * 10, 'testing wrapper forward', "=" * 10)
    forward_output = stage1_model_wrapper.forward(data)
    print(forward_output.xs_recon.shape, forward_output.xs_recon.min(), forward_output.xs_recon.max())
    print("=" * 10, 'testing forward', "=" * 10)
    output = stage1_model.forward(data)
    print(f'forward xs recon: shape: {output.xs_recon.shape}, min: {output.xs_recon.min()}, max: {output.xs_recon.max()} mean: {output.xs_recon.mean()} std: {output.xs_recon.std()}')
    print("=" * 10, 'testing encoding', "=" * 10)
    latent_output = stage1_model.encode(data)
    print(f'encoded zs: shape: {latent_output.zs.shape}, min: {latent_output.zs.min()}, max: {latent_output.zs.max()} mean: {latent_output.zs.mean()} std: {latent_output.zs.std()}')
    print("=" * 10, 'testing connector', "=" * 10)
    forward_output = connector.forward(latent_output)
    print(f'connected zs: shape: {forward_output.zs.shape}, min: {forward_output.zs.min()}, max: {forward_output.zs.max()} mean: {forward_output.zs.mean()} std: {forward_output.zs.std()}')
    print("=" * 10, 'testing reverse', "=" * 10)
    reverse_output = connector.reverse(forward_output)
    print(f'reverse zs: shape: {reverse_output.zs.shape}, min: {reverse_output.zs.min()}, max: {reverse_output.zs.max()} mean: {reverse_output.zs.mean()} std: {reverse_output.zs.std()}')
    print("|reverse zs - zs|:", torch.abs(reverse_output.zs - latent_output.zs).mean())
    print("=" * 10, 'testing decoding (on stage1 encoding)', "=" * 10)
    recon_output = stage1_model.decode(reverse_output)
    recon = recon_output.xs_recon
    print(recon.shape, recon.min(), recon.max())
    recon_image = ToPILImage()(recon[0].clamp(0., 1.))
    recon_image.save('./visuals/sanity_recon.png')
    print("=" * 10, 'testing stage1 loss', "=" * 10)
    loss = stage1_model.compute_loss(recon_output, data)['loss_total']
    print(loss)
    print("=" * 10, 'all set!', "=" * 10)