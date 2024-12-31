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
from torchstat import stat, ModelStat
import sys
import os
from omegaconf import OmegaConf
from check_utils import *
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
im_size = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (256, 256)
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage1_model_wrapper, _  = create_model(config, is_master=False) # does not load ckpt
    stage1_model_wrapper:Stage1ModelWrapper
    print('stage1 model:',stage1_model_wrapper)
    stage1_model = stage1_model_wrapper.stage_1_model
    connector = stage1_model_wrapper.connector
    image = get_default_image(im_size, single_image=True)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    print("=" * 10, 'testing forward', "=" * 10)
    stat(stage1_model, data, model_fn='forward', simple=True)
    print("=" * 10, 'testing encoding', "=" * 10)
    latent_output = stage1_model.encode(data)
    #print(f'encoded zs: shape: {latent_output.zs.shape}, min: {latent_output.zs.min()}, max: {latent_output.zs.max()} mean: {latent_output.zs.mean()} std: #{latent_output.zs.std()}')
    stat(stage1_model, data, model_fn='encode', simple=True)
    print("=" * 10, 'testing connector', "=" * 10)
    forward_output = connector.forward(latent_output)
    stat(connector, latent_output, model_fn='forward', simple=True)
    print("=" * 10, 'testing reverse', "=" * 10)
    reverse_output = connector.reverse(forward_output)
    stat(connector, forward_output, model_fn='reverse', simple=True)
    print("=" * 10, 'testing decoding (on stage1 encoding)', "=" * 10)
    recon_output = stage1_model.decode(reverse_output)
    stat(stage1_model, reverse_output, model_fn='decode', simple=True)
    print("=" * 10, 'testing stage1 loss', "=" * 10)
    try:
        stage1_model.compute_loss_fn = lambda inputs: stage1_model.compute_loss(inputs[0], inputs[1])['loss_total']
        loss = stage1_model.compute_loss(recon_output, data)['loss_total']
        stat(stage1_model, [recon_output, data], model_fn='compute_loss_fn', simple=True)
    except NotImplementedError as e:
        print(f"compute_loss not implemented: {e}")
    print("=" * 10, 'all set!', "=" * 10)