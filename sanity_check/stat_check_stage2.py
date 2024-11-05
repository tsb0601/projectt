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
def count_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
im_size = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (256, 256)
detailed_stat = len(sys.argv) > 3 and sys.argv[3] == 'detailed'
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage2_model_wrapper, _  = create_model(config, is_master=False) # does not load ckpt
    stage2_model_wrapper:Stage2ModelWrapper
    print('stage2_model_wrapper:', stage2_model_wrapper) if detailed_stat else None
    param, trainable_param = count_params(stage2_model_wrapper)
    print(f"Total params: {param/1e6:.2f}M, Trainable params: {trainable_param/1e6:.2f}M") if detailed_stat else None
    stage1_model = stage2_model_wrapper.stage_1_model
    connector = stage2_model_wrapper.connector
    stage2_model = stage2_model_wrapper.stage_2_model
    image = get_default_image(im_size)[0:1, ...] # B \equiv 1
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    #print("=" * 10, 'testing stage1 forward', "=" * 10)
    #stat(stage1_model, data, model_fn='forward', simple=True)
    #print("=" * 10, 'testing stage1 encoding', "=" * 10)
    latent_output = stage1_model.encode(data)
    #stat(stage1_model, data, model_fn='encode', simple=True)
    print("=" * 10, 'testing connector', "=" * 10) 
    forward_output = connector.forward(latent_output)
    stat(connector, latent_output, model_fn='forward', simple=True)
    print("=" * 10, 'testing reverse', "=" * 10)
    reverse_output = connector.reverse(forward_output)
    stat(connector, forward_output, model_fn='reverse', simple=True)
    print("=" * 10, 'testing stage1 decoding (on stage1 encoding)', "=" * 10)
    recon_output = stage1_model.decode(reverse_output)
    stat(stage1_model, reverse_output, model_fn='decode', simple=True)
    print("=" * 10, 'testing stage1 loss', "=" * 10)
    stage1_model.compute_loss_fn = lambda inputs: stage1_model.compute_loss(inputs[0], inputs[1])['loss_total']
    try:
        loss = stage1_model.compute_loss(recon_output, data)['loss_total'] # some models may not support stage1 loss
        stat(stage1_model, [recon_output, data], model_fn='compute_loss_fn', simple=True)
    except Exception as e:
        print(e)
    print("=" * 10, 'testing stage2 forward', "=" * 10)
    stage2_model.forward_fn = lambda inputs: stage2_model.forward(inputs[0], inputs[1])
    stat(stage2_model, [latent_output, data], model_fn='forward_fn', simple=True)
    print("=" * 10, 'testing stage2 loss', "=" * 10)
    stage2_model.compute_loss_fn = lambda inputs: stage2_model.compute_loss(inputs[0], inputs[1], inputs[2])['loss_total']
    loss = stage2_model.compute_loss(forward_output, reverse_output, data)['loss_total']
    stat(stage2_model, [forward_output, reverse_output, data], model_fn='compute_loss_fn', simple=True)
    print("=" * 10, 'all set!', "=" * 10)