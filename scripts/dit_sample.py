from enum import auto
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData
from rqvae.models import create_model
from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from rqvae.models.interfaces import *
import sys
import os
from omegaconf import OmegaConf
from rqvae.models.DiT.models.DiT import DiT
from torch_xla.amp import autocast
import torch_xla.runtime as xr
xr.initialize_cache('/home/bytetriper/.cache/xla_compile/stage2_DiT')
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
save_dir = sys.argv[2] if len(sys.argv) > 2 else './visuals/samples'
os.makedirs(save_dir, exist_ok=True)
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage2_model_wrapper, stage2_model_wrapper_ema  = create_model(config, is_master=True, ema = float(config.get('ema',0.114514))) # load ckpt if available, use ema by default
    if config.get('EMA_PATH', None) is not None:
        ckpt_path = config.EMA_PATH
        state_dict = torch.load(ckpt_path, map_location='cpu')
        stage2_model_wrapper_ema.load_state_dict(state_dict) # load the model (will re-define the model if the ckpt is a nn.Module)
    else:
        raise ValueError('EMA_PATH not found in config')
    if stage2_model_wrapper_ema is not None:
        stage2_model_wrapper = stage2_model_wrapper_ema.module
    stage2_model_wrapper:Stage2ModelWrapper
    stage2_model_wrapper = stage2_model_wrapper.to(xm.xla_device())
    DiT_model = stage2_model_wrapper.stage_2_model.model
    assert isinstance(DiT_model, DiT), 'stage2 model must be DiT, got: ' + str(type(DiT_model))
    n_samples = 10
    #labels = torch.randint(389, 390, (n_samples,)).to(xm.xla_device()).long()
    labels = torch.Tensor([1,3,4,114,512,777,888,11,13,66]).to(xm.xla_device()).long()
    noises = torch.randn(n_samples, 768, 16, 16)
    if os.path.isfile('visuals/samples/noises.pt'):
        print('loading noises...')
        noises = torch.load('visuals/samples/noises.pt')
    else:
        print('saving noises...')
        torch.save(noises, 'visuals/samples/noises.pt')
    noises = noises.to(xm.xla_device())
    data = LabeledImageData(condition=labels, img=noises)
    # do inference!
    with autocast(device = xm.xla_device()):
        generation = stage2_model_wrapper.infer(data)
    xs = generation.xs_recon.cpu().float().clamp(0., 1.)
    print('generated:', xs.shape, xs.min(), xs.max())
    # save the generated image
    for i in range(xs.shape[0]):
        image = xs[i].cpu().clamp(0., 1.)
        image = ToPILImage()(image)
        image.save(os.path.join(save_dir, f'sample_{i}.png'))