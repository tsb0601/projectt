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
from rqvae.models.ema import ExponentialMovingAverage
import sys
import os
from omegaconf import OmegaConf
config_path = sys.argv[1]
im_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
config = OmegaConf.load(config_path).arch
stage2_model_wrapper, stage2_model_ema  = create_model(config, config.ema)
stage2_model_wrapper:Stage2ModelWrapper
stage2_model_ema: Optional[ExponentialMovingAverage]
require_grad_params = [param for name, param in stage2_model_wrapper.named_parameters() if param.requires_grad]
require_grad_params_names = [name for name, param in stage2_model_wrapper.named_parameters() if param.requires_grad]
opt = torch.optim.AdamW(require_grad_params, lr=1e-4)
print('stage2_model_wrapper:', stage2_model_wrapper)
print('trainable params:', require_grad_params_names)
print('opt:', opt)
stage1_model = stage2_model_wrapper.stage_1_model
connector = stage2_model_wrapper.connector
stage2_model = stage2_model_wrapper.stage_2_model
image_path = '/home/bytetriper/VAE-enhanced/test.png'
image = Image.open(image_path).resize((im_size, im_size)).convert('RGB')
#repeat 2 times to asssure model works with batch size > 1
image = ToTensor()(image).unsqueeze(0).repeat(2,1,1,1)
print(image.shape, image.min(), image.max())
#image = (image * 2) - 1.
#noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
data = LabeledImageData(img=image)
print("=" * 10, 'testing encoding', "=" * 10)
latent_output = stage1_model.encode(data)
print('encoded zs:', latent_output.zs.shape, latent_output.zs.min(), latent_output.zs.max())
print("=" * 10, 'testing connector', "=" * 10)
latent_output = connector.forward(latent_output)
print('connected zs:', latent_output.zs.shape, latent_output.zs.min(), latent_output.zs.max())
print("=" * 10, 'testing forward', "=" * 10)
forward_output = stage2_model.forward(latent_output, data)
print('forward zs_pred:', forward_output.zs_pred.shape, forward_output.zs_pred.min(), forward_output.zs_pred.max())
print("=" * 10, 'testing reverse', "=" * 10)
reverse_output = connector.reverse(forward_output)
print('reverse zs:', reverse_output.zs.shape, reverse_output.zs.min(), reverse_output.zs.max())
print("=" * 10, 'testing decoding (on stage2 output)', "=" * 10)
recon_output = stage1_model.decode(reverse_output)
recon = recon_output.xs_recon
print(recon.shape, recon.min(), recon.max())
print("=" * 10, 'testing decoding (on stage1 encoding)', "=" * 10)
reverse_output = connector.reverse(latent_output)
recon_output = stage1_model.decode(reverse_output)
recon = recon_output.xs_recon
print(recon.shape, recon.min(), recon.max())
print("=" * 10, 'testing stage1 loss', "=" * 10)
try:
    loss = stage1_model.compute_loss(recon_output, data)['loss_total']
    print(loss)
except NotImplementedError:
    print('loss not implemented')
print("=" * 10, 'testing stage2 loss', "=" * 10)
loss = stage2_model_wrapper.compute_loss(latent_output ,forward_output, data)['loss_total']
print(loss)
print("=" * 10, 'testing stage2 infer (skipped)', "=" * 10)
#with autocast(device=xm.xla_device()):
#    generated_output = stage2_model.infer(data)
#print(generated_output.zs_pred.shape, generated_output.zs_pred.min(), generated_output.zs_pred.max())
print("=" * 10, 'testing stage2 backward', "=" * 10)
loss.backward()
opt.step()
print("=" * 10, 'testing EMA', "=" * 10)
if stage2_model_ema:
    stage2_model_ema.update(stage2_model_wrapper, step=None)
else:
    print('EMA not available')
print("=" * 10, 'all set!', "=" * 10)