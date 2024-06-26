from transformers import ViTMAEForPreTraining, ViTImageProcessor
import sys
import os
import torch_xla.core.xla_model as xm
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.models.MAE import Stage1MAE
from PIL import Image
from torchvision.transforms import ToTensor
ckpt_path = sys.argv[1]
assert os.path.exists(ckpt_path)
mae = Stage1MAE(ckpt_path=ckpt_path).to(xm.xla_device())
load_path = '/home/bytetriper/VAE-enhanced/ckpt/MAE_256_ft/MAE/26062024_073033/epoch6_model.pt'
ckpt = torch.load(load_path, map_location='cpu')['state_dict']
#mae.load_state_dict(ckpt, strict=True)
img = './test.png'
img = Image.open(img).convert('RGB').resize((256, 256))
img = ToTensor()(img).unsqueeze(0).to(xm.xla_device())
print(img.shape,img.min(),img.max())
mae.eval()
print(mae.model.config.mask_ratio)
with torch.no_grad():
    #latent = mae.encode(img)[0]
    #recon_img = mae.decode(latent)[0]
    recon_img = mae(img)[0]
    print(recon_img.shape, recon_img.min(), recon_img.max())
    recon_img = recon_img.squeeze(0).clamp(0, 1).cpu().numpy()
recon_img = (recon_img * 255).astype('uint8').transpose(1, 2, 0)
recon_img = Image.fromarray(recon_img)
recon_img.save('./recon.png')