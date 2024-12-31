import os
import sys
sys.path.insert(0, './')
from rqvae.models.klvae import Stage1_KLVAE
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData
ckpt_path = '../model_zoo/klvae/mse'
with torch.no_grad():
    model = Stage1_KLVAE(ckpt_path)
    image_path = './test.png'
    image = Image.open(image_path).resize((256, 256)).convert('RGB')
    #repeat 20 times
    #image = image.repeat(1, 1, 1, 1)
    image = ToTensor()(image).unsqueeze(0)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    latent_output = model.encode(data)
    latent = latent_output.zs
    empty_cls_token = torch.ones(latent.shape[0], 1, latent.shape[-1]) * 0.
    #replace the first token with empty token
    #latent_output.zs = torch.cat([empty_cls_token, latent[:, 1:]], dim=1)
    print(latent.shape)
    recon_output = model.decode(latent_output)
    recon = recon_output.xs_recon
    loss = (recon.clamp(0,1) - image.clamp(0,1)).abs().mean()
    #loss = model.compute_loss(recon_output, image)['loss_total']
    print(recon.shape, recon.min(), recon.max(), loss)
    recon_image = ToPILImage()(recon[0].clamp(0., 1.))
    recon_image.save('./visuals/recon_klvae.png')