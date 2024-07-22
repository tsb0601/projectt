import os
import sys
# add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from rqvae.models.utils import instantiate_from_config
from rqvae.models.interfaces import *
from rqvae.models import create_model
from omegaconf import OmegaConf
from rqvae.img_datasets.interfaces import LabeledImageData
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
config_path = '/home/bytetriper/VAE-enhanced/configs/imagenet256/stage2/DiTwklvae.yaml'
def main():
    config = OmegaConf.load(config_path)
    model, _  = create_model(config.arch,stage=2)
    model: Stage2ModelWrapper
    model.to('cpu')
    print(model)
    img_path = './test.png'
    pil_image = Image.open(img_path).resize((256,256))
    x = ToTensor()(pil_image).unsqueeze(0).repeat(4,1,1,1)
    #x = torch.randn(4,3,256,256)
    y = torch.randint(0,1000,(4,))
    data = LabeledImageData(
        img=x,
        condition=y,
        additional_attr= {}
    )
    encodings = model.stage_1_model.encode(data)
    zs = encodings.zs
    print(zs.shape)
    zs = zs.permute(0,2,3,1).reshape(4, -1 , 4)
    print(zs.shape)
    for i in range(zs.shape[1]):
        print('i:',zs[0,i,].norm(p=2))
    print('zs:',zs[0].norm(p=2))
    #outputs = model.infer(data)
    #print(outputs.xs_recon.shape)
if __name__ == '__main__':
    main()