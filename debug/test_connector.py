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
config_path = '/home/bytetriper/VAE-enhanced/configs/imagenet256/stage2/DiTwklvae.yaml'
def main():
    config = OmegaConf.load(config_path)
    model, _  = create_model(config.arch,stage=2)
    model: Stage2ModelWrapper
    model.to('cpu')
    print(model)
    x = torch.randn(4,3,256,256)
    y = torch.randint(0,1000,(4,))
    data = LabeledImageData(
        img=x,
        condition=y,
        additional_attr= {}
    )
    encodings, outputs = model.forward(data)
    zs = encodings.zs
    print(zs.shape)
    xs_recon = outputs.zs_pred
    print(xs_recon.shape)
    outputs = model.infer(data)
    print(outputs.xs_recon.shape)
if __name__ == '__main__':
    main()