import os
import sys
# add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.models.utils import instantiate_from_config
from omegaconf import OmegaConf
import torch
def instantiate_model(config_path: str):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config)
    return model

def test_forward(xs, config_path):
    model = instantiate_model(config_path)
    model.eval()
    output = model(xs)
    return output

def main():
    xs = torch.randn(20, 3, 256, 256)
    config_path = '/home/bytetriper/VAE-enhanced/configs/imagenet256/linear_probe/klvae_r.yaml'
    output = test_forward(xs, config_path)
    print(output.shape, output.mean(), output.std())

if __name__ == '__main__':
    main()