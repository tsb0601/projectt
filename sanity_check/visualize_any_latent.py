"""
this files takes in a stage1/2 model, and an image
if the model is a stage1 model, it takes a image and returns the latent visualization
if the model is a stage2 model, it generates a latent and returns the visualization
visualization is done by PCA
"""
import sys
import os
from typing import Any, Union
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # add the root directory to the path
from PIL import Image
import torch
from rqvae.models.ema import ExponentialMovingAverage
from rqvae.models.interfaces import Stage1ModelWrapper, Stage2ModelWrapper
from rqvae.models import _create_according_to_config, create_model
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
import torch_xla.core.xla_model as xm
IM_SIZE = 256
class latent_visualizer():
    def __init__(self):
        pass
    def normalize(self, latent: torch.Tensor):
        """
        normalize latent to 0-1
        """
        #std_value = (latent - latent.mean()) / latent.std()
        #normalized_value = 1 / (1 + torch.exp(-std_value))
        #return normalized_value
        B, C, H, W = latent.shape
        per_instance_min = latent.flatten(1).min(1).values
        per_instance_max = latent.flatten(1).max(1).values
        per_instance_min = per_instance_min.view(B, 1, 1, 1)
        per_instance_max = per_instance_max.view(B, 1, 1, 1)
        print('per_instance_min shape:', per_instance_min.shape, 'per_instance_max shape:', per_instance_max.shape, 'latent shape:', latent.shape)
        return (latent - per_instance_min) / (per_instance_max - per_instance_min)
    def __call__(self, latent:torch.Tensor,remove_potential_cls: bool = True) -> torch.Tensor:
        """
        latent: B, L, D or B, C, H, W
        """
        pca = PCA(n_components=3)
        assert len(latent.shape) == 3 or len(latent.shape) == 4, 'latent shape not supported'
        if len(latent.shape) == 4:
            latent = latent.permute(0, 2, 3, 1).contiguous().reshape(latent.shape[0], -1, latent.shape[1])
        B, L, D = latent.shape
        if remove_potential_cls and latent.shape[1] % 2 == 1:
            latent = latent[:, 1:]
        latent = latent.reshape(-1, latent.shape[-1]) # B*L, D
        latent = latent.cpu().numpy()
        pca.fit(latent)
        latent = pca.transform(latent)
        latent = torch.tensor(latent) # B*L, 3
        print('pca transformed latent shape:', latent.shape)
        # reshape back
        latent = latent.reshape(B, L, 3)
        # reshape to square
        H = W = int(L ** 0.5)
        assert H * W == latent.shape[1], 'latent shape not square'
        latent = latent.reshape(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
        latent = torch.nn.functional.interpolate(latent, size=IM_SIZE, mode='nearest')
        latent = self.normalize(latent)
        return latent

def prepare_latent(model:Union[Stage1ModelWrapper, Stage2ModelWrapper, ExponentialMovingAverage], data: LabeledImageData) -> torch.Tensor:
    if isinstance(model, ExponentialMovingAverage):
        model = model.module # get the underlying model
        print('model is EMA, underlying model:', model)
    if isinstance(model, Stage1ModelWrapper):
        zs = model.encode(data)
        latent = zs.zs
        print('model is stage1, zs shape:', zs.zs.shape)
    elif isinstance(model, Stage2ModelWrapper):
        zs = model.encode(data)
        infer_zs = model.infer(data)
        latent = infer_zs.xs_recon
        print('model is stage2, infer zs shape:', infer_zs.xs_recon.shape)
    return latent

def load_model(config_path: str) -> Union[Stage1ModelWrapper, Stage2ModelWrapper, ExponentialMovingAverage]:
    config = OmegaConf.load(config_path)
    model, model_ema = create_model(config.arch, config.arch.ema)
    return model if model_ema is None else model_ema # if ema is not None, return ema

def main(config_path: str, image_path: str, save_path: str, condition: Any = None, use_tpu: bool = False):
    device = 'cpu' if not use_tpu else xm.xla_device()
    print('using device:', device)
    model = load_model(config_path)
    img = Image.open(image_path).resize((IM_SIZE,IM_SIZE), Image.Resampling.BICUBIC) # load image
    img = ToTensor()(img).unsqueeze(0) # to tensor
    print('image shape:', img.shape)
    data = LabeledImageData(img, condition, image_path) # create data
    model.eval()
    model.to(device)
    data._to(device)
    with torch.no_grad():
        latent = prepare_latent(model, data) # get latent
    print('latent shape:', latent.shape)
    latent = latent_visualizer()(latent) # visualize latent
    latent = ToPILImage()(latent[0]) # to image
    latent.save(save_path) # save image
    print('latent visualization saved at:', save_path)

if __name__ == '__main__':
    config_path, image_path, save_path = sys.argv[1], sys.argv[2], sys.argv[3]
    condition = None if len(sys.argv) < 5 else sys.argv[4]
    use_tpu = True if len(sys.argv) < 6 else sys.argv[5]
    main(config_path, image_path, save_path, condition, use_tpu)