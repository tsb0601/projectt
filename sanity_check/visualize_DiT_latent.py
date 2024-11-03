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
import sys
import os
from omegaconf import OmegaConf
from check_utils import *
from rqvae.models.DiT.models import DiT
import torch.nn.functional as F
from sklearn.decomposition import PCA
from timm.models.vision_transformer import  Attention
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
im_size = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (256, 256)
class AttentionAggregator():
    def __init__(self):
        self.attns = []
    def __call__(self, attn):
        #print('calling AttentionAggregator, attn shape:', attn.shape)
        attn: torch.Tensor # B, H, N , N
        attn = attn.squeeze().permute(1, 0, 2).reshape(attn.shape[-2], -1) # H, B, N, N -> N, H*N
        #print('reshaped attn shape:', attn.shape)
        self.attns.append(attn) # B should always be 1
        return attn
    def normalize(self, latent: torch.Tensor):
        """
        normalize latent to 0-1
        """
        #std_value = (latent - latent.mean()) / latent.std()
        #normalized_value = 1 / (1 + torch.exp(-std_value))
        #return normalized_value
        return (latent - latent.min()) / (latent.max() - latent.min())
    def visualize(self):
        """
        called after all attentions are collected
        """
        pca = PCA(n_components=3) # r, g, b
        attns = torch.stack(self.attns, dim=0) # T, N, H*N
        print('stacked attns shape:', attns.shape)
        T, N, _ = attns.shape
        attn_for_learning = attns
        attn_for_learning = attn_for_learning.reshape(-1, attns.shape[-1]) # T*N, N*H
        print('reshaped attns shape:', attn_for_learning.shape)
        pca.fit(attn_for_learning.cpu().numpy())
        attns = attns.reshape(-1, attns.shape[-1]) # T*N, N*H
        attns = pca.fit_transform(attns.cpu().numpy())
        print('pca transformed attns shape:', attns.shape)
        # reshape back
        attns = attns.reshape(T, N, 3) # T, N, 3
        attns = torch.tensor(attns)
        for t in range(T):
            attn = attns[t]
            #print('t attn shape:', attn.shape)
            h = w = int(attn.shape[0] ** 0.5)
            attn = attn.reshape(h, w, 3).permute(2, 0, 1) # 3, H, W
            #print('reshaped attn shape:', attn.shape)
            attn = F.interpolate(attn.unsqueeze(0), size=im_size, mode='nearest').squeeze()
            #print('interpolated attn shape:', attn.shape)
            # rescale to 0-255
            attn = self.normalize(attn) * 255
            attn = attn.to(torch.uint8)
            img = ToPILImage()(attn)
            img.save(f'./visuals/attns/attn_{t}.png')
    def visualize_per_t(self):
        """
        called after all attentions are collected
        """
        pca = PCA(n_components=3) # r, g, b
        for t in range(len(self.attns)):
            attn = self.attns[t]
            #print('t attn shape:', attn.shape)
            pca = PCA(n_components=3) # r, g, b
            pca.fit(attn.cpu().numpy())
            attn = pca.transform(attn.cpu().numpy())
            attn = torch.tensor(attn)
            #print('pca transformed attn shape:', attn.shape)
            #attn = attns[t]
            #print('t attn shape:', attn.shape)
            h = w = int(attn.shape[0] ** 0.5)
            attn = attn.reshape(h, w, 3).permute(2, 0, 1) # 3, H, W
            #print('reshaped attn shape:', attn.shape)
            attn = F.interpolate(attn.unsqueeze(0), size=im_size, mode='nearest').squeeze()
            #print('interpolated attn shape:', attn.shape)
            # rescale to 0-255
            attn = self.normalize(attn) * 255
            attn = attn.to(torch.uint8)
            img = ToPILImage()(attn)
            img.save(f'./visuals/attns_random/attn_{t}.png')
attn_aggregator = AttentionAggregator()
class AttentionwHook(nn.Module):
    def __init__(
            self,
            attn: Attention
    ) -> None:
        super().__init__()
        for k, v in attn.__dict__.items():
            setattr(self, k, v) # copy all attributes
    def construct_from_block(self, attn):
        self.qkv = attn.qkv
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_aggregator(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage2_model_wrapper, _  = create_model(config, is_master=True) # load ckpt if available
    stage2_model_wrapper:Stage2ModelWrapper
    DiT_model = stage2_model_wrapper.stage_2_model.model
    assert isinstance(DiT_model, DiT), 'stage2 model must be DiT, got: ' + str(type(DiT_model))
    for block in DiT_model.blocks:
        if hasattr(block, 'attn'):
            block.attn = AttentionwHook(block.attn)
    print(DiT_model)
    image = get_default_image(im_size)
    print('image (shape min max):', image.shape, image.min(), image.max())
    data = LabeledImageData(img=image)
    encodings = stage2_model_wrapper.encode(data)
    print('encodings:', encodings.zs.shape)    
    labels = torch.tensor([1000] * encodings.zs.shape[0], device=encodings.zs.device)
    t = torch.zeros_like(labels)
    model_kwargs = dict(y=labels)
    terms = stage2_model_wrapper.stage_2_model.diffusion.training_losses(DiT_model, encodings.zs, t, model_kwargs)
    print('loss', terms["loss"].mean())
    print('attns:', len(attn_aggregator.attns), attn_aggregator.attns[0].shape)
    attn_aggregator.visualize_per_t()