from transformers import ViTMAEForPreTraining, ViTImageProcessor, ViTMAEModel
import torch
import torch_xla.core.xla_model as xm
from torch import nn
import numpy as np
torch.set_default_dtype(torch.bfloat16)
ckpt_path = '/home/bytetriper/model_zoo/mae_base_256'
dtype = torch.bfloat16
MAE = ViTMAEModel.from_pretrained(ckpt_path).to(xm.xla_device()).to(dtype)
print(MAE.device,MAE.dtype)
x = torch.randn(1, 3, 256, 256).to(xm.xla_device()).to(dtype)
print(x.device,x.dtype)
y = MAE(x)
print(y.last_hidden_state.device,y.last_hidden_state.dtype)
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
class dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, 197, 768), requires_grad=False
        )
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], 14, add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    def forward(self, x):
        print(self.position_embeddings.shape, self.position_embeddings.dtype)
        print(self.position_embeddings[:,1:,:].shape, self.position_embeddings[:,1:,:].dtype)
        x = x + self.position_embeddings[:,1:,:]
        print(x.shape, x.dtype)
        return x
dummy = dummy().to(dtype).to(xm.xla_device())
x = torch.randn(1, 196, 768).to(dtype).to(xm.xla_device())
print(x.shape, x.dtype)
y = dummy(x)
print(y.shape, y.dtype)