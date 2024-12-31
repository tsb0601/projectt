import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn as nn


class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """
    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        # self.gamma = nn.Parameter(torch.FloatTensor(1, 1, 1))
        # self.beta = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class DTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb
class ViTDiscriminator(nn.Module):
    def __init__(self,
        in_channels = 3,
        patch_size = 8,
        extend_size = 2,
        dim = 384,
        blocks = 6,
        num_heads = 6,
        image_size = 256,
        dim_head = None,
        use_cls: bool = False,
        dropout = 0
    ):
        super(ViTDiscriminator, self).__init__()
        self.patch_size = patch_size + 2 * extend_size
        padding = extend_size
        self.token_dim = in_channels * (self.patch_size ** 2)
        #self.project_patches = nn.Linear(self.token_dim, dim)
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=self.patch_size,
            stride= patch_size, # w/o overlap
            padding=padding
        )
        token_num = (image_size // patch_size) ** 2
        self.emb_dropout = nn.Dropout(dropout)
        self.use_cls = use_cls
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(token_num + 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        self.dim = dim
        self.token_num = token_num
        self.Transformer_Encoder = DTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)
        self.final_sigmoid = nn.Sigmoid()
        self.init_weights()
    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.dim, int(self.token_num ** .5), cls_token=True, extra_tokens=1)
        print(pos_embed.shape)
        self.pos_emb1D.data.copy_(torch.from_numpy(pos_embed))
        
        #init the last layer with 0
        nn.init.constant_(self.mlp_head[-1].weight, 0)
        nn.init.constant_(self.mlp_head[-1].bias, 0)
    def classify(self, img):
        # Generate overlappimg image patches
        image_patches = self.patch_embed(img)
        image_patches = rearrange(image_patches, 'b c h w -> b (h w) c')
        batch_size, tokens, _ = image_patches.shape

        # Prepend the classifier token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        image_patches = torch.cat((cls_token, image_patches), dim = 1)

        # Plus the positional embedding
        image_patches = image_patches + self.pos_emb1D[: tokens + 1, :]
        image_patches = self.emb_dropout(image_patches)

        result = self.Transformer_Encoder(image_patches)
        logits = self.mlp_head(result[:, 1:, :])
        #logits = nn.Sigmoid()(logits)
        return logits
    def forward(self, x, y = None):
        return self.classify(x), self.classify(y) if y is not None else None
def main():
    discriminator = ViTDiscriminator()
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    out = discriminator(x, y)
    print(out[0].shape, out[1].shape if out[1] is not None else None)
    print(out[0][:,0,0])
if __name__ == '__main__':
    main()