# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from rqvae.models.interfaces import Stage2Model
from .diffusion import create_diffusion
from .blocks import ConvEncoder, ConvDecoder
from header import *
from rqvae.models.basicblocks.basics import ConvMlp
from transformers import Dinov2Model
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
class DiTBlockOnlyMlp(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x
class DiTBlockWoAdaLN(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    def forward(self, x, c):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class DiTBlockwoAdaLNAttn(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
    def forward(self, x, c):
        x = x + self.mlp(self.norm(x))
        return x
class DiTBlockwConv(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, use Conv to replace MLP. When kernel_size=1, it is equivalent to DiTBlock.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, kernel_size:int = 1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = ConvMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0, kernel_size=kernel_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
class FinalLayerSimple(nn.Module):
    """
    simple linear layer
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    def forward(self, x, c):
        x = self.linear(x)
        return x
class FinalLayerIdentity(nn.Module):
    """
    identity layer
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.hidden_size = hidden_size
        self.output_channels = patch_size * patch_size * out_channels
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    def forward(self, x, c):
        if x.shape[-1] != self.output_channels:
            x = self.linear(x)
        return x
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class DiTwSkipConnection(DiT):
    """
    only difference is that it has skip connection in forward function
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.blocks) % 2 == 0, "The number of transformer blocks must be even for skip connection."
    # reload forward
    def forward(self, x, t, y):
        """
        Forward pass of DiT with long skip connection. For a total of N transformer blocks, the skip connection connects the ith and (N-i)th blocks.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        depth = len(self.blocks)
        half_depth = depth // 2 # must be divisible by 2
        hs = []
        for i, block in enumerate(self.blocks):
            if i < half_depth:
                hs.append(x)
            x = block(x, c)                      # (N, T, D)
            if i >= half_depth:
                x = x + hs.pop()
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
class DiTwSkipConnectionConv(DiT):
    """
    only difference is that it has skip connection in forward function
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.blocks) % 2 == 1, "The number of transformer blocks must be even for skip connection."
        self.lns = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, self.hidden_size) for _ in range(len(self.blocks) // 2)
        ])
        
    # reload forward
    def forward(self, x, t, y):
        """
        Forward pass of DiT with long skip connection. For a total of N transformer blocks, the skip connection connects the ith and (N-i)th blocks.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        depth = len(self.blocks)
        half_depth = depth // 2 
        down_blocks = self.blocks[:half_depth]
        mid_block = self.blocks[half_depth]
        up_blocks = self.blocks[half_depth + 1:]
        skips = []
        for i, block in enumerate(down_blocks):
            x = block(x, c)                      # (N, T, D)
            skips.append(x)
        x = mid_block(x, c)
        for i, block in enumerate(up_blocks):
            x = torch.cat([x, skips.pop()], dim= -1)   # (N, T, 2*D)
            x = self.lns[i](x) # (N, T, D)
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
class DiTwoAttn(DiT):
    """
    DiT without attention mechanism, simply stacking MLPs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.blocks:
            block.attn = nn.Identity()
class DiTonlyMlp(DiT):
    """
    only MLP + AdaLN-Zero, no final layer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_size = self.hidden_size
        mlp_ratio = kwargs.get('mlp_ratio', None)
        depth = kwargs.get('depth', None)
        self.blocks = nn.ModuleList([
            DiTBlockOnlyMlp(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        self.final_layer = FinalLayerSimple(hidden_size, self.patch_size, self.out_channels)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
class CustomDiTBlock(nn.Module):
    """
    h1 -> linear -> h2 -> MLP -> h2 -> linear -> h1
    """
    def __init__(self, h1, h2, mlp_ratio=4.0, modulate_pos: int = 0, gate_pos: int = -1):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.mlp = nn.ModuleList([
            nn.Linear(h1, h2, bias=True),
            nn.Linear(h2, int(h2 * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(h2 * mlp_ratio), h2, bias=True),
            nn.Linear(h2, h1, bias=True)
        ])
        self.modulate_dim = h1 if modulate_pos == 0 else h2
        self.gate_dim = h1 if gate_pos == -1 else h2
        self.modulate_pos = modulate_pos
        self.gate_pos = gate_pos
        assert modulate_pos in [0,1] and gate_pos in [-1, -2]
        self.adaln = nn.ModuleList([
            nn.Linear(h1, self.modulate_dim, bias=True),
            nn.Linear(h1, self.modulate_dim, bias=True),
            nn.Linear(h1, self.gate_dim, bias=True)
        ])
        self.norm = nn.LayerNorm(self.modulate_dim, elementwise_affine=False, eps=1e-6)
        nn.init.constant_(self.adaln[-1].weight, 0)
        nn.init.constant_(self.adaln[-1].bias, 0)
    def forward(self, x, c):
        shift, scale, gate = self.adaln[0](c), self.adaln[1](c), self.adaln[2](c)
        orig_ = x
        for i, block in enumerate(self.mlp):
            if i == self.modulate_pos:
            #    print('modulate, shift, scale', shift.shape, scale.shape)
                x = modulate(self.norm(x), shift, scale)
            x = block(x)
            #print('block', i, x.shape)
            if i == self.gate_pos + len(self.mlp):
                x = x * gate.unsqueeze(1)
            #    print('gate', x.shape)
        x = orig_ + x
        return x
class SimplyMLP(nn.Module):
    """
    self-designed MLP
    """
    def __init__(self, mlp_ratio: float = 4.0, hidden_size: int = 768, in_channels: int = 768):
        super().__init__()
        self.pre_projection = nn.Linear(hidden_size, in_channels, bias=True) # let's try a compression
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, in_channels, bias=True), # let's try a compression
            nn.Linear(in_channels, int(hidden_size * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size, bias=True),
        )
        self.after_projection = nn.Linear(in_channels, hidden_size, bias=True) # let's try a compression
        #self.adaln = nn.Sequential(
        #    nn.SiLU(),
        #    nn.Linear(hidden_size, 3 * in_channels, bias=True)
        #)
        #nn.init.constant_(self.after_projection.weight, 0)
        #nn.init.constant_(self.after_projection.bias, 0)
        self.adaln = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Linear(hidden_size, hidden_size, bias=True)
        ])
        # init adaln-Zero
        nn.init.constant_(self.adaln[-1].weight, 0)
        nn.init.constant_(self.adaln[-1].bias, 0)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    def forward(self, x, c):
        #shift, scale, gate = self.adaln(c).chunk(3, dim=1)
        shift = self.adaln[0](c)
        scale = self.adaln[1](c)
        gate = self.adaln[2](c)
        x = modulate(self.norm(x), shift, scale)
        x = self.pre_projection(x)
        x = self.mlp(x)
        x = self.after_projection(x) * gate.unsqueeze(1) 
        return x
class DiTSimplyMlp(DiT):
    """
    one-block self-designed MLP
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_size = self.hidden_size
        mlp_ratio = kwargs.get('mlp_ratio', None)
        in_channels = kwargs.get('in_channels', None)
        pn = kwargs.get('patch_size', None)
        in_channels = in_channels * pn * pn # patch_size ** 2 * in_channels
        self.blocks = nn.ModuleList([
            SimplyMLP(mlp_ratio=mlp_ratio, hidden_size=hidden_size, in_channels=in_channels)
        ])
        #self.final_layer = FinalLayerIdentity(hidden_size, self.patch_size, self.out_channels)
        self.final_layer = FinalLayerSimple(hidden_size, self.patch_size, self.out_channels)
        #nn.init.constant_(self.final_layer.linear.weight, 0)
        #nn.init.constant_(self.final_layer.linear.bias, 0)
class DiTCustomSingleBlock(DiT):
    """
    DiT but with a single custom block
    """
    def __init__(self, **kwargs):
        h2 = kwargs.pop('h2', None)
        modulate_pos = kwargs.pop('modulate_pos', 0)
        gate_pos = kwargs.pop('gate_pos', -1)
        super().__init__(**kwargs)
        hidden_size = self.hidden_size
        mlp_ratio = kwargs.get('mlp_ratio', None)
        h1 = hidden_size
        self.blocks = nn.ModuleList([
            CustomDiTBlock(h1, h2, mlp_ratio=mlp_ratio, modulate_pos=modulate_pos, gate_pos=gate_pos) for _ in range(len(self.blocks))
        ])
        self.final_layer = FinalLayerSimple(hidden_size, self.patch_size, self.out_channels)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
class DiTWoAdaLN(DiT):
    def __init__(self, **kwargs):
        t_emb_dim = kwargs.pop('t_emb_dim', 256)
        wo_attn = kwargs.pop('wo_attn', False)
        super().__init__(**kwargs)
        mlp_ratio = kwargs.get('mlp_ratio', None)
        self.hidden_size += t_emb_dim # actual embed dim, we concat t_emb to x_emb
        self.t_embedder = TimestepEmbedder(t_emb_dim) # re-def t_embedder
        block_cls = DiTBlockwoAdaLNAttn if wo_attn else DiTBlockWoAdaLN
        self.blocks = nn.ModuleList([
            block_cls(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(len(self.blocks))
        ])
        self.final_layer = FinalLayerSimple(self.hidden_size, self.patch_size, self.out_channels)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        # t: (N, D) -> (N, L, D)
        t = t.unsqueeze(1).expand(-1, x.shape[1], -1)
        # drop y
        x = torch.cat([x, t], dim=-1) # concat t to x
        for block in self.blocks:
            x = block(x, None)
        x = self.final_layer(x, None)
        x = self.unpatchify(x)
        return x
class DiTwWideBlockonLargeTimeStep(DiT):
    """
    use a thin DiT on small time step, and a wide DiT on large time step
    """
    def __init__(self, **kwargs):
        self.t_sep = kwargs.pop('t_sep', 800)
        wide_channel = kwargs.pop('wide_channel', 1024) # wide hidden size for wide-DiT
        super().__init__(**kwargs)
        hidden_size = self.hidden_size
        mlp_ratio = kwargs.get('mlp_ratio', None)
        depth = kwargs.get('depth', None)
        input_size = kwargs.get('input_size', None)
        num_classes = kwargs.get('num_classes', None)
        class_dropout_prob = kwargs.get('class_dropout_prob', None)
        self.thin_blocks = nn.ModuleList([
            DiTBlock(hidden_size, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.wide_blocks = nn.ModuleList([
            DiTBlock(wide_channel, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.thin_final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)
        self.wide_final_layer = FinalLayer(wide_channel, self.patch_size, self.out_channels)
        self.wide_x_embedder = PatchEmbed(input_size, self.patch_size, self.in_channels, wide_channel, bias=True)
        self.wide_pos_embed = nn.Parameter(torch.zeros(1, self.wide_x_embedder.num_patches, wide_channel), requires_grad=False)
        self.wide_t_embedder = TimestepEmbedder(wide_channel)
        self.wide_y_embedder = LabelEmbedder(num_classes, wide_channel, class_dropout_prob)
        w = self.wide_x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.wide_x_embedder.proj.bias, 0)
        nn.init.normal_(self.wide_y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.wide_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.wide_t_embedder.mlp[2].weight, std=0.02)
        #pos embed
        pos_embed = get_2d_sincos_pos_embed(self.wide_pos_embed.shape[-1], int(self.wide_x_embedder.num_patches ** 0.5))
        self.wide_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)        
        for block in self.thin_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.wide_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.thin_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.thin_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.thin_final_layer.linear.weight, 0)
        nn.init.constant_(self.thin_final_layer.linear.bias, 0)
        nn.init.constant_(self.wide_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.wide_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.wide_final_layer.linear.weight, 0)
        nn.init.constant_(self.wide_final_layer.linear.bias, 0)
    def forward(self, x, t, y):
        thin_x = self.x_embedder(x) + self.pos_embed
        thin_t = self.t_embedder(t)
        thin_y = self.y_embedder(y, self.training)
        thin_c = thin_t + thin_y
        wide_x = self.wide_x_embedder(x) + self.wide_pos_embed
        wide_t = self.wide_t_embedder(t)
        wide_y = self.wide_y_embedder(y, self.training)
        wide_c = wide_t + wide_y
        for block in self.thin_blocks:
            thin_x = block(thin_x, thin_c)
        thin_x = self.thin_final_layer(thin_x, thin_c)
        for block in self.wide_blocks:
            wide_x = block(wide_x, wide_c)
        wide_x = self.wide_final_layer(wide_x, wide_c)
        # thin_x: (N, T, patch_size ** 2 * out_channels)
        # wide_x: (N, T, patch_size ** 2 * out_channels)
        # combine in batch dimension
        # t: (N,) -> (N, *x.shape[1:]), x: (N, 
        empty_x = torch.zeros_like(thin_x)
        """for i in range(len(x)):
            if t[i] < self.t_sep:
                empty_x[i] = thin_x[i]
            else:
                empty_x[i] = wide_x[i]"""
        # use tensor op to replace for loop
        # t: (N,) 
        # t_sep: scalar
        # thin_x: (N, T, patch_size ** 2 * out_channels)
        # wide_x: (N, T, patch_size ** 2 * out_channels)
        # first expand t to (N, T, patch_size ** 2 * out_channels)
        t = t.reshape(-1, 1, 1).expand(-1, thin_x.shape[1], thin_x.shape[2])
        # then compare t with t_sep
        empty_x = torch.where(t < self.t_sep, thin_x, wide_x)
        final_x = self.unpatchify(empty_x)
        return final_x
class DiTwoMlp(DiT):
    """
    DiT without MLPs, simply stacking attention mechanisms.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.blocks:
            block.mlp = nn.Identity()
class DiT_convMLP(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        conv_kernel_size:int = 1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlockwConv(hidden_size, num_heads, mlp_ratio=mlp_ratio, kernel_size=conv_kernel_size) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
from einops.layers.torch import Rearrange
class ConvStem(nn.Module):
    """
    Follow implementation of https://github.com/Jack-Etheredge/early_convolutions_vit_pytorch/blob/main/vitc/early_convolutions.py#L84
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        n_filter_list = (in_channels, 768, 768, 768, 768)  # hardcoding for now because that's what the paper used
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=3,  # hardcoding for now because that's what the paper used
                          stride=2,  # hardcoding for now because that's what the paper used
                          padding=1),  # hardcoding for now because that's what the paper used
            )
                for i in range(len(n_filter_list)-1)
            ])
        total_stride = 1
        compression_factor = 2 ** (len(n_filter_list)-1) 
        self.patch_size = (compression_factor, compression_factor) 
        # those conv layers should equals to a 16x16 patching
        self.conv_layers.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                    out_channels=out_channels, 
                                    stride=1,  # hardcoding for now because that's what the paper used 
                                    kernel_size=1,  # hardcoding for now because that's what the paper used 
                                    padding=0))  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module("flatten image", 
                                    Rearrange('batch channels height width -> batch (height width) channels'))
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
class ConvStemAsPatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=16):
        super().__init__()
        dim = in_channels * stride ** 2
        self.conv = nn.Conv2d(in_channels, dim, kernel_size, stride, padding=0, bias=True) # stride = patch_size
        self.ln_in = nn.Linear(dim, out_channels)
        self.patch_size = (stride, stride)
    def forward(self, x):
        x = self.conv(x) # (N, C, H, W) -> (N, D, H', W')
        x = x.flatten(2).transpose(1, 2) # (N, D, H', W') -> (N, H'*W', D)
        x = self.ln_in(x) # (N, H'*W', D) -> (N, H'*W', out_channels)
        return x
class DiTConvStem(DiT):
    """
    replace patch embedding with a convolutional stem
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_channels = kwargs.get('in_channels', None)
        out_channels = kwargs.get('hidden_size', None)
        self.x_embedder = ConvStem(3, out_channels) # replace patch embedding with a convolutional stem, in channel should be 3
        self.in_channels = 3 # reload for generation
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.patch_size[0] ** 2, out_channels), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.patch_size[0]))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    def unpatchify(self, x):
        """
        x: (N, H*W/p**2, p**2*3)
        imgs: (N, H, W, 3)
        """
        c = 3 # for pixel rgb
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
class DiTConvStemAsPatchEmbed(DiT):
    """
    replace patch embedding with a convolutional stem
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_channels = kwargs.get('in_channels', None)
        out_channels = kwargs.get('hidden_size', None)
        self.x_embedder = ConvStemAsPatchEmbed(3, out_channels) # replace patch embedding with a convolutional stem, in channel should be 3
        self.in_channels = 3 # reload for generation
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.patch_size[0] ** 2, out_channels), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.patch_size[0]))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    def unpatchify(self, x):
        """
        x: (N, H*W/p**2, p**2*3)
        imgs: (N, H, W, 3)
        """
        c = 3 # for pixel rgb
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

class DiTwReg(DiT):
    """
    DiT with registers
    """
    def __init__(self, **kwargs):
        register_size = kwargs.pop('register_size', 256)
        super().__init__(**kwargs)
        hidden_size = self.hidden_size
        self.register_size = register_size
        self.register = nn.Parameter(torch.zeros(1, register_size, hidden_size))
    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        # cat register to x
        reg = self.register.expand(x.shape[0], -1, -1)
        x = torch.cat([reg, x], dim=1)
        for block in self.blocks:
            x = block(x, c)
        # drop register
        x = x[:, self.register_size:]
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
from transformers import Dinov2Model, ViTImageProcessor, ViTMAEModel
class DiTwDino(DiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_size = self.hidden_size
        dino_ckpt = kwargs.pop('dino_ckpt', None)
        self.dino = Dinov2Model.from_pretrained(dino_ckpt)
        dino_processor = ViTImageProcessor.from_pretrained(dino_ckpt)
        dino_mean, dino_std = dino_processor.image_mean, dino_processor.image_std
        self.dino_mean = self.register_buffer('dino_mean', torch.tensor(dino_mean).view(1, 3, 1, 1))
        self.dino_std = self.register_buffer('dino_std', torch.tensor(dino_std).view(1, 3, 1, 1))
        assert hidden_size == self.dino.config.hidden_size, 'DiT hidden size should be the same as dino'
        patch_size = self.x_embedder.patch_size[0]
        assert patch_size == self.dino.config.patch_size, 'DiT patch size should be the same as dino'
    def forward(self, x, t, y):
        # x: (N, C, H, W)
        x = (x - self.dino_mean) / self.dino_std
        x = self.dino(x).last_hidden_state # (N, T + 1, D)
        # remove CLS
        x = x[:, 1:] # (N, T, D)
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c) # (N, T, D)
        x = self.final_layer(x, c)
        x = self.unpatchify(x) # (N, C, H, W)
        return x
    def unpatchify(self, x):
        """
        x: (N, H*W/p**2, p**2*3)
        imgs: (N, H, W, 3)
        """
        c = 3 # for pixel rgb
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

class DiTwMAE(DiT):
    """
    DiT with MAE
    """
    def __init__(self, **kwargs):
        mae_ckpt = kwargs.pop('mae_ckpt', None)
        super().__init__(**kwargs)
        self.in_channels = 3 # hack
        hidden_size = self.hidden_size
        input_size = kwargs.get('input_size', None)
        self.mae = ViTMAEModel.from_pretrained(mae_ckpt)
        mae_processor = ViTImageProcessor.from_pretrained(mae_ckpt)
        mae_mean, mae_std = mae_processor.image_mean, mae_processor.image_std
        self.register_buffer('mae_mean', torch.tensor(mae_mean).view(1, 3, 1, 1))
        self.register_buffer('mae_std', torch.tensor(mae_std).view(1, 3, 1, 1))
        self.patch_size = self.mae.config.patch_size
        patch_num = (input_size // self.patch_size) **2
        noise = torch.arange(patch_num)
        self.register_buffer('noise', noise)
        assert hidden_size == self.mae.config.hidden_size, 'DiT hidden size should be the same as mae, got %s and %s' % (hidden_size, self.mae.config.hidden_size)
    def forward(self, x, t, y):
        # x: (N, C, H, W)
        # input is in [-1, 1]
        # first normalize to [0, 1]
        #print('input x', x.shape)
        x = (x + 1) / 2
        # then normalize to mae
        x = (x - self.mae_mean) / self.mae_std
        noise = self.noise.unsqueeze(0).expand(x.shape[0], -1)
        #print('x', x.shape)
        x = self.mae(x, noise = noise, interpolate_pos_encoding = True).last_hidden_state # (N, T + 1, D)
        #print('after mae', x.shape)
        # remove CLS
        x = x[:, 1:] # (N, T, D)
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c) # (N, T, D)
        x = self.final_layer(x, c)
        #print('final layer', x.shape)
        x = self.unpatchify(x) # (N, C, H, W)
        #print('unpatchify', x.shape)
        return x
    def unpatchify(self, x):
        """
        x: (N, H*W/p**2, p**2*3)
        imgs: (N, H, W, 3)
        """
        c = 6 if self.learn_sigma else 3 
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

class DiTWideAtLast(DiT):
    """
    normal DiT at first, few wide blocks at last
    """    
    def __init__(self, **kwargs):
        second_patch_size = kwargs.pop('second_patch_size', 16)
        second_depth = kwargs.pop('second_depth', 4)
        second_hidden_size = kwargs.pop('second_hidden_size', 768)
        self.second_patch_size = second_patch_size
        super().__init__(**kwargs)
        self.factor = self.patch_size // second_patch_size
        assert self.patch_size % second_patch_size == 0, 'second patch size should be divisible by first patch size'
        second_num_patches = (self.x_embedder.num_patches * self.factor ** 2)
        self.second_num_patches = second_num_patches
        self.second_blocks = nn.ModuleList([
            DiTBlock(second_hidden_size, self.num_heads, mlp_ratio=4.0) for _ in range(second_depth)
        ])
        self.second_pos_embed = nn.Parameter(torch.zeros(1, second_num_patches, second_hidden_size), requires_grad=False)
        self.second_final_layer = FinalLayer(second_hidden_size, second_patch_size, self.out_channels)
        #init pos embed
        pos_embed = get_2d_sincos_pos_embed(self.second_pos_embed.shape[-1], int(second_num_patches ** 0.5))
        self.second_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # init DiT blocks
        for block in self.second_blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # init final layer
        nn.init.constant_(self.second_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.second_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.second_final_layer.linear.weight, 0)
        nn.init.constant_(self.second_final_layer.linear.bias, 0)
        #self.pixel_shuffle = nn.PixelShuffle(self.factor)
        second_input_size = self.x_embedder.num_patches ** 0.5 * self.factor
        self.second_x_embedder = PatchEmbed(second_input_size, 1, self.hidden_size // self.factor**2, second_hidden_size, bias=True)
        self.second_t_embedder = TimestepEmbedder(second_hidden_size)
        self.second_y_embedder = LabelEmbedder(self.num_classes, second_hidden_size, self.class_dropout_prob)
        # init second x embedder
        w = self.second_x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.second_x_embedder.proj.bias, 0)
        # init second t embedder
        nn.init.normal_(self.second_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.second_t_embedder.mlp[2].weight, std=0.02)
        # init second y embedder
        nn.init.normal_(self.second_y_embedder.embedding_table.weight, std=0.02)
    def forward(self, x, t, y):
        N, C, H, W = x.shape
        timestep = t 
        condition = y
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(timestep)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        # x: (N, T, D)
        x = self.unpatchify(x)
        x = self.second_x_embedder(x) + self.second_pos_embed
        second_t = self.second_t_embedder(timestep)
        second_y = self.second_y_embedder(condition, self.training)
        second_c = second_t + second_y
        for block in self.second_blocks:
            x = block(x, second_c)
        x = self.second_final_layer(x, second_c)
        x = self.second_unpatchify(x)
        return x
    def unpatchify(self, x):
        """
        x: (N, H*W/p**2, p**2*D)
        imgs: (N, H, W, D)
        """
        c = self.hidden_size // self.factor ** 2
        p = self.factor
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    def second_unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.second_patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
        
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

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
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)