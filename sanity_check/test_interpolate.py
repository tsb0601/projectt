

from torch import nn
import torch
patch_pos_embed = torch.randn(1, 1, 1, 196)
num_positions = 196
embeddings_positions = 256
dim = patch_pos_embed.shape[1]
print('patch_pos_embed', patch_pos_embed.shape)
# Interpolating the decoder position embeddings shape wrt embeddings shape i.(x).
# 1 keeps the other dimension constant
patch_pos_embed = nn.functional.interpolate(
    patch_pos_embed,
    size = (1, embeddings_positions),
    #scale_factor=(1, embeddings_positions * 1. / num_positions),
    mode="bicubic",
    align_corners=False,
)
print('patch_pos_embed', patch_pos_embed.shape)
# Converting back to the original shape
patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
print('patch_pos_embed', patch_pos_embed.shape)