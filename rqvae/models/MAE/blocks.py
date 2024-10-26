from functools import partial
from torch import nn
import torch
from rqvae.models.basicblocks.utils import zero_module
from rqvae.models.basicblocks.basics import ConvResnetBlock
class SimpleMLP(nn.Module):
    def __init__(self, input_dim:int, layers:int = 1, mlp_ratio: float = 4.0, bottleneck_ratio: float = 16.0):
        super(SimpleMLP, self).__init__()
        bottle_dim = int(input_dim // bottleneck_ratio)
        cur_dim = input_dim
        each_layer_downsample_ratio = int(bottleneck_ratio ** (1.0 / layers)) if layers > 1 else 1
        self.down = nn.ModuleList()
        norm = partial(nn.LayerNorm, eps = 1e-6)
        act = nn.SiLU
        for i in range(layers):
            next_dim = int(cur_dim / each_layer_downsample_ratio)
            if i == layers - 1:
                next_dim = bottle_dim
            hidden_dim = int(cur_dim * mlp_ratio)
            self.down.append(nn.Sequential(
                nn.Linear(cur_dim, next_dim),
                #norm(hidden_dim),
                #act(),
                #nn.Linear(hidden_dim, next_dim),
                norm(next_dim),
                act(),
            ))
            cur_dim = next_dim
        self.linear_in = nn.Linear(cur_dim, bottle_dim)
        cur_dim = bottle_dim
        self.up = nn.ModuleList()
        for i in range(layers):
            next_dim = int(cur_dim * each_layer_downsample_ratio)
            if i == layers - 1:
                next_dim = input_dim
            hidden_dim = int(cur_dim * mlp_ratio)
            self.up.append(nn.Sequential(
                nn.Linear(cur_dim, next_dim),
                #norm(hidden_dim),
                #act(),
                #nn.Linear(hidden_dim, next_dim),
                norm(next_dim),
                act(),
            ))
            cur_dim = next_dim
        self.linear_out = zero_module(nn.Linear(cur_dim, input_dim))
    def forward(self, x):
        for layer in self.down:
            x = layer(x)
        x = self.linear_in(x)
        for layer in self.up:
            x = layer(x)
        x = self.linear_out(x)
        return x
    def encode(self, x):
        for layer in self.down:
            x = layer(x)
        return self.linear_in(x)
    def decode(self, x):
        for layer in self.up:
            x = layer(x)
        return self.linear_out(x)
def L_to_P(zs:torch.Tensor, split:float = 1)-> torch.Tensor:
    """
    zs: [batch_size, seq_len, hidden_size]
    return: [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    """
    # reshape it to square
    batch_size, num_patches, hidden_size = zs.shape
    pn = int(num_patches ** 0.5)
    zs = zs.view(batch_size, pn, pn, hidden_size)
    #zs = self.forward_norm(zs)
    # channel goes first
    zs = zs.permute(0,3,1,2).contiguous() # [batch_size, hidden_size, patch_size, patch_size]
    sqrt_split = int(split ** 0.5)
    split_c = int(hidden_size // split)
    split_pn = pn * sqrt_split
    # reshape to bsz, split_c, split_pn, split_pn
    # first split to split_c, sqrt_split, sqrt_split, pn, pn
    zs = zs.view(batch_size, split_c, sqrt_split, sqrt_split, pn, pn)
    # then permute to split_c, split_pn, sqrt_split, split_pn, sqrt_split
    zs = zs.permute(0,1,4,2,5,3).contiguous()
    # then reshape to bsz, hidden_size, split_pn, split_pn
    zs = zs.reshape(batch_size, split_c, split_pn, split_pn)
    return zs.contiguous()
def P_to_L(zs:torch.Tensor, split:float = 1) -> torch.Tensor:
    """
    zs: [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    return: [batch_size, seq_len, hidden_size]
    """
    batch_size, c , pn, pn = zs.shape
    aggregated_c = c * split
    sqrt_split = int(split ** 0.5)
    split_pn = int(pn // sqrt_split)
    #zs = zs.view(batch_size, c, sqrt_split, split_pn, sqrt_split, split_pn)
    zs = zs.reshape(batch_size, c, split_pn, sqrt_split, split_pn, sqrt_split)
    #try reshape back to see diff
    # do a reverse permute to (0,1,4,2,5,3)
    zs = zs.permute(0,1,3,5,2,4).contiguous()
    zs = zs.view(batch_size, aggregated_c, split_pn, split_pn)
    zs = zs.permute(0,2,3,1).contiguous()
    zs = zs.view(batch_size, split_pn, split_pn, aggregated_c)
    zs = zs.view(batch_size, split_pn*split_pn, aggregated_c)
    return zs.contiguous()
class ConvUp(nn.Module):
    def __init__(self, in_channels:int, out_channel: int , kernel_size: int = 1,  layers_channels: list = []):
        super(ConvUp, self).__init__()
        if layers_channels == []:
            layers_channels = [out_channel] # default to one layer
        layers = len(layers_channels)
        padding = kernel_size // 2
        self.up_conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.up = nn.ModuleList()
        cur_dim = in_channels
        for i in range(layers):
            next_dim = layers_channels[i]
            if i == layers - 1:
                next_dim = out_channel
            self.up.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size, res_first=False))
            cur_dim = next_dim
        self.up_conv_out = zero_module(nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding))
    def forward(self, x):
        need_transform = len(x.shape) == 3
        include_cls = False
        if need_transform:
            # [B, T, D]
            # check if CLS token is included
            if x.shape[1] % 2 == 1:
                x = x[:,1:] # remove the first token
                include_cls = True
            x = L_to_P(x)
        x = self.up_conv_in(x)
        for layer in self.up:
            x = layer(x)
        x = self.up_conv_out(x)
        if need_transform:
            x = P_to_L(x)
            if include_cls:
                x = torch.cat([torch.zeros_like(x[:,0:1]), x], dim=1)
        return x