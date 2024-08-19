from functools import partial
from math import ceil
from numpy import identity
from torch import nn
import torch
from rqvae.models.basicblocks.utils import zero_module
class SimpleMLP(nn.Module):
    def __init__(self, input_dim:int, layers:int = 1, bottleneck_ratio: float = 16.0):
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

def Normalize(in_channels, num_groups=32):
    num_groups = in_channels // 32 # adaptive number of groups
    if num_groups == 0: # avoid zero division
        num_groups = 1 
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ConvResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                dropout, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        padding = kernel_size // 2
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.act = nn.SiLU()
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
class SimpleConv(nn.Module):
    def __init__(self, in_channels:int, layers:int = 1, bottleneck_ratio: float = 16.0, kernel_size: int = 3, final_norm: bool = False):
        super(SimpleConv, self).__init__()
        bottle_dim = int(in_channels // bottleneck_ratio)
        padding = kernel_size // 2
        each_layer_downsample_ratio = int(bottleneck_ratio ** (1.0 / layers)) if layers > 1 else 1
        self.down = nn.ModuleList()
        cur_dim = in_channels
        self.down_conv_in = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        for i in range(layers):
            next_dim = int(cur_dim / each_layer_downsample_ratio)
            if i == layers - 1:
                next_dim = bottle_dim
            self.down.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size))
            cur_dim = next_dim
        self.down_conv_out = torch.nn.Conv2d(bottle_dim, bottle_dim, kernel_size=kernel_size, stride=1, padding=padding)
        if final_norm:
            self.norm = nn.GroupNorm(num_groups=1, num_channels=bottle_dim, eps=1e-6, affine=False) # non-affine norm
        else:
            self.norm = nn.Identity()
        self.up_conv_in = torch.nn.Conv2d(bottle_dim, bottle_dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.up = nn.ModuleList()
        for i in range(layers):
            next_dim = int(cur_dim * each_layer_downsample_ratio)
            if i == layers - 1:
                next_dim = in_channels
            self.up.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size))
            cur_dim = next_dim
        self.up_conv_out = zero_module(torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding))
    def forward(self, x):
        for layer in self.down:
            x = layer(x)
        x = self.down_conv_out(x)
        x = self.norm(x)
        x = self.up_conv_in(x)
        for layer in self.up:
            x = layer(x)
        x = self.up_conv_out(x)
        return x
    def encode(self, x):
        for layer in self.down:
            x = layer(x)
        x = self.down_conv_out(x)
        return self.norm(x)
    def decode(self, x):
        x = self.up_conv_in(x)
        for layer in self.up:
            x = layer(x)
        x = self.up_conv_out(x)
        return x