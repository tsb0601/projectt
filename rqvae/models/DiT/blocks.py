from typing import List
import torch
from torch import nn
from ..basicblocks.utils import zero_module
from ..basicblocks.basics import ConvResnetBlock

class ConvEncoder(nn.Module):
    def __init__(self, in_channels:int, layers:int = 1, bottleneck_ratio: float = 16.0, kernel_size: int = 3, final_norm: bool = False):
        super(ConvEncoder, self).__init__()
        bottle_dim = int(in_channels // bottleneck_ratio)
        each_layer_downsample_ratio = int(bottleneck_ratio ** (1.0 / layers)) if layers > 1 else 1
        self.in_channels = in_channels
        self.down = nn.ModuleList()
        cur_dim = in_channels
        for i in range(layers):
            next_dim = int(cur_dim / each_layer_downsample_ratio)
            if i == layers - 1:
                next_dim = bottle_dim
            self.down.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size))
            cur_dim = next_dim
    def forward(self, x: torch.Tensor, return_hidden_states: bool = False)-> torch.Tensor:
        hs = []
        for layer in self.down:
            if return_hidden_states:
                hs.append(x)
            x = layer(x)
        if return_hidden_states:
            hs.append(x) # add the last hidden state
        if return_hidden_states:
            return x, hs
        return x
class ConvDecoder(nn.Module):
    def __init__(self, bottle_dim:int, layers:int = 1, upsample_ratio: float = 16.0, kernel_size: int = 3):
        super(ConvDecoder, self).__init__()
        in_channels = bottle_dim * upsample_ratio
        each_layer_upsample_ratio = int(upsample_ratio ** (1.0 / layers)) if layers > 1 else 1
        self.up = nn.ModuleList()
        cur_dim = bottle_dim
        for i in range(layers):
            next_dim = int(cur_dim * each_layer_upsample_ratio)
            if i == layers - 1:
                next_dim = in_channels
            self.up.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size, res_first=True))
            cur_dim = next_dim
        self.out_channels = in_channels
    def forward(self, x):
        for layer in self.up:
            x = layer(x)
        return x
class ConvDecoder_wSkipConnection(nn.Module):
    def __init__(self, bottle_dim:int, layers:int = 1, upsample_ratio: float = 16.0, kernel_size: int = 3):
        super(ConvDecoder_wSkipConnection, self).__init__()
        in_channels = bottle_dim * upsample_ratio
        each_layer_upsample_ratio = int(upsample_ratio ** (1.0 / layers)) if layers > 1 else 1
        self.up = nn.ModuleList()
        cur_dim = bottle_dim
        for i in range(layers):
            next_dim = int(cur_dim * each_layer_upsample_ratio)
            if i == layers - 1:
                next_dim = in_channels
            self.up.append(ConvResnetBlock(in_channels=cur_dim * 2, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size, res_first=True)) # *2 for skip connection
            cur_dim = next_dim
        self.out_channels = in_channels
    def forward(self, x:torch.Tensor, hs:List[torch.Tensor])-> torch.Tensor:
        for layer in self.up:
            x = torch.cat([x, hs.pop()], dim=1)
            x = layer(x)
        x = torch.cat([x, hs.pop()], dim=1) # add the last hidden state
        return x
class SimpleConv(nn.Module):
    def __init__(self, in_channels:int, layers:int = 1, bottleneck_ratio: float = 16.0, kernel_size: int = 3, final_norm: bool = False, double_channel_output: bool = False):
        super(SimpleConv, self).__init__()
        bottle_dim = int(in_channels // bottleneck_ratio)
        assert in_channels % bottle_dim == 0, "in_channels must be divisible by bottleneck_ratio"
        padding = kernel_size // 2
        self.down = nn.ModuleList()
        self.down_conv_in = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.down = ConvEncoder(in_channels=in_channels, layers=layers, bottleneck_ratio=bottleneck_ratio, kernel_size=kernel_size, final_norm=final_norm)
        #for i in range(layers):
        #    next_dim = int(cur_dim / each_layer_downsample_ratio)
        #    if i == layers - 1:
        #        next_dim = bottle_dim
        #    self.down.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size))
        #    cur_dim = next_dim
        self.down_conv_out = torch.nn.Conv2d(bottle_dim, bottle_dim, kernel_size=kernel_size, stride=1, padding=padding)
        if final_norm:
            self.norm = nn.GroupNorm(num_groups=1, num_channels=bottle_dim, eps=1e-6, affine=False) # non-affine norm
        else:
            self.norm = nn.Identity()
        self.up_conv_in = torch.nn.Conv2d(bottle_dim, bottle_dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.up = ConvDecoder(bottle_dim=bottle_dim, layers=layers, upsample_ratio=bottleneck_ratio, kernel_size=kernel_size)
        #self.up = nn.ModuleList()
        #for i in range(layers):
        #    next_dim = int(cur_dim * each_layer_downsample_ratio)
        #    if i == layers - 1:
        #        next_dim = in_channels
        #    self.up.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size))
        #    cur_dim = next_dim
        self.up_conv_out = zero_module(torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding))
    def forward(self, x):
        #for layer in self.down:
        #    x = layer(x)
        x = self.down_conv_in(x)
        x = self.down(x)
        x = self.down_conv_out(x)
        x = self.norm(x)
        x = self.up_conv_in(x)
        x = self.up(x)
        #for layer in self.up:
        #    x = layer(x)
        x = self.up_conv_out(x)
        return x
    def encode(self, x):
        x = self.down_conv_in(x)
        x = self.down(x)
        #for layer in self.down:
        #    x = layer(x)
        x = self.down_conv_out(x)
        return self.norm(x)
    def decode(self, x):
        x = self.up_conv_in(x)
        x = self.up(x)
        #for layer in self.up:
        #    x = layer(x)
        x = self.up_conv_out(x)
        return x