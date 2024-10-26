from functools import partial
from torch import nn
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
class ConvUp(nn.Module):
    def __init__(self, in_channels:int, layers:int , out_channel: int , kernel_size: int = 1, final_norm: bool = False):
        super(ConvUp, self).__init__()
        padding = kernel_size // 2
        bottleneck_ratio = out_channel / in_channels
        each_layer_downsample_ratio = int(bottleneck_ratio ** (1.0 / layers)) if layers > 1 else 1
        self.up_conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.up = nn.ModuleList()
        for i in range(layers):
            next_dim = int(cur_dim * each_layer_downsample_ratio)
            if i == layers - 1:
                next_dim = in_channels
            self.up.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size,res_first=False))
            cur_dim = next_dim
        self.up_conv_out = zero_module(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding))
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