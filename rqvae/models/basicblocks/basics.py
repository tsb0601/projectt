from functools import partial
from torch import nn
from timm.models.vision_transformer import PatchEmbed, DropPath
import torch_xla
import torch

from .utils import *
from timm.models.vision_transformer import Mlp as MLP # Mlp ?? absoultely MLP
norms = {
    'batch': nn.BatchNorm2d,
    'layer': partial(nn.LayerNorm, eps=1e-6), # LayerNorm with default eps
}
class Attention(nn.Module):
    """
    borrowed from https://github.com/LTH14/mage/blob/main/models_mage.py#L16
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        with torch_xla.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale # float32 for numerical stability
        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class AttnMlpBlock(nn.Module):
    """
    borrowed from https://github.com/LTH14/mage/blob/main/models_mage.py#L47
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
import collections.abc
from itertools import repeat
"""
borrowed from timm.layers.helpers.py#L10
"""
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)
class ConvMlp(nn.Module):
    """
    a module like timm.layers.Mlp but with Conv1d with adjustable kernel size
    partially borrowed from timm.layers.mlp.py#L13
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=None, bias=True, drop=0., kernel_size=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        conv_ln = partial(nn.Conv1d, kernel_size=kernel_size)
        self.fc1 = conv_ln(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = conv_ln(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class MlpResBlock(nn.Module):
    """
    Borrowed from https://github.com/LTH14/rcg/blob/main/rdm/modules/diffusionmodules/latentmlp.py#L9
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param mid_channels: the number of middle channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    """

    def __init__(
        self,
        channels,
        mid_channels,
        emb_channels,
        dropout,
        use_context=False,
        context_channels=512
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, mid_channels, bias=True),
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Linear(mid_channels, channels, bias=True)
            ),
        )

        self.use_context = use_context
        if use_context:
            self.context_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_channels, mid_channels, bias=True),
        )

    def forward(self, x, emb, context):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        if self.use_context:
            context_out = self.context_layers(context)
            h = h + emb_out + context_out
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return x + h
def Normalize(in_channels, num_groups=32):
    num_groups = in_channels // 32 # adaptive number of groups
    if num_groups == 0: # avoid zero division
        num_groups = 1 
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
class ConvResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                dropout, kernel_size=3, res_first:bool = False):
        super().__init__()
        self.in_channels = in_channels
        padding = kernel_size // 2
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.act = nn.SiLU()
        self.norm1 = Normalize(in_channels)
        conv1_channel = in_channels if res_first else out_channels
        self.conv1 = torch.nn.Conv2d(in_channels,
                                    conv1_channel,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding)
        self.norm2 = Normalize(conv1_channel)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(conv1_channel,
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
            self.up.append(ConvResnetBlock(in_channels=cur_dim, out_channels=next_dim, dropout=0.0, kernel_size=kernel_size,res_first=False))
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