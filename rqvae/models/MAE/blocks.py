from functools import partial
from torch import nn
import torch
from rqvae.models.basicblocks.utils import zero_module
from rqvae.models.basicblocks.basics import ConvResnetBlock, DCAE_ChannelDownsampleLayer, DCAE_ChannelUpsampleLayer
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
class DCAE_ChannelDownsampleLayerwReshape(DCAE_ChannelDownsampleLayer):
    def forward(self, x):
        need_transform = len(x.shape) == 3
        include_cls = False
        if need_transform:
            # [B, T, D]
            # check if CLS token is included
            if x.shape[1] % 2 == 1:
                x = x[:,1:]
                include_cls = True
            x = L_to_P(x)
        x = super().forward(x)
        if need_transform:
            x = P_to_L(x)
            if include_cls:
                x = torch.cat([torch.zeros_like(x[:,0:1]), x], dim=1)
        return x
class DCAE_ChannelUpsampleLayerwReshape(DCAE_ChannelUpsampleLayer):
    def forward(self, x):
        need_transform = len(x.shape) == 3
        include_cls = False
        if need_transform:
            # [B, T, D]
            # check if CLS token is included
            if x.shape[1] % 2 == 1:
                x = x[:,1:]
                include_cls = True
            x = L_to_P(x)
        x = super().forward(x)
        if need_transform:
            x = P_to_L(x)
            if include_cls:
                x = torch.cat([torch.zeros_like(x[:,0:1]), x], dim=1)
        return x
import torch.nn.functional as F
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    r"""
    Borrowed from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L15
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + x # remove drop path for now
        return x

class ConvNextDownSampler(nn.Module):
    """
    do DCAE downsampling with ConvNeXt block
    """
    def __init__(self, in_channels:int, out_channels:int, layer_channels:list = None, stage_depths:list = None):
        if layer_channels is None:
            layer_channels = [out_channels]
            assert stage_depths is None, "if layer_channels is not None, stage_depths must be None"
            stage_depths = [1]
        assert len(layer_channels) == len(stage_depths), "layer_channels and stage_depths must have the same length"
        super(ConvNextDownSampler, self).__init__()
        stages = len(layer_channels)
        self.stages = stages
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1) # do a small conv first
        self.down = nn.ModuleList()
        cur_dim = in_channels
        for i in range(stages):
            next_dim = layer_channels[i]
            assert cur_dim % next_dim == 0, f"input dim {cur_dim} must be divisible by next dim {next_dim} at stage {i}"
            downsample_factor = cur_dim // next_dim
            down_sample_block = DCAE_ChannelDownsampleLayer(in_channels=cur_dim, downsample_factor=downsample_factor)
            self.down.append(down_sample_block)
            cur_dim = next_dim
        self.main_stage = nn.ModuleList()
        for i in range(stages):
            dim = layer_channels[i]
            depth = stage_depths[i]
            this_stage = nn.Sequential(*[ConvNeXtBlock(dim, layer_scale_init_value=1e-6) for _ in range(depth)])
            self.main_stage.append(this_stage)
    def forward(self, x):
        need_transform = len(x.shape) == 3
        include_cls = False
        #print(f'Downsample input shape: {x.shape}')
        if need_transform:
            # [B, T, D]
            # check if CLS token is included
            if x.shape[1] % 2 == 1:
                x = x[:,1:]
                include_cls = True
            x = L_to_P(x)
        x = self.conv_in(x)
        #print(f'After conv_in: {x.shape}')
        for i in range(self.stages):
            #print(f'downsample, stage {i}: {x.shape}')
            x = self.down[i](x)
            x = self.main_stage[i](x)
        if need_transform:
            x = P_to_L(x)
            if include_cls:
                x = torch.cat([torch.zeros_like(x[:,0:1]), x], dim=1)
        return x
class ConvNextUpSampler(nn.Module):
    """
    do DCAE upsampling with ConvNeXt block
    """
    def __init__(self, in_channels:int, out_channels:int, layer_channels:list = None, stage_depths:list = None):
        if layer_channels is None:
            layer_channels = [out_channels]
            assert stage_depths is None, "if layer_channels is not None, stage_depths must be None"
            stage_depths = [1]
        assert layer_channels[-1] == out_channels, "last element of layer_channels must be equal to out_channels"
        assert len(layer_channels) - 1 == len(stage_depths), "layer_channels must have one more element than stage_depths"
        super(ConvNextUpSampler, self).__init__()
        stages = len(layer_channels)
        self.stages = stages
        self.up = nn.ModuleList()
        cur_dim = in_channels
        for i in range(stages):
            next_dim = layer_channels[i]
            assert next_dim % cur_dim == 0, f"input dim {next_dim} must be divisible by next dim {cur_dim} at stage {i}"
            upsample_factor = next_dim // cur_dim
            up_sample_block = DCAE_ChannelUpsampleLayer(in_channels=cur_dim, upsample_factor=upsample_factor)
            self.up.append(up_sample_block)
            cur_dim = next_dim
        self.main_stage = nn.ModuleList()
        for i in range(stages - 1):
            dim = layer_channels[i]
            depth = stage_depths[i]
            this_stage = nn.Sequential(*[ConvNeXtBlock(dim, layer_scale_init_value=1e-6) for _ in range(depth)])
            self.main_stage.append(this_stage)
        self.conv_out = zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        need_transform = len(x.shape) == 3
        include_cls = False
        #print(f'Upsample input shape: {x.shape}')
        if need_transform:
            # [B, T, D]
            # check if CLS token is included
            if x.shape[1] % 2 == 1:
                x = x[:,1:]
                include_cls = True
            x = L_to_P(x)
        for i in range(self.stages - 1):
            #print(f'upsample, stage {i}: {x.shape}')
            x = self.up[i](x)
            x = self.main_stage[i](x)
        x = self.up[-1](x)
        x = self.conv_out(x)
        #print(f'Upsample output shape: {x.shape}')
        if need_transform:
            x = P_to_L(x)
            if include_cls:
                x = torch.cat([torch.zeros_like(x[:,0:1]), x], dim=1)
        return x
class ConvNextMidBlocks(nn.Module):
    """
    stack a few ConvNeXt blocks
    """
    def __init__(self, in_channels:int, depth:int):
        super(ConvNextMidBlocks, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ConvNeXtBlock(in_channels))
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x