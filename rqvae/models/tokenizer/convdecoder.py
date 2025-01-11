import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return self.shortcut(residual) + x


class AttnBlock(nn.Module):
    """
    A simple Self-Attention block for a [B, C, H, W] feature map.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, C, H, W]
        residual = x
        x = self.norm(x)

        q = self.q(x)  # [B, C, H, W]
        k = self.k(x)  # [B, C, H, W]
        v = self.v(x)  # [B, C, H, W]

        B, C, H, W = q.shape
        
        # Reshape for batch matrix multiplication
        q = q.view(B, C, H*W)            # [B, C, HW]
        q = q.permute(0, 2, 1)          # [B, HW, C]
        k = k.view(B, C, H*W)           # [B, C, HW]

        # Compute attention map
        attn = torch.bmm(q, k)          # [B, HW, HW]
        attn = attn * (C**-0.5)         # scale by sqrt(C)
        attn = F.softmax(attn, dim=2)   # [B, HW, HW]

        # Apply to values
        v = v.view(B, C, H*W)           # [B, C, HW]
        attn = attn.permute(0, 2, 1)    # [B, HW, HW], swap to match v shape
        x = torch.bmm(v, attn)          # [B, C, HW]
        x = x.view(B, C, H, W)          # [B, C, H, W]

        x = self.proj_out(x)
        return residual + x


class ConvDecoder(nn.Module):
    """
    A 'VAE-like' decoder that:
      - Takes an input of shape [B, N, D], where N=H*W (square number of tokens),
        and rearranges to [B, D, H, W].
      - Maps D to 'hidden_channels' with a conv layer.
      - Upsamples from (H, W) to (256, 256) through a series of ResNet & Attn blocks.
    """
    def __init__(
        self,
        in_channels,        # e.g. D dimension (number of features per token)
        out_resolution=256, # final image resolution
        num_tokens=256,     # default tokens => 16x16
        hidden_channels=128 # base channel count
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.out_resolution = out_resolution
        self.hidden_channels = hidden_channels

        # We'll assume num_tokens = H*W is a perfect square
        self.init_hw = int(math.sqrt(num_tokens))  # e.g. sqrt(256) = 16
        assert self.init_hw * self.init_hw == num_tokens, "num_tokens must be a perfect square"

        # 1) Conv that maps [B, in_channels, H, W] -> [B, hidden_channels, H, W]
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # 2) Middle block: ResNet + Attn + ResNet
        self.mid = nn.ModuleList([
            ResnetBlock(hidden_channels, hidden_channels),
            AttnBlock(hidden_channels),
            ResnetBlock(hidden_channels, hidden_channels)
        ])

        # We want to upsample from (init_hw) to 256.
        # For 16 -> 256, we need 4x upsampling in total (16->32->64->128->256),
        # so let's define 4 "up blocks".
        self.up_blocks = nn.ModuleList()

        # Let's define a helper function to keep track of how many times we've upsampled
        # so far. We'll do:
        #   16 -> 32 -> 64 -> 128 -> 256
        # That is 4 times doubling. If your init_hw differs, you can tweak these blocks.
        current_hw = self.init_hw
        target_hw = self.out_resolution

        # We'll keep halving the "hidden_channels" as we go, or you can set your own schedule
        ch = hidden_channels

        # Keep doubling until we reach the out_resolution
        while current_hw < target_hw:
            next_hw = current_hw * 2
            # If we do a standard stable-diffusion approach, we might do something like:
            out_ch = max(ch // 2, 32)  # don't go too low
            block = nn.ModuleList([
                ResnetBlock(ch, ch),
                ResnetBlock(ch, ch),
                AttnBlock(ch),                # optional
                nn.Upsample(scale_factor=2, mode='nearest'),
                # Then maybe map channels down after upsample
                ResnetBlock(ch, out_ch)
            ])
            self.up_blocks.append(block)
            ch = out_ch
            current_hw = next_hw

        # 3) Final block: map 'ch' to 3 channels
        self.final = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x):
        """
        x: [B, N, D], where N = H*W (square).
        We'll rearrange into [B, D, H, W], convolve, upsample, and produce [B, 3, 256, 256].
        """
        b, n, d = x.shape
        hw = int(math.sqrt(n))
        # Reshape tokens to 2D
        # x -> [B, D, H, W]
        x = rearrange(x, 'b (h w) d -> b d h w', h=hw, w=hw)

        # 1) Initial conv
        x = self.conv_in(x)  # [B, hidden_channels, H, W]

        # 2) Middle block
        for block in self.mid:
            x = block(x)

        # 3) Up blocks
        for blocklist in self.up_blocks:
            for blk in blocklist:
                x = blk(x)

        # 4) Final
        for blk in self.final:
            x = blk(x)

        # Should be [B, 3, 256, 256]
        return x

