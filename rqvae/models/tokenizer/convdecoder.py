import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return self.shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) 
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class ConvDecoder(nn.Module):
    def __init__(
        self,
        input_dim=1152,          # SigLIP hidden dimension
        latent_channels=4,       # SD uses 4 channels in latent space
        hidden_channels=128,     # Base channel count
        output_resolution=256    # Output image size
    ):
        super().__init__()
        self.output_resolution = output_resolution
        
        # Calculate initial spatial dimensions (same as SD - start from 32x32)
        self.initial_height = output_resolution // 8
        self.initial_width = output_resolution // 8
        
        # Project from flat latent to spatial latent
        self.proj_in = nn.Linear(input_dim, latent_channels * self.initial_height * self.initial_width)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)

        # Middle block (same as SD)
        self.mid = nn.ModuleList([
            ResnetBlock(hidden_channels, hidden_channels),
            AttnBlock(hidden_channels),
            ResnetBlock(hidden_channels, hidden_channels)
        ])

        # Decoder blocks
        self.up = nn.ModuleList([
            # Block 1: hidden_channels -> hidden_channels
            nn.ModuleList([
                ResnetBlock(hidden_channels, hidden_channels),
                ResnetBlock(hidden_channels, hidden_channels),
                ResnetBlock(hidden_channels, hidden_channels),
                AttnBlock(hidden_channels),
                nn.Upsample(scale_factor=2, mode="nearest")
            ]),
            # Block 2: hidden_channels -> hidden_channels//2
            nn.ModuleList([
                ResnetBlock(hidden_channels, hidden_channels//2),
                ResnetBlock(hidden_channels//2, hidden_channels//2),
                ResnetBlock(hidden_channels//2, hidden_channels//2),
                AttnBlock(hidden_channels//2),
                nn.Upsample(scale_factor=2, mode="nearest")
            ]),
            # Block 3: hidden_channels//2 -> hidden_channels//4
            nn.ModuleList([
                ResnetBlock(hidden_channels//2, hidden_channels//4),
                ResnetBlock(hidden_channels//4, hidden_channels//4),
                ResnetBlock(hidden_channels//4, hidden_channels//4),
                nn.Upsample(scale_factor=2, mode="nearest")
            ])
        ])

        # Final block
        self.final = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels//4, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels//4, 3, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        batch_size = x.shape[0]
        
        # We probably want to aggregate across the sequence dimension before projecting
        # Could use mean, max, or other aggregation
        x = x.mean(dim=1)  # Shape: [batch, embed_dim]
        
        # Project to spatial latents
        x = self.proj_in(x)  # Shape: [batch, spatial_dims]
        x = x.view(batch_size, 4, self.initial_height, self.initial_width)
        
        # Rest of processing remains the same...
        x = self.conv_in(x)
        
        for block in self.mid:
            x = block(x)

        for up_block in self.up:
            for block in up_block:
                x = block(x)

        for block in self.final:
            x = block(x)

        return x  # Shape will be [batch, 3, 256, 256]