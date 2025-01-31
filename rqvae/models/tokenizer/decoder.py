import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        # x shape: [B, N, D]
        x_norm = self.norm1(x)
        # Transpose for attention
        x_t = x_norm.transpose(0, 1)
        attn_output, _ = self.attn(x_t, x_t, x_t)
        # Transpose back
        attn_output = attn_output.transpose(0, 1)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

class VAEDecoder(nn.Module):
    def __init__(
        self,
        input_dim=1152,
        num_tokens=256,
        output_resolution=256,
        patch_size=16,
        size="large",  # "large", "huge", "giant"
        decoder_dim=None,
        num_layers=None,
        num_heads=None,
        mlp_ratio=None,
    ):
        super().__init__()
        # Size configuration presets
        size_configs = {
            "large": {
                "decoder_dim": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "mlp_ratio": 4.0
            },
            "huge": {
                "decoder_dim": 1280,
                "num_layers": 32,
                "num_heads": 16,
                "mlp_ratio": 4.0
            },
            "giant": {
                "decoder_dim": 1536,
                "num_layers": 48,
                "num_heads": 24,
                "mlp_ratio": 8.0
            }
        }
        
        # Get config for selected size
        config = size_configs[size]
        
        # Set parameters (use explicit args if provided, else use size config)
        self.decoder_dim = decoder_dim or config["decoder_dim"]
        self.num_layers = num_layers or config["num_layers"]
        self.num_heads = num_heads or config["num_heads"]
        self.mlp_ratio = mlp_ratio or config["mlp_ratio"]

        self.num_tokens = num_tokens
        self.patch_size = patch_size
        self.output_resolution = output_resolution
        self.output_tokens = (output_resolution // patch_size) ** 2

        # Projection and transformer blocks
        self.input_proj = nn.Linear(input_dim, self.decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.output_tokens, self.decoder_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.decoder_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio
            ) for _ in range(self.num_layers)
        ])

        # Final layers
        self.norm = nn.LayerNorm(self.decoder_dim)
        self.head = nn.Sequential(
            nn.Linear(self.decoder_dim, patch_size * patch_size * 3),
            nn.Tanh()
        )

    def interpolate_pos_encoding(self, pos_embed, height, width):
        pos_embed = pos_embed.reshape(1, int(self.output_tokens ** 0.5), int(self.output_tokens ** 0.5), -1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(height, width), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        pos_embed = pos_embed.flatten(1, 2)
        return pos_embed

    def forward(self, x):
        B = x.shape[0]
        
        # Project to ViT-L dimension
        x = self.input_proj(x)
        
        # Handle different token counts if needed
        if self.num_tokens != self.output_tokens:
            input_size = int(self.num_tokens ** 0.5)
            output_size = int(self.output_tokens ** 0.5)
            
            # Interpolate pos embeddings
            pos_embed = self.interpolate_pos_encoding(self.pos_embed, output_size, output_size)
            
            # Interpolate feature tokens
            x = x.transpose(1, 2)
            x = x.reshape(B, -1, input_size, input_size)
            x = F.interpolate(x, size=(output_size, output_size), mode='bilinear', align_corners=False)
            x = x.reshape(B, -1, output_size * output_size)
            x = x.transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        # Add position embeddings
        x = x + pos_embed

        # ViT-L transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final norm and projection
        x = self.norm(x)
        x = self.head(x)

        # Reshape to image
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                     h=self.output_resolution//self.patch_size,
                     w=self.output_resolution//self.patch_size,
                     p1=self.patch_size,
                     p2=self.patch_size,
                     c=3)
        
        return x