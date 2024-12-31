import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import AutoModel
import numpy as np
from typing import Optional, List, Dict, Tuple
from rqvae.models.tokenizer.diffaugment import DiffAugment

class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)

class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x.view(shape)

def make_conv_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        self.conv = make_conv_block(channels, kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)

class DiscriminatorHead(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, num_blocks: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Initial projection if needed
        self.proj = nn.Sequential(
            make_conv_block(in_channels, kernel_size=1),
            *[ResidualBlock(in_channels) for _ in range(num_blocks)]
        )
        
        # Conditional projection
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(0.2, True)
        )
        
        # Final classification
        self.classifier = SpectralConv1d(in_channels, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj(x)
        
        if cond is not None:
            # Project condition and apply as channel-wise modulation
            c = self.embed_proj(cond).unsqueeze(-1)
            x = x * c
            
        return self.classifier(x)

class DINOv2Discriminator(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        hooks: List[int] = [5, 11, 17, 23],
        img_size: int = 256,
        use_augment: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.hooks = hooks
        self.img_size = img_size
        self.use_augment = use_augment
        self.n_heads = len(hooks)
        
        # Load DINOv2
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        
        # Get model dimensions
        self.hidden_dim = self.backbone.config.hidden_size
        # print("hidden dim is:", self.hidden_dim)
        
        # Register hooks for feature extraction
        self.features = {}
        self._register_hooks()
        
        # Create discriminator heads
        self.disc_heads = nn.ModuleList([
            DiscriminatorHead(self.hidden_dim, 1152)
            for _ in range(self.n_heads)
        ])
        
    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
            
        for i, block_idx in enumerate(self.hooks):
            self.backbone.encoder.layer[block_idx].register_forward_hook(
                get_hook(f"layer_{i}")
            )
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DiffAugment to input images during training"""
        if self.training and self.use_augment:
            # DiffAugment expects input in range [-1, 1]
            if x.min() < -1 or x.max() > 1:
                x = (x * 2) - 1
            x = DiffAugment(x, policy='color,translation,cutout', channels_first=True)
        return x
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Clear previous features
        self.features.clear()
        
        # Ensure input is properly scaled
        if x.min() < -1 or x.max() > 1:
            x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # DINOv2 forward pass
        self.backbone(x)
        return self.features
    
    def forward(
        self, 
        x: torch.Tensor, 
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Apply augmentation
        x = self.augment(x)
        
        print("buggggggggggggggggg")
        # Extract features from different layers
        features = self.extract_features(x)
        
        
        print("features", features)
        print("in for looooooop")
        # Apply discriminator heads
        logits = []
        for i, head in enumerate(self.disc_heads):
            

            feat = features[f"layer_{i}"]
            if isinstance(feat, tuple):
                feat = feat[0]

            print("This is, ", i, feat.shape)

            # DINOv2 output is [batch, seq_len, channels]
            # Need to convert to [batch, channels, seq_len] for Conv1d
            if len(feat.shape) == 3:
                b, seq_len, c = feat.shape
                feat = feat.transpose(1, 2)
            elif len(feat.shape) == 4:
                b, c, h, w = feat.shape
                feat = feat.view(b, c, -1)
                
            logits.append(head(feat, cond))
        
        # Combine predictions from all heads
        return torch.cat(logits, dim=1).mean(dim=1)

# Convenience function for creating the discriminator
def create_dinov2_discriminator(
    model_size: str = "large",
    img_size: int = 256,
    use_augment: bool = True
) -> DINOv2Discriminator:
    """
    Create a DINOv2 discriminator with specified configuration
    
    Args:
        model_size: One of ['small', 'base', 'large', 'giant']
        img_size: Input image size
        use_augment: Whether to use augmentation during training
    """
    model_name = f"facebook/dinov2-{model_size}"
    return DINOv2Discriminator(
        model_name=model_name,
        img_size=img_size,
        use_augment=use_augment
    )