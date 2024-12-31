from operator import inv
import torch
import math
import torch.nn as nn
class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier Embedding for timesteps. 
    """
    embedding_size: int = 256
    scale: float = 1.0
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    def forward(self, t):
        with torch.no_grad():
            W = self.W # stop gradient manually
        t = t[:, None] * W[None, :] * 2 * torch.pi
        # Concatenate sine and cosine transformations
        t_embed =  torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        print(freqs.shape, t.shape)
        args = t[:, None].float() * freqs[None]
        print(args.shape)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
class Inv():
    def __init__(self):
        self.in_channels = 2
    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.in_channels:
            # do a unpatchify to get the image
            print('stage0 trying to patchify:', x.shape)
            need_patchify = True
            x = x.permute(0, 2, 3, 1) # to (N, H, W, C)
            out_channel = x.shape[-1]
            p = int((x.shape[-1] // self.in_channels) ** 0.5)
            assert (p**2) * self.in_channels == x.shape[-1], f'stage0 trying to patchify: p={p}, x.shape={x.shape}'
            assert len(x.shape) == 4, f'stage0 trying to patchify a 3D tensor: x.shape={x.shape}'
            h, w = x.shape[1], x.shape[2]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(shape=(x.shape[0], self.in_channels, h * p, w * p))
            print('stage0 unpatchified:', x.shape)
        if need_patchify:
            # patchify image from (N, self.in_channels, H, W) to (N, out_channel, h', w')
            x = x.reshape(shape=(x.shape[0], self.in_channels, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(x.shape[0], h, w, out_channel))
            x = x.permute(0, 3, 1, 2) # to (N, C, H, W)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_features, H, W, eps=1e-5, momentum=0.1):
        super(LayerNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

        # Initialize learnable parameters gamma and beta
        self.gamma = nn.Parameter(torch.ones(1, num_features, H, W))
        self.beta = nn.Parameter(torch.zeros(1, num_features, H, W))

        # Initialize running mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features, H, W))
        self.register_buffer('running_var', torch.ones(num_features, H, W))

    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.num_features, "Input channel dimension must match the number of features"

        if self.training:
            # Compute mean and variance along (B) dimension, keep (C, H, W) dimensions
            mean = x.mean(dim=0, keepdim=False)
            var = x.var(dim=0, keepdim=False, unbiased=False)

            # Update running estimates
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running mean and variance during evaluation
            mean = self.running_mean
            var = self.running_var

        # Normalize the input
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable parameters gamma and beta
        out = self.gamma * x_normalized + self.beta

        return out
def main():
    embedding = GaussianFourierEmbedding(768)
    #embedding = TimestepEmbedder(768)
    t = torch.arange(10)
    t_embed = embedding(t)
    print(t_embed.shape)
    print(t_embed)
def inv_main():
    inv = Inv()
    x = torch.randn(2, 512, 16 , 16) # (N, H, W, C)
    y = inv.forward(x)
    print(y.shape)
    assert torch.allclose(x, y), f'|x - y|={torch.abs(x - y).mean()}'
def ln_main():
    # Example usage
    x = torch.randn(8, 3, 32, 32)  # (B, C, H, W) input
    target_layer_norm = LayerNorm2d(num_features=3, H=32, W=32)
    output = target_layer_norm(x)

    # Show mean and std before and after normalization
    B, C, H, W = x.size()
    mean = x.mean(dim=0, keepdim=False)
    var = x.var(dim=0, keepdim=False, unbiased=False)
    print(f"Mean before normalization: {mean.mean().item():.4f}, Std before normalization: {torch.sqrt(var).mean().item():.4f}")
    print(f"Mean after normalization: {output.mean().item():.4f}, Std after normalization: {output.std().item():.4f}")
    
    print(output.shape)  # should be (8, 3, 32, 32)

if __name__ == "__main__":
    #main()
    #inv_main()
    ln_main()