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
def main():
    embedding = GaussianFourierEmbedding(768)
    #embedding = TimestepEmbedder(768)
    t = torch.arange(10)
    t_embed = embedding(t)
    print(t_embed.shape)
    print(t_embed)
    
if __name__ == "__main__":
    main()