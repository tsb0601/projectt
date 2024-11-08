from sqlite3 import Time
from typing import Union
from .DiT import DiTBlock, TimestepEmbedder, GaussianFourierEmbedding, PatchEmbed, LabelEmbedder, get_2d_sincos_pos_embed
import torch.nn as nn
import torch
from transformers import VitDetModel
def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of query and key sizes.

    Args:
        q_size (`int`):
            Size of query q.
        k_size (`int`):
            Size of key k.
        rel_pos (`torch.Tensor`):
            Relative position embeddings (num_embeddings, num_channels).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel position embeddings.
        rel_pos_resized = nn.functional.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_relative_positions(attn, queries, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings as introduced in
    [MViT2](https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py).

    Args:
        attn (`torch.Tensor`):
            Attention map.
        queries (`torch.Tensor`):
            Query q in the attention layer with shape (batch_size, queries_height * queries_width, num_channels).
        rel_pos_h (`torch.Tensor`):
            Relative position embeddings (Lh, num_channels) for height axis.
        rel_pos_w (`torch.Tensor`):
            Relative position embeddings (Lw, num_channels) for width axis.
        q_size (`Tuple[int]`):
            Spatial sequence size of query q with (queries_height, queries_width).
        k_size (`Tuple[int]`):
            Spatial sequence size of key k with (keys_height, keys_width).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    queries_height, queries_width = q_size
    keys_height, keys_width = k_size
    relative_height = get_rel_pos(queries_height, keys_height, rel_pos_h)
    relative_width = get_rel_pos(queries_width, keys_width, rel_pos_w)

    batch_size, _, dim = queries.shape
    r_q = queries.reshape(batch_size, queries_height, queries_width, dim)
    relative_height = torch.einsum("bhwc,hkc->bhwk", r_q, relative_height)
    relative_weight = torch.einsum("bhwc,wkc->bhwk", r_q, relative_width)

    attn = (
        attn.view(batch_size, queries_height, queries_width, keys_height, keys_width)
        + relative_height[:, :, :, :, None]
        + relative_weight[:, :, :, None, :]
    ).view(batch_size, queries_height * queries_width, keys_height * keys_width)

    return attn
class VitDetAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self,
        hidden_size: int,
        num_attention_heads: int,
        input_size: int,
        qkv_bias: bool = False,
        use_relative_position_embeddings: bool = False,
    ):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            input_size (`Tuple[int]`, *optional*):
                Input resolution, only required in case relative position embeddings are added.
        """
        super().__init__()

        dim = hidden_size
        num_heads = num_attention_heads

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_relative_position_embeddings = use_relative_position_embeddings
        if self.use_relative_position_embeddings:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, hidden_state, output_attentions=False):
        batch_size, height, width, _ = hidden_state.shape
        # qkv with shape (3, batch_size, num_heads, height * width, num_channels)
        qkv = self.qkv(hidden_state).reshape(batch_size, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # queries, keys and values have shape (batch_size * num_heads, height * width, num_channels)
        queries, keys, values = qkv.reshape(3, batch_size * self.num_heads, height * width, -1).unbind(0)

        attention_scores = (queries * self.scale) @ keys.transpose(-2, -1)

        if self.use_relative_position_embeddings:
            attention_scores = add_decomposed_relative_positions(
                attention_scores, queries, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attention_probs = attention_scores.softmax(dim=-1)

        hidden_state = attention_probs @ values
        hidden_state = hidden_state.view(batch_size, self.num_heads, height, width, -1)
        hidden_state = hidden_state.permute(0, 2, 3, 1, 4)
        hidden_state = hidden_state.reshape(batch_size, height, width, -1)
        hidden_state = self.proj(hidden_state)

        if output_attentions:
            attention_probs = attention_probs.reshape(
                batch_size, self.num_heads, attention_probs.shape[-2], attention_probs.shape[-1]
            )
            outputs = (hidden_state, attention_probs)
        else:
            outputs = (hidden_state,)

        return outputs

class MultiStageDiT(nn.Module):
    def __init__(self,
        input_size :int =32,
        class_dropout_prob: float=0.1,
        learn_sigma : bool =True,
        shared_adaln: bool = False,
        in_channels: int = 3, # number of input channels
        inflated_size: int = 256, # size of the inflated latent/image
        patch_sizes : Union[list[float], tuple[float]] = (2, 16, 2), 
        depths: Union[list[int], tuple[int]] = (2, 2, 2),
        widths: Union[list[int], tuple[int]] = (64, 1024, 64),
        num_heads: Union[list[int], tuple[int]] = (4, 16, 4),
        mlp_ratios: Union[list[float], tuple[float]] = (4.0, 4.0, 4.0),
    ):
        super().__init__()
        self.input_size = input_size
        self.class_dropout_prob = class_dropout_prob
        self.learn_sigma = learn_sigma
        self.patch_sizes = patch_sizes
        self.depths = depths
        self.widths = widths
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.inflated_size = inflated_size
        self.shared_adaln = shared_adaln
        self.in_channels = in_channels
    @staticmethod
    def adaln_zero(hidden_size: int ):
        """
        return a zero-initialized AdaLN layer
        """
        return nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    def build_single_stage( # build blocks and positional encodings for a single stage
        self,
        input_size: int,
        patch_size: float,
        input_token_dimension: int,
        depth: int,
        width: int,
        num_heads: int,
        mlp_ratio: float,
        num_classes: int,
        class_dropout_prob: float,
    ):
        x_embedder = PatchEmbed(
            img_size= input_size,
            patch_size= patch_size,
            in_chans= input_token_dimension,
            embed_dim= width
        )
        pos_embed = nn.Parameter(torch.zeros(1, x_embedder.num_patches, width))
        t_embedder = GaussianFourierEmbedding(width)
        y_embedder = LabelEmbedder(num_classes, width, class_dropout_prob)
        adaLN = self.adaln_zero(width) if self.shared_adaln else None
        blocks = nn.ModuleList([
            DiTBlock(
                dim = width,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                adaLN = adaLN,
            ) for _ in range(depth)
        ])
        # use a linear final layer instead of a mlp
        final_layer = nn.Linear(
            in_features = width,
            out_features = patch_size * patch_size * input_token_dimension,
            bias = True
        )
        stage = nn.ModuleList([x_embedder, pos_embed, t_embedder, y_embedder, blocks, final_layer])
        self.init_single_stage(stage)
        return stage
    def unpatchify(self, x: torch.Tensor, stage: nn.ModuleList):
        """
        Unpatchify a tensor of shape (N, T, D) 
        to (N, in_channels, inflated_size, inflated_size)
        """
        x_embedder: PatchEmbed = stage[0]
        p = x_embedder.patch_size[0]
        c = self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    def init_single_stage(self, stage: nn.ModuleList):
        x_embedder, pos_embed, t_embedder, y_embedder, blocks, final_layer = stage
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for block in blocks:
            block: DiTBlock
            block.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        position_embedding = get_2d_sincos_pos_embed(pos_embed.shape[-1], int(x_embedder.num_patches ** 0.5))
        pos_embed.data.copy_(torch.from_numpy(position_embedding).float().unsqueeze(0))
        
        x_embedder: PatchEmbed
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(x_embedder.proj.bias, 0)

        y_embedder: LabelEmbedder
        # Initialize label embedding table:
        nn.init.normal_(y_embedder.embedding_table.weight, std=0.02)

        
        t_embedder: GaussianFourierEmbedding
        # Initialize timestep embedding MLP:
        nn.init.normal_(t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        final_layer: nn.Linear
        # Zero-out output layers:
        nn.init.constant_(final_layer.weight, 0)
        nn.init.constant_(final_layer.bias, 0)

    def single_stage_forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, stage: nn.ModuleList):
        x_embedder, pos_embed, t_embedder, y_embedder, blocks, final_layer = stage
        x = x_embedder(x) + pos_embed
        t = t_embedder(t)
        y = y_embedder(y)
        c = t + y 
        for block in blocks:
            x = block(x, c)
        x = final_layer(x)
        return x

    
        