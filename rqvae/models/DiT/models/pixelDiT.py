from typing import Union
from .DiT import DiT, DiTBlock, TimestepEmbedder, GaussianFourierEmbedding, FinalLayer
import torch.nn as nn

class MultiStageDiT(nn.Module):
    def __init__(self,
        input_size :int =32,
        class_dropout_prob: float=0.1,
        learn_sigma : bool =True,
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
    def build_single_stage( # build blocks and positional encodings for a single stage
        self,
        inflated_size: int,
        patch_size: float,
        depth: int,
        width: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        pass