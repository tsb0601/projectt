import torch
from torch import nn
from .compress.blocks import SimpleConv, SimpleMLP
from ..interfaces import *
"""
all connectors should accept a Stage1ModelEncoding and return a Stage1ModelEncoding
"""
def update_additional_attr(additional_attr: Optional[dict], new_pairs: dict):
    if additional_attr is None:
        additional_attr = {}
    additional_attr.update(new_pairs)
    return additional_attr
class id_connector(base_connector):
    def __init__(self):
        super().__init__()

    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings:
        return encodings # do nothing
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings:
        if isinstance(encodings, Stage1Encodings):
            stage1_encodings = encodings
        elif isinstance(encodings, Stage2ModelOutput):
            stage1_encodings = Stage1Encodings(
                zs = encodings.zs_pred,
                additional_attr = update_additional_attr(encodings.additional_attr, {'zs_degraded': encodings.zs_degraded})
            )
        return stage1_encodings
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
def P_to_P(zs:torch.Tensor, split:float = 1)-> torch.Tensor:
    """
    zs: [batch_size, hidden_size, patch_size, patch_size]
    return: [batch_size, hidden_size//split, sqrt(patch_size*split), sqrt(patch_size*split)]
    """
    batch_size, hidden_size, pn, _ = zs.shape
    sqrt_split = int(split ** 0.5)
    split_c = int( hidden_size//split )
    split_pn = pn * sqrt_split
    # reshape to bsz, split_c, split_pn, split_pn
    # first split to split_c, sqrt_split, sqrt_split, pn, pn
    zs = zs.view(batch_size, split_c, sqrt_split, sqrt_split, pn, pn)
    # then permute to split_c, split_pn, sqrt_split, split_pn, sqrt_split
    zs = zs.permute(0,1,4,2,5,3).contiguous()
    # then reshape to bsz, hidden_size, split_pn, split_pn
    zs = zs.view(batch_size, split_c, split_pn, split_pn)
    return zs.contiguous()
class ReshapeAndSplit_connector(base_connector):
    def __init__(self, split:int = 1, remove_cls:bool = True):
        super().__init__()
        #self.forward_norm = nn.BatchNorm1d(hidden_size, affine=False)
        #self.reverse_norm = nn.BatchNorm1d(hidden_size, affine=True)
        self.split = split
        self.remove_cls = remove_cls
        assert int(split**0.5) == split**0.5, 'split should be a square number'
    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings:
        zs = encodings.zs # zs : [batch_size, num_patches + 1, hidden_size]
        # remove cls
        if self.remove_cls:
            zs = zs[:,1:]
        zs = L_to_P(zs, self.split)
        """# reshape it to square
        batch_size, num_patches, hidden_size = zs.shape
        pn = int(num_patches ** 0.5)
        zs = zs.view(batch_size, pn, pn, hidden_size)
        #zs = self.forward_norm(zs)
        # channel goes first
        zs = zs.permute(0,3,1,2).contiguous() # [batch_size, hidden_size, patch_size, patch_size]
        sqrt_split = int(self.split ** 0.5)
        split_c = hidden_size // self.split
        split_pn = pn * sqrt_split
        # reshape to bsz, split_c, split_pn, split_pn
        # first split to split_c, sqrt_split, sqrt_split, pn, pn
        zs = zs.view(batch_size, split_c, sqrt_split, sqrt_split, pn, pn)
        # then permute to split_c, split_pn, sqrt_split, split_pn, sqrt_split
        zs = zs.permute(0,1,2,4,3,5).contiguous()
        # then reshape to bsz, hidden_size, split_pn, split_pn
        zs = zs.view(batch_size, split_c, split_pn, split_pn)"""
        return Stage1Encodings(zs=zs, additional_attr=encodings.additional_attr)
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings:
        zs = encodings.zs if isinstance(encodings, Stage1Encodings) else encodings.zs_pred
        if len(zs.shape) == 4:
            # been reshaped, we reshape them back & add a zero cls token
            """batch_size, c , pn, pn = zs.shape
            aggregated_c = c * self.split
            sqrt_split = int(self.split ** 0.5)
            split_pn = pn // sqrt_split
            zs = zs.view(batch_size, c, sqrt_split, split_pn, sqrt_split, split_pn)
            zs = zs.permute(0,1,2,4,3,5).contiguous()            
            zs = zs.view(batch_size, aggregated_c, split_pn, split_pn)
            zs = zs.view(batch_size, aggregated_c, split_pn*split_pn).permute(0,2,1).contiguous()"""
            zs = P_to_L(zs, self.split)
            empty_cls = torch.zeros(zs.shape[0], 1, zs.shape[2], device=zs.device).to(zs.dtype)
            zs = torch.cat([empty_cls, zs], dim=1)
        else:
            raise ValueError('reverse input should be a 4D tensor')
        additional_attr = encodings.additional_attr if isinstance(encodings, Stage1Encodings) else update_additional_attr(encodings.additional_attr, {'zs_degraded': encodings.zs_degraded})
        stage1_encodings = Stage1Encodings(zs=zs, additional_attr=additional_attr)
        return stage1_encodings

class Downsample_with_MLP_Connector(base_connector):
    def __init__(self, split:int = 1, hidden_size:int = 512, layers:int = 1,  bottleneck_ratio: float = 16.0, remove_cls: bool = True, patch_as_input: bool = False):
        super().__init__()
        self.split = split
        self.layers = layers
        self.hidden_size = hidden_size
        self.mlp = SimpleMLP(hidden_size, layers, bottleneck_ratio)
        self.remove_cls = remove_cls
        self.patch_as_input = patch_as_input
        assert patch_as_input & remove_cls == False, 'patch_as_input and remove_cls should not be true at the same time'
    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings:
        zs = encodings.zs # bsz, seq_len, hidden_size
        # remove cls 
        if self.patch_as_input:
            zs = P_to_L(zs, 1)
        if self.remove_cls:
            zs = zs[:,1:]
        zs = self.mlp.encode(zs) 
        zs = L_to_P(zs, self.split)
        return Stage1Encodings(zs=zs, additional_attr=encodings.additional_attr)
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings:
        zs = encodings.zs if isinstance(encodings, Stage1Encodings) else encodings.zs_pred
        if len(zs.shape) == 4:
            zs = P_to_L(zs, self.split)
            zs = self.mlp.decode(zs)
            if self.remove_cls:    
                empty_cls = torch.zeros(zs.shape[0], 1, zs.shape[2], device=zs.device).to(zs.dtype)
                zs = torch.cat([empty_cls, zs], dim=1)
            if self.patch_as_input:
                zs = L_to_P(zs, 1)
        else:
            raise ValueError('reverse input zs should a 4-d tensor')
        additional_attr = encodings.additional_attr if isinstance(encodings, Stage1Encodings) else update_additional_attr(encodings.additional_attr, {'zs_degraded': encodings.zs_degraded})
        stage1_encodings = Stage1Encodings(zs=zs, additional_attr=additional_attr)
        return stage1_encodings

class Downsample_with_Conv_Connector(base_connector):
    def __init__(self, split:int = 1, hidden_size:int = 512, layers:int = 1,  bottleneck_ratio: float = 16.0, remove_cls: bool = True, patch_as_input: bool = False, kernel_size:int = 3, final_norm: bool = False):
        super().__init__()
        self.split = split
        self.layers = layers
        self.hidden_size = hidden_size
        self.conv = SimpleConv(hidden_size, layers, bottleneck_ratio, kernel_size, final_norm)
        self.remove_cls = remove_cls
        self.patch_as_input = patch_as_input
        assert patch_as_input & remove_cls == False, 'patch_as_input and remove_cls should not be true at the same time'
    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings:
        zs = encodings.zs # bsz, seq_len, hidden_size
        # remove cls 
        if self.remove_cls:
            zs = zs[:,1:]
        if not self.patch_as_input:
            zs = L_to_P(zs, self.split)
        zs = self.conv.encode(zs) 
        return Stage1Encodings(zs=zs, additional_attr=encodings.additional_attr)
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings:
        zs = encodings.zs if isinstance(encodings, Stage1Encodings) else encodings.zs_pred
        if len(zs.shape) == 4:
            zs = self.conv.decode(zs)
            if not self.patch_as_input:
                zs = P_to_L(zs, self.split)
            if self.remove_cls:    
                empty_cls = torch.zeros(zs.shape[0], 1, zs.shape[2], device=zs.device).to(zs.dtype)
                zs = torch.cat([empty_cls, zs], dim=1)
        else:
            raise ValueError('reverse input should be a 4D tensor')
        additional_attr = encodings.additional_attr if isinstance(encodings, Stage1Encodings) else update_additional_attr(encodings.additional_attr, {'zs_degraded': encodings.zs_degraded})
        stage1_encodings = Stage1Encodings(zs=zs, additional_attr=additional_attr)
        return stage1_encodings