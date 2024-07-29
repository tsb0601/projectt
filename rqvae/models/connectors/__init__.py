from numpy import empty
import torch
from torch import nn
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
class MAE_Diffusion_connector(base_connector):
    def __init__(self, split:int = 1):
        super().__init__()
        #self.forward_norm = nn.BatchNorm1d(hidden_size, affine=False)
        #self.reverse_norm = nn.BatchNorm1d(hidden_size, affine=True)
        self.split = split
        assert int(split**0.5) == split**0.5, 'split should be a square number'
    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings:
        zs = encodings.zs # zs : [batch_size, num_patches + 1, hidden_size]
        # remove cls
        zs = zs[:,1:]
        # reshape it to square
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
        zs = zs.view(batch_size, split_c, split_pn, split_pn)
        return Stage1Encodings(zs=zs, additional_attr=encodings.additional_attr)
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings:
        zs = encodings.zs if isinstance(encodings, Stage1Encodings) else encodings.zs_pred
        if len(zs.shape) == 4:
            # been reshaped, we reshape them back & add a zero cls token
            batch_size, c , pn, pn = zs.shape
            aggregated_c = c * self.split
            sqrt_split = int(self.split ** 0.5)
            split_pn = pn // sqrt_split
            zs = zs.view(batch_size, c, sqrt_split, split_pn, sqrt_split, split_pn)
            zs = zs.permute(0,1,2,4,3,5).contiguous()            
            zs = zs.view(batch_size, aggregated_c, split_pn, split_pn)
            zs = zs.view(batch_size, aggregated_c, split_pn*split_pn).permute(0,2,1).contiguous()
            empty_cls = torch.zeros(batch_size, 1, aggregated_c, device=zs.device).to(zs.dtype)
            zs = torch.cat([empty_cls, zs], dim=1)
        additional_attr = encodings.additional_attr if isinstance(encodings, Stage1Encodings) else update_additional_attr(encodings.additional_attr, {'zs_degraded': encodings.zs_degraded})
        stage1_encodings = Stage1Encodings(zs=zs, additional_attr=additional_attr)
        return stage1_encodings