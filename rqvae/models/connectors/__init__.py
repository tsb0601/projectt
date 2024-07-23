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
    def __init__(self):
        super().__init__()
        #self.forward_norm = nn.BatchNorm1d(hidden_size, affine=False)
        #self.reverse_norm = nn.BatchNorm1d(hidden_size, affine=True)
    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings:
        zs = encodings.zs # zs : [batch_size, num_patches + 1, hidden_size]
        # remove cls
        zs = zs[:,1:]
        # reshape it to square
        batch_size, num_patches, hidden_size = zs.shape
        patch_size = int(num_patches ** 0.5)
        zs = zs.view(batch_size, patch_size, patch_size, hidden_size)
        #zs = self.forward_norm(zs)
        # channel goes first
        zs = zs.permute(0,3,1,2).contiguous() # [batch_size, hidden_size, patch_size, patch_size]
        #zs = zs.mul_(0.08838)
        return Stage1Encodings(zs=zs, additional_attr=encodings.additional_attr)
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings:
        zs = encodings.zs if isinstance(encodings, Stage1Encodings) else encodings.zs_pred
        #zs = zs.div_(0.08838)
        if len(zs.shape) == 4:
            # been reshaped, we reshape them back & add a zero cls token
            batch_size, hidden_size, patch_size, _ = zs.shape
            num_patches = patch_size ** 2
            zs = zs.permute(0,2,3,1).contiguous().view(batch_size, num_patches, hidden_size)
            empty_cls = torch.zeros(batch_size, 1, hidden_size, device=zs.device).to(zs.dtype)
            zs = torch.cat([empty_cls, zs], dim=1)
        additional_attr = encodings.additional_attr if isinstance(encodings, Stage1Encodings) else update_additional_attr(encodings.additional_attr, {'zs_degraded': encodings.zs_degraded})
        stage1_encodings = Stage1Encodings(zs=zs, additional_attr=additional_attr)
        return stage1_encodings