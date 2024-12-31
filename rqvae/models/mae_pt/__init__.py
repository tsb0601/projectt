import rqvae.models.mae_pt.models_vit as models_vit
import torch
from rqvae.models.interfaces import Stage1Model, Stage1Encodings, Stage1ModelOutput, Stage2ModelOutput
import torch_xla.core.xla_model as xm
from torch import nn
from header import *
class Stage1MAE_For_Probing_PT(nn.Module):
    def __init__(self, model_type, nb_classes, global_pool: bool,ckpt_path:str)->None:
        super().__init__()
        assert model_type in ['vit_base_patch16','vit_large_patch16', 'vit_huge_patch14'], 'model type should be either vit_base_patch16 or vit_large_patch16'
        model = models_vit.__dict__[model_type](
            num_classes=nb_classes,
            global_pool=global_pool,
        )
        self.global_pool = global_pool
        self.num_classes = nb_classes
        self.model:models_vit.VisionTransformer = model
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        keys = self.model.load_state_dict(ckpt['model'], strict=False)
        if global_pool:
            assert set(keys.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(keys.missing_keys) == {'head.weight', 'head.bias'}
        model.head = nn.Identity()
    def forward(self, xs:torch.Tensor)-> Stage1ModelOutput:
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        prob_latent = self.model(xs)
        return prob_latent
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)
