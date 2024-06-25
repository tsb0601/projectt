import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch
from rqvae.models.interfaces import Stage1Model,Stage2Model
class dummy_model(Stage1Model):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.dummy_conv = nn.Conv2d(3,3,1)
    def forward(self, xs , *args, **kwargs):
        xs_recon = self.dummy_conv(xs)
        return xs_recon, 0.114
    
    def encode(self, xs, *args, **kwargs):
        xs_recon = self.dummy_conv(xs)
        return xs_recon, 0.114
    def decode(self, zs, *args, **kwargs):
        return zs
    def compute_loss(self, xs_recon, *args, xs ):
        loss= (xs_recon - xs).abs().mean()
        output_dict = {
            'loss_total' : loss,
            'loss_recon' : loss,
            'loss_latent': loss,
        }
        return output_dict
    def get_last_layer (self):
        return self.dummy_conv.weight
    def get_recon_imgs(self,x, xs, *args, **kwargs):
        return x, xs
class dummy_model_stage2(Stage2Model):
    def __init__(self, hidden_size:int):
        super().__init__()
        self.dummy_linear = nn.Linear(hidden_size,hidden_size)
    def forward(self, zs , *args, **kwargs):
        zs = self.dummy_linear(zs)
        return (zs, )
    def compute_loss(self, zs_output, zs , *args, **kwargs):
        loss = (zs_output - zs).square().mean()
        return {
            'loss_total': loss,
        }
    def get_recon_imgs(self, x, xs, *args, **kwargs):
        return x.clamp(0,1), xs.clamp(0,1)