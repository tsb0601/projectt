import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch
from rqvae.models.interfaces import Stage1Model
class dummy_model(Stage1Model):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.dummy_conv = nn.Conv2d(3,3,1)
    def forward(self, xs , *args, **kwargs):
        xs_recon = self.dummy_conv(xs)
        return xs_recon, 0.114
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