import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch
from rqvae.models.interfaces import Stage1Model
class dummy_loss:
    def __init__(self) -> None:
        self.fix_code = torch.Tensor([1,4,5]).long()
    def get_last_layer (self):
        return nn.Identity()
class dummy_model(Stage1Model):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.dummy_conv = nn.Conv2d(3,3,1)
        self.fix_code = torch.ones((8,8,4)).long()
    def forward(self, xs , *args, **kwargs):
        xs_recon = self.dummy_conv(xs)
        # expand the code to the same batchsize as the input
        self.fix_code = self.fix_code.expand(xs.shape[0],-1,-1,-1)
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