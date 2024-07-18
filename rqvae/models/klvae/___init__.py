from diffusers import AutoencoderKL
import torch
from ..interfaces import *
class KLVAE(Stage1Model):
    def __init__(self, vae_type:str):
        super().__init__()
        ckpt_path = f'stabilityai/sd-vae-ft-{vae_type}'
        vae = AutoencoderKL.from_pretrained(ckpt_path)
        vae: AutoencoderKL
        self.vae = vae
    def forward(self, inputs: LabeledImageData):
        x = inputs.img
        x = x.mul_(2).sub_(1)
        x = self.vae(x).sample
        x = x.mul_(0.5).add_(0.5)
        return (x,)
    def encode(self, inputs: LabeledImageData):
        x = inputs.img
        x = x.mul_(2).sub_(1)
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        return (x,)
    def decode(self, outputs: Union[Stage1Encodings,Stage2ModelOutput]):
        z = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred
        z = z.div_(0.18215)
        x = self.vae.decode(z).sample
        x = x.mul_(0.5).add_(0.5)
        return (x,)
    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError, 'KLVAE does not support training'