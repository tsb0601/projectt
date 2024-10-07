from ...img_datasets.interfaces import LabeledImageData
from ..interfaces import Stage1ModelOutput
from .models.autoencoder import AutoencoderKL
from ..interfaces import *
from typing import Any, Dict, Tuple

class AutoEncoderKL_Stage1(Stage1Model):
    """
    A simple wrapper for the AutoencoderKL model to be used in the Stage 1
    """
    def __init__(
        self, ddconfig: Dict[str, Any], 
        embed_dim: int, 
        kl_weight: float,
        sample: bool = True
    ):
        
        super(AutoEncoderKL_Stage1, self).__init__()
        self.model = AutoencoderKL(ddconfig, embed_dim)
        self.do_sample = sample # if not return the mean of the latent space
        self.kl_weight = kl_weight
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        xs = inputs.img
        latent = self.model.encode(xs)
        sample = latent.sample() if self.do_sample else latent.mode()
        encodings = Stage1Encodings(
            zs = sample,
            additional_attr={
                'distr': latent,
            }
        )
        return encodings
    def decode(self, outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs
        recon = self.model.decode(zs)
        return Stage1ModelOutput(recon,
                additional_attr={
                    'distr': outputs.additional_attr['distr'],
                })
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData, **kwargs) -> dict:
        img = inputs.img
        recon = outputs.xs_recon
        distr = outputs.additional_attr['distr']
        rec_loss = torch.abs(img.contiguous() - recon.contiguous())
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0] # take mean over the batch
        #rec_loss = rec_loss.mean() # we take the mean over the pixels
        kl_loss = distr.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] # take mean over the batch
        #kl_loss = kl_loss.mean() # we take the mean over the latent dimensions
        total_loss = rec_loss + self.kl_weight * kl_loss
        return {
            'loss_total': total_loss,
            'loss_latent': kl_loss,
            'loss_recon': rec_loss,
        }
    def forward(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        encodings = self.encode(inputs)
        recons = self.decode(encodings)
        return recons
    def get_last_layer(self):
        return self.model.get_last_layer()
    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        return self.forward(inputs)
    def get_recon_imgs(self, x, xs, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.clamp(0, 1), xs.clamp(0, 1)