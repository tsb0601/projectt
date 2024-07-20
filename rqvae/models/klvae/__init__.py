from diffusers import AutoencoderKL
from ..interfaces import *
class Stage1_KLVAE(Stage1Model):
    def __init__(self, ckpt_path:str):
        super().__init__()
        #ckpt_path = f'stabilityai/sd-vae-ft-{vae_type}'
        vae = AutoencoderKL.from_pretrained(ckpt_path)
        vae: AutoencoderKL
        self.vae = vae
    def forward(self, inputs: LabeledImageData):
        x = inputs.img
        x = x.mul_(2).sub_(1)
        x = self.vae(x).sample
        x = x.mul_(0.5).add_(0.5)
        return Stage1ModelOutput(xs_recon=x, additional_attr={})
    def encode(self, inputs: LabeledImageData):
        x = inputs.img
        x = x.mul_(2).sub_(1)
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        return Stage1Encodings(zs=x, additional_attr={})
    def decode(self, outputs: Union[Stage1Encodings,Stage2ModelOutput]):
        z = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred
        z = z.div_(0.18215)
        x = self.vae.decode(z).sample
        x = x.mul_(0.5).add_(0.5)
        return Stage1ModelOutput(xs_recon=x, additional_attr={})
    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError('KLVAE does not support training')
    @torch.no_grad()
    def infer(self, inputs: LabeledImageData):
        return self(inputs)
    def get_recon_imgs(self, x: torch.Tensor, xs: torch.Tensor):
        return x.clamp(0, 1), xs.clamp(0, 1)
    
    
class Stage1_KLVAE_ForProbing(nn.Module):
    def __init__(self, ckpt_path:str, latent_size:int, global_pool:bool = False): # global pool only
        super().__init__()
        vae = AutoencoderKL.from_pretrained(ckpt_path)
        vae: AutoencoderKL
        self.vae = vae
        self.hidden_size = latent_size * latent_size * vae.config.latent_channels
        self.fc_norm = nn.LayerNorm(self.hidden_size)
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = xs.mul_(2).sub_(1)
        xs = self.vae.encode(xs).latent_dist.sample().mul_(0.18215)
        latent = xs.reshape(xs.shape[0], -1)
        latent = self.fc_norm(latent)
        return latent