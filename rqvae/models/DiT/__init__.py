from models import DiT
import torch
from ..interfaces import Stage2Model
from diffusion import create_diffusion
from typing import List, Optional
class DiT_Stage2(Stage2Model):
    def __init__(self, latent_size:int, num_classes:int, **kwargs):
        super().__init__()
        self.model = DiT(num_classes=num_classes, input_size=latent_size, **kwargs) 
        # like DiT we only support square images
        self.diffusion = create_diffusion(timestep_respacing="") # like DiT we set default 1000 timesteps
        self.model.requires_grad_(True)
        self.latent_size = latent_size
    def forward(self, zs, labels, **kwargs):
        return zs # in this case, we don't need to do anything
    def compute_loss(self, zs_pred, zs, labels, **kwargs):
        t = torch.randint(0, self.diffusion.num_timesteps, (zs_pred.shape[0],), device=zs_pred.device)
        model_kwargs = dict(y=labels)
        loss_dict = self.diffusion.training_losses(self.model, zs_pred, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        return loss
    def get_recon_imgs(self, xs_real, xs, **kwargs):
        return xs_real.clamp(0, 1), xs.clamp(0, 1)
    @torch.no_grad()
    def infer(self, labels: Optional[List[int] ]= None, cfg: float = 0.0, n:int = 1):
        device = xm.xla_device() # default to TPU
        if labels is None:
            labels = torch.randint(0, self.model.num_classes, (n,), device=device)
        else:
            assert len(labels) == n
        z = torch.randn(n, self.model.in_channels, self.latent_size, self.latent_size, device=device)
        using_cfg = cfg > 1.0
        if using_cfg: # this means we use cfg
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * labels.shape[0], device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg)
            sample_fn = self.model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = self.model.forward
        # Sample images:
        samples = self.diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return samples