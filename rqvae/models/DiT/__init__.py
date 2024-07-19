from .models import DiT
import torch
from ..interfaces import *
from .diffusion import create_diffusion
from typing import List, Optional
import torch_xla.core.xla_model as xm
class DiT_Stage2(Stage2Model):
    def __init__(self,hidden_size:int, input_size:int, num_classes:int, depth:int, **kwargs):
        super().__init__()
        self.model = DiT(num_classes=num_classes, input_size=input_size, hidden_size = hidden_size, depth=depth, **kwargs) 
        # like DiT we only support square images
        self.diffusion = create_diffusion(timestep_respacing="") # like DiT we set default 1000 timesteps
        self.model.requires_grad_(True)
        self.input_size = input_size
        self.cfg = kwargs.get("cfg", .0)
        self.use_cfg = self.cfg > 1.
        self.n_samples = kwargs.get("n_samples", 1)
    def forward(self, stage1_encodings: Stage1Encodings, inputs: LabeledImageData
    ) -> Stage2ModelOutput:
        zs = stage1_encodings.zs
        return Stage2ModelOutput(
            zs_pred = zs,
            zs_degraded= zs, # no degradation this time 
            additional_attr = {}
        )
    def compute_loss(self, stage1_encodings: Stage1Encodings, stage2_output: Stage2ModelOutput, inputs: LabeledImageData, valid:bool = False, **kwargs
    ) -> dict:
        zs = stage1_encodings.zs
        labels = inputs.condition
        t = torch.randint(0, self.diffusion.num_timesteps, (zs.shape[0],), device=zs.device)
        model_kwargs = dict(y=labels)
        loss_dict = self.diffusion.training_losses(self.model, zs, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        return {
            "loss_total": loss,
        }
    def get_recon_imgs(self, xs_real, xs, **kwargs):
        return xs_real.clamp(0, 1), xs.clamp(0, 1)
    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage2ModelOutput:
        device = xm.xla_device() # default to TPU
        labels = inputs.condition
        n = self.n_samples
        cfg = self.cfg
        if labels is None:
            labels = torch.randint(0, self.model.num_classes, (self.n_samples,), device=device)
        else:
            n = labels.shape[0]
        y = labels
        z = torch.randn(n, self.model.in_channels, self.input_size, self.input_size, device=device)
        if self.use_cfg: # this means we use cfg
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
        if self.use_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return Stage2ModelOutput(
            zs_pred = samples,
            zs_degraded = None,
            additional_attr = {
                "labels": labels,
            }
        )
    def get_last_layer(self):
        return self.model.final_layer.linear.weight