from rqvae.models.basicblocks.utils import zero_module
from .models.DiT import *
import torch
from ..interfaces import *
from .diffusion import create_diffusion, SimpleDiffusion
from typing import List, Optional
import torch_xla.core.xla_model as xm
from .blocks import ConvEncoder, ConvDecoder, ConvDecoder_wSkipConnection
from rqvae.models.utils import get_obj_from_str
from .models.pixelDiT import MultiStageDiT
class DiT_Stage2(Stage2Model):
    def __init__(
        self,
        input_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes=1000,
        learn_sigma=True,
        timestep_respacing: str = "",
        noise_schedule: str = "linear",
        cfg: float = 0.0,
        inference_step: int = 250,
        n_samples: int = 125,
        do_beta_rescaling: bool = False,
        use_simple_diffusion: bool = False,
        use_loss_weighting: bool = False,
        use_schedule_shift: bool = False,
        gamma: float = 0.3,
        class_cls_str: str = "rqvae.models.DiT.models.DiT.DiT",
        **kwargs,
    ):
        super().__init__()
        self.timestep_respacing = str(timestep_respacing)
        self.cfg = cfg
        # learn_sigma = kwargs.pop("learn_sigma", True) # learn sigma is True by default and is a required argument in DiT
        # noise_schedule = kwargs.pop("noise_schedule", "linear")
        if do_beta_rescaling:
            base_dim = 128 * 128 * 3  # 3 following https://arxiv.org/pdf/2301.11093
            input_dim = input_size * input_size * in_channels  # input channels
            input_base_dimension_ratio = math.sqrt(base_dim / input_dim) # do rescaling
        else:
            input_base_dimension_ratio = 1.0 # do not do rescaling
        self.hidden_size = hidden_size
        DiT_model_cls:DiT = get_obj_from_str(class_cls_str)
        # assert this class inherits from DiT
        assert issubclass(DiT_model_cls, DiT), f"[!]DiT_Stage2: class_cls_str: {class_cls_str} must inherit from DiT"
        self.model = DiT_model_cls(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            **kwargs,
        )
        self.model.requires_grad_(True)
        self.inference_step = inference_step
        # like DiT we only support square images
        self.diffusion = create_diffusion(
            timestep_respacing=self.timestep_respacing,
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
            input_base_dimension_ratio=input_base_dimension_ratio,
            use_simple_diffusion=use_simple_diffusion,
            use_loss_weighting=use_loss_weighting,
            use_schedule_shift=use_schedule_shift,
            #predict_xstart= True
        )  # like DiT we set default 1000 timesteps
        self.infer_diffusion = create_diffusion(
            timestep_respacing=str(self.inference_step),
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
            input_base_dimension_ratio=input_base_dimension_ratio,
            use_simple_diffusion=use_simple_diffusion,
            use_loss_weighting=use_loss_weighting,
            use_schedule_shift=use_schedule_shift,
            #predict_xstart= True
        )
        self.use_simple_diffusion = use_simple_diffusion
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_cfg = self.cfg >= 1.0
        self.gamma = gamma # only used in simple diffusion
        self.n_samples = n_samples
        print(
            f"[!]DiT_Stage2: Using cfg: {self.use_cfg}, n_samples: {self.n_samples}, cfg: {self.cfg}, timestep_respacing: {self.timestep_respacing}, learn_sigma: {learn_sigma}", f"noise_schedule: {noise_schedule}", f"dim_ratio: {input_base_dimension_ratio}"
        )

    def forward(
        self, stage1_encodings: Stage1Encodings, inputs: LabeledImageData
    ) -> Stage2ModelOutput:
        zs = stage1_encodings.zs
        return Stage2ModelOutput(
            zs_pred=zs, zs_degraded=zs, additional_attr={}  # no degradation this time
        )

    def compute_loss(
        self,
        stage1_encodings: Stage1Encodings,
        stage2_output: Stage2ModelOutput,
        inputs: LabeledImageData,
        valid: bool = False,
        **kwargs,
    ) -> dict:
        zs = stage1_encodings.zs
        labels = inputs.condition
        if labels is None:
            # we set null labels to num_classes
            labels = torch.tensor(
                [self.num_classes] * zs.shape[0], device=zs.device
            ).long()
        t = torch.randint(
            0, self.diffusion.num_timesteps, (zs.shape[0],), device=zs.device
        ) if not self.use_simple_diffusion else torch.rand(zs.shape[0], device=zs.device) 
        model_kwargs = dict(y=labels)
        terms = self.diffusion.training_losses(self.model, zs, t, model_kwargs)
        loss = terms["loss"].mean()
        return {
            "loss_total": loss,
            "t": t,
        }

    def get_recon_imgs(self, xs_real, xs, **kwargs):
        return xs_real.clamp(0, 1), xs.clamp(0, 1)

    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage2ModelOutput:
        device = xm.xla_device()  # default to TPU
        labels = inputs.condition
        noise = None
        if isinstance(labels, torch.Tensor):
            device = labels.device  # sp hack
        if isinstance(inputs.img, torch.Tensor):
            device = inputs.img.device  # sp hack
            noise = inputs.img
        n = self.n_samples
        cfg = self.cfg
        if labels is None:
            labels = torch.randint(
                0, self.num_classes, (self.n_samples,), device=device
            )
        else:
            n = labels.shape[0]
            # labels = torch.randint(0, self.num_classes, (n,), device=device) # we still do random label sampling
        y = labels
        z = torch.randn(
            n, self.model.in_channels, self.input_size, self.input_size, device=device
        ) if noise is None else noise
        if self.use_cfg:  # this means we use cfg
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([self.num_classes] * labels.shape[0], device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg)
            sample_fn = self.model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)  # do conditional sampling
            sample_fn = self.model.forward
        simple_diffusion_kwargs = dict(gamma=self.gamma) if self.use_simple_diffusion else {}
        # Sample images:
        samples = self.infer_diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            **simple_diffusion_kwargs
        )
        if self.use_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return Stage2ModelOutput(
            zs_pred=samples,
            zs_degraded=None,
            additional_attr={
                "labels": labels,
            },
        )

    def get_last_layer(self):
        return self.model.final_layer.linear.weight

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        # self.model.pos_embed.requires_grad_(False) # always freeze positional embeddings
        


class MultiStageDiT_Stage2(Stage2Model):
    def __init__(
        self,
        input_size :int =32,
        num_classes: int = 1000,
        class_dropout_prob: float=0.1,
        learn_sigma : bool =True,
        shared_adaln: bool = False,
        in_channels: int = 3, # number of input channels
        inflated_size: int = 256, # size of the inflated latent/image
        patch_sizes : Union[list[float], tuple[float]] = (2, 16, 2), 
        depths: Union[list[int], tuple[int]] = (2, 2, 2),
        widths: Union[list[int], tuple[int]] = (64, 1024, 64),
        num_heads: Union[list[int], tuple[int]] = (4, 16, 4),
        mlp_ratios: Union[list[float], tuple[float]] = (4.0, 4.0, 4.0),
        window_sizes: Union[list[int], tuple[int]] = (16, 256, 16),
        timestep_respacing: str = "",
        noise_schedule: str = "linear",
        cfg: float = 0.0,
        inference_step: int = 250,
        n_samples: int = 125,
        do_beta_rescaling: bool = False,
        use_simple_diffusion: bool = False,
        use_loss_weighting: bool = False,
        use_schedule_shift: bool = False,
        class_cls_str: str = "rqvae.models.DiT.models.pixelDiT.MultiStageDiT",
        **kwargs,
    ):
        super().__init__()
        self.timestep_respacing = str(timestep_respacing)
        self.cfg = cfg
        # learn_sigma = kwargs.pop("learn_sigma", True) # learn sigma is True by default and is a required argument in DiT
        # noise_schedule = kwargs.pop("noise_schedule", "linear")
        if do_beta_rescaling:
            base_dim = 128 * 128 * 3  # 3 following https://arxiv.org/pdf/2301.11093
            input_dim = input_size * input_size * in_channels  # input channels
            input_base_dimension_ratio = math.sqrt(base_dim / input_dim) # do rescaling
        else:
            input_base_dimension_ratio = 1.0 # do not do rescaling
        DiT_model_cls : MultiStageDiT = get_obj_from_str(class_cls_str)
        # assert this class inherits from DiT
        assert issubclass(DiT_model_cls, MultiStageDiT), f"[!]DiT_Stage2: class_cls_str: {class_cls_str} must inherit from DiT"
        self.model = DiT_model_cls(
            input_size=input_size,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            shared_adaln=shared_adaln,
            in_channels=in_channels,
            inflated_size=inflated_size,
            patch_sizes=patch_sizes,
            depths=depths,
            widths=widths,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            window_sizes=window_sizes,
            **kwargs,
        )
        self.model.requires_grad_(True)
        self.inference_step = inference_step
        # like DiT we only support square images
        self.diffusion = create_diffusion(
            timestep_respacing=self.timestep_respacing,
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
            input_base_dimension_ratio=input_base_dimension_ratio,
            use_simple_diffusion=use_simple_diffusion,
            use_loss_weighting=use_loss_weighting,
            use_schedule_shift=use_schedule_shift,
            #predict_xstart= True
        )  # like DiT we set default 1000 timesteps
        self.infer_diffusion = create_diffusion(
            timestep_respacing=str(self.inference_step),
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
            input_base_dimension_ratio=input_base_dimension_ratio,
            use_simple_diffusion=use_simple_diffusion,
            use_loss_weighting=use_loss_weighting,
            use_schedule_shift=use_schedule_shift,
            #predict_xstart= True
        )
        self.use_simple_diffusion = use_simple_diffusion
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_cfg = self.cfg >= 1.0
        self.n_samples = n_samples
        print(
            f"[!]DiT_Stage2: Using cfg: {self.use_cfg}, n_samples: {self.n_samples}, cfg: {self.cfg}, timestep_respacing: {self.timestep_respacing}, learn_sigma: {learn_sigma}", f"noise_schedule: {noise_schedule}", f"dim_ratio: {input_base_dimension_ratio}"
        )

    def forward(
        self, stage1_encodings: Stage1Encodings, inputs: LabeledImageData
    ) -> Stage2ModelOutput:
        zs = stage1_encodings.zs
        return Stage2ModelOutput(
            zs_pred=zs, zs_degraded=zs, additional_attr={}  # no degradation this time
        )

    def compute_loss(
        self,
        stage1_encodings: Stage1Encodings,
        stage2_output: Stage2ModelOutput,
        inputs: LabeledImageData,
        valid: bool = False,
        **kwargs,
    ) -> dict:
        zs = stage1_encodings.zs
        labels = inputs.condition
        if labels is None:
            # we set null labels to num_classes
            labels = torch.tensor(
                [self.num_classes] * zs.shape[0], device=zs.device
            ).long()
        t = torch.randint(
            0, self.diffusion.num_timesteps, (zs.shape[0],), device=zs.device
        ) if not self.use_simple_diffusion else torch.rand(zs.shape[0], device=zs.device) 
        model_kwargs = dict(y=labels)
        terms = self.diffusion.training_losses(self.model, zs, t, model_kwargs)
        loss = terms["loss"].mean()
        return {
            "loss_total": loss,
            "t": t,
        }

    def get_recon_imgs(self, xs_real, xs, **kwargs):
        return xs_real.clamp(0, 1), xs.clamp(0, 1)

    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage2ModelOutput:
        device = xm.xla_device()  # default to TPU
        labels = inputs.condition
        if isinstance(labels, torch.Tensor):
            device = labels.device  # sp hack
        if isinstance(inputs.img, torch.Tensor):
            device = inputs.img.device  # sp hack
        n = self.n_samples
        cfg = self.cfg
        if labels is None:
            labels = torch.randint(
                0, self.num_classes, (self.n_samples,), device=device
            )
        else:
            n = labels.shape[0]
            # labels = torch.randint(0, self.num_classes, (n,), device=device) # we still do random label sampling
        y = labels
        z = torch.randn(
            n, self.model.in_channels, self.input_size, self.input_size, device=device
        )
        if self.use_cfg:  # this means we use cfg
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([self.num_classes] * labels.shape[0], device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg)
            sample_fn = self.model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)  # do conditional sampling
            sample_fn = self.model.forward
        # Sample images:
        samples = self.infer_diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if self.use_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return Stage2ModelOutput(
            zs_pred=samples,
            zs_degraded=None,
            additional_attr={
                "labels": labels,
            },
        )

    def get_last_layer(self):
        return self.model.final_layer.weight

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        # self.model.pos_embed.requires_grad_(False) # always freeze positional embeddings


class simplewrapper(nn.Module):
    def __init__(
        self,
        conv_encoder: ConvEncoder,
        conv_decoder: Union[ConvDecoder, ConvDecoder_wSkipConnection],
        model: DiT,
    ):
        super(simplewrapper, self).__init__()
        self.conv_encoder = conv_encoder
        self.model = model
        self.conv_decoder = conv_decoder
        self.skip_connect = isinstance(conv_decoder, ConvDecoder_wSkipConnection)
        # assert self.conv_encoder.in_channels == self.conv_decoder.out_channels, f'[!]simplewrapper: conv_encoder.in_channels: {self.conv_encoder.#in_channels} must be equal to conv_out.out_channels: {self.conv_decoder.out_channels}'
        # self.conv_out = zero_module(torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)) # zero out the output
        out_channels = (
            self.conv_decoder.out_channels * 2
            if self.skip_connect
            else self.conv_decoder.out_channels
        )
        channels = self.conv_encoder.in_channels
        self.out = nn.Sequential(  # following https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#682
            nn.GroupNorm(
                num_groups=min(out_channels, 32),
                num_channels=out_channels,
                eps=1e-6,
                affine=True,
            ),
            nn.SiLU(),
            zero_module(
                nn.Conv2d(
                    out_channels, channels, kernel_size=3, stride=1, padding=1
                )  # do a small conv
            ),
        )

    def forward(
        self, x_t, t, **model_kwargs
    ):  # follow the calling convention of diffusion
        if self.skip_connect:
            x_t, hs = self.conv_encoder(x_t, return_hidden_states=True)
        else:
            x_t = self.conv_encoder(x_t)
            hs = None
        pred = self.model(x_t, t, **model_kwargs)
        pred = (
            self.conv_decoder(pred)
            if not self.skip_connect
            else self.conv_decoder(pred, hs)
        )
        # pred = self.conv_out(pred)
        pred = self.out(pred)
        return pred


class DiTwConv_Stage2(Stage2Model):
    def __init__(
        self,
        downsample_ratio: int,
        layers: int,
        kernel_size: int,
        skip_connect: bool,
        input_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes=1000,
        learn_sigma=True,
        timestep_respacing: str = "",
        noise_schedule: str = "linear",
        cfg: float = 0.0,
        inference_step: int = 250,
        n_samples: int = 125,
        do_beta_rescaling: bool = False,
    ):
        super().__init__()
        self.timestep_respacing = str(timestep_respacing)
        self.cfg = cfg
        noise_schedule = noise_schedule
        self.hidden_size = hidden_size
        assert kernel_size in [
            1,
            3,
        ], f"[!]DiTwConv_Stage2: kernel_size: {kernel_size} must be 1 or 3"
        assert (
            in_channels % downsample_ratio == 0
        ), f"[!]DiTwConv_Stage2: in_channels: {in_channels} must be divisible by downsample_ratio: {downsample_ratio}"
        downsampled_channels = in_channels // downsample_ratio
        self.in_channels = in_channels
        self.downsample_ratio = downsample_ratio
        self.downsampled_channels = downsampled_channels
        # compressor = SimpleConv(in_channels=in_channels, layers=layers, bottleneck_ratio=downsample_ratio, kernel_size=kernel_size)
        conv_encoder = ConvEncoder(
            in_channels=in_channels,
            layers=layers,
            bottleneck_ratio=downsample_ratio,
            kernel_size=kernel_size,
        )
        model = DiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=downsampled_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        )
        self.downsampled_out_channels = model.out_channels
        decoder_cls = ConvDecoder_wSkipConnection if skip_connect else ConvDecoder
        conv_decoder = decoder_cls(
            bottle_dim=self.downsampled_out_channels,
            layers=layers,
            upsample_ratio=downsample_ratio,
            kernel_size=kernel_size,
        )
        model.requires_grad_(True)
        conv_decoder.requires_grad_(True)
        conv_encoder.requires_grad_(True)
        self.model = simplewrapper(conv_encoder, conv_decoder, model)
        self.model.requires_grad_(True)  # joint training
        # like DiT we only support square images
        self.diffusion = create_diffusion(
            timestep_respacing=self.timestep_respacing,
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
        )  # like DiT we set default 1000 timesteps
        self.inferencing_step = inference_step
        self.infer_diffusion = create_diffusion(
            timestep_respacing=str(self.inferencing_step),
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
        )
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_cfg = self.cfg > 1.0
        self.n_samples = n_samples
        print(
            f"[!]DiT_Stage2_wConv: Using cfg: {self.use_cfg}, n_samples: {self.n_samples}, cfg: {self.cfg}, timestep_respacing: {self.timestep_respacing}, learn_sigma: {learn_sigma}"
        )

    def forward(
        self, stage1_encodings: Stage1Encodings, inputs: LabeledImageData
    ) -> Stage2ModelOutput:
        zs = stage1_encodings.zs
        return Stage2ModelOutput(
            zs_pred=zs, zs_degraded=zs, additional_attr={}  # no degradation this time
        )

    def compute_loss(
        self,
        stage1_encodings: Stage1Encodings,
        stage2_output: Stage2ModelOutput,
        inputs: LabeledImageData,
        valid: bool = False,
        **kwargs,
    ) -> dict:
        zs = stage1_encodings.zs
        labels = inputs.condition
        if labels is None:
            # we set null labels to num_classes
            labels = torch.tensor(
                [self.num_classes] * zs.shape[0], device=zs.device
            ).long()
        t = torch.randint(
            0, self.diffusion.num_timesteps, (zs.shape[0],), device=zs.device
        )
        model_kwargs = dict(y=labels)
        terms = self.diffusion.training_losses(self.model, zs, t, model_kwargs)
        loss = terms["loss"].mean()
        return {
            "loss_total": loss,
            "t": t,
        }

    def get_last_layer(self):
        if len(self.model.conv_decoder.up) == 0:
            return self.model.model.final_layer.linear.weight
        return self.model.conv_decoder.up[-1].conv2.weight

    def get_recon_imgs(self, xs_real, xs, **kwargs):
        return xs_real.clamp(0, 1), xs.clamp(0, 1)

    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage2ModelOutput:
        device = xm.xla_device()  # default to TPU
        labels = inputs.condition
        if isinstance(labels, torch.Tensor):
            device = labels.device  # sp hack
        if isinstance(inputs.img, torch.Tensor):
            device = inputs.img.device  # sp hack
        n = self.n_samples
        cfg = self.cfg
        if labels is None:
            labels = torch.randint(
                0, self.num_classes, (self.n_samples,), device=device
            )
        else:
            n = labels.shape[0]
            # labels = torch.randint(0, self.num_classes, (n,), device=device) # we still do random label sampling
        y = labels
        z = torch.randn(
            n, self.in_channels, self.input_size, self.input_size, device=device
        )
        if self.use_cfg:  # this means we use cfg
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * labels.shape[0], device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg)
            sample_fn = self.model.forward_with_cfg
        else:
            # y_null = torch.tensor([1000] * labels.shape[0], device=device)
            model_kwargs = dict(y=y)  # do unconditional sampling
            sample_fn = self.model.forward
        # Sample images:
        samples = self.infer_diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if self.use_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return Stage2ModelOutput(
            zs_pred=samples,
            zs_degraded=None,
            additional_attr={
                "labels": labels,
            },
        )
