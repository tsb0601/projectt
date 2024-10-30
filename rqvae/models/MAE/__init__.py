from .modeling_vit_mae import ViTMAEForPreTraining, ViTMAEModel, ViTMAEForPreTrainingwBottleNeck
from transformers import ViTImageProcessor, AutoConfig
import torch
from rqvae.models.interfaces import *
import torch_xla.core.xla_model as xm
from .modeling_vit_mae import ViTMAEDecoderOutput
from torch import nn
from header import *
from safetensors import safe_open
from .blocks import SimpleMLP
from safetensors.torch import load_model, save_model
from .blocks import *
def load_model_from_ckpt(model:nn.Module, ckpt_path:str, strict:bool = True) -> nn.Module:
    if ckpt_path.endswith('.pt'):
        ckpt = torch.load(ckpt_path)
        keys = model.load_state_dict(ckpt['model'], strict = strict)    
    elif ckpt_path.endswith('.safetensors'):
        keys = load_model(model, ckpt_path, strict = strict)
    return model, keys
def custom_forward(
    self,
    hidden_states,
    ids_restore,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
    interpolate_pos_encoding:bool = False # glad Transformers add that as an argument for ViTs
):
    # embed tokens
    x = self.decoder_embed(hidden_states)

    # append mask tokens to sequence
    if x.shape[1] < ids_restore.shape[1] + 1:
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    else:
        x_ = x[:, 1:, :] 
    # unshuffle
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    # add pos embed
    hidden_states = x + self.decoder_pos_embed

    # apply Transformer layers (blocks)
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    for i, layer_module in enumerate(self.decoder_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.__call__,
                hidden_states,
                None,
                output_attentions,
            )
        else:
            layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    hidden_states = self.decoder_norm(hidden_states)

    # predictor projection
    logits = self.decoder_pred(hidden_states)

    # remove cls token
    logits = logits[:, 1:, :]

    if not return_dict:
        return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
    return ViTMAEDecoderOutput(
        logits=logits,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
class Stage1MAE(Stage1Model):
    def __init__(self, ckpt_path:str, mask_ratio: float = 0., train_encoder:bool = False, no_cls:bool = False, loss_type: str = 'l1')->None:
        super().__init__()
        tensor_path = os.path.join(ckpt_path, 'model.safetensors')
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.isfile(tensor_path):
            tensor_path = os.path.join(ckpt_path, 'model.pt') # try another name
        if not os.path.isfile(tensor_path):
            print(f'init from scratch according to {config_path}')
        else:
            print(f'init from {tensor_path} according to {config_path}')
        config = AutoConfig.from_pretrained(config_path)
        model = ViTMAEForPreTraining(config)
        if no_cls:
            model.decoder.set_trainable_cls_token()
        self.model = model
        self.model.config.mask_ratio = mask_ratio
        assert mask_ratio >= 0. and mask_ratio <= 1., 'mask ratio should be between 0 and 1, but got {}'.format(mask_ratio)
        self.model.vit.requires_grad_(train_encoder) # freeze encoder
        self.model.vit.embeddings.position_embeddings.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        if os.path.isfile(tensor_path):
            _, keys = load_model_from_ckpt(self.model, tensor_path, strict = False)
            print(f'missing keys: {keys[0]}, unexpected keys: {keys[1]}')
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        noise = torch.arange(patch_num)
        default_id_restore = torch.arange(patch_num)
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer('noise', noise)
        self.register_buffer('default_id_restore', default_id_restore)
        # get the final layernorm's affine parameters
        self.no_cls = no_cls
        self.loss = lambda x, y: (x - y).abs().mean() if loss_type == 'l1' else (x - y).square().mean()
        assert loss_type in ['l1', 'l2'], 'loss type should be either l1 or l2, but got {}'.format(loss_type)
        print(f'Stage1MAE model loaded with mean {processor.image_mean} and std {processor.image_std}, mask ratio {mask_ratio}')
    def forward(self, inputs: LabeledImageData)-> Stage1ModelOutput:
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model(xs, noise, drop_cls_token = self.no_cls) if self.model.config.mask_ratio == 0. else self.model(xs)
        logits = outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        output = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr= {'outputs': outputs}
        )
        return output
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        # mask_ratio must be zero
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model.vit(xs, noise=noise)
        latent = outputs.last_hidden_state # bsz, num_patches, hidden_size
        encodings = Stage1Encodings(
            zs = latent,
            additional_attr = {'outputs': outputs,
        }
        )
        return encodings
    def decode(self, outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred # still we can pass Stage2ModelOutput
        ids_restore = self.default_id_restore.unsqueeze(0).expand(zs.shape[0],-1)
        image_mean = self.image_mean.expand(zs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(zs.shape[0], -1, -1, -1)
        outputs = self.model.decoder(zs,ids_restore, drop_cls_token=self.no_cls)
        logits = outputs.logits
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        outputs = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr = {'outputs': outputs}
        )
        return outputs
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData , valid:bool = True) -> dict:
        xs = inputs.img
        xs_recon = outputs.xs_recon
        MAE_outputs = outputs.additional_attr['outputs']
        loss_recon = self.loss(xs_recon, xs) if self.model.config.mask_ratio == 0. else MAE_outputs.loss
        #loss_recon = (xs_recon - xs).square().mean() if self.model.config.mask_ratio == 0. else MAE_outputs.loss # L2
        loss_latent = torch.Tensor([0.]).to(xs.device)
        return {
            'loss_total': loss_recon + loss_latent,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent
        }
    def get_recon_imgs(self, xs: torch.Tensor , xs_recon: torch.Tensor):
        xs = xs.clamp(0, 1)
        xs_recon = xs_recon.clamp(0, 1)
        return xs , xs_recon
    def get_last_layer(self) ->torch.Tensor:
        return self.model.decoder.decoder_pred.weight
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)
    
class MAEEncoder_ForProbing(nn.Module):
    def __init__(self, ckpt_path:str, global_pool:bool = False):
        super().__init__()
        self.model = ViTMAEModel.from_pretrained(ckpt_path)
        self.model.requires_grad_(False)
        self.model.config.mask_ratio = 0.
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        self.register_buffer('noise', torch.arange(patch_num))
        self.global_pool = global_pool
        if global_pool:
            self.model.layernorm = nn.Identity() # remove layernorm
            self.model.fc_norm = nn.LayerNorm(self.model.config.hidden_size)
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1).to(xs.device).to(xs.dtype)
        outputs = self.model(xs, noise)
        latent = outputs.last_hidden_state
        if self.global_pool:
            latent = latent[:, 1: , :]
            latent = self.model.fc_norm(latent)
            latent = latent.mean(dim=1)
        else: # take the cls token
            latent = latent[:, 0, :]
        return latent
    def requires_grad_(self, requires_grad:bool = True):
        self.model.requires_grad_(requires_grad)
        self.model: ViTMAEModel
        self.model.embeddings.position_embeddings.requires_grad_(False)
class Stage1MAEwEnhancedDec(Stage1Model):
    def __init__(self, ckpt_path:str, mask_ratio: float = 0., train_encoder:bool = False, no_cls:bool = False, layers_channels :list = None, use_conv:bool = False, loss_type: str = 'l1', do_decoder_embed_in_encode: bool = False)->None:
        super().__init__()
        tensor_path = os.path.join(ckpt_path, 'model.safetensors')
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.isfile(tensor_path):
            tensor_path = os.path.join(ckpt_path, 'model.pt') # try another name
        if not os.path.isfile(tensor_path):
            print(f'init from scratch according to {config_path}')
        else:
            print(f'init from {tensor_path} according to {config_path}')
        config = AutoConfig.from_pretrained(config_path)
        model = ViTMAEForPreTraining(config)
        if no_cls:
            model.decoder.set_trainable_cls_token()
        self.model = model
        self.model.config.mask_ratio = mask_ratio
        assert mask_ratio >= 0. and mask_ratio <= 1., 'mask ratio should be between 0 and 1, but got {}'.format(mask_ratio)
        self.model.vit.requires_grad_(train_encoder) # freeze encoder
        self.model.vit.embeddings.position_embeddings.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        if os.path.isfile(tensor_path):
            _, keys = load_model_from_ckpt(self.model, tensor_path, strict = False)
            print(f'missing keys: {keys[0]}, unexpected keys: {keys[1]}')
        out_channels = self.model.config.patch_size * self.model.config.patch_size * self.model.config.num_channels
        conv_up_decoder = ConvUp(in_channels = self.model.config.decoder_hidden_size, out_channel = out_channels, kernel_size = 3 if use_conv else 1, layers_channels = layers_channels)
        self.model.decoder.decoder_pred = conv_up_decoder # replace the decoder_pred with the upsampled version
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        noise = torch.arange(patch_num)
        default_id_restore = torch.arange(patch_num)
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer('noise', noise)
        self.register_buffer('default_id_restore', default_id_restore)
        # get the final layernorm's affine parameters
        self.no_cls = no_cls
        self.loss = lambda x, y: (x - y).abs().mean() if loss_type == 'l1' else (x - y).square().mean()
        self.do_encoder_embed_in_decode = do_decoder_embed_in_encode
        assert loss_type in ['l1', 'l2'], 'loss type should be either l1 or l2, but got {}'.format(loss_type)
        print(f'Stage1MAE model loaded with mean {processor.image_mean} and std {processor.image_std}, mask ratio {mask_ratio}')
    def forward(self, inputs: LabeledImageData)-> Stage1ModelOutput:
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model(xs, noise, drop_cls_token = self.no_cls) if self.model.config.mask_ratio == 0. else self.model(xs)
        logits = outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        output = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr= {'outputs': outputs}
        )
        return output
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        # mask_ratio must be zero
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model.vit(xs, noise=noise)
        latent = outputs.last_hidden_state # bsz, num_patches, hidden_size
        latent = self.model.decoder.decoder_embed(latent) if self.do_encoder_embed_in_decode else latent
        encodings = Stage1Encodings(
            zs = latent,
            additional_attr = {'outputs': outputs,
        }
        )
        return encodings
    def decode(self, outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred # still we can pass Stage2ModelOutput
        ids_restore = self.default_id_restore.unsqueeze(0).expand(zs.shape[0],-1)
        image_mean = self.image_mean.expand(zs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(zs.shape[0], -1, -1, -1)
        outputs = self.model.decoder(zs,ids_restore, drop_cls_token=self.no_cls, do_decoder_embed = not self.do_encoder_embed_in_decode)
        logits = outputs.logits
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        outputs = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr = {'outputs': outputs}
        )
        return outputs
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData , valid:bool = True) -> dict:
        xs = inputs.img
        xs_recon = outputs.xs_recon
        MAE_outputs = outputs.additional_attr['outputs']
        loss_recon = self.loss(xs_recon, xs) if self.model.config.mask_ratio == 0. else MAE_outputs.loss
        #loss_recon = (xs_recon - xs).square().mean() if self.model.config.mask_ratio == 0. else MAE_outputs.loss # L2
        loss_latent = torch.Tensor([0.]).to(xs.device)
        return {
            'loss_total': loss_recon + loss_latent,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent
        }
    def get_recon_imgs(self, xs: torch.Tensor , xs_recon: torch.Tensor):
        xs = xs.clamp(0, 1)
        xs_recon = xs_recon.clamp(0, 1)
        return xs , xs_recon
    def get_last_layer(self) ->torch.Tensor:
        decoder_pred: ConvUp = self.model.decoder.decoder_pred
        return decoder_pred.up_conv_out.weight
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)
class Stage1MAEwEnhancedEncDec(Stage1Model):
    def __init__(self, ckpt_path:str, mask_ratio: float = 0., train_encoder:bool = False, no_cls:bool = False, up_layers_channels :list = None, down_layers_channels :list = None ,use_conv:bool = False, loss_type: str = 'l1', do_decoder_embed_in_encode: bool = False)->None:
        super().__init__()
        tensor_path = os.path.join(ckpt_path, 'model.safetensors')
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.isfile(tensor_path):
            tensor_path = os.path.join(ckpt_path, 'model.pt') # try another name
        if not os.path.isfile(tensor_path):
            print(f'init from scratch according to {config_path}')
        else:
            print(f'init from {tensor_path} according to {config_path}')
        config = AutoConfig.from_pretrained(config_path)
        model = ViTMAEForPreTraining(config)
        if no_cls:
            model.decoder.set_trainable_cls_token()
        self.model = model
        self.model.config.mask_ratio = mask_ratio
        assert mask_ratio >= 0. and mask_ratio <= 1., 'mask ratio should be between 0 and 1, but got {}'.format(mask_ratio)
        self.model.vit.requires_grad_(train_encoder) # freeze encoder
        self.model.vit.embeddings.position_embeddings.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        if os.path.isfile(tensor_path):
            _, keys = load_model_from_ckpt(self.model, tensor_path, strict = False)
            print(f'missing keys: {keys[0]}, unexpected keys: {keys[1]}')
        out_channels = self.model.config.patch_size * self.model.config.patch_size * self.model.config.num_channels
        conv_up_decoder = ConvUp(in_channels = self.model.config.decoder_hidden_size, out_channel = out_channels, kernel_size = 3 if use_conv else 1, layers_channels = up_layers_channels)
        conv_down_encoder = ConvUp(in_channels = self.model.config.hidden_size, out_channel = self.model.config.decoder_hidden_size, kernel_size = 3 if use_conv else 1, layers_channels = down_layers_channels)
        self.model.decoder.decoder_pred = conv_up_decoder # replace the decoder_pred with the upsampled version
        self.model.decoder.decoder_embed = conv_down_encoder # replace the decoder_embed with the downsampled version
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        noise = torch.arange(patch_num)
        default_id_restore = torch.arange(patch_num)
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer('noise', noise)
        self.register_buffer('default_id_restore', default_id_restore)
        # get the final layernorm's affine parameters
        self.no_cls = no_cls
        self.do_encoder_embed_in_decode = do_decoder_embed_in_encode
        self.loss = lambda x, y: (x - y).abs().mean() if loss_type == 'l1' else (x - y).square().mean()
        assert loss_type in ['l1', 'l2'], 'loss type should be either l1 or l2, but got {}'.format(loss_type)
        print(f'Stage1MAE model loaded with mean {processor.image_mean} and std {processor.image_std}, mask ratio {mask_ratio}')
    def forward(self, inputs: LabeledImageData)-> Stage1ModelOutput:
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model(xs, noise, drop_cls_token = self.no_cls) if self.model.config.mask_ratio == 0. else self.model(xs)
        logits = outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        output = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr= {'outputs': outputs}
        )
        return output
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        # mask_ratio must be zero
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model.vit(xs, noise=noise)
        latent = outputs.last_hidden_state # bsz, num_patches, hidden_size
        latent = self.model.decoder.decoder_embed(latent) if self.do_encoder_embed_in_decode else latent
        encodings = Stage1Encodings(
            zs = latent,
            additional_attr = {'outputs': outputs,
        }
        )
        return encodings
    def decode(self, outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred # still we can pass Stage2ModelOutput
        ids_restore = self.default_id_restore.unsqueeze(0).expand(zs.shape[0],-1)
        image_mean = self.image_mean.expand(zs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(zs.shape[0], -1, -1, -1)
        outputs = self.model.decoder(zs,ids_restore, drop_cls_token=self.no_cls, do_decoder_embed = not self.do_encoder_embed_in_decode)
        logits = outputs.logits
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        outputs = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr = {'outputs': outputs}
        )
        return outputs
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData , valid:bool = True) -> dict:
        xs = inputs.img
        xs_recon = outputs.xs_recon
        MAE_outputs = outputs.additional_attr['outputs']
        loss_recon = self.loss(xs_recon, xs) if self.model.config.mask_ratio == 0. else MAE_outputs.loss
        #loss_recon = (xs_recon - xs).square().mean() if self.model.config.mask_ratio == 0. else MAE_outputs.loss # L2
        loss_latent = torch.Tensor([0.]).to(xs.device)
        return {
            'loss_total': loss_recon + loss_latent,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent
        }
    def get_recon_imgs(self, xs: torch.Tensor , xs_recon: torch.Tensor):
        xs = xs.clamp(0, 1)
        xs_recon = xs_recon.clamp(0, 1)
        return xs , xs_recon
    def get_last_layer(self) ->torch.Tensor:
        decoder_pred: ConvUp = self.model.decoder.decoder_pred
        return decoder_pred.up_conv_out.weight
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)
class Stage1MAEwDCAE(Stage1Model):
    def __init__(self, ckpt_path:str, mask_ratio: float = 0., train_encoder:bool = False, no_cls:bool = False, loss_type: str = 'l1', do_decoder_embed_in_encode: bool = False)->None:
        super().__init__()
        tensor_path = os.path.join(ckpt_path, 'model.safetensors')
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.isfile(tensor_path):
            tensor_path = os.path.join(ckpt_path, 'model.pt') # try another name
        if not os.path.isfile(tensor_path):
            print(f'init from scratch according to {config_path}')
        else:
            print(f'init from {tensor_path} according to {config_path}')
        config = AutoConfig.from_pretrained(config_path)
        model = ViTMAEForPreTraining(config)
        if no_cls:
            model.decoder.set_trainable_cls_token()
        self.model = model
        self.model.config.mask_ratio = mask_ratio
        assert mask_ratio >= 0. and mask_ratio <= 1., 'mask ratio should be between 0 and 1, but got {}'.format(mask_ratio)
        self.model.vit.requires_grad_(train_encoder) # freeze encoder
        self.model.vit.embeddings.position_embeddings.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        if os.path.isfile(tensor_path):
            _, keys = load_model_from_ckpt(self.model, tensor_path, strict = False)
            print(f'missing keys: {keys[0]}, unexpected keys: {keys[1]}')
        out_channels = self.model.config.patch_size * self.model.config.patch_size * self.model.config.num_channels
        downsample_ratio = self.model.config.hidden_size // self.model.config.decoder_hidden_size
        assert self.model.config.hidden_size % self.model.config.decoder_hidden_size == 0, 'hidden size should be divisible by decoder hidden size'
        dcae_down_encoder = DCAE_ChannelDownsampleLayerwReshape(in_channels = self.model.config.hidden_size, downsample_factor = downsample_ratio)
        dcae_up_decoder = DCAE_ChannelUpsampleLayerwReshape(in_channels = self.model.config.decoder_hidden_size, upsample_factor = downsample_ratio)
        self.model.decoder.decoder_pred = dcae_up_decoder # replace the decoder_pred with the upsampled version
        self.model.decoder.decoder_embed = dcae_down_encoder # replace the decoder_embed with the downsampled version
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        noise = torch.arange(patch_num)
        default_id_restore = torch.arange(patch_num)
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer('noise', noise)
        self.register_buffer('default_id_restore', default_id_restore)
        # get the final layernorm's affine parameters
        self.no_cls = no_cls
        self.do_encoder_embed_in_decode = do_decoder_embed_in_encode
        self.loss = lambda x, y: (x - y).abs().mean() if loss_type == 'l1' else (x - y).square().mean()
        assert loss_type in ['l1', 'l2'], 'loss type should be either l1 or l2, but got {}'.format(loss_type)
        print(f'Stage1MAE model loaded with mean {processor.image_mean} and std {processor.image_std}, mask ratio {mask_ratio}')
    def forward(self, inputs: LabeledImageData)-> Stage1ModelOutput:
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model(xs, noise, drop_cls_token = self.no_cls) if self.model.config.mask_ratio == 0. else self.model(xs)
        logits = outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        output = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr= {'outputs': outputs}
        )
        return output
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        # mask_ratio must be zero
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model.vit(xs, noise=noise)
        latent = outputs.last_hidden_state # bsz, num_patches, hidden_size
        latent = self.model.decoder.decoder_embed(latent) if self.do_encoder_embed_in_decode else latent
        encodings = Stage1Encodings(
            zs = latent,
            additional_attr = {'outputs': outputs,
        }
        )
        return encodings
    def decode(self, outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred # still we can pass Stage2ModelOutput
        ids_restore = self.default_id_restore.unsqueeze(0).expand(zs.shape[0],-1)
        image_mean = self.image_mean.expand(zs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(zs.shape[0], -1, -1, -1)
        outputs = self.model.decoder(zs,ids_restore, drop_cls_token=self.no_cls, do_decoder_embed = not self.do_encoder_embed_in_decode)
        logits = outputs.logits
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        outputs = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr = {'outputs': outputs}
        )
        return outputs
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData , valid:bool = True) -> dict:
        xs = inputs.img
        xs_recon = outputs.xs_recon
        MAE_outputs = outputs.additional_attr['outputs']
        loss_recon = self.loss(xs_recon, xs) if self.model.config.mask_ratio == 0. else MAE_outputs.loss
        #loss_recon = (xs_recon - xs).square().mean() if self.model.config.mask_ratio == 0. else MAE_outputs.loss # L2
        loss_latent = torch.Tensor([0.]).to(xs.device)
        return {
            'loss_total': loss_recon + loss_latent,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent
        }
    def get_recon_imgs(self, xs: torch.Tensor , xs_recon: torch.Tensor):
        xs = xs.clamp(0, 1)
        xs_recon = xs_recon.clamp(0, 1)
        return xs , xs_recon
    def get_last_layer(self) ->torch.Tensor:
        decoder_pred: DCAE_ChannelUpsampleLayerwReshape = self.model.decoder.decoder_pred
        return decoder_pred.conv.weight
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)

class Stage1MAEwConvNextDCAE(Stage1Model):
    def __init__(self, ckpt_path:str, mask_ratio: float = 0., train_encoder:bool = False, no_cls:bool = False, loss_type: str = 'l1', down_layers_channels :list = None, up_layers_channels :list = None, down_depths: list = None, up_depths: list = None,do_decoder_embed_in_encode: bool = False)->None:
        super().__init__()
        tensor_path = os.path.join(ckpt_path, 'model.safetensors')
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.isfile(tensor_path):
            tensor_path = os.path.join(ckpt_path, 'model.pt') # try another name
        if not os.path.isfile(tensor_path):
            print(f'init from scratch according to {config_path}')
        else:
            print(f'init from {tensor_path} according to {config_path}')
        config = AutoConfig.from_pretrained(config_path)
        model = ViTMAEForPreTraining(config)
        if no_cls:
            model.decoder.set_trainable_cls_token()
        self.model = model
        self.model.config.mask_ratio = mask_ratio
        assert mask_ratio >= 0. and mask_ratio <= 1., 'mask ratio should be between 0 and 1, but got {}'.format(mask_ratio)
        self.model.vit.requires_grad_(train_encoder) # freeze encoder
        self.model.vit.embeddings.position_embeddings.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        if os.path.isfile(tensor_path):
            _, keys = load_model_from_ckpt(self.model, tensor_path, strict = False)
            print(f'missing keys: {keys[0]}, unexpected keys: {keys[1]}')
        out_channels = self.model.config.patch_size * self.model.config.patch_size * self.model.config.num_channels
        assert self.model.config.hidden_size % self.model.config.decoder_hidden_size == 0, 'hidden size should be divisible by decoder hidden size'
        Convnext_DCAE_encoder = ConvNextDownSampler(in_channels = self.model.config.hidden_size, out_channels = self.model.config.decoder_hidden_size, layer_channels = down_layers_channels, stage_depths = down_depths)
        Convnext_DCAE_decoder = ConvNextUpSampler(in_channels = self.model.config.decoder_hidden_size, out_channels = out_channels, layer_channels = up_layers_channels, stage_depths = up_depths)
        self.model.decoder.decoder_pred = Convnext_DCAE_decoder # replace the decoder_pred with the upsampled version
        self.model.decoder.decoder_embed = Convnext_DCAE_encoder # replace the decoder_embed with the downsampled version
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        noise = torch.arange(patch_num)
        default_id_restore = torch.arange(patch_num)
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer('noise', noise)
        self.register_buffer('default_id_restore', default_id_restore)
        # get the final layernorm's affine parameters
        self.no_cls = no_cls
        self.do_encoder_embed_in_decode = do_decoder_embed_in_encode
        self.loss = lambda x, y: (x - y).abs().mean() if loss_type == 'l1' else (x - y).square().mean()
        assert loss_type in ['l1', 'l2'], 'loss type should be either l1 or l2, but got {}'.format(loss_type)
        print(f'Stage1MAE model loaded with mean {processor.image_mean} and std {processor.image_std}, mask ratio {mask_ratio}')
    def forward(self, inputs: LabeledImageData)-> Stage1ModelOutput:
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model(xs, noise, drop_cls_token = self.no_cls) if self.model.config.mask_ratio == 0. else self.model(xs)
        logits = outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        output = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr= {'outputs': outputs}
        )
        return output
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        # mask_ratio must be zero
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model.vit(xs, noise=noise)
        latent = outputs.last_hidden_state # bsz, num_patches, hidden_size
        latent = self.model.decoder.decoder_embed(latent) if self.do_encoder_embed_in_decode else latent
        encodings = Stage1Encodings(
            zs = latent,
            additional_attr = {'outputs': outputs,
        }
        )
        return encodings
    def decode(self, outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred # still we can pass Stage2ModelOutput
        ids_restore = self.default_id_restore.unsqueeze(0).expand(zs.shape[0],-1)
        image_mean = self.image_mean.expand(zs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(zs.shape[0], -1, -1, -1)
        outputs = self.model.decoder(zs,ids_restore, drop_cls_token=self.no_cls, do_decoder_embed = not self.do_encoder_embed_in_decode)
        logits = outputs.logits
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        outputs = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr = {'outputs': outputs}
        )
        return outputs
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData , valid:bool = True) -> dict:
        xs = inputs.img
        xs_recon = outputs.xs_recon
        MAE_outputs = outputs.additional_attr['outputs']
        loss_recon = self.loss(xs_recon, xs) if self.model.config.mask_ratio == 0. else MAE_outputs.loss
        #loss_recon = (xs_recon - xs).square().mean() if self.model.config.mask_ratio == 0. else MAE_outputs.loss # L2
        loss_latent = torch.Tensor([0.]).to(xs.device)
        return {
            'loss_total': loss_recon + loss_latent,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent
        }
    def get_recon_imgs(self, xs: torch.Tensor , xs_recon: torch.Tensor):
        xs = xs.clamp(0, 1)
        xs_recon = xs_recon.clamp(0, 1)
        return xs , xs_recon
    def get_last_layer(self) ->torch.Tensor:
        decoder_pred: ConvNextUpSampler = self.model.decoder.decoder_pred
        return decoder_pred.conv_out.weight
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vit.train(False) # freeze the encoder
        self.model.decoder.decoder_embed.train(False) # freeze the decoder embed
class Stage1MAEwBottleNeck(Stage1Model):
    def __init__(self, ckpt_path:str, mask_ratio: float = 0., train_encoder:bool = False, no_cls:bool = False, mlp_layers:int = 0, mlp_ratio:float = 4, bottleneck_ratio:float = 4)->None:
        super().__init__()
        tensor_path = os.path.join(ckpt_path, 'model.safetensors')
        if not os.path.isfile(tensor_path):
            tensor_path = os.path.join(ckpt_path, 'model.pt') # try another name
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.isfile(tensor_path):
            print(f'init from scratch according to {config_path}')
        else:
            print(f'init from {tensor_path} according to {config_path}')
        config = AutoConfig.from_pretrained(config_path)
        model = ViTMAEForPreTrainingwBottleNeck(config, mlp_layers, mlp_ratio, bottleneck_ratio)
        if no_cls:
            model.decoder.set_trainable_cls_token()
        self.model = model
        self.model.config.mask_ratio = mask_ratio
        assert mask_ratio >= 0. and mask_ratio <= 1., 'mask ratio should be between 0 and 1, but got {}'.format(mask_ratio)
        self.model.vit.requires_grad_(train_encoder) # freeze encoder
        self.model.vit.embeddings.position_embeddings.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        if os.path.isfile(tensor_path):
            _, keys = load_model_from_ckpt(self.model, tensor_path, strict = False)
            print(f'missing keys: {keys[0]}, unexpected keys: {keys[1]}')
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        self.model.compressor.requires_grad_(True)
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        noise = torch.arange(patch_num)
        default_id_restore = torch.arange(patch_num)
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer('noise', noise)
        self.register_buffer('default_id_restore', default_id_restore)
        self.no_cls = no_cls
        print(f'Stage1MAE model loaded with mean {processor.image_mean} and std {processor.image_std}, mask ratio {mask_ratio}')
    def forward(self, inputs: LabeledImageData)-> Stage1ModelOutput:
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model(xs, noise, drop_cls_token = self.no_cls) if self.model.config.mask_ratio == 0. else self.model(xs)
        logits = outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        output = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr= {'outputs': outputs}
        )
        return output
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        # mask_ratio must be zero
        xs = inputs.img
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model.vit(xs, noise=noise)
        latent = outputs.last_hidden_state # bsz, num_patches, hidden_size
        latent = self.model.compressor.encode(latent)
        encodings = Stage1Encodings(
            zs = latent,
            additional_attr = {'outputs': outputs,
        }
        )
        return encodings
    def decode(self, outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs if isinstance(outputs, Stage1Encodings) else outputs.zs_pred # still we can pass Stage2ModelOutput
        zs = self.model.compressor.decode(zs)
        #zs = (zs - self.layernorm_mean) / self.layernorm_std
        ids_restore = self.default_id_restore.unsqueeze(0).expand(zs.shape[0],-1)
        image_mean = self.image_mean.expand(zs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(zs.shape[0], -1, -1, -1)
        outputs = self.model.decoder(zs,ids_restore, drop_cls_token=self.no_cls)
        logits = outputs.logits
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        outputs = Stage1ModelOutput(
            xs_recon = xs_recon,
            additional_attr = {'outputs': outputs}
        )
        return outputs
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData , valid:bool = True) -> dict:
        xs = inputs.img
        xs_recon = outputs.xs_recon
        MAE_outputs = outputs.additional_attr['outputs']
        loss_recon = self.loss(xs_recon, xs) if self.model.config.mask_ratio == 0. else MAE_outputs.loss
        #loss_recon = (xs_recon - xs).square().mean() if self.model.config.mask_ratio == 0. else MAE_outputs.loss # L2
        loss_latent = torch.Tensor([0.]).to(xs.device)
        return {
            'loss_total': loss_recon + loss_latent,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent
        }
    def get_recon_imgs(self, xs: torch.Tensor , xs_recon: torch.Tensor):
        xs = xs.clamp(0, 1)
        xs_recon = xs_recon.clamp(0, 1)
        return xs , xs_recon
    def get_last_layer(self) ->torch.Tensor:
        return self.model.decoder.decoder_pred.weight
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)
    def train(self, mode: bool = True):
        super().train(mode)