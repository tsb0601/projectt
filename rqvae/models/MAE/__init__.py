from transformers import ViTMAEForPreTraining, ViTImageProcessor, ViTMAEModel
import torch
from rqvae.models.interfaces import Stage1Model
import torch_xla.core.xla_model as xm
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoderOutput
from torch import nn
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
    #mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.#shape[1], 1)
    #x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
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
    def __init__(self, ckpt_path:str ,mask_ratio: float = 0. )->None:
        super().__init__()
        self.model = ViTMAEForPreTraining.from_pretrained(ckpt_path)
        self.model.decoder.forward = custom_forward.__get__(self.model.decoder) # original forward method would cause XLA error when mask ratio is zero, we replace it with a custom forward method
        self.model.config.mask_ratio = mask_ratio
        self.model.vit.requires_grad_(False) # freeze encoder
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        noise = torch.arange(256)
        default_id_restore = torch.arange(256)
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer('noise', noise)
        self.register_buffer('default_id_restore', default_id_restore)
        print(f'Stage1MAE model loaded with mean {image_mean} and std {image_std}')
        self.image_mean = self.image_mean
        self.image_std = self.image_std
        #take out mean and std from processor
    def forward(self, xs:torch.Tensor)-> tuple:
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model(xs, noise)
        logits = outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        return (xs_recon ,)
    def encode(self, xs:torch.Tensor)->tuple:
        image_mean = self.image_mean.expand(xs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(xs.shape[0], -1, -1, -1)
        xs = (xs - image_mean) / image_std
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1)
        outputs = self.model.vit(xs,noise=noise)
        return (outputs.last_hidden_state,)
    def decode(self, zs:torch.Tensor)->tuple:
        ids_restore = self.default_id_restore.unsqueeze(0).expand(zs.shape[0],-1)
        image_mean = self.image_mean.expand(zs.shape[0], -1, -1, -1)
        image_std = self.image_std.expand(zs.shape[0], -1, -1, -1)
        outputs = self.model.decoder(zs,ids_restore)
        logits = outputs.logits
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * image_std + image_mean
        return (xs_recon,)
    def compute_loss(self, xs_recon, xs , valid:bool = True) -> dict:
        loss_recon = (xs_recon - xs).abs().mean()
        loss_latent = torch.Tensor([0.]).to(xs.device)
        return {
            'loss_total': loss_recon + loss_latent,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent
        }
    def get_recon_imgs(self, xs , xs_recon) -> tuple:
        xs = xs.clamp(0, 1)
        xs_recon = xs_recon.clamp(0, 1)
        return xs , xs_recon
    def get_last_layer(self):
        return self.model.decoder.decoder_pred.weight
    @torch.no_grad()
    def infer(self, xs):
        return self(xs)
    
class MAEEncoder_ForProbing(nn.Module):
    def __init__(self, ckpt_path:str):
        super().__init__()
        self.model = ViTMAEModel.from_pretrained(ckpt_path)
        self.model.requires_grad_(False)
        self.model.config.mask_ratio = 0.
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        self.register_buffer('noise', torch.arange(patch_num))
    def forward(self, xs:torch.Tensor)->tuple:
        noise = self.noise.unsqueeze(0).expand(xs.shape[0],-1).to(xs.device).to(xs.dtype)
        outputs = self.model(xs, noise)
        return outputs.last_hidden_state