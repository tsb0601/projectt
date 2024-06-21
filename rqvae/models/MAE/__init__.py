from transformers import ViTMAEForPreTraining, ViTImageProcessor
import torch
from rqvae.models.interfaces import Stage1Model
import torch_xla.core.xla_model as xm
class Stage1MAE(Stage1Model):
    def __init__(self, ckpt_path:str ,mask_ratio: float = 0. )->None:
        super().__init__()
        self.model = ViTMAEForPreTraining.from_pretrained(ckpt_path)
        self.model.config.mask_ratio = mask_ratio
        self.model.vit.requires_grad_(False) # freeze encoder
        self.model.decoder.requires_grad_(True)
        self.model.decoder.decoder_pos_embed.requires_grad_(False) # this is a hack to make sure that the positional embeddings are not trained
        processor = ViTImageProcessor.from_pretrained(ckpt_path)
        self.noise = torch.arange(256).to(xm.xla_device())
        image_mean, image_std = processor.image_mean, processor.image_std
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        xm.master_print(f'Stage1MAE model loaded with mean {image_mean} and std {image_std}')
        self.image_mean = self.image_mean.to(xm.xla_device())
        self.image_std = self.image_std.to(xm.xla_device())
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
    def compute_loss(self, xs_recon, xs) -> dict:
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