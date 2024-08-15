from requests import patch
from .contrastive_models import *
from .decoder import GeneralDecoder
from ..interfaces import *
from transformers import AutoConfig, AutoImageProcessor
class ContrastiveModelwithDecoder(Stage1Model):
    def __init__(self, pretrained_encoder_path: str, general_decoder_config:str, num_patches:int):
        super().__init__()
        self.encoder = None
        # see which key in MODEL_ARCH is in the pretrained_encoder_path
        for key in MODEL_ARCH.keys():
            if key in pretrained_encoder_path.lower():
                self.encoder = MODEL_ARCH[key](pretrained_encoder_path)
                break
        assert self.encoder is not None, f"Model not found in MODEL_ARCH for {pretrained_encoder_path}"
        encoder_config = AutoConfig.from_pretrained(pretrained_encoder_path)
        patch_size = encoder_config.patch_size
        config = AutoConfig.from_pretrained(general_decoder_config)
        config.patch_size = patch_size # reload patch size
        # get mean and std
        image_processor = AutoImageProcessor.from_pretrained(pretrained_encoder_path)
        self.image_mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
        self.decoder = GeneralDecoder(config, num_patches=num_patches)
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(True)
        self.decoder.decoder_pos_embed.requires_grad_(False) # freeze positional embedding
    def forward(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        x = inputs.img
        x = (x - self.image_mean) / self.image_std
        latent = self.encoder(x)
        decoder_output = self.decoder(latent)
        logits = decoder_output.logits
        xs_recon = self.decoder.unpatchify(logits)
        xs_recon = xs_recon * self.image_std + self.image_mean
        return Stage1ModelOutput(xs_recon, additional_attr={})
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        x = inputs.img
        x = (x - self.image_mean) / self.image_std
        return Stage1Encodings(self.encoder(x), additional_attr={})
    def decode(self,outputs: Stage1Encodings) -> Stage1ModelOutput:
        decoder_output = self.decoder(outputs.zs)
        logits = decoder_output.logits
        xs_recon = self.decoder.unpatchify(logits)
        xs_recon = xs_recon * self.image_std + self.image_mean
        return Stage1ModelOutput(xs_recon, additional_attr={})
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData, valid:bool = False) -> torch.Tensor:
        xs = inputs.img
        xs_recon = outputs.xs_recon
        recon_loss = (xs - xs_recon).square().mean()
        return {
            'loss_total': recon_loss,
            'loss_recon': recon_loss,
            'loss_latent': torch.Tensor([0.0]).to(recon_loss.device),
        }
    def get_recon_imgs(self, x, xs) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.clamp(0, 1), xs.clamp(0, 1)
    def get_last_layer(self) -> torch.Tensor:
        return self.decoder.decoder_pred.weight
    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        return self(inputs)