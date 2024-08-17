import torch
from .models import fcmae
from ..interfaces import *
from .models.fcmae import FCMAE
class Stage1FCMAE(Stage1Model):
    def __init__(self,model_name:str = 'convnextv2_base', pretrained_model_path: str= None):
        super().__init__()
        init_method = fcmae.__dict__[model_name] # get the model from the fcmae module
        self.model:FCMAE = init_method()
        if pretrained_model_path:
            self.model.encoder.load_state_dict(torch.load(pretrained_model_path))
        self.model.eval()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.model.decoder.requires_grad_(True)
        self.model.encoder.requires_grad_(False)
    def forward(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        xs = inputs.img
        xs = (xs - self.image_mean) / self.image_std
        zs = self.model.forward_encoder(xs)
        logits = self.model.forward_decoder(zs)
        logits = self.model.patch_to_L(logits)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * self.image_std + self.image_mean
        return Stage1ModelOutput(xs_recon=xs_recon, additional_attr={})
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        xs = inputs.img
        xs = (xs - self.image_mean) / self.image_std
        zs = self.model.forward_encoder(xs)
        return Stage1Encodings(zs=zs)
    def decode(self, encodings: Stage1Encodings) -> Stage1ModelOutput:
        logits = self.model.forward_decoder(encodings.zs)
        logits = self.model.patch_to_L(logits)
        xs_recon = self.model.unpatchify(logits)
        xs_recon = xs_recon * self.image_std + self.image_mean
        return Stage1ModelOutput(xs_recon=xs_recon, additional_attr={})
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData, valid: bool = False) -> dict:
        xs_recon = outputs.xs_recon
        mse_loss = (xs_recon - inputs.img).square().mean()
        return {'loss_total': mse_loss,
                'loss_recon': mse_loss,
                'loss_latent': torch.tensor(0.).to(mse_loss.device)
        }
    def get_recon_imgs(self, x, xs, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.clamp(0, 1), xs.clamp(0, 1)
    def get_last_layer(self) -> torch.Tensor:
        return self.model.pred.weight
    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        return self.forward(inputs)

class Stage1FCMAE_forProbing(nn.Module):
    def __init__(self,model_name:str = 'convnextv2_base', pretrained_model_path: str= None, hidden_dim:int = 512, global_pool:bool = True):
        """
        CNN only support global pooling
        """
        super().__init__()
        init_method = fcmae.__dict__[model_name]
        self.model:FCMAE = init_method()
        if pretrained_model_path:
            self.model.encoder.load_state_dict(torch.load(pretrained_model_path))
        self.model.requires_grad_(False)
        self.model.eval()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.hidden_dim = hidden_dim
        # global average pooling
        self.norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        #self.head = nn.Linear(self.hidden_dim, nb_classes)
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = (xs - self.image_mean) / self.image_std
        zs = self.model.forward_encoder(xs) # no mask
        zs = self.norm(zs.mean([-2, -1]))
        return zs
    def requires_grad_(self, requires_grad: bool):
        self.head.requires_grad_(requires_grad)