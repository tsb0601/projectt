import torch
from ..interfaces import *

class Identity_Stage1(Stage1Model):
    def __init__(self, hidden_dim: int, input_size:int, in_channels:int):
        """
        assume the input is a 3D tensor of shape (B, in_channel, input_size, input_size)
        """
        super().__init__()
        patch_size = int((hidden_dim // in_channels) ** 0.5)
        patch_num = input_size // patch_size
        assert patch_size ** 2 * in_channels == hidden_dim, "hidden_dim should be divisible by in_channel"
        assert patch_num * patch_size == input_size, "input_size should be divisible by patch_size"
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.in_channels = in_channels
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.dummy_param = torch.nn.Parameter(torch.empty(0)) # require a dummy parameter for optimizer
    def forward(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        """
        Just return x
        """
        output = Stage1ModelOutput(
            xs_recon=inputs.img,
            additional_attr={}
        )
        return output
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        """
        reshape the input to [B, hidden_dim, patch_num, patch_num]
        """
        x = inputs.img
        pn, pz , c = self.patch_num, self.patch_size, self.in_channels
        x = x.view(-1, c, pn, pz, pn, pz).permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, c * pz * pz, pn , pn)
        x = (x * 2) - 1. # normalize to [-1, 1]
        return Stage1Encodings(zs=x, additional_attr={})
    def decode(self,outputs: Stage1Encodings) -> Stage1ModelOutput:
        zs = outputs.zs # zs : [B, hidden_dim, patch_num, patch_num]
        iz, c = self.input_size, self.in_channels
        pn, pz = self.patch_num, self.patch_size
        x = zs.view(-1, c, pz, pz, pn, pn).permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, c, iz, iz)
        x = (x + 1) / 2. # denormalize to [0, 1]
        output = Stage1ModelOutput(
            xs_recon=x,
            additional_attr={}
        )
        return output
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData):
        xs = inputs.img
        xs_recon = outputs.xs_recon
        loss = (xs - xs_recon).abs().mean() # l1 loss, but normally should be zero
        return {
            "loss_total": loss,
            "loss_recon": loss,
            "loss_latent": loss
        }
    def get_recon_imgs(self, x, xs):
        return x, xs
    def get_last_layer(self):
        return self.dummy_param # no parameter to return
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        return self.forward(inputs)