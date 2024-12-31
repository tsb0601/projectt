import os
import sys
from numpy import NaN
from transformers import ViTMAEForPreTraining, ViTImageProcessor
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder
import torch
import torch.nn as nn
def decoder_interpolate_pos_encoding(self:ViTMAEDecoder, embeddings: torch.Tensor) -> torch.Tensor:
    """
    This method is a modified version of the interpolation function for ViT-mae model at the deocder, that
    allows to interpolate the pre-trained decoder position encodings, to be able to use the model on higher
    resolution images.
    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """
    # -1 removes the class dimension since we later append it without interpolation
    embeddings_positions = embeddings.shape[1] - 1
    num_positions = self.decoder_pos_embed.shape[1] - 1
    # Separation of class token and patch tokens
    class_pos_embed = self.decoder_pos_embed[:, 0, :]
    patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
    # To retain the final 3d tensor with the required dimensions
    dim = self.decoder_pos_embed.shape[-1]
    # Increasing a dimension to enable bicubic interpolation
    patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)
    # permute to bring the dimension to be interpolated, to the last
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    # Interpolating the decoder position embeddings shape wrt embeddings shape i.e (x).
    # 1 keeps the other dimension constant
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(1, embeddings_positions),
        mode="bicubic",
        align_corners=False,
    )
    # Converting back to the original shape
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    # Adding the class token back
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
def load_and_remove(ckpt_path:str, save_path:str, target_h_w: int):
    model:ViTMAEForPreTraining = ViTMAEForPreTraining.from_pretrained(ckpt_path)
    encoder_embed = model.vit.embeddings.position_embeddings
    decoder_embed = model.decoder.decoder_pos_embed
    print('before:',encoder_embed.shape, decoder_embed.shape)
    x = torch.zeros((1, target_h_w, encoder_embed.shape[-1]))
    encoder_embed = model.vit.embeddings.interpolate_pos_encoding(x, target_h_w, target_h_w)
    target_h_w//= model.config.patch_size
    x = torch.zeros((1, target_h_w* target_h_w + 1, decoder_embed.shape[-1]))
    print('interpote using ', x.shape)
    decoder_embed = decoder_interpolate_pos_encoding(model.decoder, x) #
    print('interpolated', encoder_embed.shape, decoder_embed.shape)
    model.vit.embeddings.position_embeddings = nn.Parameter(encoder_embed)
    model.decoder.decoder_pos_embed = nn.Parameter(decoder_embed)
    model.config.image_size = target_h_w * model.config.patch_size
    if not os.path.exists(save_path):
        # ask for permission to create the directory
        user_check = input(f"Directory {save_path} does not exist. Do you want to create it? [y/n]: ")
        if user_check == 'y':
            os.makedirs(save_path)
        else:
            print("Exiting...")
            sys.exit(0)
    else:
        user_check = input(f"Directory {save_path} already exists. Do you want to overwrite it? [y/n]: ")
        if user_check == 'n':
            print("Exiting...")
        else:
            print(f"Saving model to {save_path}")
            model.save_pretrained(save_path)

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    save_path = sys.argv[2]
    target_h_w = int(sys.argv[3])
    load_and_remove(ckpt_path, save_path, target_h_w)