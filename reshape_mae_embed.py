from transformers import ViTMAEForPreTraining, ViTImageProcessor,ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed
import torch
import sys
import os

config = ViTMAEConfig.from_pretrained(sys.argv[1])
ori_resol = config.image_size
save_path = sys.argv[2]
os.makedirs(save_path,exist_ok=True)
target_resol = int(sys.argv[3])
patch_size = config.patch_size
hidden_size = config.hidden_size
decoder_hidden_size = config.decoder_hidden_size
num_patches = target_resol**2 // patch_size**2
pos_embed = get_2d_sincos_pos_embed(
    hidden_size , int(num_patches**0.5), add_cls_token=True
)
decoder_pos_embed = get_2d_sincos_pos_embed(
    decoder_hidden_size , int(num_patches**0.5), add_cls_token=True
)
print(pos_embed.shape, decoder_pos_embed.shape)
model = ViTMAEForPreTraining.from_pretrained(sys.argv[1])
model.config.image_size = target_resol
model.vit.embeddings.position_embeddings = torch.nn.Parameter(torch.zeros_like(torch.from_numpy(pos_embed)).float().unsqueeze(0))
model.vit.embeddings.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
# do the same to the decoder
model.decoder.decoder_pos_embed = torch.nn.Parameter(torch.zeros_like(torch.from_numpy(decoder_pos_embed)).float().unsqueeze(0))
model.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
model.save_pretrained(save_path)
print(f"Model saved to {save_path}")