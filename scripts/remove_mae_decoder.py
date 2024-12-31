from copy import deepcopy
import os
import sys
from transformers import ViTMAEForPreTraining, ViTImageProcessor
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder, ViTMAEConfig
import torch
from safetensors.torch import load_file
import torch.nn as nn
def load_and_remove(ckpt_path:str, save_path:str):
    config = ViTMAEConfig.from_pretrained(ckpt_path)
    model:ViTMAEForPreTraining = ViTMAEForPreTraining(config)
    state_dict = load_file(os.path.join(ckpt_path,'model.safetensors'),device='cpu')
    keys = list(state_dict.keys())
    print(f"Loaded model from {ckpt_path}, keys: {keys}")
    for k in keys:
        if 'decoder' in k:
            state_dict.pop(k)
    keys = state_dict.keys()
    print(f"decoder removed, keys: {keys}")
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"keys that are not loaded: {keys}")
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
    load_and_remove(ckpt_path, save_path)