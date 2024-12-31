import os
import torch
import sys
from safetensors.torch import load_model, save_model, save_file, load_file
from safetensors import safe_open
from typing import Tuple, List
from torch import nn

def load_from_ckpt(ckpt_path:str, strict:bool = True) -> Tuple[nn.Module]:
    if ckpt_path.endswith('.pt'):
        ckpt = torch.load(ckpt_path)
        if isinstance(ckpt, dict):
            state_dict = ckpt
        elif isinstance(ckpt, nn.Module):
            model = ckpt
            state_dict = model.state_dict()
    elif ckpt_path.endswith('.safetensors'):
        state_dict = load_file(ckpt_path)
    else:
        raise ValueError(f'[!]ERROR: ckpt_path should end with .pt or .safetensors, but got {ckpt_path}')
    return state_dict
def save_to_ckpt(state_dict, ckpt_path:str):
    if ckpt_path.endswith('.pt'):
        torch.save(state_dict, ckpt_path)
    elif ckpt_path.endswith('.safetensors'):
        save_file(state_dict, ckpt_path)
    else:
        raise ValueError(f'[!]ERROR: ckpt_path should end with .pt or .safetensors, but got {ckpt_path}')
def load_state_dict(path:str):
    return torch.load(path, map_location='cpu')

def save_state_dict(state_dict, path:str):
    torch.save(state_dict, path)
    
def rm_dec_weight(state_dict: safe_open) -> dict:
    new_state_dict = {}
    for k in state_dict.keys():
        if 'decoder' not in k:
            new_state_dict[k] = state_dict[k]
    return new_state_dict

def main():
    if len(sys.argv) != 3:
        print('Usage: python rm_dec_weight.py <input_path> <output_path>')
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    state_dict = load_from_ckpt(input_path)
    #assert isinstance(state_dict, dict), 'state_dict is not a dictionary, but a %s' % type(state_dict)
    print(state_dict.keys())
    new_state_dict = rm_dec_weight(state_dict)
    print(new_state_dict.keys())
    save_to_ckpt(new_state_dict, output_path)

if __name__ == '__main__':
    main()