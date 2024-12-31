"""
transform the MAE stage1model to hgf format checkpoint
"""
import os
import sys
#set working directory to be the parent directory of the current file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.models.MAE import Stage1MAE
from transformers import ViTMAEForPreTraining, ViTImageProcessor
import torch
DEFAULT_PROCESSOR = ViTImageProcessor.from_pretrained('/home/bytetriper/model_zoo/mae_base_256_ft')


def convert(ckpt_path:str, save_path:str):
    global DEFAULT_PROCESSOR
    mae = Stage1MAE(ckpt_path=ckpt_path).cpu()
    mae.model.config.mask_ratio = .75
    #load_ckpt = torch.load(load_path, map_location='cpu')
    #keys = list(load_ckpt.keys())
    #for key in keys:
    #    if 'module.' in key:
    #        load_ckpt[key.replace('module.', '', 1)] = load_ckpt.pop(key)
    #print(load_ckpt.keys())
    #keys = mae.load_state_dict(load_ckpt, strict=False)
    #print(f'keys that are not loaded: {keys}')
    #mae.model.savespretrained(save_path)
    #DEFAULT_PROCESSOR.save_pretrained(save_path)
    torch.save(mae.state_dict(), os.path.join(save_path, '0-model.pt'))
if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    save_path = sys.argv[2]
    assert os.path.exists(ckpt_path)
    os.makedirs(save_path, exist_ok=True)
    convert(ckpt_path, save_path)