"""
This script is used to dump the imagenet latent representation of the data into latent dataset.
"""
from PIL import Image
import torch
from rqvae.models.utils import instantiate_from_config
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData, LabeledImageDatasetWrapper
from rqvae.img_datasets import create_dataset
from rqvae.models import create_model
from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from rqvae.models.connectors import base_connector
from rqvae.models.interfaces import *
import sys
from torch_xla.distributed.parallel_loader import ParallelLoader as pl
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import argparse
import torch_xla.distributed.xla_backend
import torch_xla.distributed.xla_multiprocessing as xmp
def get_stage1_model(config:dict)->Stage1ModelWrapper:
    stage1_model_wrapper, _  = create_model(config, is_master=True)
    stage1_model_wrapper:Stage1ModelWrapper
    return stage1_model_wrapper

def extract_features(stage1_model_wrapper:Stage1ModelWrapper, dataset: LabeledImageDatasetWrapper, output_dir:str, use_connector:bool = True, is_ddp:bool = False):
    device = xm.xla_device()
    stage1_model_wrapper.to(device)
    os.makedirs(output_dir, exist_ok=True)
    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else dataset.default_collate_fn
    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, sampler=sampler, collate_fn=collate_fn)
        dataloader = pl(dataloader, [device]).per_device_loader(device)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn) # 1 for batch size to batch effect
    encode_func = stage1_model_wrapper.encode if use_connector else stage1_model_wrapper.stage_1_model.encode
    tbar = tqdm(dataloader, total=len(dataloader), desc='Extracting features', disable= not is_ddp or not xm.is_master_ordinal())
    with torch.no_grad():
        for i, data in enumerate(tbar): 
            img_path = data.img_path[0]
            img_name = os.path.basename(img_path).split('.')[0] + '.npz' # replace .jpg with .npz
            # if exists, skip
            if os.path.exists(os.path.join(output_dir, img_name)):
                continue
            data: LabeledImageData
            data._to(device)
            encodings = encode_func(data)
            zs = encodings.zs.squeeze(0).cpu().numpy()
            condition = data.condition.squeeze(0).cpu().numpy()
           
            #print(f'img_name: {img_name}, zs shape: {zs.shape}, condition shape: {condition.shape}')
            np.savez(os.path.join(output_dir, img_name), latent=zs, condition=condition)
            #tbar.set_postfix({'img':img_name})
    print('Done!')
def parse_args():
    parser = argparse.ArgumentParser(description='Dump latent representation of the dataset')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--use_connector', action='store_true', help='Use connector')
    parser.add_argument('--is_ddp', action='store_true', help='Use DDP')
    parser.add_argument('--train_split', action='store_true', help='Use train split')
    return parser.parse_args()
def main(rank, config_path:str, output_dir:str, use_connector:bool = True, is_ddp:bool = False, train_split:bool = False):
    config = OmegaConf.load(config_path)
    stage1_model_wrapper = get_stage1_model(config.arch)
    dataset = create_dataset(config)[0 if train_split else 1] # 0 for train, 1 for val
    extract_features(stage1_model_wrapper, dataset, output_dir, use_connector, is_ddp)
if __name__ == '__main__':
    args = parse_args()
    if args.is_ddp:
        xmp.spawn(main, args=(args.config_path, args.output_dir, args.use_connector, args.is_ddp, args.train_split), start_method='fork')
    else:
        main(0, args.config_path, args.output_dir, args.use_connector, args.is_ddp, args.train_split)