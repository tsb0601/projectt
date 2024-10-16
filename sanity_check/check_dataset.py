import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.img_datasets import create_dataset

from PIL import Image
import torch
from rqvae.models.interfaces import *
import sys
import os
from omegaconf import OmegaConf


config_path = sys.argv[1]

config = OmegaConf.load(config_path)

dataset_trn, dataset_val = create_dataset(config)
print('dataset_trn:', len(dataset_trn),type(dataset_trn))
print('dataset_val:', len(dataset_val),type(dataset_val))
# test one sample
sample = dataset_trn[114514]
sample: LabeledImageData
print(f'train data sample:\
img shape: {sample.img.shape},\
img dtype: {sample.img.dtype},\
condition shape: {sample.condition.shape},\
condition dtype: {sample.condition.dtype},\
img[0,0]: {sample.img[0,0]},\
condition: {sample.condition},\
img_path: {sample.img_path}')
sample = dataset_val[0]
sample: LabeledImageData
print(f'train data sample:\
img shape: {sample.img.shape},\
img dtype: {sample.img.dtype},\
condition shape: {sample.condition.shape},\
condition dtype: {sample.condition.dtype},\
img[0,0]: {sample.img[0,0]},\
condition: {sample.condition},\
img_path: {sample.img_path}')