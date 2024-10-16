from .interfaces import LabeledImageData
from torch.utils.data import Dataset
import os
import numpy as np
import torch
class LatentDataset(Dataset):
    """
    https://github.com/chuanyangjin/fast-DiT/blob/main/train.py#L97
    """
    def __init__(self, features_dir):
        self.features_dir = features_dir

        self.features_files = sorted(os.listdir(features_dir))

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        features = np.load(os.path.join(self.features_dir, feature_file))
        latent  = torch.from_numpy(features['latent']).float()
        condition = torch.from_numpy(features['condition']) 
        return LabeledImageData(img=latent, condition=condition, img_path=feature_file)
    
    
        

