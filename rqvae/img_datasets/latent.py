from .interfaces import LabeledImageData
from torch.utils.data import Dataset
import os
import numpy as np
import torch
class CustomDataset(Dataset):
    """
    https://github.com/chuanyangjin/fast-DiT/blob/main/train.py#L97
    """
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]
        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()
        return LabeledImageData(img=features, condition=labels, img_path=feature_file)
    
    
        

