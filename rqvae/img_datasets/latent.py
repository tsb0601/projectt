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
    
# do multiprocess load to speed up, read all features into memory
import multiprocessing 
def read_file(args):
    feature_file, features_dir = args
    with open(os.path.join(features_dir, feature_file), 'rb') as f:
        features = np.load(f, allow_pickle=True)
        latent  = torch.from_numpy(features['latent']).float()
        condition = torch.from_numpy(features['condition']) 
    return LabeledImageData(img=latent, condition=condition, img_path=feature_file)

def multiprocess_read_files(file_paths:list, features_dir:str):
    # Limit the number of open files by batching the file paths
    batch_size = 12800
    file_contents = []
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        print('batch:', i, len(batch))
        with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count() // 4)) as pool: # use 1/4 of the cpu cores as TPU has 4 cores for DDP
            file_contents.extend(pool.map(read_file, [(file_path, features_dir) for file_path in batch]))
    return file_contents
import time
class LatentDatasetPreLoaded(Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.features_files = sorted(os.listdir(features_dir))
        # load all features
        start_load = time.time()
        self.features = multiprocess_read_files(self.features_files, features_dir)
        print('load features time:', time.time()-start_load)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
    
    
