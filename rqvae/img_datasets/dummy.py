import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from .interfaces import LabeledImageData
class Dummy_Dataset(Dataset):
    """
    This dataset returns both PIL.Image and it's path, w/o label
    """
    def __init__(self, root, transform=None, img_ext = ('.jpg', '.jpeg', '.png')):
        self.root = root
        self.transform = transform
        self.imgs = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(img_ext)]
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return LabeledImageData(img=img, img_path=img_path)
    