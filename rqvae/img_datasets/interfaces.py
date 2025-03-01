from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import Optional, Any, List
from numpy import ndarray
@dataclass
class LabeledImageData:
    img: torch.Tensor = None
    condition: Any = None
    img_path: str = None
    additional_attr: Optional[str] = None
    def _to(self, device_or_dtype): # inplace
        if self.img is not None and isinstance(self.img, torch.Tensor):
            self.img = self.img.to(device_or_dtype)
        if (self.condition is not None) and isinstance(self.condition, torch.Tensor):
            self.condition = self.condition.to(device_or_dtype).long() # shame on me! This is so bad
        return self
    def to(self, device_or_dtype):
        data = LabeledImageData(
            img=self.img.to(device_or_dtype) if (self.img is not None) and isinstance(self.img, torch.Tensor) else self.img,
            condition=self.condition.to(device_or_dtype).long() if (self.condition is not None) and isinstance(self.condition, torch.Tensor) else self.condition,
            img_path=self.img_path,
            additional_attr=self.additional_attr
        )
        return data
    def __getitem__(self, idx):
        return LabeledImageData(
            img=self.img[idx] if (self.img is not None) and isinstance(self.img, torch.Tensor) else self.img,
            condition=self.condition[idx] if (self.condition is not None) and isinstance(self.condition, torch.Tensor) else self.condition,
            img_path=self.img_path[idx] if self.img_path is not None else self.img_path,
            additional_attr=self.additional_attr[idx] if self.additional_attr is not None else self.additional_attr
        )
    def __len__(self):
        img_len = len(self.img) if (self.img is not None) and isinstance(self.img, torch.Tensor) else 1
        condition_len = len(self.condition) if (self.condition is not None) and isinstance(self.condition, torch.Tensor) else 1
        return max(img_len, condition_len)
    def __setitem__(self, idx, value):
        raise ValueError('Cannot set item to LabeledImageData')
class LabeledImageDatasetWrapper(Dataset):
    """
    We assume the base dataset returns a tuple of (img, label, img_path, additional_attr), the last two being optional.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        if hasattr(dataset, 'collate_fn'):
            self.collate_fn = dataset.collate_fn # override the collate_fn
        else:
            self.collate_fn = self.default_collate_fn
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, LabeledImageData):
            return item
        elif isinstance(item, tuple):
            return LabeledImageData(*item)
        elif isinstance(item, dict):
            return LabeledImageData(**item)
        elif isinstance(item, torch.Tensor):
            return LabeledImageData(img=item)
        else:
            raise ValueError(f'Unsupported return type from base dataset: {type(item)}')
    def default_collate_fn(self, batch: List[LabeledImageData]) -> LabeledImageData:
        img, condition, img_path, additional_attr = zip(*[(x.img, x.condition, x.img_path, x.additional_attr) for x in batch])
        # if condition is tensor then stack
        if condition[0] is not None:
            if isinstance(condition[0], torch.Tensor):
                condition = torch.stack(condition)
            elif isinstance(condition[0], int) or isinstance(condition[0], float) or isinstance(condition[0], bool) or isinstance(condition[0], ndarray):
                condition = torch.Tensor(condition)
        img = torch.stack(img)
        return LabeledImageData(img=img, condition=condition, img_path=img_path, additional_attr=additional_attr)