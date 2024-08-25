# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import Subset
import torchvision
from torchvision.datasets import ImageNet,ImageFolder
from .imagenet import ImageNet_wImagepath, ImageNet_Fake
from .dummy import Dummy_Dataset
from .lsun import LSUNClass
from .ffhq import FFHQ
from .transforms import create_transforms
from .interfaces import LabeledImageDatasetWrapper
SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))
def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split='val', is_eval=is_eval)

    root = config.dataset.get('root', None)

    if config.dataset.type == 'imagenet':
        root = root if root else 'data/imagenet'
        dataset_trn = ImageNet_wImagepath(root, split='train', transform=transforms_trn)
        dataset_val = ImageNet_wImagepath(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'imagenet_recon':
        root = root if root else 'data/imagenet/val_256' # special judge
        dataset_trn = Dummy_Dataset(root, transform=transforms_val)
        dataset_val = Dummy_Dataset(root, transform=transforms_val)
    elif config.dataset.type == 'imagenet_test':
        root = root if root else 'data/imagenet'
        dataset_trn = ImageNet_wImagepath(root, split='val', transform=transforms_trn)
        dataset_val = ImageNet_wImagepath(root, split='val', transform=transforms_val)
        #dataset_trn = Subset(dataset_trn, torch.randperm(len(dataset_trn))[:1024])
        # choose the first image and repeat it for 1024 time for training
        dataset_trn = Subset(dataset_trn, [0]*512) # lets get a bit more samples
        dataset_val = Subset(dataset_val, list(range(16)))
    elif config.dataset.type == 'imagenet_u':
        root = root if root else 'data/imagenet'
        dataset_trn = ImageNet_Fake(root, split='train', transform=transforms_trn)
        dataset_val = ImageNet_Fake(root, split='val', transform=transforms_val)
        dataset_trn = Subset(dataset_trn, torch.randperm(len(dataset_trn))[:512])
        dataset_val = Subset(dataset_val, torch.randperm(len(dataset_val))[:256])
    elif config.dataset.type == 'ffhq':
        root = root if root else 'data/ffhq'
        dataset_trn = FFHQ(root, split='train', transform=transforms_trn)
        dataset_val = FFHQ(root, split='val', transform=transforms_val)
    elif config.dataset.type in ['LSUN-cat', 'LSUN-church', 'LSUN-bedroom']:
        root = root if root else 'data/lsun'
        category_name = config.dataset.type.split('-')[-1]
        dataset_trn = LSUNClass(root, category_name=category_name, transform=transforms_trn)
        dataset_val = LSUNClass(root, category_name=category_name, transform=transforms_trn)
    else:
        raise ValueError('%s not supported...' % config.dataset.type)
    dataset_trn = LabeledImageDatasetWrapper(dataset_trn)
    dataset_val = LabeledImageDatasetWrapper(dataset_val)
    if logger is not None:
        logger.info(f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')
    return dataset_trn, dataset_val
