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

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def create_transforms(config, split='train', is_eval=False):
    if config.transforms.type.startswith('imagenetDiT'):
        # parse resolution from 'imagenet{}x{}'.format(resolution, resolution)
        resolution = int(config.transforms.type.split('x')[-1])
        if split == 'train' and not is_eval:
            #weak data augmentation
            transforms_ = [
                #transforms.Resize(resolution),
                #transforms.RandomCrop(resolution),
                lambda x: center_crop_arr(x, resolution), # following DiT
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        else:
            transforms_ = [
                lambda x: center_crop_arr(x, resolution),
                #transforms.Resize(256),
                #transforms.CenterCrop(256),
                #transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif config.transforms.type.startswith('imagenetweak'):
        # parse resolution from 'imagenet{}x{}'.format(resolution, resolution)
        resolution = int(config.transforms.type.split('x')[-1])
        if split == 'train' and not is_eval:
            #weak data augmentation
            transforms_ = [
                transforms.Resize(int(resolution * 1.1), interpolation=3),
                transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        else:
            transforms_ = [
                lambda x: center_crop_arr(x, resolution),
                #transforms.Resize(256),
                #transforms.CenterCrop(256),
                #transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif config.transforms.type.startswith('imagenet'):
        # parse resolution from 'imagenet{}x{}'.format(resolution, resolution)
        resolution = int(config.transforms.type.split('x')[-1])
        if split == 'train' and not is_eval:
            #weak data augmentation
            transforms_ = [
                transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), interpolation=3),  # following MAE pretraining
                transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        else:
            transforms_ = [
                lambda x: center_crop_arr(x, resolution),
                #transforms.Resize(256),
                #transforms.CenterCrop(256),
                #transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif 'ffhq' in config.transforms.type:
        resolution = int(config.transforms.type.split('_')[0].split('x')[-1])
        if split == 'train' and not is_eval:
            transforms_ = [
                transforms.RandomResizedCrop(resolution, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        else:
            transforms_ = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif config.transforms.type in ['LSUN', 'LSUN-cat', 'LSUN-church', 'LSUN-bedroom']:
        resolution = 256 # only 256 resolution is supoorted for LSUN
        transforms_ = [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    elif config.transforms.type == 'none':
        transforms_ = []
    else:
        raise NotImplementedError('%s not implemented..' % config.transforms.type)

    transforms_ = transforms.Compose(transforms_)

    return transforms_
