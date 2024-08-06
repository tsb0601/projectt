
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.distributed.xla_backend
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
from torch_xla.amp import syncfree, autocast
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--epochs', type=int, default=3)
    return parser.parse_known_args()
args, extra_args = parse_args()

#from torch_xla.amp import syncfree
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.size = size
    def __getitem__(self, index):
        random_data = torch.randn(self.size)
        random_label = 0
        return random_data, random_label
    def __len__(self):
        return self.len
class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
class Simple_Linear(nn.Module):
    def __init__(self, size:int):
        super(Simple_Linear, self).__init__()
        self.linear = nn.Linear(size, 2)
        self.size = size
    def forward(self, x):
        return self.linear(x)
def dist_setup(rank, world_size):
    disenv = (xm.get_ordinal(), xm.xrt_world_size())
    return disenv
def main(rank, args, extra_args):
    device = xm.xla_device()
    disenv = dist_setup(rank, args.world_size)
    print(f'Rank: {rank}, World Size: {xm.xrt_world_size()}, Disenv: {disenv}, device: {device}')
    dataset = RandomDataset((256), 10000)
    print(f'Rank: {rank}, Dataset Size: {len(dataset)}')
    model = Simple_Linear(256)
    print(f'Rank: {rank}, Model: {model}')
    if xm.xrt_world_size() > 1:
        sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
        dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.requires_grad_(True)
    model = model.to(device)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.load_path != '':
        model.load_state_dict(torch.load(args.load_path, map_location='cpu'))
    #optimizer = optim.AdamW(trainable_params, lr=1e-3)
    optimizer = syncfree.AdamW(trainable_params, lr=1e-2)
    print(f'Rank: {rank}, Optimizer: {optimizer.state_dict()}')
    
    xm.broadcast_master_param(model)
    model.train()
    pl_loader = pl.MpDeviceLoader(dataloader, device)
    for epoch in range(args.epochs):
        #pl_loader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
        for data, target in pl_loader:
            optimizer.zero_grad()
            with autocast(xm.xla_device()):
                output = model(data)
                loss = criterion(output, target)
            acc = (output.detach().argmax(1) == target).sum() / target.size(0)
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.master_print(f'Loss: {loss.item()}, Acc: {acc.item()}')
    if args.save_path != '':
        state_dict = xm._maybe_convert_to_cpu(model.state_dict())
        if xm.get_ordinal() == 0:
            torch.save(state_dict, args.save_path)
def _mp_fn(rank, args, extra_args):
    main(rank, args, extra_args)
if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(args, extra_args))