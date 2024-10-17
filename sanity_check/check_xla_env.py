"""
check the xla environment
"""
# first import a bunch of torch and numpy stuff
import torch
import torch_xla.core.xla_model as xm
import sys
from torch_xla.distributed.parallel_loader import ParallelLoader as pl
import os
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import torch_xla.distributed.xla_backend
import torch_xla.distributed.xla_multiprocessing as xmp

def check_xla_env(rank):
    device = xm.xla_device()
    local_rank = xm.get_local_ordinal()
    global_rank = xm.get_ordinal()
    print(f'rank: {rank}, local_rank: {local_rank}, global_rank: {global_rank}')
    print(f'xla device: {xm.xla_real_devices([str(device)])[0]}')
    

if __name__ == '__main__':
    xmp.spawn(check_xla_env, args=(), start_method='fork')