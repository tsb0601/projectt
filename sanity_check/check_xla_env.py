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
    world_size = xm.xrt_world_size()
    print(f'rank: {rank}, local_rank: {local_rank}, global_rank: {global_rank}')
    print(f'xla device: {xm.xla_real_devices([str(device)])[0]}')
    # try rendevous
    xm.rendezvous('rend test')
    xm.master_print('rendezvous done')
    is_world_master = rank == 0
    signup_sheet = torch.zeros(world_size, dtype=torch.int32) # a tensor to store the signup sheet
    signup_sheet[global_rank] = 1 # mark the signup sheet
    # reduce the signup sheet
    signup_sheet = signup_sheet.to(device) # move the signup sheet to the device
    signup_sheet = xm.all_reduce('sum', signup_sheet) # reduce the signup sheet
    if is_world_master:
        if not torch.all(signup_sheet == 1):
            print('Error: not all the devices signed up')
            print(signup_sheet)
        else:
            print('All devices signed up')
    

if __name__ == '__main__':
    xmp.spawn(check_xla_env, args=(), start_method='fork')