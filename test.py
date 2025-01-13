import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.runtime as xr

def dataparallel_and_sync(model, find_unused_parameters=True):
    """Unified function for DDP wrapping and parameter synchronization"""
    model = DDP(
        model,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True
    )
    # Broadcast parameters from rank 0
    for _, param in model.state_dict().items():
        dist.broadcast(param, 0)
   
    xm.mark_step()
    return model

def _mp_fn(index):
    device = xm.xla_device()
    dist.init_process_group('xla', init_method='xla://')

    torch.manual_seed(42)
    model = nn.Linear(128, 10).to(device)

    # xm.broadcast_master_param(model)
    model = dataparallel_and_sync(model)
    # xm.mark_step()

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=.001)

    for i in range(10):
        data, target = torch.randn((128, 128), device=device), torch.randn((128, 10), device=device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()
        xm.mark_step()

    # Print mean parameters so we can confirm they're the same across replicas
    print([p.mean() for p in model.parameters()])

if __name__ == '__main__':
    os.environ['PJRT_DEVICE'] = 'TPU'
    torch_xla.launch(_mp_fn)
