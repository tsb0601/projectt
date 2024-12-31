from dataclasses import dataclass
import datetime
import os
import collections
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla as xla
import torch_xla.distributed.xla_backend # must be imported for xla init_process_group
import torch_xla.core.xla_model as xm
from torch import nn
from tqdm import tqdm
import torch
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
xr.initialize_cache('~/.cache/xla_compile', readonly=False)


@dataclass
class DistEnv:
    world_size: int
    world_rank: int
    local_rank: int
    num_gpus: int
    master: bool
    device_name: str
    TPU: bool


def initialize(rank, world_size):
    rank = int(os.environ.get('RANK', 0))
    #world_size = int(os.environ.get('WORLD_SIZE', 1))
    #local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        #os.environ['RANK'] = str(args.rank)
        #os.environ['WORLD_SIZE'] = str(args.world_size)
        #os.environ['LOCAL_RANK'] = str(args.local_rank)
        #local_rank = os.environ['LOCAL_RANK']
        print(f'[dist] Distributed: wait dist process group:{rank}')
        dist.init_process_group(backend='xla', init_method='xla://',
        timeout=datetime.timedelta(0, 2000))
        print(
            f'[dist] Distributed: success device:{rank}, ',
            f'{dist.get_rank()}/{dist.get_world_size()}'
        )
        distenv = DistEnv(world_size=dist.get_world_size(), # or xm.xrt_world_size()
                          world_rank=dist.get_rank(), # or xm.get_ordinal()
                          local_rank=rank,
                          num_gpus=1,
                          master=(dist.get_rank() == 0), # or xm.is_master_ordinal()
                          device_name=str(xm.xla_real_devices([str(xm.xla_device())])[0]),
                          TPU=True
                          )
    else:
        print('[dist] Single processed')
        raise ValueError('Single processed is currently not supported')
        distenv = DistEnv(1, 0, 0, xm.xrt_world_size(), True, f'TPU:{str(xm.xla_device())[-1]}')

    print(f'[dist] {distenv}')

    return distenv


def dataparallel_and_sync(distenv, model, find_unused_parameters=True):
    if dist.is_initialized():
        model = DDP(
            model,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True
        )
        for _, param in model.state_dict().items():
            dist.broadcast(param, 0)
        # could be replaced by xm.broadcast_master_param, but this is a feature for torchxla > 2.0
        # xm.broadcast_master_param(model.parameters(), 0)
        dist.barrier()
        xm.mark_step() # not sure if this is necessary
    else:
        model = torch.nn.DataParallel(model)
    #torch.cuda.synchronize()
    return model

class dummy_model(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
def main(rank:int , world_size: int= 4):
    print(' rank {} ; world size {}'.format(rank, world_size))
    distenv = initialize(rank, world_size)
    model = dummy_model().to(xm.xla_device())
    model = dataparallel_and_sync(distenv, model) # now model is DDP
    cross_entropy = nn.CrossEntropyLoss()
    tbar = tqdm(range(10000),disable=not distenv.master)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    for i in tbar:
        random_data = torch.randn(1024, 10).to(xm.xla_device())
        output = model(random_data)
        random_label = torch.randint(0, 5, (1024,)).to(xm.xla_device())
        loss = cross_entropy(output, random_label)
        tbar.set_description(f'loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        xm.mark_step()
    print('local rank {} ; rank {} ; world size {}'.format(distenv.local_rank, distenv.world_rank, distenv.world_size))

if __name__ == '__main__':
    world_size = 4
    xmp.spawn(main, args=(world_size,))
    