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
import argparse
from rqvae.utils.utils import set_seed

def update_argument_parser(parser):
    parser.add_argument('--dist-backend', default='xla', type=str, help='distributed backend')
    parser.add_argument(
        '--local_rank', default=-1, type=int,
        help='Used for multi-process training. Can either be manually set ' +
             'or automatically set by using \'python -m torch.distributed.launch\'.')
    return parser


@dataclass
class DistEnv:
    world_size: int
    world_rank: int
    local_rank: int
    master: bool
    device_name: str
    TPU: bool


def initialize(args: argparse.Namespace , logger =None):
    # see if args have rank or world_size, if not, try to get from env
    rank = args.rank if hasattr(args, 'rank') else int(os.environ.get("RANK", 0))
    world_size = args.world_size if hasattr(args, 'world_size') else int(os.environ.get('WORLD_SIZE', 1))
    local_rank = args.local_rank if hasattr(args, 'local_rank') else int(os.environ.get('LOCAL_RANK', 0))
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    if args.world_size > 1:
        # recaculate rank and world_size
        local_rank = rank % 4 # we have 4 cores per TPU
        args.local_rank = local_rank
        print(f'[dist] Distributed: wait dist process group:{rank}')
        dist.init_process_group(backend=args.dist_backend, init_method='xla://', world_size=world_size, rank=rank,
        timeout=datetime.timedelta(0, args.timeout))
        print(
            f"""[dist] Distributed: success device:{rank}, """,
            f"""{local_rank}/{dist.get_rank()}/{dist.get_world_size()}"""
        )
        distenv = DistEnv(world_size=dist.get_world_size(), # or xm.xrt_world_size()
                          world_rank=dist.get_rank(), # or xm.get_ordinal()
                          local_rank=local_rank,
                          master=(dist.get_rank() == 0), # or xm.is_master_ordinal()
                          device_name=str(xm.xla_real_devices([str(xm.xla_device())])[0]),
                          TPU=True
                          )
        # set seed for each process to avoid same seed
        # the seed would be a function of (args.seed, dist.get_rank()) and should be random enough
        # for example, generate random seed from args.seed for (rank + 1) times
        seed = args.seed
        for _ in range(rank + 1):
            seed = hash((seed, rank)) 
            # convert seed to [0, 2^32 - 1]
            seed = seed % (2**32 - 1)
        set_seed(seed) # set seed for each process
    else:
        print('[dist] Single processed')
        raise ValueError('Single processed is currently not supported')
        distenv = DistEnv(1, 0, 0, xm.xrt_world_size(), True, f'TPU:{str(xm.xla_device())[-1]}')

    print(f'[dist] {distenv}')

    if logger is not None:
        logger.info(distenv)

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


def param_sync(param):
    dist.broadcast(param, 0)
    dist.barrier()
    #torch.cuda.synchronize()



@torch.no_grad()
def all_gather_cat(distenv, tensor, dim=0):
    if distenv.world_size == 1:
        return tensor
    #use xm instead of dist to avoid bug
    g_tensor = xm.all_gather(tensor, dim=dim, pin_layout=False) # pin_layout = True sometimes causes error in xla
    #g_tensor = torch.cat(g_tensor, dim=dim)
    return g_tensor
