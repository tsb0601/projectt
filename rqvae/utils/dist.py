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
import itertools
from torch_xla.core.xla_model import collective_broadcast
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
    use_ddp: bool

def initialize(args: argparse.Namespace , logger =None):
    # see if args have rank or world_size, if not, try to get from env
    #rank = args.rank if hasattr(args, 'rank') else int(os.environ.get("RANK", 0))
    #world_size = args.world_size if hasattr(args, 'world_size') else int(os.environ.get('WORLD_SIZE', 1))
    #local_rank = args.local_rank if hasattr(args, 'local_rank') else int(os.environ.get('LOCAL_RANK', 0))
    args.rank = xm.get_ordinal()
    args.world_size = xm.xrt_world_size()
    args.local_rank = xm.get_local_ordinal()
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    if args.world_size > 1:
        # recaculate rank and world_size
        print(f'[dist] Distributed: wait dist process group:{args.rank}')
        if args.use_ddp:
            dist.init_process_group(backend=args.dist_backend, init_method='xla://', world_size=args.world_size, rank=args.rank,
            timeout=datetime.timedelta(0, args.timeout))
        print(
            f"""[dist] Distributed: success device:{args.rank}, """,
            f"""{args.local_rank}/{args.rank}/{args.world_size}""",
        )
        distenv = DistEnv(world_size=args.world_size, # or xm.xrt_world_size()
                          world_rank=args.rank, # or xm.get_ordinal()
                          local_rank=args.local_rank,
                          master=(args.rank == 0), # or xm.is_master_ordinal()
                          device_name=str(xm.xla_real_devices([str(xm.xla_device())])[0]),
                          TPU=True,
                          use_ddp=args.use_ddp
                          )
        # set seed for each process to avoid same seed
        # the seed would be a function of (args.seed, xm.get_ordinal()) and should be random enough
        # for example, generate random seed from args.seed for (rank + 1) times
        seed = args.seed
        rank = args.rank
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

def broadcast_master_param(model: torch.nn.Module) -> None:
  """
  Broadcast the model parameters from master process to other processes
  """
  parameters_and_buffers = list(
      itertools.chain(model.parameters(), model.buffers()))
  collective_broadcast(parameters_and_buffers, pin_layout=True)
  xm.mark_step()
def dataparallel_and_sync(distenv, model, find_unused_parameters=True):
    if distenv.use_ddp:
        assert dist.is_initialized(), 'DistributedDataParallel requires torch.distributed to be initialized.'
        model = DDP(
            model,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True
        )
        #broadcast_master_param(model)
        for _, param in model.state_dict().items():
            dist.broadcast(param, 0)
        # could be replaced by xm.broadcast_master_param, but this is a feature for torchxla > 2.0
        # xm.broadcast_master_param(model.parameters(), 0)
        #dist.barrier()
    else:
        broadcast_master_param(model) # significantly slower than dist.broadcast before first compilation, so caching is really important
        #model = torch.nn.DataParallel(model)
    # do a check to see all models are the same
    test_param = next(model.parameters())
    test_param = test_param.detach().clone()
    xm.master_print(f'[dist] test_param: {test_param.shape}')
    if not distenv.use_ddp:
        all_params = xm.all_gather(test_param, dim=0)
        if distenv.master:
            for i, param in enumerate(all_params):
                if not torch.allclose(param, test_param, atol=1e-6):
                    print(f'[dist] rank {i} model is not the same as master model')
                    raise ValueError(f'[dist] rank {i} model is not the same as master model')
    xm.mark_step() # mark step for sync
    return model


#def param_sync(param):
#    xm.broadcast_master_param(param)
#    xm.mark_step()
    #dist.broadcast(param, 0)
    #dist.barrier()
    #torch.cuda.synchronize()



@torch.no_grad()
def all_gather_cat(distenv, tensor, dim=0):
    if distenv.world_size == 1:
        return tensor
    #use xm instead of dist to avoid bug
    g_tensor = xm.all_gather(tensor, dim=dim, pin_layout=False) # pin_layout = True sometimes causes error in xla
    #g_tensor = torch.cat(g_tensor, dim=dim)
    return g_tensor
