from dataclasses import dataclass
import datetime
import os
import collections
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla as xla
import torch_xla.distributed.xla_backend # must be imported as init
import torch_xla.core.xla_model as xm


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
    num_gpus: int
    master: bool
    device_name: str
    TPU: bool


def initialize(args, logger=None):

    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if args.world_size > 1:
        #os.environ["RANK"] = str(args.rank)
        #os.environ["WORLD_SIZE"] = str(args.world_size)
        #os.environ["LOCAL_RANK"] = str(args.local_rank)
        local_rank = os.environ["LOCAL_RANK"]
        print(f'[dist] Distributed: wait dist process group:{local_rank}')
        dist.init_process_group(backend=args.dist_backend, init_method='xla://',
        timeout=datetime.timedelta(0, args.timeout))
        print(
            f"""[dist] Distributed: success device:{local_rank}, """,
            f"""{dist.get_rank()}/{dist.get_world_size()}"""
        )
        distenv = DistEnv(world_size=dist.get_world_size(),
                          world_rank=dist.get_rank(),
                          local_rank=local_rank,
                          num_gpus=1,
                          master=(dist.get_rank() == 0),
                          device_name=str(xm.xla_real_devices([str(xm.xla_device())])[0]),
                          TPU=True
                          )
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
