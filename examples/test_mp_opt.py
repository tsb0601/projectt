from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch_xla
import torch_xla.amp
import torch_xla.amp.syncfree
import torch_xla.core
from torchvision.models import VisionTransformer
import torch_xla.core.xla_model as xm
import itertools
from torch_xla.distributed.parallel_loader import ParallelLoader
from torch_xla.core.xla_model import collective_broadcast
from tqdm import tqdm
from torch_xla.distributed import xla_backend
from torch_xla.distributed import xla_multiprocessing as xmp
import torch_xla.runtime as xr
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
import os
from contextlib import nullcontext
cache_compile_path = './xla_compile/test_mp_opt' 
os.makedirs(cache_compile_path, exist_ok=True)
xr.initialize_cache(cache_compile_path, readonly=False)
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met
from torch_xla.debug.profiler import trace
import torch.autograd.profiler as profiler
profile_log_path = './profile' 
tracing = True
os.makedirs(profile_log_path, exist_ok=True) 
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = int(length)
        self.size = size
    def __getitem__(self, index):
        return torch.randn(*self.size)
    def __len__(self):
        return self.len
    def _collate_fn(self, batch):
        return torch.stack(batch)
### below are some vision models with seq length 64 ###
def model_S(): # 5.8M
    return VisionTransformer(
        image_size = 256,
        patch_size = 32,
        num_layers= 6,
        num_heads = 16,
        hidden_dim = 256,
        mlp_dim = 1024
    )
def model_M(): # 
    return VisionTransformer(
        image_size = 256,
        patch_size = 32,
        num_layers= 12,
        num_heads = 16,
        hidden_dim = 256,
        mlp_dim = 1024
    )
def model_L(): # 88.24M
    return VisionTransformer(
        image_size = 256,
        patch_size = 32,
        num_layers= 12,
        num_heads = 16,
        hidden_dim = 768,
        mlp_dim = 3072
    )
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def broadcast_master_param(model: torch.nn.Module) -> None:
  """
  Broadcast the model parameters from master process to other processes
  """
  parameters_and_buffers = list(
      itertools.chain(model.parameters(), model.buffers()))
  collective_broadcast(parameters_and_buffers, pin_layout=True)
  xm.mark_step()
def main(rank:int = None):
    print(f'xm.ordinal: {xm.get_ordinal()}')
    device = xm.xla_device() # current device
    print(f'[!]device: {device}')
    global profile_log_path , tracing
    world_size = xm.xrt_world_size() # world size
    rank = xm.get_ordinal() # rank
    cpu_model = model_S()
    model = cpu_model.to(device)
    # broadcast the model to all devices
    xm.rendezvous('init')
    xm.master_print(f'[!]broadcasting model parameters...')
    broadcast_master_param(model)
    param_count = count_parameters(model)
    xm.master_print(f'[!]model broadcasted, total trainable parameters: {param_count/1e6:.2f}M')
    xm.mark_step()
    # create optimizer
    optimizer_cls_dict = {
        'naive': optim.AdamW,
        'syncfree': torch_xla.amp.syncfree.AdamW,
        'zero': partial(ZeroRedundancyOptimizer, optimizer_class=optim.AdamW),
    }
    choice = 'naive' # change to see other optimizers' performance
    optimizer = optimizer_cls_dict[choice](model.parameters(), lr=1e-4)
    #optimizer = ZeroRedundancyOptimizer(
    #    model.parameters(),
    #    optimizer_class=optim.AdamW,
    #    lr=1e-4,
    #)
    xm.master_print(f'[!]optimizer created')
    xm.mark_step()
    # create data loader
    dataset = RandomDataset((3, 256, 256), 1e4) # 1e4 is large enough to see the performance
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    loader_train = DataLoader(
        dataset,
        batch_size=8, # small local batch size 8
        sampler=sampler_train,
        num_workers=4,
        drop_last=True,
        collate_fn=dataset._collate_fn
    )
    xm.mark_step()
    # start training
    server = xp.start_server(9012, only_on_master=True) if tracing else None
    model.train()
    loss_fct = nn.CrossEntropyLoss()
    for epoch in range(1):
        met.clear_all()
        loader = ParallelLoader(loader_train, [device]).per_device_loader(device)       
        tbar = tqdm(loader, total=len(loader), desc=f'[!]epoch {epoch}', disable=not xm.is_master_ordinal())
        for i, data in enumerate(tbar):
            if i == 0 and xm.is_master_ordinal() and tracing:
                xp.trace_detached('localhost:9012', profile_log_path, duration_ms = 100000) # trace 100s, change the duration if needed
            with xp.StepTrace('train_cls',step_num=i) if tracing else nullcontext():
                with xp.Trace('build_graph') if tracing else nullcontext():
                    data = data.to(device)
                    labels = torch.zeros(data.size(0), dtype=torch.long).to(data.device) # pseudo labels
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fct(output, labels)
                with xp.Trace('backward') if tracing else nullcontext():
                    loss.backward()
                with xp.Trace('reduce_grad') if tracing else nullcontext():
                    if choice != 'zero': # zero optimizer has its own reduce_gradients
                        xm.reduce_gradients(optimizer)
                with xp.Trace('step') if tracing else nullcontext():
                    optimizer.step()
                if not tracing: # if tracing StepTrace will do the mark step
                    xm.mark_step()
            tbar.set_postfix({'loss': loss.item()}) # fetching is not good but it won't hurt much
    xm.mark_step()
    xm.master_print(met.metrics_report())
    xm.master_print(f'[!]training finished')  
if __name__ == '__main__':
    xmp.spawn(main, args=(), start_method='fork')
    #main()