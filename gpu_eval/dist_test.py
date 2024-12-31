
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
import os
def _mp_fn(rank, flags):
    torch.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=2)
    print("Rank", rank)
    this_device = f"cuda:{rank}"
    print("Device", this_device)
    test_tensor = torch.tensor([rank]).cuda(this_device) 
    gathered_tensor = [torch.zeros_like(test_tensor) for _ in range(2)] if rank == 1 else None
    print("Rank", rank, "test_tensor", test_tensor,"gathered_tensor", gathered_tensor)
    g_tensor = dist.gather(test_tensor, gathered_tensor, dst=1)
    print("Rank", rank, "gathered_tensor", gathered_tensor)
    dist.destroy_process_group()

if __name__ == "__main__":
    flags = {}
    mp.spawn(_mp_fn, args=(flags,), nprocs=2, start_method="fork")