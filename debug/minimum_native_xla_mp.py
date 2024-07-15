import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch.optim as optim
import torch.nn as nn
import torch
from torch_xla.distributed import xla_backend
from torch.utils.data import DataLoader, Dataset
#from torch_xla.amp import syncfree
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.size = size
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        random_data = self.data[index]
        random_label = int(random_data.sum() > 0)
        return random_data, random_label
    def __len__(self):
        return self.len

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
    print(f'Rank: {rank}, World Size: {args.world_size}, Disenv: {disenv}, device: {device}')
    dataset = RandomDataset((256), 1000)
    print(f'Rank: {rank}, Dataset Size: {len(dataset)}')
    model = Simple_Linear(256)
    print(f'Rank: {rank}, Model: {model}')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.requires_grad_(True)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.AdamW(trainable_params, lr=1e-3)
    print(f'Rank: {rank}, Optimizer: {optimizer.state_dict()}')
    if args.load_path != '':
        model.load_state_dict(torch.load(args.load_path, map_location='cpu'))
    model = model.to(device)
    model.train()
    
    for epoch in range(args.epochs):
        #pl_loader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
        pl_loader = pl.MpDeviceLoader(dataloader, device)
        for data, target in pl_loader:
            optimizer.zero_grad(set_to_none=True)
            print(f'Rank: {rank}, Data: {data.shape}, Target: {target.shape}')
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            #print(f'Rank: {rank}, grads: {model.linear.weight.grad}')
            #xm.optimizer_step(optimizer, pin_layout=False)
    if args.save_path != '':
        state_dict = xm._maybe_convert_to_cpu(model.state_dict())
        if xm.get_ordinal() == 0:
            torch.save(state_dict, args.save_path)
from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--epochs', type=int, default=1)
    return parser.parse_known_args()
if __name__ == '__main__':
    args, extra_args = parse_args()
    xmp.spawn(main, args=(args, extra_args))