import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
path = sys.argv[1]
model = torch.load(path)
print(model)
C = model.bn.running_var.shape[0]
running_mean = model.bn.running_mean.view(C, -1)
running_var = model.bn.running_var.view(C, -1)
# print the mean of var of running_mean
print(running_mean.var(dim=-1).mean(), running_mean.abs().mean(),running_mean.device)
# print the var of var of running_var
print(running_var.var(dim=-1).mean(), running_var.mean())