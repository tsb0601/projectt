import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rqvae.models.connectors import ReshapeAndSplit_connector
path = sys.argv[1]

a = torch.load(path)

reshape_and_split = ReshapeAndSplit_connector(a, remove_cls = True)
bn = a.bn
running_mean = bn.running_mean
running_var = bn.running_var
