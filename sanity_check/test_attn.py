from torch.nn.functional import scaled_dot_product_attention   
import torch_xla.core.xla_model as xm

import torch

device = 'cpu'

query = torch.randn(2, 4, 3).to(device)
key = torch.randn(2, 4, 3).to(device)
value = torch.randn(2, 4, 3).to(device)

output, attention = scaled_dot_product_attention(query, key, value)

print(output.shape, attention.shape)