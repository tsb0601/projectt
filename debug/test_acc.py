import torch_xla.core.xla_model as xm
device = xm.xla_device()
import torch
import torch.nn as nn
import torch_xla
linear_layer = nn.Linear(2, 2)
print(linear_layer.weight.dtype)
#torch_xla._XLAC._xla_set_mat_mul_precision('high')
print(torch.get_float32_matmul_precision())
def print_tensor(x: torch.Tensor, precision: int = 4, in_bf16: bool = False):
    # print x with precision as matrix
    assert x.dim() == 2
    if in_bf16:
        bf16_unit = 0.0078125
    else:
        bf16_unit = 1.
    for row in x:
        print(' '.join([f'{val/bf16_unit:.{precision}f}' for val in row.tolist()]))
        
x = torch.randn(2, 2)

weight = torch.eye(2) # identity matrix

id_x = x

x = x @ weight

assert torch.allclose(x, id_x), f"[CPU]Expected {id_x} but got {x}"
x = torch.randn(2, 2, device=device)
linear_layer = nn.Linear(2, 2).to(device)
weight = linear_layer.weight
pweight = torch.pinverse(weight)
id_mat = weight @ pweight
print_tensor(id_mat, 10)
id_x = x
x = x @ id_mat
x = x.cpu()
id_x = id_x.cpu()
if not torch.allclose(x, id_x):
    print('[TPU]Expected:')
    print_tensor(id_x,10, True)
    print('[TPU]Got:')
    print_tensor(x,10, True)