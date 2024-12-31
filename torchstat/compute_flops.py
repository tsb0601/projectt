import torch.nn as nn
import torch
import numpy as np


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_flops(module, inp, out)
    elif isinstance(module, nn.GELU):
        return compute_GELU_flops(module, inp, out)
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out)
    elif isinstance(module, nn.GroupNorm):
        return compute_GroupNorm_flops(module, inp, out)
    elif isinstance(module, nn.Identity):
        return 0 # Identity layer does not have any flops
    elif isinstance(module, nn.LayerNorm):
        return compute_LayerNorm_flops(module, inp, out)
    elif isinstance(module, nn.Embedding):
        return compute_Embedding_flops(module, inp, out)
    else:
        unsupported_ops_flops:set = globals().get('unsupported_ops_flops')
        if unsupported_ops_flops is None:
            globals()['unsupported_ops_flops'] = set()
            unsupported_ops_flops = globals()['unsupported_ops_flops']
        if type(module).__name__ not in unsupported_ops_flops:
            unsupported_ops_flops.add(type(module).__name__)
            print(f"[Flops]: {type(module).__name__} is not supported!")
        return 0
    pass


def compute_Conv2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops

def compute_LayerNorm_flops(module, inp, out):
    assert isinstance(module, nn.LayerNorm)
    #print('compute_LayerNorm_flops', inp.size(), out.size())
    assert len(inp.size()) == len(out.size()), f'input and output should have same dimensions, inp: {inp.size()}, out: {out.size()}'
    
    # Extract dimensions
    num_elements = inp.numel()
    num_features = module.normalized_shape[-1]
    
    # FLOPs include both multiply and add operations, calculated similarly to MADD
    mean_flops = num_elements  # One add per element for mean
    variance_flops = num_elements * 2  # One subtract and one square per element for variance
    normalize_flops = num_elements * 5  # Extra multiply for standard deviation correction (sqrt)

    # Total FLOPs
    flops = mean_flops + variance_flops + normalize_flops
    return flops
def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    in_c, in_h, in_w = inp.size()[1:]
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops

def compute_GroupNorm_flops(module, inp, out):
    assert isinstance(module, nn.GroupNorm)
    #assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    #in_c, in_h, in_w = inp.size()[1:]
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops

def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for s in inp.size()[1:]:
        active_elements_count *= s

    return active_elements_count

def compute_GELU_flops(module, inp, out):
    assert isinstance(module, nn.GELU)
    #print('compute_GELU_flops', inp.size(), out.size())
    assert len(inp.size()) == len(out.size()), f'input and output should have same dimensions, inp: {inp.size()}, out: {out.size()}'
    
    # FLOPs calculations for GELU
    # Including tanh approximation: 6 multiplies, 4 adds, 1 exp (approximated by 4 flops)
    num_elements = inp.numel()
    
    flops_per_element = 6 + 4 + 4  # 6 multiplications, 4 additions, and 4 for the exp approximation in tanh
    flops = num_elements * flops_per_element
    return flops

def compute_Pool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    assert len(inp.size()) <= 4 and len(out.size()) <= 4
    batch_size = inp.size()[0]
    seq_len = 1
    for i in range(1, len(inp.size())-1):
        seq_len *= inp.size()[i]
    return batch_size * inp.size()[-1] * out.size()[-1] * seq_len

def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for s in output_size.shape[1:]:
        output_elements_count *= s

    return output_elements_count

def compute_Embedding_flops(module, inp, out):
    assert isinstance(module, nn.Embedding)
    return 0 # Embedding layer does not have any flops
