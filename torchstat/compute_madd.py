"""
compute Multiply-Adds(MAdd) of each leaf module
"""

import torch.nn as nn


def compute_Conv2d_madd(module, inp, out):
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    # ops per output element
    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
    kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_ConvTranspose2d_madd(module, inp, out):
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
    kernel_add_group = kernel_add * in_h * in_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_BatchNorm2d_madd(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]

    
    # 1. sub mean
    # 2. div standard deviation
    # 3. mul alpha
    # 4. add beta
    
    return 4 * in_c * in_h * in_w

def compute_GroupNorm_madd(module, inp, out):
    """
    ref: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    Input can be (N, C , *), where * is any number of additional dimensions
    Output will be (N, C, *)
    """
    assert isinstance(module, nn.GroupNorm)
    #assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    
    #in_c, in_h, in_w = inp.size()[1:]
    in_c = inp.size()[1]
    input_shapes = inp.size()[2:]
    input_dim_prod = 1
    for s in input_shapes:
        input_dim_prod *= s
    # 1. sub mean
    # 2. div standard deviation
    # 3. mul alpha
    # 4. add beta
    return 4 * in_c * input_dim_prod

def compute_MaxPool2d_madd(module, inp, out):
    assert isinstance(module, nn.MaxPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    return (k_h * k_w - 1) * out_h * out_w * out_c


def compute_AvgPool2d_madd(module, inp, out):
    assert isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    kernel_add = k_h * k_w - 1
    kernel_avg = 1

    return (kernel_add + kernel_avg) * (out_h * out_w) * out_c


def compute_ReLU_madd(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6))

    count = 1
    for i in inp.size()[1:]:
        count *= i
    return count


def compute_Softmax_madd(module, inp, out):
    assert isinstance(module, nn.Softmax)
    assert len(inp.size()) > 1

    count = 1
    for s in inp.size()[1:]:
        count *= s
    exp = count
    add = count - 1
    div = count
    return exp + add + div


def compute_Linear_madd(module, inp, out): # linear can be applied to 1d, 2d tensors
    assert isinstance(module, nn.Linear)
    #print('compute_Linear_madd',inp.size(),out.size())
    assert len(inp.size()) == len(out.size()), f'input and output should have same dimensions, inp: {inp.size()}, out: {out.size()}'
    assert len(inp.size()) <= 4 and len(out.size()) <= 4, f'inp: {inp.size()}, out: {out.size()}, only 1d, 2d, tensors are supported for Linear layer'
    num_in_features = inp.size()[-1]
    num_out_features = out.size()[-1]
    L = 1
    for i in range(1, len(inp.size())-1):
        L *= inp.size()[i]
    mul = num_in_features
    add = num_in_features - 1
    return num_out_features * (mul + add) * L


def compute_Bilinear_madd(module, inp1, inp2, out):
    assert isinstance(module, nn.Bilinear)
    assert len(inp1.size()) == len(inp2.size()) == len(out.size()), f'input and output should have same dimensions, inp1: {inp1.size()}, inp2: {inp2.size()}, out: {out.size()}'
    assert len(inp1.size()) <= 3 and len(inp2.size()) <= 3 and len(out.size()) <= 3, f'inp1: {inp1.size()}, inp2: {inp2.size()}, out: {out.size()}, only 1d, 2d, tensors are supported for Bilinear layer'

    num_in_features_1 = inp1.size()[-1]
    num_in_features_2 = inp2.size()[-1]
    num_out_features = out.size()[-1]
    if len(inp1.size()) == 3:
        L = inp1.size()[1]
    else:
        L = 1
    mul = num_in_features_1 * num_in_features_2 + num_in_features_2
    add = num_in_features_1 * num_in_features_2 + num_in_features_2 - 1
    return num_out_features * (mul + add) * L


def compute_madd(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_madd(module, inp, out)
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_madd(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_madd(module, inp, out)
    elif isinstance(module, nn.MaxPool2d):
        return compute_MaxPool2d_madd(module, inp, out)
    elif isinstance(module, nn.AvgPool2d):
        return compute_AvgPool2d_madd(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return compute_ReLU_madd(module, inp, out)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_madd(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_madd(module, inp, out)
    elif isinstance(module, nn.Bilinear):
        return compute_Bilinear_madd(module, inp[0], inp[1], out)
    elif isinstance(module, nn.GroupNorm):
        return compute_GroupNorm_madd(module, inp, out)
    elif isinstance(module, nn.Identity):
        return 0 # Identity layer does not have any MAdd
    else:
        unsupported_op_madd:set = globals().get('unsupported_ops_madd')
        if unsupported_op_madd is None:
            globals()['unsupported_ops_madd'] = set()
            unsupported_op_madd = globals()['unsupported_ops_madd']
        if type(module).__name__ not in unsupported_op_madd:
            unsupported_op_madd.add(type(module).__name__)
            print(f"[MAdd]: {type(module).__name__} is not supported!")
        return 0
