import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from zmq import has

from torchstat import compute_madd
from torchstat import compute_flops
from torchstat import compute_memory


class ModelHook(object):
    def __init__(self, model, example_input, model_fn:str = 'forward'):
        assert isinstance(model, nn.Module)
        #assert isinstance(input_size, (list, tuple))

        self._model = model
        self._input_size = example_input
        self._origin_call = dict()  # sub module call hook
        self._hook_model()
        #x = torch.rand(1, *self._input_size)  # add module duration time
        x = example_input
        self._model.eval()
        calling_fn = getattr(self._model, model_fn)
        calling_fn(x)
    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)
        def _register_buff(module:nn.Module, name:str, value:torch.Tensor):
            module: nn.Module
            #module.name = value # initialize the buffer
            module.__setattr__(name, value)
            #if hasattr(module, name):
            #    module.name = value # initialize the buffer
            #else:
            #    module.register_buffer(name, value)
        if len(list(module.children())) > 0:
            return
        #module.register_buffer('_input_shape', torch.zeros(3).int())
        #module.register_buffer('_output_shape', torch.zeros(3).int())
        #module.register_buffer('parameter_quantity', torch.zeros(1).int())
        #module.register_buffer('inference_memory', torch.zeros(1).long())
        #module.register_buffer('MAdd', torch.zeros(1).long())
        #module.register_buffer('duration', torch.zeros(1).float())
        #module.register_buffer('Flops', torch.zeros(1).long())
        #module.register_buffer('Memory', torch.zeros(2).long())
        _register_buff(module, '_input_shape', torch.zeros(3).int())
        _register_buff(module, '_output_shape', torch.zeros(3).int())
        _register_buff(module, 'parameter_quantity', torch.zeros(1).int())
        _register_buff(module, 'inference_memory', torch.zeros(1).long())
        _register_buff(module, 'MAdd', torch.zeros(1).long())
        _register_buff(module, 'duration', torch.zeros(1).float())
        _register_buff(module, 'Flops', torch.zeros(1).long())
        _register_buff(module, 'Memory', torch.zeros(2).long())
        
    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            # Itemsize for memory
            itemsize = input[0].detach().numpy().itemsize

            start = time.time()
            output = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))

            module._input_shape = torch.from_numpy(
                np.array(input[0].size()[1:], dtype=np.int32))
            module._output_shape = torch.from_numpy(
                np.array(output.size()[1:], dtype=np.int32))

            parameter_quantity = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                parameter_quantity += (0 if p is None else torch.numel(p.data))
            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.longlong))

            inference_memory = 1
            for s in output.size()[1:]:
                inference_memory *= s
            # memory += parameters_number  # exclude parameter memory
            inference_memory = inference_memory * 4 / (1024 ** 2)  # shown as MB unit
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            if len(input) == 1:
                madd = compute_madd(module, input[0], output)
                flops = compute_flops(module, input[0], output)
                Memory = compute_memory(module, input[0], output)
            elif len(input) > 1:
                madd = compute_madd(module, input, output)
                flops = compute_flops(module, input, output)
                Memory = compute_memory(module, input, output)
            else:  # error
                madd = 0
                flops = 0
                Memory = (0, 0)
            module.MAdd = torch.from_numpy(
                np.array([madd], dtype=np.int64))
            module.Flops = torch.from_numpy(
                np.array([flops], dtype=np.int64))
            Memory = np.array(Memory, dtype=np.int32) * itemsize
            module.Memory = torch.from_numpy(Memory)

            return output

        for module in self._model.modules():
            if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    @staticmethod
    def _retrieve_leaf_modules(model):
        leaf_modules = []
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:
                leaf_modules.append((name, m))
        return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self._retrieve_leaf_modules(self._model))
