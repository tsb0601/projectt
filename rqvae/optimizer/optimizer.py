# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_xla.core.xla_model as xm
from torch_xla.amp import syncfree
def create_resnet_optimizer(model, config):
    optimizer_type = config.type.lower()
    trainable_params_wname = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable_param_names = [name for name, _ in trainable_params_wname]
    trainable_params = [param for _, param in trainable_params_wname]
    all_param_cnt = sum([param.numel() for param in trainable_params])
    xm.master_print(f'All trainable parameters: {all_param_cnt}')
    xm.master_print(f'Creating optimizer with type {optimizer_type}')
    xm.master_print(f'trainable_params: {trainable_param_names}')
    optim_class = syncfree if config.amp else torch.optim
    xm.master_print(f'Using {optim_class} optimizer')
    xm.mark_step()
    if optimizer_type == 'adamw':
        optimizer = optim_class.AdamW(
            trainable_params, lr=config.init_lr, weight_decay=config.weight_decay,
            betas=config.betas
        )
    elif optimizer_type == 'adam':
        optimizer = optim_class.Adam(
            trainable_params, lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type == 'sgd':
        optimizer = optim_class.SGD(
            trainable_params, lr=config.init_lr, weight_decay=config.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f'{optimizer_type} invalid..')
    return optimizer


def create_optimizer(model, config):
    optimizer = create_resnet_optimizer(model, config.optimizer)
    return optimizer
