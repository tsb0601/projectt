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

from .ema import ExponentialMovingAverage
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch
import os
import torch.distributed as dist
from rqvae.models.interfaces import Stage1Model
from omegaconf import DictConfig
from typing import Optional, Tuple
DEBUGING = os.environ.get('DEBUG', False)
from .utils import *
def xm_step_every_layer(model:nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Sequential):
            for mm in m:
                mm.register_forward_hook(lambda m, i, o: xm.mark_step())
        else:
            m.register_forward_hook(lambda m, i, o: xm.mark_step())
def create_model(config:DictConfig, ema:bool=False)->Tuple[Stage1Model, Optional[ExponentialMovingAverage]]:
    # config: OmegaConf.DictConfig
    # config to dict for model init
    model = instantiate_from_config(config)
    model_ema = instantiate_from_config(config) if ema else None
    if DEBUGING: # add xm_step for faster compilation and more reusable compilation cache
        if dist.is_initialized() and dist.get_rank() == 0:
            print('[!]DEBUGGING: Adding xm_step to every layer. This will slow down the training.')
        xm_step_every_layer(model)
        if ema:
            xm_step_every_layer(model_ema)
    if ema:
        raise NotImplementedError('Exponential Moving Average is not implemented yet.')
        model_ema = ExponentialMovingAverage(model_ema, config.ema.mu)
        model_ema.eval()
        model_ema.update(model, step=-1)
    return model, model_ema
