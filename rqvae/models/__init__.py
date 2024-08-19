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
import os
import torch.distributed as dist
from rqvae.models.interfaces import *
from omegaconf import DictConfig
from rqvae.models.utils import load_model_from_ckpt
from typing import Optional, Tuple
import torch
DEBUGING = os.environ.get('DEBUG', False)
from .utils import *
def xm_step_every_layer(model:nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Sequential):
            for mm in m:
                mm.register_forward_hook(lambda m, i, o: xm.mark_step())
        else:
            m.register_forward_hook(lambda m, i, o: xm.mark_step())
def _create_according_to_config(config:DictConfig, use_ema:bool, stage:int)->Tuple[XLA_Model, Optional[ExponentialMovingAverage]]:
    assert stage in (0, 1, 2), f'[!]ERROR: Stage should be 0, 1 or 2 (0 is the connector), but got {stage}'
    model = instantiate_from_config(config)
    if config.get('ckpt_path', False):
        ckpt_path = config.ckpt_path
        _, keys = load_model_from_ckpt(model, ckpt_path, strict = False)
        xm.master_print(f'[!]INFO: Loaded Stage{stage} model from {ckpt_path} with keys: {keys}')
    model_ema = instantiate_from_config(config) if use_ema else None
    if DEBUGING: # add xm_step for faster compilation and more reusable compilation cache
        if dist.is_initialized() and dist.get_rank() == 0:
            print('[!]DEBUGGING: Adding xm_step to every layer. This will slow down the training.')
        xm_step_every_layer(model)
        if use_ema:
            xm_step_every_layer(model_ema)
    if use_ema:
        raise NotImplementedError('Exponential Moving Average is not implemented yet.')
        model_ema = ExponentialMovingAverage(model_ema, config.ema)
        model_ema.eval()
        model_ema.update(model, step=-1)
    model: XLA_Model
    model_ema: Optional[ExponentialMovingAverage]
    model = model
    if use_ema:
        model_ema = model_ema
    return model, model_ema
def create_model(config:DictConfig, ema:float=0.114514)->Tuple[XLA_Model, Optional[ExponentialMovingAverage]]:
    # config: OmegaConf.DictConfig
    # config to dict for model init    
    use_ema = (ema != 0.114514)
    stage_1_model, stage_1_ema = _create_according_to_config(config.stage_1, use_ema, stage=1)
    stage_1_model: Stage1Model
    stage_1_ema: Optional[ExponentialMovingAverage]
    if hasattr(config, 'connector'):
        connector, connector_ema = _create_according_to_config(config.connector, use_ema, stage=0)
    else:
        connector = None
    connector: Optional[base_connector]
    if hasattr(config, 'stage_2'):
        stage_2_model, stage_2_ema = _create_according_to_config(config.stage_2, use_ema, stage=2)
        stage_2_model: Stage2Model
        stage_2_ema: Optional[ExponentialMovingAverage]
        stage2model = Stage2ModelWrapper(stage_1_model, stage_2_model, connector)
        stage2ema = Stage2ModelWrapper(stage_1_ema, stage_2_ema, connector_ema) if use_ema else None
        if config.get('ckpt_path', False):
            ckpt_path = config.ckpt_path
            _, keys = load_model_from_ckpt(stage2model, ckpt_path, strict = False)
            xm.master_print(f'[!]INFO: Loaded Stage2Wrapper from {ckpt_path} with keys: {keys}')
            assert keys.unexpected_keys == [], f'[!]ERROR: Unexpected keys: {keys.unexpected_keys}'
        return stage2model, stage2ema
    else:
        stage1model = Stage1ModelWrapper(stage_1_model, connector)
        stage1ema = Stage1ModelWrapper(stage_1_ema, connector_ema) if use_ema else None
        if config.get('ckpt_path', False):
            ckpt_path = config.ckpt_path
            _, keys = load_model_from_ckpt(stage2model, ckpt_path, strict = False)
            xm.master_print(f'[!]INFO: Loaded Stage1Wrapper from {ckpt_path} with keys: {keys}')
            assert keys.unexpected_keys == [], f'[!]ERROR: Unexpected keys: {keys.unexpected_keys}'
        return stage1model, stage1ema
    #if stage == 2:
    #    stage_1_model,stage_1_ema = create_model(config.stage_1, ema=ema, stage=1) 
    #    stage_2_model, stage_2_ema = create_model(config.stage_2, ema=ema, stage=1)
    #    if hasattr(config, 'connector'):
    #        connector = instantiate_from_config(config.connector)
    #    else:
    #        connector = None
    #    connector: Optional[base_connector]
    #    stage2model = Stage2ModelWrapper(stage_1_model, stage_2_model, connector)
    #    stage2model_ema = Stage2ModelWrapper(stage_1_ema, stage_2_ema, connector) if use_ema else None 
    #    if config.get('ckpt_path', False):
    #        ckpt_path = config.ckpt_path
    #        _, keys = load_model_from_ckpt(stage2model, ckpt_path, strict = False)
    #        xm.master_print(f'[!]INFO: Loaded Stage2Wrapper from {ckpt_path} with keys: {keys}')
    #    return stage2model, stage2model_ema
    return model, model_ema
