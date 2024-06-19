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
from .rqvae import get_rqvae
#from .rqtransformer import get_rqtransformer
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch
import os
DEBUGING = os.environ.get('DEBUG', False)
class dummy_loss:
    def __init__(self) -> None:
        self.fix_code = torch.Tensor([1,4,5]).long()
    def get_last_layer (self):
        return nn.Identity()
class dummy_model(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.module = dummy_loss()
        self.dummy_conv = nn.Conv2d(3,3,1)
        self.fix_code = torch.ones((8,8,4)).long()
    def forward(self, xs , *args, **kwargs):
        xs_recon = self.dummy_conv(xs)
        # expand the code to the same batchsize as the input
        self.fix_code = self.fix_code.expand(xs.shape[0],-1,-1,-1)
        return xs_recon, 0.114
    def compute_loss(self, *args, xs ):
        output_dict = {
            'loss_total' : xs.mean(),
            'loss_recon' : xs.mean(),
            'loss_latent': xs.mean(),
            'codes': self.fix_code
        }
        return output_dict
    def get_last_layer (self):
        return self.dummy_conv.weight
def xm_step_every_layer(model:nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Sequential):
            for mm in m:
                mm.register_forward_hook(lambda m, i, o: xm.mark_step())
        else:
            m.register_forward_hook(lambda m, i, o: xm.mark_step())
def create_model(config, ema=False):
    model_type = config.type.lower()
    if model_type == 'rq-transformer':
        raise ValueError(f'{model_type} is invalid..')
        #model = get_rqtransformer(config)
        #model_ema = get_rqtransformer(config) if ema else None
    elif model_type == 'rq-vae':
        model = get_rqvae(config)
        model_ema = get_rqvae(config) if ema else None
        #model = dummy_model()
        #model_ema = dummy_model() if ema else None
        # add hook to call the mark_step function after forward to every module in the model
    else:
        raise ValueError(f'{model_type} is invalid..')
    if DEBUGING: # add xm_step for faster compilation and more reusable compilation cache
        xm_step_every_layer(model)
        if ema:
            xm_step_every_layer(model_ema)
    if ema:
        model_ema = ExponentialMovingAverage(model_ema, config.ema)
        model_ema.eval()
        model_ema.update(model, step=-1)
    return model, model_ema
