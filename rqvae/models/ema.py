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

import logging
from typing import OrderedDict
import torch

logger = logging.getLogger(__name__)

class ExponentialMovingAverage(torch.nn.Module):
    def __init__(self, init_module, mu):
        super(ExponentialMovingAverage, self).__init__()

        self.module = init_module
        # only keep the param that requires grad
        self.requires_grad_params = [name for name, param in self.module.named_parameters() if param.requires_grad]
        self.module.eval()
        self.module.requires_grad_(False) # make sure the model is not trainable
        self.mu = mu

    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)
    def infer(self, *args, **kwargs):
        return self.module.infer(*args, **kwargs)
    @torch.no_grad()
    def sync_when_init(self, module):
        keys = module.state_dict().keys()
        self_keys = self.module.state_dict().keys()
        for key in keys:
            if key in self_keys:
                self.module.state_dict()[key].copy_(module.state_dict()[key])
    @torch.no_grad()
    def update(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            mu = min(self.mu, (1. + step) / (10. + step))
        ema_params = OrderedDict(self.module.named_parameters())
        params = OrderedDict(module.named_parameters())
        for name in self.requires_grad_params:
            if name not in params:
                raise ValueError(f'[ExponentialMovingAverage] not found {name} in the model')
            if name not in ema_params:
                raise ValueError(f'[ExponentialMovingAverage] not found {name} in the EMA model')
            param = params[name]
            ema_params[name].mul_(mu).add_(param.data, alpha=1 - mu)
        #state_dict = {}
        #with torch.no_grad():
        #    for (name, m1), (name2, m2) in zip(self.module.state_dict().items(), module.state_dict().items()):
        #        if name != name2:
        #            logger.warning('[ExpoentialMovingAverage] not matched keys %s, %s', name, name2)
#
        #        if step is not None and step < 0:
        #            state_dict[name] = m2.clone().detach()
        #        else:
        #            state_dict[name] = ((mu * m1) + ((1.0 - mu) * m2)).clone().detach()
        #self.module.load_state_dict(state_dict)

    def compute_loss(self, *args, **kwargs):
        return self.module.compute_loss(*args, **kwargs)

    def get_recon_imgs(self, *args, **kwargs):
        return self.module.get_recon_imgs(*args, **kwargs)
