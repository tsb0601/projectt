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

import abc

from torch import nn
import torch
from typing import Tuple
# create a dataclass for ModelOutput
from dataclasses import dataclass
@dataclass
class Stage1ModelOutput:
    xs_recon: torch.Tensor
    additional_attr: dict

@dataclass
class Stage2ModelOutput:
    zs_pred: torch.Tensor
    zs_degraded: torch.Tensor
    additional_attr: dict
    
@dataclass
class Stage1Encodings:
    zs: torch.Tensor
    additional_attr: dict
class XLA_Model(nn.Module, metaclass=abc.ABCMeta):
    """
    an abstract class for STage1 and Stage2 models
    """
    @abc.abstractmethod
    def infer(self, *args, **kwargs):
        """Inference the model.
        """
        pass
    @abc.abstractmethod
    def get_recon_imgs(self, *args, **kwargs)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        don't actually need this, but for the sake of consistency
        """
        raise NotImplementedError
class Stage1Model(XLA_Model):

    #@abc.abstractmethod
    #def get_codes(self, *args, **kwargs):
    #    """Generate the code from the input."""
    #    pass

    #@abc.abstractmethod
    #def decode_code(self, *args, **kwargs):
    #    """Generate the decoded image from the given code."""
    #    pass
    # for vq based mode you should use the above two, but for more general models only the below two are needed
    @abc.abstractmethod
    def get_recon_imgs(self, *args, **kwargs)-> Tuple[torch.Tensor, torch.Tensor]:
        """Scales the real and recon images properly.
        """
        pass
    @abc.abstractmethod
    def encode(self, *args, **kwargs)-> Stage1Encodings:
        """Encode the input image to the latent space.
        """
        pass
    @abc.abstractmethod
    def decode(self, *args, **kwargs)-> Stage1ModelOutput:
        """Decode the latent code to the image space.
        """
        pass
    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs)-> dict:
        """Compute the losses necessary for training.

        return {
            'loss_total': ...,
            'loss_recon': ...,
            'loss_latent': ...,
            'codes': ...,
            ...
        }
        """
        pass
    @abc.abstractmethod
    def forward(self, *args, **kwargs)-> Stage1ModelOutput:
        """Forward pass of the model.
        """
        pass
    @abc.abstractmethod
    def get_last_layer(self, *args, **kwargs)-> torch.Tensor:
        """Get the last layer of the model.
        """
        pass
class Stage2Model(XLA_Model):
    """A template for the Stage2 model."""
    
    @abc.abstractmethod
    def compute_loss(self, zs_pred, zs ,*args, **kwargs)-> dict:
        """Compute the losses necessary for training.
        Typically, it would be the cross-entropy of the AR prediction w.r.t. the ground truth.
        """
        raise NotImplementedError
    @abc.abstractmethod
    def forward(self, *args, **kwargs)-> Stage2ModelOutput:
        """Forward pass of the model.
        """
        pass
    @abc.abstractmethod
    def infer(self, *args, **kwargs)-> Stage2ModelOutput:
        """Inference the model.
        """
        pass
class Stage2ModelWrapper(XLA_Model):
    """
    Wrap a Stage2 model with a Stage1 model.
    """
    def __init__(self, stage_1_model: Stage1Model, stage_2_model: Stage2Model):
        super().__init__()
        self.stage_1_model = stage_1_model
        self.stage_2_model = stage_2_model
        self.stage_1_model.requires_grad_(False) # freeze the stage 1 model
        self.stage_2_model.requires_grad_(True) # train the stage 2 model
    def forward(self, *args, **kwargs) -> Tuple[Stage1Encodings, Stage2ModelOutput]:
        with torch.no_grad():
            stage1_encodings = self.stage_1_model.encode(*args, **kwargs)
        stage2_output = self.stage_2_model(stage1_encodings, *args, **kwargs)
        return stage1_encodings, stage2_output
    def compute_loss(self, zs_pred, zs , *args, **kwargs) -> dict:
        return self.stage_2_model.compute_loss(zs_pred, zs, *args, **kwargs)
    @torch.no_grad()
    def get_recon_imgs(self, zs, zs_pred) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.stage_1_model.decode(zs).xs_recon
        xs_pred = self.stage_1_model.decode(zs_pred).xs_recon
        return xs, xs_pred
    @torch.no_grad()
    def infer(self, *args, **kwargs) -> Stage1ModelOutput:
        stage_2_gen = self.stage_2_model.infer(*args, **kwargs)
        zs_gen = stage_2_gen.zs_pred
        stage_1_gen = self.stage_1_model.decode(zs_gen)
        return stage_1_gen

    

