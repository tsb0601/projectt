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

class XLA_Model(nn.Module, metaclass=abc.ABCMeta):
    """
    an abstract class for STage1 and Stage2 models
    """

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
    def get_recon_imgs(self, *args, **kwargs):
        """Scales the real and recon images properly.
        """
        pass
    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        """Encode the input image to the latent space.
        """
        pass
    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        """Decode the latent code to the image space.
        """
        pass
    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
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
    def get_last_layer(self, *args, **kwargs):
        """Get the last layer of the model.
        """
        pass

class Stage2Model(XLA_Model):
    """A template for the Stage2 model."""
    
    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        """Compute the losses necessary for training.
        Typically, it would be the cross-entropy of the AR prediction w.r.t. the ground truth.
        """
        raise NotImplementedError
    @abc.abstractmethod
    def get_recon_imgs(self, *args, **kwargs):
        """
        don't actually need this, but for the sake of consistency
        """
        raise NotImplementedError
    def get_block_size(self):
        raise NotImplementedError

class Stage2ModelWrapper(XLA_Model):
    """
    Wrap a Stage2 model with a Stage1 model.
    """
    def __init__(self, stage_1_model: Stage1Model, stage_2_model: Stage2Model):
        super().__init__()
        self.stage_1_model = stage_1_model
        self.stage_2_model = stage_2_model
        self.stage_1_model.requires_grad_(False) # freeze the stage 1 model

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            stage_1_output = self.stage_1_model(*args, **kwargs)
        stage_2_output = self.stage_2_model(*stage_1_output, *args, **kwargs)
        return stage_2_output
    def compute_loss(self, *args, **kwargs):
        return self.stage_2_model.compute_loss(*args, **kwargs)
    def get_recon_imgs(self, *args, **kwargs):
        return self.stage_2_model.get_recon_imgs(*args, **kwargs)
