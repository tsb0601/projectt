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

from numpy import shape
from torch import nn
import torch
from header import *
# create a dataclass for ModelOutput
from dataclasses import dataclass
from rqvae.img_datasets.interfaces import LabeledImageData
@dataclass
class Stage1ModelOutput:
    xs_recon: torch.Tensor
    additional_attr: dict = None


@dataclass
class Stage2ModelOutput:
    zs_pred: torch.Tensor
    zs_degraded: torch.Tensor
    additional_attr: dict = None


@dataclass
class Stage1Encodings:
    zs: torch.Tensor
    additional_attr: dict = None


class XLA_Model(nn.Module, metaclass=abc.ABCMeta):
    """
    an abstract class for STage1 and Stage2 models
    """

    @abc.abstractmethod
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        """Inference the model."""
        pass

    @abc.abstractmethod
    def get_recon_imgs(
        self, x, xs, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        don't actually need this, but for the sake of consistency
        """
        raise NotImplementedError
    @abc.abstractmethod
    def get_last_layer(self) -> torch.Tensor:
        """Get the last layer of the model."""
        pass
class base_connector(nn.Module, metaclass=abc.ABCMeta): # for connecting stage1 and stage2 models
    def __init__(self, bn: nn.modules.batchnorm._BatchNorm = None):
        super().__init__()
        self.bn = bn  
        if bn is not None:
            # register forward hook
            self.hook_forward()
    def hook_forward(self):
        self.__class__.__call__ = self.wrap_call # wrap the forward call
    @abc.abstractmethod
    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings: # from stage1 to stage2
        raise NotImplementedError
    @abc.abstractmethod
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings: # from stage2 to stage1
        raise NotImplementedError
    @torch.no_grad()
    def _forward_hook(self, encodings: Stage1Encodings) -> Stage1Encodings:
        latent = encodings.zs
        normed_latent = self.bn(latent)
        return Stage1Encodings(zs=normed_latent, additional_attr=encodings.additional_attr)
    @torch.no_grad()
    def normalize(self, encodings: Stage1Encodings) -> Stage1Encodings:
        if self.bn is None:
            return encodings
        zs = encodings.zs
        # expand to match zs dim: B,C ,L or B,C, H,W
        running_mean = self.bn.running_mean.view(1, -1, *(1 for _ in range(zs.dim() - 2)))
        running_var = self.bn.running_var.view(1, -1, *(1 for _ in range(zs.dim() - 2)))
        zs = (zs - running_mean) / torch.sqrt(running_var + self.bn.eps)
        return Stage1Encodings(zs=zs, additional_attr=encodings.additional_attr)
    @torch.no_grad()
    def unnormalize(self, encodings: Union[Stage1Encodings, Stage2ModelOutput]) -> Stage1Encodings:
        if self.bn is None:
            return encodings
        zs = encodings.zs if isinstance(encodings, Stage1Encodings) else encodings.zs_pred
        # expand to match zs dim: B,C ,L or B,C, H,W
        running_mean = self.bn.running_mean.view(1, -1, *(1 for _ in range(zs.dim() - 2)))
        running_var = self.bn.running_var.view(1, -1, *(1 for _ in range(zs.dim() - 2)))
        zs = zs * torch.sqrt(running_var + self.bn.eps) + running_mean
        return Stage1Encodings(zs=zs, additional_attr=encodings.additional_attr)
    def wrap_call(self, encodings: Stage1Encodings):
        encodings = self.forward(encodings)
        self._forward_hook(encodings)
        return encodings
    #add a hook on forward to get mean and var
from .connectors import id_connector
class Stage1Model(XLA_Model):

    @abc.abstractmethod
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        """Encode the input image to the latent space."""
        pass

    @abc.abstractmethod
    def decode(self,outputs: Stage1Encodings) -> Stage1ModelOutput:
        """Decode the latent code to the image space."""
        pass

    @abc.abstractmethod
    def compute_loss(self,  outputs: Stage1ModelOutput, inputs: LabeledImageData, **kwargs) -> dict:
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
    def forward(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        """Forward pass of the model."""
        pass



class Stage2Model(XLA_Model):
    """A template for the Stage2 model."""

    @abc.abstractmethod
    def compute_loss(
        self, stage1_encodings: Stage1Encodings, stage2_output: Stage2ModelOutput, inputs: LabeledImageData
    ) -> dict:
        """Compute the losses necessary for training.
        Typically, it would be the cross-entropy of the AR prediction w.r.t. the ground truth.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self, stage1_encodings: Stage1Encodings, inputs: LabeledImageData
    ) -> Stage2ModelOutput:
        """Forward pass of the model."""
        pass

    @abc.abstractmethod
    def infer(self, *args, **kwargs) -> Stage2ModelOutput:
        """Inference the model."""
        pass

class Stage1ModelWrapper(Stage1Model):
    def __init__(self, stage_1_model: Stage1Model, connector: Optional[base_connector] = None):
        super().__init__()
        self.stage_1_model = stage_1_model
        if connector is None:
            connector = id_connector()
        self.connector = connector
        self.connector.requires_grad_(True) # train the connector
        self.train()
    def forward(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        encodings = self.stage_1_model.encode(inputs)
        encodings = self.connector(encodings) # call __call__ instead of forward to make hook work
        encodings = self.connector.reverse(encodings)
        outputs = self.stage_1_model.decode(encodings)
        return outputs
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        encodings = self.stage_1_model.encode(inputs)
        encodings = self.connector(encodings) # call __call__ instead of forward to make hook work
        return encodings
    def decode(self, encodings: Stage1Encodings) -> Stage1ModelOutput:
        encodings = self.connector.reverse(encodings)
        return self.stage_1_model.decode(encodings)
    def compute_loss(self, outputs: Stage1ModelOutput, inputs: LabeledImageData, **kwargs) -> dict:
        return self.stage_1_model.compute_loss(outputs, inputs, **kwargs)
    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        return self.forward(inputs)
    @torch.no_grad()
    def get_recon_imgs(self, x, xs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stage_1_model.get_recon_imgs(x, xs)
    def get_last_layer(self) -> torch.Tensor:
        return self.stage_1_model.get_last_layer()
class Stage2ModelWrapper(Stage2Model):
    """
    Wrap a Stage2 model with a Stage1 model.
    """

    def __init__(self, stage_1_model: Stage1Model, stage_2_model: Stage2Model, connector: Optional[base_connector] = None, do_normalize: bool = False):
        super().__init__()
        self.stage_1_model = stage_1_model
        self.stage_2_model = stage_2_model
        if connector is None:
            connector = id_connector()
        self.connector = connector 
        self.stage_1_model.requires_grad_(False)  # freeze the stage 1 model
        self.stage_2_model.requires_grad_(True)  # train the stage 2 model
        self.connector.requires_grad_(False)  # freeze the connector
        self.do_normalize = do_normalize
        self.stage_1_model.eval()
        self.connector.eval()
        self.stage_2_model.train()
    def encode(self, inputs: LabeledImageData) -> Stage1Encodings:
        with torch.no_grad():
            stage1_encodings = self.stage_1_model.encode(inputs)
            stage1_encodings = self.connector.forward(stage1_encodings)
            if self.do_normalize:
                stage1_encodings = self.connector.normalize(stage1_encodings) # normalize the latent space
        return stage1_encodings
    def forward(self, inputs: LabeledImageData) -> Tuple[Stage1Encodings, Stage2ModelOutput]:
        stage1_encodings = self.encode(inputs)
        stage2_output = self.stage_2_model(stage1_encodings, inputs)
        return stage1_encodings, stage2_output
    def decode(self, stage1_encodings: Stage2ModelOutput) -> Stage1ModelOutput:
        if self.do_normalize:
            stage1_encodings = self.connector.unnormalize(stage1_encodings)
        stage1_encodings = self.connector.reverse(stage1_encodings)
        stage1_gen = self.stage_1_model.decode(stage1_encodings)
        return stage1_gen
    def compute_loss(self, stage1_encodings: Stage1Encodings ,stage2_output: Stage2ModelOutput , inputs: LabeledImageData , **kwargs) -> dict:
        return self.stage_2_model.compute_loss(stage1_encodings, stage2_output, inputs, **kwargs)

    @torch.no_grad()
    def get_recon_imgs(self, zs, zs_pred) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stage_2_model.get_recon_imgs(zs, zs_pred)

    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        stage_2_gen = self.stage_2_model.infer(inputs)
        stage_1_gen = self.decode(stage_2_gen)
        return stage_1_gen
    def get_last_layer(self) -> torch.Tensor:
        return self.stage_2_model.get_last_layer()