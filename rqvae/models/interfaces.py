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

from git import Optional
from torch import nn
import torch
from typing import Tuple, Union

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
    def __init__(self):
        super().__init__()

    def forward(self, encodings: Stage1Encodings) -> Stage1Encodings: # from stage1 to stage2
        raise NotImplementedError
    def reverse(self, encodings: Union[Stage1Encodings,Stage2ModelOutput]) -> Stage1Encodings: # from stage2 to stage1
        raise NotImplementedError
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


class Stage2ModelWrapper(XLA_Model):
    """
    Wrap a Stage2 model with a Stage1 model.
    """

    def __init__(self, stage_1_model: Stage1Model, stage_2_model: Stage2Model, connector: Optional[base_connector] = None):
        super().__init__()
        self.stage_1_model = stage_1_model
        self.stage_2_model = stage_2_model
        if connector is None:
            connector = id_connector()
        self.connector = connector 
        self.stage_1_model.requires_grad_(False)  # freeze the stage 1 model
        self.stage_2_model.requires_grad_(True)  # train the stage 2 model
    def forward(self, inputs: LabeledImageData) -> Tuple[Stage1Encodings, Stage2ModelOutput]:
        with torch.no_grad():
            stage1_encodings = self.stage_1_model.encode(inputs)
            stage1_encodings = self.connector.forward(stage1_encodings)
        stage2_output = self.stage_2_model(stage1_encodings, inputs)
        return stage1_encodings, stage2_output

    def compute_loss(self, stage1_encodings: Stage1Encodings ,stage2_output: Stage2ModelOutput , inputs: LabeledImageData , **kwargs) -> dict:
        return self.stage_2_model.compute_loss(stage1_encodings, stage2_output, inputs, **kwargs)

    @torch.no_grad()
    def get_recon_imgs(self, zs, zs_pred) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stage_2_model.get_recon_imgs(zs, zs_pred)

    @torch.no_grad()
    def infer(self, inputs: LabeledImageData) -> Stage1ModelOutput:
        stage_2_gen = self.stage_2_model.infer(inputs)
        stage_1_encodings = self.connector.reverse(stage_2_gen)
        stage_1_gen = self.stage_1_model.decode(stage_1_encodings)
        return stage_1_gen
    def get_last_layer(self) -> torch.Tensor:
        return self.stage_2_model.get_last_layer()