from numpy.core.multiarray import CLIP
from transformers import Dinov2Model, CLIPVisionModel
from torch import nn
import torch


class Dinov2Wrapper(nn.Module):
    def __init__(self, model_name):
        super(Dinov2Wrapper, self).__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
    def forward(self, x):
        return self.model(x).last_hidden_state

    

class ClipWrapper(nn.Module):
    def __init__(self, model_name):
        super(ClipWrapper, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
    def forward(self, x):
        return self.model(x).last_hidden_state
    
MODEL_ARCH = {
    'dinov2': Dinov2Wrapper,
    'clip': ClipWrapper
}