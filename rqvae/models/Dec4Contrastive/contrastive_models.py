from numpy.core.multiarray import CLIP
from transformers import Dinov2Model, CLIPVisionModel, SiglipModel, SiglipImageProcessor
from torch import nn
import torch
import torch.nn.functional as F
class SigLIPEncoder(nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", num_tokens=64):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.hidden_size = 1152  # SigLIP-SO400M hidden size        
        self.load_model()
        self.vision_tower.eval()
    def load_model(self):
        model = SiglipModel.from_pretrained(self.model_name)
        processor = SiglipImageProcessor.from_pretrained(self.model_name)
        
        self.vision_tower = model.vision_model
        self.processor = processor
    @torch.no_grad() # encoder is always frozen
    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)

        outputs = self.vision_tower(images, output_hidden_states=True)
        image_features = outputs.hidden_states[-1]
        
        b, num_tokens, dim = image_features.shape
        h = w = int(num_tokens**0.5)
        target_h = target_w = int(self.num_tokens**0.5)

        if self.num_tokens!=729:
            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2)
            image_features = F.interpolate(image_features, size=(target_h, target_w), mode='bilinear', align_corners=False)
            image_features = image_features.permute(0, 2, 3, 1).contiguous().view(b, self.num_tokens, dim)

        # Normalize vision if needed
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features
    @torch.no_grad()
    def encode_image(self, image):

        features = self(image)
        
        return features

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

class SiglipWrapper(nn.Module):
    def __init__(self, model_name):
        super(SiglipWrapper, self).__init__()
        self.model = SigLIPEncoder(model_name)
        self.empty_cls = torch.zeros(1, 1, self.model.hidden_size) # should be frozen
    def forward(self, x):
        latent = self.model(x)
        cls = self.empty_cls.expand(latent.size(0), -1, -1)
        return torch.cat([cls, latent], dim=1)
MODEL_ARCH = {
    'dinov2': Dinov2Wrapper,
    'clip': ClipWrapper,
    'siglip': SiglipWrapper,
}