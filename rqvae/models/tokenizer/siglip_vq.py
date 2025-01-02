
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import torch_xla.core.xla_model as xm
from .quantizer import VectorQuantizer



class SigLIPVQEncoder(nn.Module):
    def __init__(
        self, 
        model_name="google/siglip-so400m-patch14-384",
        num_tokens=64,
        embedding_dim=1152,  # SigLIP's native dimension
        num_codebook_vectors=8192,  # Size of VQ codebook
        commitment_cost=0.25,
        use_commitment=False,
        trainable=False,
        progressive_unfreeze=False,
        unfreeze_after_steps=50000,
        unfreeze_strategy='gradual',  # 'all' or 'gradual'
        device=None
    ):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.hidden_size = embedding_dim
        self.device = device
        self.trainable = trainable
        self.progressive_unfreeze = progressive_unfreeze
        self.unfreeze_after_steps = unfreeze_after_steps
        self.unfreeze_strategy = unfreeze_strategy
        
        # Load SigLIP model
        self.load_model()
        
        # Add VQ layer
        self.vq = VectorQuantizer(
            num_embeddings=num_codebook_vectors,
            embedding_dim=embedding_dim,
            use_commitment=use_commitment,
            commitment_cost=commitment_cost
        )
        
        # Move VQ to device explicitly
        if self.device:
            self.vq = self.vq.to(self.device)

            
        # Initially freeze all parameters if not trainable
        if not trainable:
            self.freeze_encoder()
 

    def load_model(self):
        xm.rendezvous('pre_load_model')
        model = AutoModel.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.vision_tower = model.vision_model
        if self.device:
            self.vision_tower = self.vision_tower.to(self.device)
        self.processor = processor
        xm.rendezvous('post_load_model')

    def freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.vision_tower.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters"""
        for param in self.vision_tower.parameters():
            param.requires_grad = True
            
    def unfreeze_layer(self, layer_idx):
        """Unfreeze a specific layer"""
        for param in self.encoder_layers[layer_idx].parameters():
            param.requires_grad = True
            
    def update_freeze_status(self, global_step):
        """Update which layers are frozen based on training progress"""
        if not self.progressive_unfreeze or global_step < self.unfreeze_after_steps:
            return
            
        xm.rendezvous('pre_update_freeze')
        
        if self.unfreeze_strategy == 'all':
            self.unfreeze_encoder()
            self.progressive_unfreeze = False  # Done unfreezing
      
                
        xm.rendezvous('post_update_freeze')

    def forward(self, images):
        # Handle single image
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Get SigLIP embeddings with appropriate grad handling
        with torch.set_grad_enabled(self.trainable):
            outputs = self.vision_tower(images, output_hidden_states=True)
            image_features = outputs.hidden_states[-1]
        
        # Reshape if necessary
        b, num_tokens, dim = image_features.shape
        h = w = int(num_tokens**0.5)
        target_h = target_w = int(self.num_tokens**0.5)

        if self.num_tokens != num_tokens:
            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2)
            image_features = F.interpolate(
                image_features, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            image_features = image_features.permute(0, 2, 3, 1).contiguous()
            image_features = image_features.view(b, self.num_tokens, dim)

        # Normalize embeddings before VQ
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # Apply VQ
        quantized, vq_loss, encoding_indices = self.vq(image_features)
        
        return quantized, vq_loss, encoding_indices

    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.processor(images=image, return_tensors="pt")['pixel_values']
        if self.device:
            image = image.to(self.device)
        
        with torch.no_grad():
            quantized, _, indices = self(image)
        return quantized

    def get_codebook_usage(self):
        """Get codebook usage statistics"""
        return self.vq.get_usage_stats()