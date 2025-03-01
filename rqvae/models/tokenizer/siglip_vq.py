
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import torch_xla.core.xla_model as xm
from .quantizer import VectorQuantizer
from .enhanced_quantizer import EnhancedVectorQuantizer


class SigLIPVQEncoder(nn.Module):

    def __init__(
        self, 
        model_name="google/siglip-so400m-patch14-384",
        num_tokens=64,
        embedding_dim=1152,
        num_codebook_vectors=8192,
        commitment_cost=0.25,
        use_commitment=False,
        clean_embedding_weight=1.0,
        trainable=False,
        progressive_unfreeze=False,
        unfreeze_after_steps=50000,
        unfreeze_strategy='gradual',
        device=None,
        use_vq=True,  # New parameter to control VQ usage
        kmeans_path=None,  # New parameter
        trainable_codebook=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.hidden_size = embedding_dim
        self.device = device
        self.trainable = trainable
        self.clean_embedding_weight = clean_embedding_weight
        self.use_vq = use_vq  # Store VQ mode
        
        # Store progressive unfreezing parameters
        self.progressive_unfreeze = progressive_unfreeze
        self.unfreeze_after_steps = unfreeze_after_steps
        self.unfreeze_strategy = unfreeze_strategy
        self.is_unfrozen = False
        
        # Load SigLIP model twice - one for training, one for reference
        self.load_model()
        
        # Add VQ layer only if using VQ mode
        if self.use_vq:
            # self.vq = VectorQuantizer(
            #     num_embeddings=num_codebook_vectors,
            #     embedding_dim=embedding_dim,
            #     use_commitment=use_commitment,
            #     commitment_cost=commitment_cost
            # )

            self.vq = EnhancedVectorQuantizer(
                num_embeddings=num_codebook_vectors,
                embedding_dim=self.hidden_size,
                kmeans_path=kmeans_path,  # Pass kmeans path
                trainable=trainable_codebook,
                use_commitment=use_commitment,
                commitment_cost=commitment_cost
            )

            if self.device:
                self.vq = self.vq.to(self.device)
        
        # Initialize in frozen state if using progressive unfreezing
        if self.progressive_unfreeze:
            self.freeze_encoder()

    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Get clean reference embeddings (always frozen)
        if self.use_vq:

            with torch.no_grad():
                ref_outputs = self.ref_vision_tower(images, output_hidden_states=True)
                ref_features = ref_outputs.hidden_states[-1]
                ref_features = self.process_features(ref_features)
        
        # Get trainable embeddings
        with torch.set_grad_enabled(self.trainable):
            outputs = self.vision_tower(images, output_hidden_states=True)
            image_features = outputs.hidden_states[-1]
            image_features = self.process_features(image_features)
        
        if self.use_vq:
            # VQ mode: apply vector quantization
            quantized, vq_loss, encoding_indices = self.vq(image_features)
            clean_loss = F.mse_loss(quantized, ref_features.detach())
            total_loss = vq_loss + self.clean_embedding_weight * clean_loss
            return quantized, total_loss, encoding_indices, clean_loss, vq_loss
        else:
            # Autoencoder mode: return features directly
            return (
                image_features,           # features instead of quantized
                torch.tensor(0.0, device=self.device),  # no vq_loss
                None,                    # no encoding indices
                torch.tensor(0.0, device=self.device),  # no vq_loss
                torch.tensor(0.0, device=self.device)  # no vq_loss
            )

    def load_model(self):
        xm.rendezvous('pre_load_model')
        
        # Load trainable model
        model = AutoModel.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)
        self.vision_tower = model.vision_model
        
        # Load reference model (always frozen)
        ref_model = AutoModel.from_pretrained(self.model_name)
        self.ref_vision_tower = ref_model.vision_model
        
        if self.device:
            self.vision_tower = self.vision_tower.to(self.device)
            self.ref_vision_tower = self.ref_vision_tower.to(self.device)
            
        # Freeze reference model
        for param in self.ref_vision_tower.parameters():
            param.requires_grad = False
            
        self.processor = processor
        xm.rendezvous('post_load_model')

    # def forward(self, images):
    #     if images.dim() == 3:
    #         images = images.unsqueeze(0)
        
    #     # Get clean reference embeddings (always frozen)
    #     with torch.no_grad():
    #         ref_outputs = self.ref_vision_tower(images, output_hidden_states=True)
    #         ref_features = ref_outputs.hidden_states[-1]
    #         ref_features = self.process_features(ref_features)
        
    #     # Get trainable embeddings
    #     with torch.set_grad_enabled(self.trainable):
    #         outputs = self.vision_tower(images, output_hidden_states=True)
    #         image_features = outputs.hidden_states[-1]
    #         image_features = self.process_features(image_features)
        
    #     # Apply VQ
    #     quantized, vq_loss, encoding_indices = self.vq(image_features)
    #     # print("encoding_indices", encoding_indices)
        
    #     # Use MSE loss instead of cosine similarity
    #     clean_loss = F.mse_loss(quantized, ref_features.detach())
    #     total_loss = vq_loss + self.clean_embedding_weight * clean_loss
        
    #     return quantized, total_loss, encoding_indices, clean_loss, vq_loss

    def process_features(self, features):
        b, num_tokens, dim = features.shape
        h = w = int(num_tokens**0.5)
        target_h = target_w = int(self.num_tokens**0.5)

        if self.num_tokens != num_tokens:
            features = features.view(b, h, w, dim)
            features = features.permute(0, 3, 1, 2)
            features = F.interpolate(
                features, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            features = features.permute(0, 2, 3, 1).contiguous()
            features = features.view(b, self.num_tokens, dim)

        return features

    def update_freeze_status(self, global_step):
        """Update which layers are frozen based on training progress"""
        if not self.progressive_unfreeze or self.is_unfrozen:
            return
            
        if global_step >= self.unfreeze_after_steps:
            xm.rendezvous('pre_update_freeze')
            
            if self.unfreeze_strategy == 'all':
                self.unfreeze_encoder()
                self.is_unfrozen = True  # Mark as unfrozen
                if xm.get_ordinal() == 0:  # Log only from master process
                    print(f"Step {global_step}: Unfreezing encoder")
            
            xm.rendezvous('post_update_freeze')

    def freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.vision_tower.parameters():
            param.requires_grad = False
            param.grad = None  # Clear any existing gradients
            
    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters"""
        for param in self.vision_tower.parameters():
            param.requires_grad = True
            
    def unfreeze_layer(self, layer_idx):
        """Unfreeze a specific layer"""
        for param in self.encoder_layers[layer_idx].parameters():
            param.requires_grad = True
            
   

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