# rqvae/models/tokenizer/quantizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_xla.core.xla_model as xm
import wandb

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_commitment: bool = False,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.use_commitment = use_commitment
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embeddings.clone())
        
        # Add buffer for tracking codebook usage
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('total_usage', torch.tensor(0))

        # Normalize embeddings initially
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)
        
    def forward(self, inputs):
        # Handle input shape and ensure it's 3D [batch_size, num_tokens, embedding_dim]
        orig_shape = inputs.shape
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        
        # Flatten if more than 3 dimensions
        if inputs.dim() > 3:
            inputs = inputs.view(-1, inputs.size(-2), inputs.size(-1))
            
        # Ensure input embeddings are normalized
        inputs = F.normalize(inputs, p=2, dim=-1)
        
        # Calculate distances
        distances = (
            torch.sum(inputs ** 2, dim=-1, keepdim=True) +
            torch.sum(self.embeddings ** 2, dim=1) -
            2 * torch.matmul(inputs, self.embeddings.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=-1)  # [batch_size * num_tokens]
        
        # Create one-hot encodings
        encodings = torch.zeros(
            encoding_indices.shape[0] * encoding_indices.shape[1],  # Flattened batch * tokens
            self.num_embeddings,
            device=inputs.device
        )
        encodings.scatter_(1, encoding_indices.view(-1, 1), 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(inputs.shape)  # Reshape back to input shape
        
        # Compute losses
        if self.use_commitment:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        else:
            loss = F.mse_loss(quantized, inputs.detach())
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Update usage statistics and EMA in training
        if self.training:
            self._update_usage_stats(encoding_indices)
            self._ema_update(encodings.view(inputs.shape[0], -1, self.num_embeddings), inputs)
            
        # Restore original shape if needed
        if len(orig_shape) != len(quantized.shape):
            quantized = quantized.view(orig_shape)
            encoding_indices = encoding_indices.view(orig_shape[:-1])
            
        return quantized, loss, encoding_indices

    def _update_usage_stats(self, encoding_indices):
        # Update usage counts
        unique_indices, counts = torch.unique(encoding_indices, return_counts=True)
        self.usage_count[unique_indices] += counts.float()
        self.total_usage += encoding_indices.numel()

    def _ema_update(self, encodings, inputs):
        xm.rendezvous('pre_ema_update')  # Sync before EMA update
        
        # Calculate new cluster sizes and sum of embeddings
        batch_cluster_size = encodings.sum(0)
        batch_embed_sum = torch.matmul(encodings.t(), inputs)
        
        # Update buffers
        self.cluster_size.data.mul_(self.decay).add_(
            batch_cluster_size, alpha=1 - self.decay)
        self.embed_avg.data.mul_(self.decay).add_(
            batch_embed_sum, alpha=1 - self.decay)
        
        # Update embeddings
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.epsilon) /
            (n + self.num_embeddings * self.epsilon) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embeddings.data.copy_(F.normalize(embed_normalized, p=2, dim=1))
        
        xm.rendezvous('post_ema_update')  # Sync after EMA update
        
    def get_usage_stats(self):
        """Returns statistics about codebook usage"""
        xm.rendezvous('pre_get_stats')  # Sync before collecting stats
        
        if self.total_usage == 0:
            stats = {
                'perplexity': 0.0,
                'usage_fraction': 0.0,
                'usage_histogram': self.usage_count.clone()
            }
        else:
            probs = self.usage_count / self.total_usage
            probs = probs[probs > 0]  # Remove zeros for entropy calculation
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            perplexity = torch.exp(entropy)
            
            used_codes = torch.sum(self.usage_count > 0).float()
            usage_fraction = used_codes / self.num_embeddings
            
            stats = {
                'perplexity': perplexity.item(),
                'usage_fraction': usage_fraction.item(),
                'usage_histogram': self.usage_count.clone()
            }
        
        xm.rendezvous('post_get_stats')  # Sync after collecting stats
        return stats

    def reset_usage_stats(self):
        """Reset usage statistics"""
        xm.rendezvous('pre_reset_stats')
        self.usage_count.zero_()
        self.total_usage.zero_()
        xm.rendezvous('post_reset_stats')

    def to(self, device):
        """Override to() to ensure all buffers move to device"""
        super().to(device)
        if hasattr(self, 'embeddings'):
            self.embeddings = self.embeddings.to(device)
        if hasattr(self, 'cluster_size'):
            self.cluster_size = self.cluster_size.to(device)
        if hasattr(self, 'embed_avg'):
            self.embed_avg = self.embed_avg.to(device)
        if hasattr(self, 'usage_count'):
            self.usage_count = self.usage_count.to(device)
        if hasattr(self, 'total_usage'):
            self.total_usage = self.total_usage.to(device)
        return self

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        return F.normalize(self.embeddings[indices], p=2, dim=-1)


class CodebookAnalyzer:
    def __init__(self, vq_layer):
        self.vq_layer = vq_layer
        self.reset_analysis()
        
    def reset_analysis(self):
        xm.rendezvous('pre_reset_analysis')
        """Reset all analysis counters"""
        self.vq_layer.reset_usage_stats()
        self.code_transitions = torch.zeros(
            (self.vq_layer.num_embeddings, self.vq_layer.num_embeddings),
            device=self.vq_layer.embeddings.device
        )
        self.spatial_usage = torch.zeros(
            (int(math.sqrt(self.vq_layer.num_embeddings)), 
             int(math.sqrt(self.vq_layer.num_embeddings))),
            device=self.vq_layer.embeddings.device
        )
        xm.rendezvous('post_reset_analysis')
        
    @torch.no_grad()
    def analyze_batch(self, indices, quantized):
        """Analyze a batch of quantized outputs and their indices"""
        # Update transition matrix (how often codes follow each other)
        if indices.dim() > 1:  # If we have spatial dimensions
            # Look at horizontal transitions
            self.code_transitions.index_add_(
                0,
                indices[:, :-1].reshape(-1),
                torch.nn.functional.one_hot(
                    indices[:, 1:].reshape(-1),
                    self.vq_layer.num_embeddings
                ).float()
            )
            
        # Update spatial usage patterns
        if indices.dim() > 1:
            h = w = int(math.sqrt(indices.shape[1]))
            spatial_indices = indices.view(-1, h, w)
            for i in range(h):
                for j in range(w):
                    self.spatial_usage[i, j] += torch.bincount(
                        spatial_indices[:, i, j],
                        minlength=self.vq_layer.num_embeddings
                    ).float()
    
    def get_analysis(self):
        """Get comprehensive analysis of codebook usage"""
        xm.rendezvous('pre_analysis')
        
        device = self.vq_layer.embeddings.device
        usage_stats = self.vq_layer.get_usage_stats()
        
        # Compute codebook similarity matrix
        similarity_matrix = F.cosine_similarity(
            self.vq_layer.embeddings.unsqueeze(1),
            self.vq_layer.embeddings.unsqueeze(0),
            dim=-1
        )
        
        # Find closest pairs
        top_k = 5
        similar_pairs = []
        values, indices = torch.topk(similarity_matrix.view(-1), k=top_k+1)
        for i in range(1, top_k+1):  # Skip first (self-similarity)
            idx = indices[i]
            code1 = idx // self.vq_layer.num_embeddings
            code2 = idx % self.vq_layer.num_embeddings
            similar_pairs.append((code1.item(), code2.item(), values[i].item()))
            
        # Compute transition probabilities
        transition_probs = self.code_transitions / (self.code_transitions.sum(dim=1, keepdim=True) + 1e-10)
        
        # Find common transitions
        top_transitions = []
        values, indices = torch.topk(transition_probs.view(-1), k=top_k)
        for i in range(top_k):
            idx = indices[i]
            code1 = idx // self.vq_layer.num_embeddings
            code2 = idx % self.vq_layer.num_embeddings
            top_transitions.append((code1.item(), code2.item(), values[i].item()))
        
        analysis = {
            **usage_stats,
            'similarity_matrix': similarity_matrix.cpu(),
            'similar_pairs': similar_pairs,
            'transition_matrix': transition_probs.cpu(),
            'top_transitions': top_transitions,
            'spatial_usage': self.spatial_usage.cpu(),
        }
        
        xm.rendezvous('post_analysis')
        return analysis
        
    def log_analysis(self, step=None, prefix=''):
        """Log analysis to wandb (only on master process)"""
        xm.rendezvous('pre_log_analysis')
        
        if xm.get_ordinal() == 0:  # Only master process logs
            analysis = self.get_analysis()
            
            log_dict = {
                f'{prefix}codebook/perplexity': analysis['perplexity'],
                f'{prefix}codebook/usage_fraction': analysis['usage_fraction'],
                f'{prefix}codebook/usage_histogram': wandb.Histogram(
                    analysis['usage_histogram'].cpu().numpy()
                ),
                f'{prefix}codebook/similarity_matrix': wandb.Image(
                    analysis['similarity_matrix'].cpu().numpy()
                ),
                f'{prefix}codebook/spatial_usage': wandb.Image(
                    analysis['spatial_usage'].cpu().numpy()
                ),
            }
            
            # Log similar pairs and transitions as text
            similar_pairs_text = '\n'.join(
                f'Code {c1} ↔ Code {c2}: {sim:.3f}'
                for c1, c2, sim in analysis['similar_pairs']
            )
            transitions_text = '\n'.join(
                f'Code {c1} → Code {c2}: {prob:.3f}'
                for c1, c2, prob in analysis['top_transitions']
            )
            
            log_dict[f'{prefix}codebook/similar_pairs'] = wandb.Html(similar_pairs_text)
            log_dict[f'{prefix}codebook/top_transitions'] = wandb.Html(transitions_text)
            
            if step is not None:
                log_dict['global_step'] = step
                
            wandb.log(log_dict)
        
        xm.rendezvous('post_log_analysis')