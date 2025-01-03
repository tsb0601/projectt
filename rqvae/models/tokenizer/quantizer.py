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

        # Initialize all buffers at once with normalized embeddings
        self.register_buffer(
            'embeddings', 
            F.normalize(torch.randn(num_embeddings, embedding_dim), p=2, dim=1)
        )
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('total_usage', torch.tensor(0))
        self.register_buffer('perplexity_ema', torch.tensor(0.0))
        self.perplexity_decay = 0.99

    def forward(self, inputs):
        # Only sync once at the start of training forward pass
        if self.training:
            xm.rendezvous('training_step')
            
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)

        # Calculate distances and quantize
        distances = torch.matmul(flat_input_norm, self.embeddings.t())
        encoding_indices = torch.argmax(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embeddings)
        
        # Reshape outputs
        quantized = quantized.view(input_shape)
        encoding_indices = encoding_indices.view(input_shape[:-1])

        if self.training:
            # Do all training computations in one block to minimize syncs
            encodings_onehot = F.one_hot(encoding_indices.view(-1), self.num_embeddings).float()
            
            # Update perplexity EMA
            avg_probs = torch.mean(encodings_onehot, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            self.perplexity_ema.mul_(self.perplexity_decay).add_(
                perplexity * (1 - self.perplexity_decay)
            )
            
            # Update embeddings and stats in single pass
            new_cluster_size = encodings_onehot.sum(0)
            new_embed_sum = torch.matmul(encodings_onehot.t(), flat_input)
            
            self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(new_embed_sum, alpha=1 - self.decay)
            
            # Update embeddings
            n = self.cluster_size.sum()
            normalized_cluster_size = (
                (self.cluster_size + self.epsilon) / 
                (n + self.num_embeddings * self.epsilon) * n
            )
            embed_normalized = self.embed_avg / normalized_cluster_size.unsqueeze(1)
            self.embeddings.copy_(F.normalize(embed_normalized, p=2, dim=1))
            
            # Update usage stats
            self.usage_count.index_add_(
                0, 
                encoding_indices.view(-1), 
                torch.ones_like(encoding_indices.view(-1), dtype=torch.float)
            )
            self.total_usage += encoding_indices.numel()

        # Compute loss
        if self.use_commitment:
            commit_loss = F.mse_loss(quantized.detach(), inputs)
            loss = self.commitment_cost * commit_loss
        else:
            loss = F.mse_loss(quantized, inputs.detach())

        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices

    def get_metrics(self):
        """Get metrics without forcing sync unless needed"""
        if self.total_usage == 0:
            return {
                'perplexity': 0.0,
                'usage_fraction': 0.0,
                'code_usage': self.usage_count.clone(),
            }
            
        used_codes = torch.sum(self.usage_count > 0).float()
        usage_fraction = used_codes / self.num_embeddings
        
        return {
            'perplexity': self.perplexity_ema.item(),
            'usage_fraction': usage_fraction.item(),
            'code_usage': self.usage_count.clone(),
        }
    
    # def get_metrics(self):
    #     """Get metrics with proper synchronization"""
    #     if self.total_usage == 0:
    #         return {
    #             'perplexity': 0.0,
    #             'usage_fraction': 0.0,
    #             'code_usage': self.usage_count.clone(),
    #         }
        
    #     # Sync metrics across TPU cores for logging
    #     synced_usage_count = xm.all_reduce('sum', self.usage_count)
    #     synced_total_usage = xm.all_reduce('sum', self.total_usage)
            
    #     used_codes = torch.sum(synced_usage_count > 0).float()
    #     usage_fraction = used_codes / self.num_embeddings
        
    #     return {
    #         'perplexity': self.perplexity_ema.item(),
    #         'usage_fraction': usage_fraction.item(),
    #         'code_usage': synced_usage_count,
    #     }
        
    def reset_metrics(self):
        """Reset all tracking metrics with single sync"""
        xm.rendezvous('reset')  # Single sync point for reset
        self.usage_count.zero_()
        self.total_usage.zero_()
        self.perplexity_ema.zero_()

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        return F.normalize(self.embeddings[indices], p=2, dim=-1)