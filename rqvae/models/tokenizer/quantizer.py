import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

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

        # Initialize embeddings without normalization
        self.register_buffer(
            'embeddings', 
            torch.randn(num_embeddings, embedding_dim)
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

        # Calculate distances using MSE instead of cosine similarity
        # Compute in a memory-efficient way
        input_sq = torch.sum(flat_input**2, dim=1, keepdim=True)
        emb_sq = torch.sum(self.embeddings**2, dim=1)
        distances = input_sq - 2 * torch.matmul(flat_input, self.embeddings.t()) + emb_sq
        encoding_indices = torch.argmin(distances, dim=1)  # Note: argmin instead of argmax
        quantized = F.embedding(encoding_indices, self.embeddings)
        
        # Reshape outputs
        quantized = quantized.view(input_shape)
        encoding_indices = encoding_indices.view(input_shape[:-1])

        if self.training:
            encodings_onehot = F.one_hot(encoding_indices.view(-1), self.num_embeddings).float()
            
            # Update perplexity EMA
            avg_probs = torch.mean(encodings_onehot, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            self.perplexity_ema.mul_(self.perplexity_decay).add_(
                perplexity * (1 - self.perplexity_decay)
            )
            
            # Update embeddings and stats
            new_cluster_size = encodings_onehot.sum(0)
            new_embed_sum = torch.matmul(encodings_onehot.t(), flat_input)
            
            self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(new_embed_sum, alpha=1 - self.decay)
            
            # Update embeddings without normalization
            n = self.cluster_size.sum()
            normalized_cluster_size = (
                (self.cluster_size + self.epsilon) / 
                (n + self.num_embeddings * self.epsilon) * n
            )
            self.embeddings.copy_(self.embed_avg / normalized_cluster_size.unsqueeze(1))
            
            # Update usage stats
            self.usage_count.index_add_(
                0, 
                encoding_indices.view(-1), 
                torch.ones_like(encoding_indices.view(-1), dtype=torch.float)
            )
            self.total_usage += encoding_indices.numel()

        # Compute MSE loss
        if self.use_commitment:
            # Codebook loss: Move codebook vectors towards encoder outputs
            codebook_loss = F.mse_loss(quantized.detach(), inputs)
            
            # Commitment loss: Move encoder outputs towards codebook vectors 
            commitment_loss = F.mse_loss(quantized, inputs.detach())
            
            # Combined loss
            loss = codebook_loss + self.commitment_cost * commitment_loss
        else:
            # Just codebook loss if no commitment
            loss = F.mse_loss(quantized.detach(), inputs)

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
    
    def reset_metrics(self):
        """Reset all tracking metrics with single sync"""
        xm.rendezvous('reset')  # Single sync point for reset
        self.usage_count.zero_()
        self.total_usage.zero_()
        self.perplexity_ema.zero_()

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        return self.embeddings[indices]  # No normalization needed