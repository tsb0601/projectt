import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm



def compute_distances(flat_input, embeddings):
    # Manual implementation of euclidean distance
    input_norm = (flat_input**2).sum(1, keepdim=True)
    emb_norm = (embeddings**2).sum(1, keepdim=True)
    distances = input_norm + emb_norm.t() - 2 * torch.mm(flat_input, embeddings.t())
    return distances.sqrt()

class EnhancedVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        kmeans_path: str = None,
        trainable: bool = True,
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
        self.trainable = trainable

        # Initialize from K-means if provided, otherwise random
        if kmeans_path:
            kmeans_state = torch.load(kmeans_path)
            centroids = kmeans_state['centroids']
            assert centroids.shape == (num_embeddings, embedding_dim), \
                f"Centroid shape {centroids.shape} doesn't match expected ({num_embeddings}, {embedding_dim})"
            
            if trainable:
                # Use Parameter for trainable embeddings
                self.embeddings = nn.Parameter(centroids)
            else:
                # Use buffer for frozen embeddings
                self.register_buffer('embeddings', centroids)
        else:
            # Random initialization
            if trainable:
                self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
            else:
                self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))

        # EMA tracking buffers (only used if trainable)
        if trainable:
            self.register_buffer('cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('embed_avg', torch.zeros(num_embeddings, embedding_dim))
        
        # Usage tracking (for both modes)
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('total_usage', torch.tensor(0))
        self.register_buffer('perplexity_ema', torch.tensor(0.0))
        self.perplexity_decay = 0.99



        
    def forward(self, inputs):
        if self.training:
            xm.rendezvous('training_step')
            
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        # distances = torch.cdist(flat_input, self.embeddings)
        distances = compute_distances(flat_input, self.embeddings)
        
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Get quantized vectors
        quantized = self.embeddings[encoding_indices].view(input_shape)

        # Update tracking metrics and embeddings if training
        if self.training:
            encodings_onehot = F.one_hot(encoding_indices, self.num_embeddings).float()
            
            # Update perplexity EMA
            avg_probs = torch.mean(encodings_onehot, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            self.perplexity_ema.mul_(self.perplexity_decay).add_(
                perplexity * (1 - self.perplexity_decay)
            )
            
            # Update embeddings if trainable
            if self.trainable:
                # Update EMA statistics
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
                self.embeddings.data.copy_(embed_normalized)
            
            # Update usage statistics (both modes)
            self.usage_count.index_add_(
                0, 
                encoding_indices.view(-1), 
                torch.ones_like(encoding_indices.view(-1), dtype=torch.float)
            )
            self.total_usage += encoding_indices.numel()

        # Compute losses
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
        """Get metrics without forcing sync"""
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
            'codebook_mode': 'trainable' if self.trainable else 'frozen'
        }
    
    def reset_metrics(self):
        """Reset tracking metrics with single sync"""
        xm.rendezvous('reset')
        self.usage_count.zero_()
        self.total_usage.zero_()
        self.perplexity_ema.zero_()

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        return self.embeddings[indices]