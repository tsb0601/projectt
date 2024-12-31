# Redefine the OnlineEnergyDistance class for CPU-only execution
import torch
from scipy.special import gamma, gammaln

class OnlineEnergyDistance:
    def __init__(self, d, sample_size=2):
        """
        Initialize the online tracker for energy distance.
        
        Parameters:
        d : int
            Dimensionality of the data.
        sample_size : int
            Number of random samples used for approximating ||X - X'|| in each mini-batch.
        """
        self.n = 0  # Total number of samples processed
        self.total_pairwise_dist = 0.0  # Sum of pairwise distances ||X - X'||
        self.total_cross_dist = 0.0  # Sum of cross distances ||X - Y||
        self.d = d  # Dimensionality of the data
        self.sample_size = sample_size  # Number of samples for pairwise distance approximation

        # Precompute E[||Y - Y'||] for standard Gaussian
        #self.constant_dist = 2 * gamma((d + 1) / 2) / gamma(d / 2)
        gammaln_ = torch.tensor(gammaln((d+1) / 2) - gammaln(d / 2), dtype=torch.float64)
        self.constant_dist = 2 * torch.exp(gammaln_)
        print(f"Constant distance: {self.constant_dist}")
    
    def update(self, X):
        """
        Update the energy distance with a new mini-batch of samples X.
        
        Parameters:
        X : torch.Tensor of shape (N, D)
            A batch of samples from the distribution P.
        """
        N, d = X.shape
        if d != self.d:
            raise ValueError("Dimensionality mismatch")

        # Pairwise distances within a sampled subset of the batch
        rand_index = torch.randperm(N)
        sample_indices = rand_index[:self.sample_size]
        second_indices = rand_index[-self.sample_size:]
        X_sample = X[sample_indices]
        X_sample_prime = X[second_indices]
        pairwise_distances = torch.cdist(X_sample, X_sample_prime, p=2)
        #print(f"Pairwise distances: {pairwise_distances}")
        self.total_pairwise_dist += pairwise_distances.mean().item()

        # Cross distances between batch and standard Gaussian
        Y = torch.randn((X_sample.shape[0], d))  # Samples from N(0, I)
        cross_distances = torch.cdist(X_sample, Y, p=2)
        #print(f"Cross distances: {cross_distances}")
        self.total_cross_dist += cross_distances.mean().item()
        #print(f"Cross distances: {cross_distances.mean().item()}, Pairwise distances: {pairwise_distances.mean().item()}")
        # Update total sample count
        self.n += 1

    def energy_distance(self):
        """
        Compute the energy distance to the standard Gaussian.
        """
        if self.n == 0:
            return 0.0
        # Compute the final energy distance
        ED = 2 * (self.total_cross_dist / self.n) - (self.total_pairwise_dist / self.n) - self.constant_dist
        return ED

# Running the class with CPU-only operations
d = 4 * 16 * 16
sample_size = 30
tracker = OnlineEnergyDistance(d, sample_size=sample_size)

# Simulate mini-batch updates
for _ in range(100):  # 10 mini-batches
    batch = torch.randn((100, d)) * 60 # Batch of size 100
    tracker.update(batch)

# Compute energy distance to standard Gaussian
energy_dist = tracker.energy_distance()
print(f"Energy distance to standard Gaussian: {energy_dist:.4f}")
