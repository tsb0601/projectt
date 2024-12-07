import torch

import torch

class OnlineVarianceTracker:
    def __init__(self):
        self.n = 0          # total number of samples seen so far
        self.mean = None    # running mean (D,)
        self.M2 = None      # running sum of squared differences (D,)

    def update(self, X: torch.Tensor):
        """
        Update the running statistics with a mini-batch of data X (N, D).
        We'll treat the incoming batch as a separate "dataset" and merge it
        with our running statistics in one step.
        """
        N = X.shape[0]
        if N == 0:
            return  # No new data in this batch

        # Compute batch mean and M2
        batch_mean = X.mean(dim=0)
        # Compute sum of squared deviations from batch mean
        # (X - batch_mean) is (N, D) -> squared -> sum over N -> (D,)
        batch_M2 = ((X - batch_mean)**2).sum(dim=0)

        if self.mean is None:
            # If no previous data, just take batch stats
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = N
        else:
            # Merge with existing stats
            n_old = self.n
            n_new = n_old + N

            delta = batch_mean - self.mean
            # Update mean
            new_mean = self.mean + delta * (N / n_new)

            # Update M2
            # M2_new = M2_old + M2_batch + delta^2 * (n_old * N / n_new)
            M2_new = self.M2 + batch_M2 + (delta**2) * (n_old * N / n_new)

            # Assign updated stats
            self.mean = new_mean
            self.M2 = M2_new
            self.n = n_new

    def trace_sigma(self):
        """
        Compute and return the trace of the covariance matrix.

        Σ = M2 / n, so Tr(Σ) = sum of variances = sum(M2 / n).
        """
        if self.n == 0:
            return torch.tensor(0.0, dtype=torch.float)
        var = self.M2 / self.n
        return var
    def __call__(self, X):
        self.update(X)
# Example usage:
# tracker = OnlineVarianceTracker()
# for batch in data_loader:  # batch is (N, D) torch.Tensor
#     tracker.update(batch)
# print("Trace of Sigma:", tracker.trace_sigma().item())


def main():
    # generate sample from N(0, I)
    torch.manual_seed(42)
    tracker = OnlineVarianceTracker() # test accuracy of OnlineVarianceTracker
    for _ in range(300):
        X = torch.randn(10, 5) * 2
        tracker(X)
    sigma = tracker.trace_sigma()
    print(sigma)
    rel_error = torch.abs(sigma - torch.ones(5) * 4).mean()
    # Expected output: tensor([1., 1., 1., 1., 1.])
    print('Relative error:', rel_error.item())
    sigma_sum = sigma.sum()
    print('Sum of variances:', sigma_sum.item())
    
if __name__ == "__main__":
    main()