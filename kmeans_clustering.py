import torch
import torch_xla.core.xla_model as xm
import os
import glob
from tqdm import tqdm
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader, IterableDataset
import time
from datetime import datetime

class StreamingDataset(IterableDataset):
    def __init__(self, embeddings_dir, batch_size):
        self.chunk_files = sorted(glob.glob(os.path.join(embeddings_dir, "embeddings_chunk_*.pt")))
        self.batch_size = batch_size
        
        if not self.chunk_files:
            raise RuntimeError(f"No chunk files found in {embeddings_dir}")
            
        first_chunk = torch.load(self.chunk_files[0], weights_only=True)
        self.embedding_dim = first_chunk.shape[1]
        del first_chunk
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Total number of chunks: {len(self.chunk_files)}")
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            chunk_files = self.chunk_files[worker_info.id::worker_info.num_workers]
        else:
            chunk_files = self.chunk_files

        for chunk_file in chunk_files:
            try:
                # print(f"Loading chunk: {os.path.basename(chunk_file)}")
                chunk = torch.load(chunk_file, weights_only=True).to(torch.float32)  # Ensure 32-bit
                
                for start_idx in range(0, chunk.shape[0], self.batch_size):
                    end_idx = min(start_idx + self.batch_size, chunk.shape[0])
                    yield chunk[start_idx:end_idx]
                
                del chunk
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing chunk {chunk_file}: {str(e)}")
                continue

class SimpleTPUKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, batch_size=1024):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.device = xm.xla_device()
        self.centroids = None
        
        print(f"Initializing K-means with {n_clusters} clusters")
        print(f"Parameters: max_iter={max_iter}, tol={tol}, batch_size={batch_size}")
    
    def _init_centroids(self, dataloader):
        print("Initializing centroids...")
        accumulated_samples = []
        samples_needed = max(self.n_clusters * 2, 100000)
        total_samples = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            total_samples += batch.shape[0]
            accumulated_samples.append(batch)
            
            if total_samples >= samples_needed:
                all_samples = torch.cat(accumulated_samples, dim=0)
                # Use numpy for random sampling to avoid TPU 64-bit issues
                indices = np.random.choice(all_samples.shape[0], self.n_clusters, replace=False)
                indices = torch.tensor(indices, device=self.device, dtype=torch.int32)
                self.centroids = all_samples[indices]
                
                print(f"Initialized centroids with shape: {self.centroids.shape}")
                print(f"Used {total_samples:,} samples for initialization")
                
                del accumulated_samples, all_samples
                return
                
        raise RuntimeError(f"Not enough samples ({total_samples}) to initialize {self.n_clusters} centroids")
    
    def _compute_distances(self, batch):
        return torch.cdist(batch, self.centroids)
    
    def fit(self, dataloader):
        if self.centroids is None:
            self._init_centroids(dataloader)
        
        print(f"Starting training...")
        
        for iteration in range(self.max_iter):
            start_time = time.time()
            old_centroids = self.centroids.clone()
            
            new_centroids = torch.zeros_like(self.centroids)
            counts = torch.zeros(self.n_clusters, device=self.device, dtype=torch.float32)
            
            n_samples = 0
            
            for batch in tqdm(dataloader, desc=f"Iteration {iteration + 1}"):
                batch = batch.to(self.device)
                n_samples += batch.shape[0]
                
                distances = self._compute_distances(batch)
                labels = torch.argmin(distances, dim=1)
                
                labels_onehot = torch.zeros(
                    batch.size(0), self.n_clusters,
                    device=self.device,
                    dtype=torch.float32  # Ensure 32-bit
                )
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                
                new_centroids += torch.matmul(labels_onehot.t(), batch)
                counts += labels_onehot.sum(dim=0)
                
                xm.mark_step()
                
                del batch, distances, labels, labels_onehot
            
            # Update centroids
            mask = counts > 0
            new_centroids[mask] /= counts[mask].unsqueeze(1)
            self.centroids = new_centroids
            
            centroid_shift = torch.norm(self.centroids - old_centroids).item()
            iteration_time = time.time() - start_time
            
            print(f"Iteration {iteration + 1}: "
                  f"shift = {centroid_shift:.6f}, "
                  f"time = {iteration_time:.2f}s, "
                  f"samples = {n_samples:,}, "
                  f"centroid_shift = {centroid_shift}")
            
            if centroid_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations!")
                break
        
        return self
    
    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dict = {
            'centroids': self.centroids.cpu(),
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'batch_size': self.batch_size,
            'timestamp': timestamp
        }
        
        save_path = os.path.join(output_dir, f"kmeans_results_{timestamp}.pt")
        torch.save(save_dict, save_path)
        print(f"Saved model to {save_path}")

def main():
    n_clusters = 16384
    batch_size = 8192
    max_iter = 200
    embeddings_dir = "/mnt/disks/peter-pd-tokenization/saved_embed"
    output_dir = os.path.join(embeddings_dir, "kmeans_results")
    
    dataset = StreamingDataset(embeddings_dir, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0
    )
    
    kmeans = SimpleTPUKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        batch_size=batch_size
    )
    
    kmeans.fit(dataloader)
    kmeans.save(output_dir)

if __name__ == "__main__":
    main()