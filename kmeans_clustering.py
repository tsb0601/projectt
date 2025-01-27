import argparse
import torch
import torch_xla.core.xla_model as xm
import os
import glob
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import time
from datetime import datetime
from queue import Queue
from threading import Thread
import wandb

# ------------------------------
# 🌟 STREAMING DATASET CLASS
# ------------------------------
class StreamingDataset(IterableDataset):
    def __init__(self, embeddings_dir, frac=1.0):
        print("🔍 Scanning for chunk files...")
        pt_files = glob.glob(os.path.join(embeddings_dir, "embeddings_chunk_*.pt"))
        npy_files = glob.glob(os.path.join(embeddings_dir, "embeddings_chunk_*.npy"))
        self.chunk_files = sorted(pt_files + npy_files)
        
        if not self.chunk_files:
            raise RuntimeError(f"❌ No chunk files found in {embeddings_dir}")

        num_chunks = int(len(self.chunk_files) * frac)
        self.chunk_files = self.chunk_files[:num_chunks]
        print(f"📂 Using {len(self.chunk_files)}/{num_chunks} chunks (frac={frac})")

    def _chunk_generator(self, chunk_files):
        next_chunk_queue = Queue(maxsize=6)
        
        def load_chunk_async(file_path):
            try:
                if file_path.endswith('.npy'):
                    chunk = torch.from_numpy(np.load(file_path)).float()
                elif file_path.endswith('.pt'):
                    chunk = torch.load(file_path, weights_only=True).float()
                next_chunk_queue.put(chunk)
            except Exception as e:
                next_chunk_queue.put(e)
        
        with tqdm(total=len(chunk_files), desc="📦 Loading chunks", unit="chunk") as pbar:
            first_file = chunk_files[0]
            current_chunk = torch.from_numpy(np.load(first_file)).float() if first_file.endswith('.npy') else torch.load(first_file, weights_only=True).float()
            pbar.update(1)
            
            for file_path in chunk_files[1:]:
                loader = Thread(target=load_chunk_async, args=(file_path,))
                loader.start()
                yield from current_chunk
                current_chunk = next_chunk_queue.get()
                if isinstance(current_chunk, Exception): raise current_chunk
                loader.join()
                pbar.update(1)
            
            yield from current_chunk

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        chunk_files = self.chunk_files[worker_info.id::worker_info.num_workers] if worker_info else self.chunk_files
        return self._chunk_generator(chunk_files)

# ------------------------------
# 🚀 TPU K-MEANS CLASS
# ------------------------------
class SimpleTPUKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, batch_size=1024, checkpoint_root=None):
        print("\n" + "="*50)
        print(f"🧠 INITIALIZING TPU K-MEANS CLUSTERER".center(50))
        print("="*50)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.device = xm.xla_device()
        self.centroids = None
        self.start_iter = 0
        
        self.checkpoint_dir = os.path.join(checkpoint_root, f"kmeans_{n_clusters}_clusters") if checkpoint_root else None
        
        print(f"🔢 Clusters: {n_clusters:,}")
        print(f"🔄 Max iterations: {max_iter}")
        print(f"🎯 Tolerance: {tol:.0e}")
        print(f"📦 Batch size: {batch_size:,}")
        print(f"📂 Checkpoint directory: {self.checkpoint_dir or 'None'}")
        
        if self.checkpoint_dir: self._load_latest_checkpoint()
        print("="*50 + "\n")
    
    def _load_latest_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            print(f"⏹️ Checkpoint directory {self.checkpoint_dir} doesn't exist")
            return
        
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_iter_*.pt"))
        if not checkpoint_files:
            print(f"⏹️ No checkpoints found in {self.checkpoint_dir}")
            return
        
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(os.path.basename(f).split("_")[2].split(".")[0]))
        print(f"🔍 Found checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint)
        if checkpoint['n_clusters'] != self.n_clusters:
            raise ValueError(f"❌ Checkpoint n_clusters {checkpoint['n_clusters']} != current {self.n_clusters}")
        
        self.centroids = checkpoint['centroids'].to(self.device)
        self.start_iter = checkpoint['iteration'] + 1
        print(f"🔄 Resuming from iteration {self.start_iter} (max_iter={self.max_iter})")
    
    def _save_checkpoint(self, iteration):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_iter_{iteration+1:04d}.pt")
        xm.save({
            'iteration': iteration,
            'centroids': self.centroids,
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'batch_size': self.batch_size
        }, checkpoint_path)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        wandb.save(checkpoint_path)
    
    def _init_centroids(self, dataloader):
        print("🎯 INITIALIZING CENTROIDS")
        accumulated_samples = []
        samples_needed = max(self.n_clusters * 2, 100000)
        total_samples = 0
        
        progress = tqdm(desc="🔄 Gathering samples", unit="samples", total=samples_needed)
        for batch in dataloader:
            batch = batch.to(self.device)
            total_samples += batch.shape[0]
            accumulated_samples.append(batch)
            progress.update(batch.shape[0])
            
            if total_samples >= samples_needed:
                all_samples = torch.cat(accumulated_samples, dim=0)
                indices = np.random.choice(all_samples.shape[0], self.n_clusters, replace=False)
                self.centroids = all_samples[torch.tensor(indices, device=self.device)]
                print(f"\n✅ Centroids initialized with {total_samples:,} samples")
                print(f"📐 Centroid shape: {self.centroids.shape}")
                del accumulated_samples, all_samples
                progress.close()
                return
                
        raise RuntimeError(f"❌ Insufficient samples ({total_samples}) for {self.n_clusters} clusters")
    
    def fit(self, dataloader):
        if self.centroids is None: self._init_centroids(dataloader)
        
        print("\n🚀 STARTING TRAINING")
        print("====================")
        
        for iteration in range(self.start_iter, self.max_iter):
            iter_start = time.time()
            old_centroids = self.centroids.clone()
            new_centroids = torch.zeros_like(self.centroids)
            counts = torch.zeros(self.n_clusters, device=self.device, dtype=torch.float32)
            n_samples = 0
            
            with tqdm(dataloader, desc=f"🌀 Iteration {iteration+1}/{self.max_iter}", unit="batch") as pbar:
                for batch in pbar:
                    batch = batch.to(self.device)
                    n_samples += batch.shape[0]
                    
                    distances = torch.cdist(batch, self.centroids)
                    labels = torch.argmin(distances, dim=1)
                    
                    labels_onehot = torch.zeros(batch.size(0), self.n_clusters, device=self.device)
                    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                    
                    new_centroids += torch.matmul(labels_onehot.t(), batch)
                    counts += labels_onehot.sum(dim=0)
                    
                    xm.mark_step()
                    pbar.set_postfix({"samples": f"{n_samples:,}"})
                    del batch, distances, labels, labels_onehot
            
            mask = counts > 0
            new_centroids[mask] /= counts[mask].unsqueeze(1)
            self.centroids = new_centroids
            
            if (iteration + 1) % 10 == 0 and self.checkpoint_dir: self._save_checkpoint(iteration)
            
            diff = self.centroids - old_centroids
            centroid_shift = torch.norm(diff, p=2, dim=1).mean().item()
            iter_time = time.time() - iter_start

            wandb.log({
                "iteration": iteration + 1,
                "centroid_shift": centroid_shift,
                "iteration_time": iter_time,
                "samples_processed": n_samples
            })

            print(f"\n⏱️  Iteration {iteration+1} Summary:")
            print(f"   ▸ Max shift: {centroid_shift:.6f}")
            print(f"   ▸ Duration: {iter_time:.2f}s")
            print(f"   ▸ Samples processed: {n_samples:,}")
            print("--------------------------------------")

            if centroid_shift < self.tol:
                print(f"🎉 CONVERGENCE ACHIEVED AFTER {iteration+1} ITERATIONS!")
                wandb.log({"converged": True, "final_iteration": iteration + 1})
                break
        
        return self
    
    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"kmeans_{self.n_clusters}_clusters_{timestamp}.pt")
        
        torch.save({
            'centroids': self.centroids.cpu(),
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'batch_size': self.batch_size,
            'timestamp': timestamp
        }, save_path)
        print(f"\n💾 Model saved to: {save_path}")
        wandb.save(save_path)


# ------------------------------
# 🏁 MAIN FUNCTION
# ------------------------------
def main():
    print("\n" + "="*50)
    print("🚀 TPU-POWERED K-MEANS CLUSTERING".center(50))
    print("="*50)
    print("        ██╗  ██╗███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗")
    print("        ██║ ██╔╝████╗ ████║██╔════╝██╔══██╗████╗  ██║██╔════╝")
    print("        █████╔╝ ██╔████╔██║█████╗  ███████║██╔██╗ ██║███████╗")
    print("        ██╔═██╗ ██║╚██╔╝██║██╔══╝  ██╔══██║██║╚██╗██║╚════██║")
    print("        ██║  ██╗██║ ╚═╝ ██║███████╗██║  ██║██║ ╚████║███████║")
    print("        ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝")
    print("="*50 + "\n")

    # Parse arguments FIRST
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', type=int, default=65536*2, 
                       help='Number of clusters (default: 131072)')
    args = parser.parse_args()

    # Define configuration parameters
    batch_size = 4096
    max_iter = 100
    embeddings_dir = "/mnt/disks/peter-pd-tokenization/saved_embed/small_chunks"
    output_dir = os.path.join(".", "kmeans_results")
    checkpoint_root = os.path.join("ckpt_gcs", "tokenizer", "kmeans_checkpoints")

    # Initialize W&B with proper ordering
    wandb.init(
        project="kmeans",
        entity=os.environ.get("WANDB_ENTITY"),
        config={
            "n_clusters": args.n_clusters,
            "batch_size": batch_size,
            "max_iter": max_iter,
            "embeddings_dir": embeddings_dir,
            "checkpoint_root": checkpoint_root
        }
    )

    print("🛠️ Preparing data pipeline...")
    dataset = StreamingDataset(embeddings_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        prefetch_factor=4,
        pin_memory=False
    )

    kmeans = SimpleTPUKMeans(
        n_clusters=args.n_clusters,
        max_iter=max_iter,
        batch_size=batch_size,
        checkpoint_root=checkpoint_root
    )
    kmeans.fit(dataloader)
    kmeans.save(output_dir)

    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!".center(50))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
