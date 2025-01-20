import torch
import os
import glob
from tqdm import tqdm
import numpy as np

def analyze_embeddings(embeddings_dir="/mnt/disks/peter-pd-tokenization/saved_embed"):
    # Get first chunk to analyze in detail
    chunk_files = sorted(glob.glob(os.path.join(embeddings_dir, "embeddings_chunk_*.pt")))
    if not chunk_files:
        raise RuntimeError(f"No chunk files found in {embeddings_dir}")
    
    print("Loading first chunk for detailed analysis...")
    chunk = torch.load(chunk_files[0], weights_only=True).to(torch.float32)
    
    # Basic info
    print(f"\nBasic Information:")
    print(f"Embedding dimension: {chunk.shape[1]}")
    print(f"Chunk size: {chunk.shape[0]} vectors")
    print(f"Total chunks: {len(chunk_files)}")

    # Calculate magnitudes
    magnitudes = torch.norm(chunk, dim=1)
    print(f"\nVector Magnitudes:")
    print(f"Average magnitude: {magnitudes.mean().item():.2f}")
    print(f"Median magnitude: {magnitudes.median().item():.2f}")
    print(f"Std of magnitudes: {magnitudes.std().item():.2f}")
    print(f"Min/Max magnitude: {magnitudes.min().item():.2f} / {magnitudes.max().item():.2f}")
    
    # Component statistics
    print(f"\nComponent Statistics:")
    print(f"Average component value: {chunk.mean().item():.4f}")
    print(f"Std of components: {chunk.std().item():.4f}")
    print(f"Min/Max component: {chunk.min().item():.4f} / {chunk.max().item():.4f}")

    # Sample distances
    print("\nCalculating pairwise distances on sample...")
    sample_size = min(1000, chunk.shape[0])
    indices = torch.randperm(chunk.shape[0])[:sample_size]
    samples = chunk[indices]
    distances = torch.cdist(samples, samples)
    
    # Distance statistics
    nonzero_distances = distances[distances > 0]
    print(f"\nPairwise Distance Statistics:")
    print(f"Average distance: {nonzero_distances.mean().item():.2f}")
    print(f"Median distance: {nonzero_distances.median().item():.2f}")
    print(f"Std of distances: {nonzero_distances.std().item():.2f}")
    print(f"Min non-zero distance: {nonzero_distances.min().item():.2f}")
    print(f"Max distance: {distances.max().item():.2f}")
    
    # Distance percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    distance_percentiles = np.percentile(nonzero_distances.cpu().numpy(), percentiles)
    print(f"\nDistance Percentiles:")
    for p, v in zip(percentiles, distance_percentiles):
        print(f"{p}th percentile: {v:.2f}")

    # Context for distance of 70
    print(f"\nContext for distance of 70:")
    percent_smaller = (nonzero_distances < 70).float().mean().item() * 100
    print(f"Percentage of distances smaller than 70: {percent_smaller:.1f}%")
    
    if chunk.shape[1] == 1152:  # If dimension matches what you mentioned
        avg_per_dim = 70 / np.sqrt(1152)
        print(f"Average difference per dimension for distance of 70: {avg_per_dim:.4f}")
        
    # Cleanup
    del chunk, samples, distances, nonzero_distances
    torch.cuda.empty_cache()

if __name__ == "__main__":
    analyze_embeddings()