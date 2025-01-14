import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeScaleKMeans:
    def __init__(self, embedding_file, n_clusters=8192, batch_size=10000, 
                 max_iter=100, random_state=42):
        self.embedding_file = embedding_file
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.kmeans = None
        
    def _load_batch(self, dataset, start_idx, end_idx):
        """Load and preprocess a batch of embeddings"""
        batch = dataset[start_idx:end_idx]
        # Convert to float32 for better numerical stability
        batch = batch.astype(np.float32)
        # Normalize the embeddings
        batch = normalize(batch)
        return batch
    
    def fit(self, output_dir="cluster_results"):
        """Fit KMeans on the embeddings using mini-batches"""
        os.makedirs(output_dir, exist_ok=True)
        start_time = datetime.now()
        
        with h5py.File(self.embedding_file, 'r') as f:
            embeddings = f['embeddings']
            total_samples = embeddings.shape[0]
            embedding_dim = embeddings.shape[1]
            
            logger.info(f"Starting clustering on {total_samples} samples "
                       f"with {embedding_dim} dimensions into {self.n_clusters} clusters")
            
            # Initialize MiniBatchKMeans
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=1
            )
            
            # Process data in batches
            for start_idx in tqdm(range(0, total_samples, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch = self._load_batch(embeddings, start_idx, end_idx)
                self.kmeans.partial_fit(batch)
        
        # Save the model and clustering results
        self._save_results(output_dir)
        
        end_time = datetime.now()
        logger.info(f"Clustering completed in {end_time - start_time}")
        
        return self
    
    def predict(self, output_dir="cluster_results"):
        """Predict clusters for all samples and analyze results"""
        if self.kmeans is None:
            self.kmeans = self._load_model(output_dir)
            
        results = {
            'labels': [],
            'distances': []
        }
        
        with h5py.File(self.embedding_file, 'r') as f:
            embeddings = f['embeddings']
            total_samples = embeddings.shape[0]
            
            # Process predictions in batches
            for start_idx in tqdm(range(0, total_samples, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch = self._load_batch(embeddings, start_idx, end_idx)
                
                batch_labels = self.kmeans.predict(batch)
                batch_distances = self.kmeans.transform(batch).min(axis=1)
                
                results['labels'].extend(batch_labels)
                results['distances'].extend(batch_distances)
        
        results['labels'] = np.array(results['labels'])
        results['distances'] = np.array(results['distances'])
        
        # Save prediction results
        self._save_predictions(results, output_dir)
        
        # Analyze clusters
        self.analyze_clusters(results, output_dir)
        
        return results
    
    def _save_results(self, output_dir):
        """Save clustering model and centers"""
        model_file = os.path.join(output_dir, 'kmeans_model.npz')
        np.savez(model_file,
                cluster_centers=self.kmeans.cluster_centers_,
                inertia=self.kmeans.inertia_)
        logger.info(f"Saved clustering model to {model_file}")
    
    def _save_predictions(self, results, output_dir):
        """Save prediction results"""
        pred_file = os.path.join(output_dir, 'cluster_predictions.npz')
        np.savez(pred_file,
                labels=results['labels'],
                distances=results['distances'])
        logger.info(f"Saved predictions to {pred_file}")
    
    def _load_model(self, output_dir):
        """Load saved clustering model"""
        model_file = os.path.join(output_dir, 'kmeans_model.npz')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No saved model found at {model_file}")
            
        data = np.load(model_file)
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state
        )
        kmeans.cluster_centers_ = data['cluster_centers']
        kmeans.inertia_ = float(data['inertia'])
        return kmeans
    
    def analyze_clusters(self, results, output_dir):
        """Analyze clustering results and generate visualizations"""
        # Compute cluster statistics
        cluster_sizes = Counter(results['labels'])
        distances = results['distances']
        
        stats = {
            'cluster_sizes': cluster_sizes,
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'min_cluster_size': min(cluster_sizes.values()),
            'max_cluster_size': max(cluster_sizes.values()),
            'empty_clusters': self.n_clusters - len(cluster_sizes)
        }
        
        # Save statistics
        stats_file = os.path.join(output_dir, 'cluster_statistics.txt')
        with open(stats_file, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        # Generate visualizations
        self._plot_cluster_size_distribution(cluster_sizes, output_dir)
        self._plot_distance_distribution(distances, output_dir)
        
        return stats
    
    def _plot_cluster_size_distribution(self, cluster_sizes, output_dir):
        """Plot distribution of cluster sizes"""
        plt.figure(figsize=(12, 6))
        sizes = list(cluster_sizes.values())
        plt.hist(sizes, bins=50)
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Size')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'cluster_size_distribution.png'))
        plt.close()
    
    def _plot_distance_distribution(self, distances, output_dir):
        """Plot distribution of distances to cluster centers"""
        plt.figure(figsize=(12, 6))
        plt.hist(distances, bins=50)
        plt.title('Distance to Cluster Center Distribution')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'distance_distribution.png'))
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True,
                        help='Path to H5 file containing embeddings')
    parser.add_argument('--output_dir', type=str, default='cluster_results',
                        help='Directory to save clustering results')
    parser.add_argument('--n_clusters', type=int, default=8192,
                        help='Number of clusters')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for processing')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    args = parser.parse_args()
    
    # Initialize and run clustering
    kmeans = LargeScaleKMeans(
        embedding_file=args.embedding_file,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        max_iter=args.max_iter
    )
    
    # Fit the model
    kmeans.fit(args.output_dir)
    
    # Generate predictions and analysis
    results = kmeans.predict(args.output_dir)
    
    logger.info("Clustering and analysis complete!")

if __name__ == "__main__":
    main()