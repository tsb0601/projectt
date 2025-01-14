import os
import io
import logging
import h5py
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor
from torchvision import transforms
from PIL import Image
import webdataset as wds
from rqvae.models.tokenizer.siglip_vq import SigLIPVQEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cc3m_shards(base_path="/mnt/disks/storage/cc3m/cc3m-wds"):
    shard_files = [
        os.path.join(base_path, f"cc3m-train-{str(i).zfill(4)}.tar")
        for i in range(481)
    ]
    existing_shards = [shard for shard in shard_files if os.path.exists(shard)]
    if len(existing_shards) == 0:
        raise RuntimeError(f"No shards found in {base_path}")
    logger.info(f"Found {len(existing_shards)} shards")
    return existing_shards

class WebDatasetAdapter:
    def __init__(self, urls, siglip_processor, args, num_samples=None):
        self.urls = urls
        self.siglip_processor = siglip_processor
        self.args = args
        self.num_samples = num_samples
        
        self.transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def create_webdataset(self, epoch):
        # Create a unique seed for each epoch that's the same across all workers
        seed = 42 + epoch
        
        # Create dataset pipeline
        dataset = (
            wds.WebDataset(self.urls, shardshuffle=1000)  # Explicitly set shardshuffle
            .shuffle(5000, seed=seed)  # Large shuffle buffer
            .decode("pil")
            .to_tuple("jpg")
            .map(self.preprocess_sample)
            .batched(self.args.batch_size, collation_fn=self.collate_fn)
        )
        
        if self.num_samples is not None:
            dataset = dataset.with_length(self.num_samples)
            
        return dataset
    
    def preprocess_sample(self, sample):
        image = sample[0]
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image)).convert('RGB')
            
        # Preprocess for SigLIP
        siglip_image = self.siglip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Return as tuple (this is key for WebDataset batching)
        return (siglip_image,)
    
    def collate_fn(self, samples):
        # Collate the tuples into batches
        siglip_images = torch.stack([sample[0] for sample in samples])
        return {"image": siglip_images}

def get_data_loader(rank, world_size, epoch, urls, siglip_processor, args):
    """
    Create a data loader for the current TPU core and epoch
    """
    # Calculate shards for this worker
    num_shards = len(urls)
    shards_per_worker = num_shards // world_size
    start_shard = rank * shards_per_worker
    end_shard = start_shard + shards_per_worker
    if rank == world_size - 1:  # Last worker takes remaining shards
        end_shard = num_shards
    
    worker_urls = urls[start_shard:end_shard]
    logger.info(f"Worker {rank}: Processing {len(worker_urls)} shards")
    
    # Estimate number of samples
    samples_per_shard = 2500  # CC3M average
    estimated_samples = len(worker_urls) * samples_per_shard
    
    # Create dataset
    dataset = WebDatasetAdapter(
        worker_urls,
        siglip_processor,
        args,
        num_samples=estimated_samples
    ).create_webdataset(epoch)
    
    # Create data loader
    cpu_loader = DataLoader(
        dataset,
        batch_size=None,  # Already batched by WebDataset
        num_workers=args.num_workers
    )
    
    # Wrap with ParallelLoader for TPU
    device = xm.xla_device()
    loader = pl.ParallelLoader(
        cpu_loader,
        [device],
        batchdim=0
    ).per_device_loader(device)
    
    return loader, estimated_samples

class EmbeddingGenerator:
    def __init__(self, args):
        self.args = args
        self.device = xm.xla_device()
        self.setup_model()
        
    def setup_model(self):
        """Initialize the SigLIP model"""
        self.model = SigLIPVQEncoder(
            model_name="google/siglip-so400m-patch14-384",
            num_tokens=256,
            embedding_dim=1152,
            use_vq=False,  # We don't need VQ for embedding generation
            device=self.device,
            trainable=False
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def process_batch(self, images):
        """Process a batch of images and return their embeddings"""
        with torch.no_grad():
            # Forward pass without VQ (will return image_features directly)
            features, _, _, _, _ = self.model(images)
            return features
            
    def generate_and_save_embeddings(self, output_path):
        """Generate embeddings for all images and save them"""
        shard_files = get_cc3m_shards(self.args.data_path)
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        
        # Create output directory if it doesn't exist
        print("Output path passed in is", output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Create HDF5 file for this worker
        worker_output = os.path.join(output_path, f"worker_{rank}.h5")
        
        with h5py.File(worker_output, 'w') as f:
            embeddings_dataset = f.create_dataset(
                'embeddings',
                shape=(0, 1152),
                maxshape=(None, 1152),
                dtype='float32',
                chunks=(1000, 1152)
            )
            
            total_tokens = 0
            
            for epoch in range(1):
                train_loader, num_samples = get_data_loader(
                    rank, world_size, epoch,
                    shard_files, self.model.processor, self.args
                )
                
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Worker {rank}")):
                    images = batch["image"]
                    
                    # Get embeddings (shape: [batch_size, 256, 1152])
                    embeddings = self.process_batch(images)
                    
                    # Reshape to [batch_size * 256, 1152]
                    embeddings = embeddings.reshape(-1, 1152)
                    
                    # Move to CPU and convert to numpy
                    embeddings_np = embeddings.cpu().numpy()
                    
                    # Resize dataset and add new embeddings
                    current_size = embeddings_dataset.shape[0]
                    new_size = current_size + embeddings_np.shape[0]
                    embeddings_dataset.resize(new_size, axis=0)
                    embeddings_dataset[current_size:new_size] = embeddings_np
                    
                    total_tokens += embeddings_np.shape[0]
                    
                    # Sync to make sure XLA operations are complete
                    xm.mark_step()
                    
                    if batch_idx % 100 == 0:
                        logger.info(f"Worker {rank}: Processed {total_tokens} tokens")
            
            logger.info(f"Worker {rank}: Completed. Total tokens: {total_tokens}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/disks/storage/cc3m/cc3m-wds")
    parser.add_argument('--output_path', type=str, default="cc3m_embeddings")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=384)
    args = parser.parse_args()
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(args)
    
    # Generate and save embeddings
    generator.generate_and_save_embeddings(args.output_path)
    
    # Sync all workers before finishing
    xm.rendezvous('embedding_generation_complete')
    
    if xm.get_ordinal() == 0:
        logger.info("Embedding generation complete for all workers")

if __name__ == "__main__":
    main()