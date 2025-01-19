import os
import io
import logging
# import h5py
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
import torch_xla.runtime as xr

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
        
        # Create dataset pipeline without max_images limitation
        dataset = (
            wds.WebDataset(self.urls, shardshuffle=1000)
            .shuffle(5000, seed=seed)
            .decode("pil")
            .to_tuple("jpg")
            .map(self.preprocess_sample)
            .batched(self.args.batch_size, collation_fn=self.collate_fn)
        )
        
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
    # print("Woker url is", worker_urls)
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
            use_vq=False,
            device=self.device,
            trainable=False
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def process_batch(self, images):
        """Process a batch of images and return their embeddings"""
        with torch.no_grad():
            features, _, _, _, _ = self.model(images)
            return features
            
    def generate_and_save_embeddings(self, output_path):
        """Generate embeddings for all images and save them in chunks"""
        shard_files = get_cc3m_shards(self.args.data_path)
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize timing metrics
        import time
        from datetime import datetime, timedelta
        start_time = time.time()
        last_update_time = start_time
        processed_since_last_update = 0
        
        # Calculate steps for chunked processing
        chunk_size = 10000  # Process 10000 images at a time
        global_batch_size = self.args.batch_size * world_size
        steps_per_chunk = chunk_size // global_batch_size
        num_chunks = (self.args.max_images + chunk_size - 1) // chunk_size
        
        logger.info(f"Worker {rank}: Processing {num_chunks} chunks of {chunk_size} images each")
        
        for chunk_idx in range(num_chunks):
            # List to store embeddings for this chunk
            chunk_embeddings = []
            
            # Calculate remaining images for last chunk
            remaining_images = min(chunk_size, 
                                self.args.max_images - chunk_idx * chunk_size)
            steps_this_chunk = (remaining_images + global_batch_size - 1) // global_batch_size
            
            logger.info(f"Worker {rank}: Starting chunk {chunk_idx + 1}/{num_chunks}, "
                    f"processing {remaining_images} images")
            
            train_loader, _ = get_data_loader(
                rank, world_size, chunk_idx,  # Use chunk_idx as epoch
                shard_files, self.model.processor, self.args
            )
            
            for batch_idx, batch in enumerate(tqdm(train_loader, 
                                                desc=f"Worker {rank} Chunk {chunk_idx + 1}")):
                if batch_idx >= steps_this_chunk:
                    break
                    
                images = batch["image"]
                
                # Get embeddings (shape: [batch_size, 256, 1152])
                embeddings = self.process_batch(images)
                
                # Convert to float16 and reshape
                embeddings = embeddings.to(torch.float16)
                embeddings = embeddings.reshape(-1, 1152)
                
                # Collect embeddings for this chunk
                chunk_embeddings.append(embeddings)
                
                # Log progress
                if batch_idx % 10 == 0:  # More frequent updates for chunks
                    current_time = time.time()
                    time_since_last_update = current_time - last_update_time
                    if time_since_last_update > 0:
                        tokens_processed = len(chunk_embeddings) * self.args.batch_size * 256
                        tokens_per_second = processed_since_last_update / time_since_last_update
                        remaining_steps = steps_this_chunk - batch_idx
                        remaining_tokens = remaining_steps * self.args.batch_size * 256
                        estimated_seconds = remaining_tokens / tokens_per_second if tokens_per_second > 0 else 0
                        eta = datetime.now() + timedelta(seconds=estimated_seconds)
                        
                        print(
                            f"Worker {rank} Chunk {chunk_idx + 1}: Processed {tokens_processed} tokens "
                            f"({tokens_per_second:.1f} tokens/sec, "
                            f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')})"
                        )
                        
                        last_update_time = current_time
                        processed_since_last_update = 0
                
                xm.mark_step()
                processed_since_last_update += embeddings.shape[0]
            
            # Concatenate embeddings for this chunk
            local_embeddings = torch.cat(chunk_embeddings, dim=0)
            
            # All-gather embeddings from all workers for this chunk
            gathered_embeddings = xm.all_gather(local_embeddings)
            
            xm.rendezvous(f'before save_complete_chunk_{chunk_idx}')
            
            # Save chunk on master only
            if rank == 0:
                chunk_filename = os.path.join(output_path, f'embeddings_chunk_{chunk_idx}.pt')
                print(f"Saving chunk {chunk_idx + 1} with shape {gathered_embeddings.shape}, at {chunk_filename}")
                torch.save(gathered_embeddings.cpu(), chunk_filename)
                logger.info(f"Master: Saved chunk {chunk_idx + 1} embeddings")
                
            
            # Clear variables to free memory
            del chunk_embeddings
            del local_embeddings
            del gathered_embeddings
            
            # Make sure all workers sync up after each chunk
            xm.rendezvous(f'save_complete_chunk_{chunk_idx}')
            logger.info(f"Worker {rank}: Completed sync after chunk {chunk_idx}")

        logger.info(f"Worker {rank}: All chunks processed")
                
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/disks/storage/cc3m/cc3m-wds")
    parser.add_argument('--output_path', type=str, default="/mnt/disks/peter-pd-tokenization/saved_embed")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=200000, help='Maximum number of images to process')
    parser.add_argument('--resolution', type=int, default=384)
    args = parser.parse_args()
    
    # Initialize embedding generator
    if xm.get_ordinal() == 0:
        print(f"Starting embedding generation for {args.max_images} images")
        print(f"Estimated storage (fp16): {args.max_images * 256 * 1152 * 2 / (1024**3):.2f} GB")
    generator = EmbeddingGenerator(args)
    
    # Generate and save embeddings
    generator.generate_and_save_embeddings(args.output_path)
    
    # Sync all workers before finishing
    xm.rendezvous('embedding_generation_complete')
    
    if xm.get_ordinal() == 0:
        logger.info("Embedding generation complete for all workers")

def _mp_fn(index):
    # cache init needs to happens inside the mp_fn.
    xr.initialize_cache(f'/tmp/xla_cache_{index}', readonly=False)
    main()
    

if __name__ == "__main__":
    torch_xla.launch(_mp_fn, args=())  # Launches multiple TPU processes




