import os
import io
import logging
import gc
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
import time 
from typing import List, Dict, Any
import gcsfs
from pathlib import Path

import PIL.Image
import shutil
import gcsfs
import tempfile
from tfrecord.torch.dataset import TFRecordDataset
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_laion_shards(base_path="gs://laion400m/webdataset"):
    """Get list of LAION-400M WebDataset shards"""
    # LAION-400M typically has ~400,000 shards (adjust based on actual count)
    total_shards = 400000  
    shard_files = [
        os.path.join(base_path, f"{i:06d}.tar") 
        for i in range(total_shards)
    ]
    
    # For cloud storage paths, verify existence if possible
    # For local paths, check existence:
    existing_shards = [s for s in shard_files if os.path.exists(s)]
    
    if not existing_shards:
        raise RuntimeError(f"No LAION shards found in {base_path}")
    logger.info(f"Found {len(existing_shards)} LAION shards")
    return existing_shards

def calculate_steps_per_epoch(total_shards, batch_size, world_size):
    """Calculate steps per epoch for proper tracking"""
    samples_per_shard = 5500  # CC3M average
    total_samples = total_shards * samples_per_shard
    return total_samples // (batch_size * world_size)



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
        seed = 42 + epoch
        dataset = (
            wds.WebDataset(self.urls, shardshuffle=1000)
            .shuffle(1000, seed=seed)  # Reduced shuffle buffer
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
            
        siglip_image = self.siglip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return (siglip_image,)
    
    def collate_fn(self, samples):
        siglip_images = torch.stack([sample[0] for sample in samples])
        return {"image": siglip_images}


from torch.utils.data import IterableDataset

class ShardedTFRecordDataset(IterableDataset):
    """
    Dataset for reading sharded TFRecord files with TPU (or multi-process) support,
    loading each shard on-the-fly to avoid high memory usage.
    """
    
    def __init__(
        self, 
        urls: List[str], 
        rank: int, 
        world_size: int, 
        processor: Any = None
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.processor = processor

        
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    


        # GCS filesystem (only if you need to download from GCS)
        self.fs = gcsfs.GCSFileSystem()
        
        # Temporary directory for local shards
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f'worker_{rank}_'))
        print(f"Worker {rank} using temp dir: {self.temp_dir}")
        
        # Calculate which shard indices this worker is responsible for
        num_shards = len(urls)
        min_shards_per_worker = num_shards // world_size
        extra_shards = num_shards % world_size
        
        if rank < extra_shards:
            start_shard = rank * (min_shards_per_worker + 1)
            shards_for_this_worker = min_shards_per_worker + 1
        else:
            start_shard = rank * min_shards_per_worker + extra_shards
            shards_for_this_worker = min_shards_per_worker
        end_shard = start_shard + shards_for_this_worker
        
        print(f"Total shards: {num_shards}, worker {rank} handling shards {start_shard}..{end_shard-1}")
        
        # Store just the shard paths this worker will process
        self.worker_urls = urls[start_shard:end_shard]
        
        if len(self.worker_urls) > 0:
            print(f"Worker {rank} first shard: {self.worker_urls[0]}")
            print(f"Worker {rank} last shard: {self.worker_urls[-1]}")

        # TFRecord description - adapt to your actual schema
        self.description = {
            "image": "byte",
        }

    def process_sample(self, sample: Dict[str, bytes]) -> torch.Tensor:
        """Process raw image bytes into a torch Tensor (or dict, etc.)."""
        try:
            # Convert bytes to PIL Image
            image = PIL.Image.open(io.BytesIO(sample["image"])).convert("RGB")

            # If you have a huggingface Processor, do it here
            if self.processor is not None:

                # print("I am here!!!!!!!!!!")
                siglip_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

                        # Preprocess for VAE
                # vae_image = self.transform(image)
                
                # print("Shape and shape are", siglip_image.shape, vae_image.shape)
                
                return siglip_image
                # return processed
            
            # Otherwise, do a trivial transform to tensor
            return torch.from_numpy(np.array(image)).permute(2, 0, 1)

        except Exception as e:
            # Return None or raise an error, depending on your preference
            print(f"Error processing image: {str(e)}")
            return None

    def __iter__(self):
        """
        Iterate through all shards assigned to this worker, and yield samples
        one by one in a streaming fashion (no big memory overhead).
        """
        for tfrecord_path in self.worker_urls:
            try:
                # Download shard from GCS to local temp path
                local_path = self.temp_dir / Path(tfrecord_path).name
                print(f"Worker {self.rank} downloading shard {tfrecord_path} to {local_path}")
                self.fs.get(tfrecord_path.replace('gs://', ''), str(local_path))

                # Create TFRecordDataset pointing to local shard
                dataset = TFRecordDataset(
                    data_path=str(local_path),
                    index_path=None,  # or set if you have .index
                    description=self.description,
                    transform=self.process_sample,  # apply transform per sample
                )

                # Iterate over *all* samples in the shard
                # This yields them one by one, so memory usage is minimal
                for sample in dataset:
                    if sample is not None:
                        yield sample

                # Optionally remove local shard after iteration to free disk
                # or you can remove them in __del__. 
                # Here we remove it right away:
                try:
                    os.remove(local_path)
                except OSError as e:
                    print(f"Error removing {local_path}: {str(e)}")

            except Exception as e:
                print(f"Error loading {tfrecord_path}: {str(e)}")
                # If a shard fails, just continue to the next
                continue

    def __del__(self):
        """Cleanup temporary directory."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary dir: {str(e)}")





def get_data_loader(rank: int, world_size: int, epoch: int, urls: List[str], 
                   siglip_processor: Any, args: Any) -> tuple:
    """Create a data loader for TPU training"""
    print(f"\nInitializing loader for rank {rank}/{world_size}")
    
    # Create dataset
    dataset = ShardedTFRecordDataset(
        urls=urls,
        rank=rank,
        world_size=world_size,
        processor=siglip_processor
    )
    
    # Create CPU data loader
    cpu_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Wrap with ParallelLoader for TPU
    device = xm.xla_device()
    loader = pl.ParallelLoader(
        cpu_loader,
        [device],
        batchdim=0
    ).per_device_loader(device)
    
    # Estimate samples (based on LAION-400M average shard size)
    estimated_samples = len(urls) * 5750 // world_size
    print(f"Worker {rank} estimated samples: {estimated_samples}")
    
    return loader, estimated_samples


def get_all_urls():
    base_path = "gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets/laion400m/images/1.0.0"
    total_shards = 66256
    urls = [f"{base_path}/laion400m-train.tfrecord-{i:05d}-of-{total_shards:05d}" 
            for i in range(total_shards)]
    return urls
class EmbeddingGenerator:
    def __init__(self, args):
        self.args = args
        self.device = xm.xla_device()
        self.setup_model()
        
    def setup_model(self):
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
        with torch.no_grad():
            features, _, _, _, _ = self.model(images)
            return features

    def generate_and_save_embeddings(self, output_path):
        # shard_files = get_cc3m_shards(self.args.data_path)
        shard_files = get_all_urls()

        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        
        os.makedirs(output_path, exist_ok=True)
        
        # 1) Create the single data loader once
        train_loader, estimated_samples = get_data_loader(
            rank,
            world_size,
            epoch=0,  # or use rank if you want different seeds per worker
            urls=shard_files,
            siglip_processor=self.model.processor,
            args=self.args
        )
        
        chunk_size = 4096*256  # for example
        embedding_buffer = []
        current_chunk_count = 0  # Number of chunk files written
        
        total_processed = 0
        max_images = self.args.max_images //4 
        total_chunks = (max_images + chunk_size - 1) // chunk_size  # Calculate total chunks for progress bar

        # Wrap the data loader with tqdm
        with tqdm(total=max_images, desc=f"Worker {rank}: Processing Images", unit="images") as pbar:
            # Single pass over the entire loader
            for batch_idx, batch in enumerate(train_loader):
                if total_processed >= max_images:
                    break  # Stop once we've processed the maximum number of images
                # print(batch)
                images = batch
                if total_processed + images.size(0) > max_images:
                    needed = max_images - total_processed
                    images = images[:needed]
                
                with torch.no_grad():
                    embeddings = self.model(images)[0]  # the first output is features
                    embeddings = embeddings.to(torch.float16).reshape(-1, 1152)
                embedding_buffer.append(embeddings.cpu())
                
                total_processed += images.size(0)
                pbar.update(images.size(0))  # Update the tqdm progress bar
                
                # Check if buffer has reached chunk_size or if we're done
                current_buffer_size = sum(t.shape[0] for t in embedding_buffer)
                if current_buffer_size >= chunk_size:
                    # Save current buffer to disk
                    chunk_embeddings = torch.cat(embedding_buffer, dim=0)
                    chunk_filename = os.path.join(output_path, f'embeddings_chunk_{rank}_{current_chunk_count}.pt')
                    torch.save(chunk_embeddings, chunk_filename)
                    
                    embedding_buffer = []  # Clear the buffer
                    current_chunk_count += 1
                    
                    # Log progress to tqdm
                    pbar.set_postfix({
                        "Chunks Saved": current_chunk_count,
                        "Current Buffer": current_buffer_size
                    })
                    
                    xm.mark_step()
            
            # Save any remaining embeddings in the buffer
            if len(embedding_buffer) > 0:
                chunk_embeddings = torch.cat(embedding_buffer, dim=0)
                chunk_filename = os.path.join(output_path, f'embeddings_chunk_{rank}_{current_chunk_count}.pt')
                torch.save(chunk_embeddings, chunk_filename)
                current_chunk_count += 1  # Increment chunk count
                
                pbar.set_postfix({
                    "Chunks Saved": current_chunk_count,
                    "Current Buffer": 0
                })
        
        logger.info(f"Worker {rank}: Processed {total_processed} images, saved {current_chunk_count} chunk files.")
        xm.rendezvous('embedding_generation_complete')

def main():
    import argparse
    import time
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/disks/storage/cc3m/cc3m-wds")
    parser.add_argument('--output_path', type=str, default="/mnt/storage/embeds/saved_embed/small_chunks")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=1500000)
    parser.add_argument('--resolution', type=int, default=384)
    args = parser.parse_args()
    
    if xm.get_ordinal() == 0:
        print(f"Starting embedding generation for {args.max_images} images")
        print(f"Estimated storage (fp16): {args.max_images * 256 * 1152 * 2 / (1024**3):.2f} GB")
    
    generator = EmbeddingGenerator(args)
    generator.generate_and_save_embeddings(args.output_path)
    xm.rendezvous('embedding_generation_complete')
    
    if xm.get_ordinal() == 0:
        logger.info("Embedding generation complete for all workers")

def _mp_fn(index):
    xr.initialize_cache(f'/tmp/xla_cache_{index}', readonly=False)
    main()

if __name__ == "__main__":
    torch_xla.launch(_mp_fn, args=())
