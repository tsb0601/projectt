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

def get_data_loader(rank, world_size, epoch, urls, siglip_processor, args):
    num_shards = len(urls)
    shards_per_worker = num_shards // world_size
    start_shard = rank * shards_per_worker
    end_shard = start_shard + shards_per_worker
    if rank == world_size - 1:
        end_shard = num_shards
    
    worker_urls = urls[start_shard:end_shard]
    logger.info(f"Worker {rank}: Processing {len(worker_urls)} shards")
    
    samples_per_shard = 2500
    estimated_samples = len(worker_urls) * samples_per_shard
    
    dataset = WebDatasetAdapter(
        worker_urls,
        siglip_processor,
        args,
        num_samples=estimated_samples
    ).create_webdataset(epoch)
    
    cpu_loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        persistent_workers=False
    )
    
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
        shard_files = get_cc3m_shards(self.args.data_path)
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
                
                images = batch["image"]
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
    # def generate_and_save_embeddings(self, output_path):
    #     shard_files = get_cc3m_shards(self.args.data_path)
    #     rank = xm.get_ordinal()
    #     world_size = xm.xrt_world_size()
        
    #     os.makedirs(output_path, exist_ok=True)
        
    #     start_time = time.time()
    #     chunk_size = 2048
    #     global_batch_size = self.args.batch_size * world_size
    #     steps_per_chunk = chunk_size // global_batch_size
    #     num_chunks = (self.args.max_images + chunk_size - 1) // chunk_size
        
    #     for chunk_idx in range(num_chunks):
    #         global_chunk_idx = rank * num_chunks + chunk_idx
    #         chunk_embeddings = []
    #         remaining_images = min(chunk_size, self.args.max_images - chunk_idx * chunk_size)
    #         steps_this_chunk = (remaining_images + global_batch_size - 1) // global_batch_size

    #         try:
    #             train_loader, _ = get_data_loader(
    #                 rank, world_size, chunk_idx,
    #                 shard_files, self.model.processor, self.args
    #             )
                
    #             for batch_idx, batch in enumerate(tqdm(train_loader, 
    #                                                  desc=f"Worker {rank} Chunk {chunk_idx + 1}/{num_chunks}",
    #                                                  total=steps_this_chunk)):
    #                 if batch_idx >= steps_this_chunk:
    #                     break
                        
    #                 images = batch["image"]
    #                 embeddings = self.process_batch(images)
    #                 embeddings = embeddings.to(torch.float16).reshape(-1, 1152)
    #                 chunk_embeddings.append(embeddings)
    #                 xm.mark_step()
                    
    #         finally:
    #             # Cleanup after processing chunk
    #             del train_loader, batch, images, embeddings
    #             gc.collect()

    #         # Save and sync
    #         local_embeddings = torch.cat(chunk_embeddings, dim=0)
    #         chunk_filename = os.path.join(output_path, f'embeddings_chunk_{global_chunk_idx}.pt')
    #         torch.save(local_embeddings.cpu(), chunk_filename)
    #         logger.info(f"Worker {rank}: Saved chunk {global_chunk_idx}")
            
    #         del local_embeddings, chunk_embeddings
    #         gc.collect()
    #         xm.rendezvous(f'save_complete_chunk_{chunk_idx}')

def main():
    import argparse
    import time
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/disks/storage/cc3m/cc3m-wds")
    parser.add_argument('--output_path', type=str, default="/mnt/disks/peter-pd-tokenization/saved_embed/small_chunks")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=200000)
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
