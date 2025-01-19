import torch_xla.runtime as xr
import os
import argparse
import math
import torch
import torch.distributed as dist
import torch_xla as xla
import wandb
from torch import nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_xla.distributed.parallel_loader import ParallelLoader

from transformers import AutoProcessor, AutoModel
from torchvision import transforms
from PIL import Image
import logging
import wandb
from tqdm import tqdm
import argparse
import json
import webdataset as wds
import io
import lpips
import math
import random
from typing import Optional, List
import torch_xla.amp as xla_amp
from torch.utils.data import IterableDataset

import PIL.Image
import shutil
import gcsfs
import tempfile
from tfrecord.torch.dataset import TFRecordDataset
from typing import List, Dict, Any
import numpy as np

xla._XLAC._xla_set_mat_mul_precision('highest') # set precision to high to assure accuracy

from rqvae.models.tokenizer.decoder import VAEDecoder
from rqvae.models.tokenizer.convdecoder import ConvDecoder

from rqvae.models.tokenizer.discriminator import create_dinov2_discriminator
# from rqvae.models.tokenizer.quantizer import CodebookAnalyzer
from rqvae.models.tokenizer.siglip_vq import SigLIPVQEncoder
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)







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
                vae_image = self.transform(image)
                
                # print("Shape and shape are", siglip_image.shape, vae_image.shape)
                
                return siglip_image, vae_image
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





CACHE_DIR = '/home/tsb/.cache/xla_compile'
project_name = 'tmp'
cache_path = os.path.join(CACHE_DIR, project_name)
cache_path = os.environ.get('XLACACHE_PATH', cache_path)



# SigLIP Encoder remains mostly the same, just device handling changes
class SigLIPEncoder(nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", num_tokens=64, device=None):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.hidden_size = 1152
        self.device = device
        
        self.load_model()

    def load_model(self):
        model = AutoModel.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.vision_tower = model.vision_model
        if self.device:
            self.vision_tower = self.vision_tower.to(self.device)
        self.processor = processor

    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        outputs = self.vision_tower(images, output_hidden_states=True)
        image_features = outputs.hidden_states[-1]
        
        b, num_tokens, dim = image_features.shape
        h = w = int(num_tokens**0.5)
        target_h = target_w = int(self.num_tokens**0.5)

        if self.num_tokens != 729:
            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2)
            image_features = F.interpolate(image_features, size=(target_h, target_w), mode='bilinear', align_corners=False)
            image_features = image_features.permute(0, 2, 3, 1).contiguous().view(b, self.num_tokens, dim)

        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.processor(images=image, return_tensors="pt")['pixel_values']
        if self.device:
            image = image.to(self.device)
        
        with torch.no_grad():
            features = self(image)
        return features

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

def preprocess_data(item, siglip_processor, args):
    image = item[0].convert('RGB')
    siglip_image = siglip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
    vae_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    vae_image = vae_transforms(image)
    return siglip_image, vae_image

def get_scheduler_lambda(total_steps=500000):
    warmup_steps = total_steps//50
    decay_start = int(0.8 * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        if current_step < decay_start:
            return 1.0
        
        num_decay_steps = total_steps - decay_start
        progress = float(current_step - decay_start) / float(max(1, num_decay_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return lr_lambda

import glob  # Add this at the top with other imports

def get_latest_checkpoint(save_dir):
    """Find the latest checkpoint in the save directory"""
    checkpoints = glob.glob(f"{save_dir}/checkpoint_step_*.pt")
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_checkpoint

def save_checkpoint(model, discriminator, optimizer, d_optimizer, scheduler, global_step, path, epoch, siglip_encoder):
    xm.rendezvous('pre_save')
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Add VQ state to checkpoint
    vae_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'siglip_encoder_state_dict': siglip_encoder.state_dict(),  # Added
        'global_step': global_step,
        'epoch': epoch
    }
    
    # Prepare the discriminator checkpoint if it exists
    d_checkpoint = None
    if discriminator is not None and d_optimizer is not None:
        d_checkpoint = {
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': d_optimizer.state_dict(),
            'global_step': global_step,
            'epoch': epoch
        }
    
    # Only master saves
    if xm.get_ordinal() == 0:
        # Save VAE checkpoint
        xm.save(vae_checkpoint, path)
        
        # Save discriminator checkpoint if it exists
        if d_checkpoint is not None:
            d_path = path.replace('checkpoint', 'discriminator')
            xm.save(d_checkpoint, d_path)
    
    xm.rendezvous('post_save')  # Final sync point




def load_checkpoint(model, discriminator, optimizer, d_optimizer, scheduler, checkpoint_path, siglip_encoder):
    xm.rendezvous('checkpoint_load')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model states including VQ
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'siglip_encoder_state_dict' in checkpoint:  # For backward compatibility
        siglip_encoder.load_state_dict(checkpoint['siglip_encoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    global_step = checkpoint['global_step']
    epoch = checkpoint['epoch']

    # If using a discriminator, load it too
    if discriminator is not None and d_optimizer is not None:
        d_path = checkpoint_path.replace('checkpoint', 'discriminator')
        if os.path.exists(d_path):
            d_checkpoint = torch.load(d_path, map_location='cpu')
            discriminator.load_state_dict(d_checkpoint['model_state_dict'])
            d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])

    # Final sync
    xm.rendezvous('post_load')

    return global_step, epoch


def get_latest_checkpoint(save_dir):
    checkpoints = glob.glob(f"{save_dir}/checkpoint_step_*.pt")
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_checkpoint

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
            wds.WebDataset(self.urls)
            .shuffle(5000, seed=seed)  # Large shuffle buffer
            .decode("pil")
            .to_tuple("jpg")
            .map(self.preprocess_sample)
            .batched(self.args.batch_size)
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
        
        # Preprocess for VAE
        vae_image = self.transform(image)
        
        return siglip_image, vae_image

# def get_data_loader(rank, world_size, epoch, urls, siglip_processor, args):
#     """
#     Create a data loader for the current TPU core and epoch
#     """
#     # Calculate shards for this worker
#     num_shards = len(urls)
#     shards_per_worker = num_shards // world_size
#     start_shard = rank * shards_per_worker
#     end_shard = start_shard + shards_per_worker
#     if rank == world_size - 1:  # Last worker takes remaining shards
#         end_shard = num_shards
    
#     worker_urls = urls[start_shard:end_shard]
    
#     # Estimate number of samples (important for proper epoch handling)
#     samples_per_shard = 2500  # CC3M average
#     estimated_samples = len(worker_urls) * samples_per_shard
    
#     # Create dataset
#     dataset = WebDatasetAdapter(
#         worker_urls,
#         siglip_processor,
#         args,
#         num_samples=estimated_samples
#     ).create_webdataset(epoch)
    
#     # Create data loader

#     # First create a regular PyTorch DataLoader
#     cpu_loader = DataLoader(
#         dataset,
#         batch_size=None,  # Already batched by WebDataset
#         num_workers=args.num_workers
#     )
    
#     # Then wrap it with ParallelLoader for TPU
#     device = xm.xla_device()
#     loader = ParallelLoader(
#         cpu_loader,
#         [device],  # List of devices where data should be sent
#         batchdim=0  # The dimension holding the batch size
#     ).per_device_loader(device)  # Get the per-device loader
    
#     return loader, estimated_samples


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

def calculate_steps_per_epoch(total_shards, batch_size, world_size):
    """Calculate steps per epoch for proper tracking"""
    samples_per_shard = 5500  # CC3M average
    total_samples = total_shards * samples_per_shard
    return total_samples // (batch_size * world_size)


def compute_d_loss(real_images, fake_images, discriminator):
    """
    Standard (non-saturating) discriminator loss:
      D_loss = E[softplus(-D(real))] + E[softplus(D(fake))]
    """
    # Concatenate real and fake
    combined = torch.cat([real_images, fake_images], dim=0)
    preds = discriminator(combined)
    real_pred, fake_pred = torch.chunk(preds, 2, dim=0)

    # Logistic loss
    d_loss = (F.softplus(-real_pred) + F.softplus(fake_pred)).mean()
    return d_loss

def compute_g_loss(fake_images, discriminator):
    """
    Standard (non-saturating) generator loss:
      G_loss = E[softplus(-D(fake))]
    """
    fake_pred = discriminator(fake_images)
    g_loss = F.softplus(-fake_pred).mean()
    return g_loss



# def train_one_step(batch, models, optimizers, state):
#     """
#     Unified training step for both VAE and GAN paths to avoid recompilation.
#     """
#     # Unpack inputs, models, and optimizers
#     siglip_images, vae_images = batch
#     vae, discriminator, siglip_encoder, lpips_loss = models
#     optimizer, d_optimizer = optimizers
#     args = state.args
#     device = xm.xla_device()

#     # Determine GAN activity status and weight
#     gan_active = 1.0 if state.global_step >= args.gan_start_steps else 0.0
#     gan_weight = compute_gan_weight(state.global_step, args) if gan_active > 0 else 0.0

#     # Enable BF16 autocast if specified
#     with xla_amp.autocast(enabled=getattr(args, 'bf16', False), dtype=torch.bfloat16, device=device):
#         # =========================================================
#         # 1) DISCRIMINATOR PHASE (Always executed, but scaled by gan_active)
#         # =========================================================
#         quantized, _, _, _, _ = siglip_encoder(siglip_images)
#         with torch.no_grad():
#             fake_images = vae(quantized).detach()

#         d_loss = compute_d_loss(
#             real_images=vae_images,
#             fake_images=fake_images,
#             discriminator=discriminator
#         )
#         d_loss *= gan_active  # Scale by GAN activity status

#         # =========================================================
#         # 2) GENERATOR (VAE) PHASE
#         # =========================================================
#         quantized, total_vq_loss, encoding_indices, clean_loss, vq_loss = siglip_encoder(siglip_images)
#         recon_images = vae(quantized)

#         # Reconstruction and perceptual losses
#         recon_loss = F.mse_loss(recon_images, vae_images)
#         perceptual_loss = lpips_loss(recon_images, vae_images).mean()

#         # Combine losses
#         total_loss = recon_loss + args.perceptual_weight * perceptual_loss + total_vq_loss

#         # Add generator adversarial loss if GAN is active
#         g_loss = compute_g_loss(fake_images=recon_images, discriminator=discriminator) if gan_active > 0 else torch.tensor(0.0, device=device)
#         total_loss += gan_weight * g_loss

#     # =========================================================
#     # Backward and Optimizer Steps
#     # =========================================================

#     # Update discriminator (if GAN active)
#     d_optimizer.zero_grad()
#     d_loss.backward()
#     xm.optimizer_step(d_optimizer)
#     xm.mark_step()

#     # Update VAE (always executed)
#     optimizer.zero_grad()
#     total_loss.backward()

#     # Gradient clipping
#     torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
#     if args.train_encoder:
#         torch.nn.utils.clip_grad_norm_(siglip_encoder.parameters(), args.max_grad_norm)

#     xm.optimizer_step(optimizer)
#     xm.mark_step()

#     # Return losses and images for logging
#     return {
#         "d_loss": float(d_loss.item()),
#         "g_loss": float(g_loss.item() * gan_active),  # Show 0 if GAN inactive
#         "recon_loss": float(recon_loss.item()),
#         "perceptual_loss": float(perceptual_loss.item()),
#         "vq_loss": float(vq_loss.item()),
#         "reference_loss": float(clean_loss.item()),
#         "total_vq_loss": float(total_vq_loss.item()),
#         "total_loss": float(total_loss.item()),
#     }, vae_images, recon_images, encoding_indices



def train_one_step(batch, models, optimizers, state):
    """
    Modified training step to handle VQ with optional BF16 autocast.
    """
    # Unpack inputs and models
    siglip_images, vae_images = batch
    vae, discriminator, siglip_encoder, lpips_loss = models
    optimizer, d_optimizer = optimizers
    args = state.args
    device = xm.xla_device()

    # Enable BF16 autocast if args.bf16 is True
    with xla_amp.autocast(enabled=getattr(args, 'bf16', False), dtype=torch.bfloat16, device=device):
        # =============================================================================
        # (1) DISCRIMINATOR PHASE
        # =============================================================================
        if args.use_gan and state.global_step >= args.gan_start_steps:
            # Get quantized embeddings
            with torch.no_grad():
                quantized, _, _, _, _ = siglip_encoder(siglip_images)
                fake_images = vae(quantized).detach()

            d_loss = compute_d_loss(
                real_images=vae_images,
                fake_images=fake_images,
                discriminator=discriminator
            )

            if state.global_step % args.d_reg_every == 0:
                d_optimizer.zero_grad()
                d_loss.backward()
                xm.optimizer_step(d_optimizer)
                xm.mark_step()
            else:
                # If not updating D this step, don't keep grads
                d_loss = d_loss.detach()
        else:
            d_loss = torch.tensor(0.0, device=device)

        # =============================================================================
        # (2) GENERATOR (VAE) PHASE
        # =============================================================================
        # print("whyyyy the shape is", siglip_images.shape)



        quantized, total_vq_loss, encoding_indices, clean_loss, vq_loss = siglip_encoder(siglip_images)
        recon_images = vae(quantized)
        # print("input embed:", quantized.shape, "recon images shape:", recon_images.shape, "vae images shape:", vae_images.shape)
        # Reconstruction + perceptual losses
        recon_loss = F.mse_loss(recon_images, vae_images)
        perceptual_loss = lpips_loss(recon_images, vae_images).mean()

        # Combine all losses
        total_loss = recon_loss + args.perceptual_weight * perceptual_loss + total_vq_loss

        # Add generator adversarial loss if in GAN mode
        if args.use_gan and state.global_step >= args.gan_start_steps:
            g_loss = compute_g_loss(fake_images=recon_images, discriminator=discriminator)
            gan_weight = compute_gan_weight(state.global_step, args)
            total_loss = total_loss + gan_weight * g_loss
        else:
            g_loss = torch.tensor(0.0, device=device)

        # Backward + update generator (VAE)
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
        if args.train_encoder:
            torch.nn.utils.clip_grad_norm_(siglip_encoder.parameters(), args.max_grad_norm)

        xm.optimizer_step(optimizer)
        xm.mark_step()

    # Return losses and images for logging
    metrics = {
        "d_loss": float(d_loss.item()),
        "g_loss": float(g_loss.item()),
        "recon_loss": float(recon_loss.item()),
        "perceptual_loss": float(perceptual_loss.item()),
        "clean_loss": float(clean_loss.item()),
        "total_loss": float(total_loss.item()),
    }
    
    # Add VQ-specific metrics only in VQ mode
    if state.args.encoder_mode == 'vq':
        metrics.update({
            "vq_loss": float(vq_loss.item()),
            "total_vq_loss": float(total_feat_loss.item()),
        })

    return metrics, vae_images, recon_images, encoding_indices


def compute_gan_weight(global_step, args):
    """
    Example ramp-up function for the GAN weight.
    Modify as needed for your schedule.
    """
    if global_step < args.gan_start_steps:
        return 0.0
    # linear or cosine ramp, etc.
    progress = float(global_step - args.gan_start_steps) / float(args.gan_rampup_steps)
    progress = max(0.0, min(1.0, progress))
    return args.gan_weight * progress


class TrainingState:
    """Class to hold training state and avoid global variables"""
    def __init__(self, args):
        self.global_step = 0
        self.epoch = 0
        self.args = args

def train_gan_step(real_images, fake_images, discriminator, args):
    """Unified GAN training step that always executes the same graph"""
    batch_size = real_images.size(0)
    
    # Always concatenate and get predictions
    with torch.no_grad():
        combined_images = torch.cat([real_images, fake_images.detach()], dim=0)
    combined_preds = discriminator(combined_images)
    real_pred, fake_pred = torch.chunk(combined_preds, 2, dim=0)
    
    # Compute discriminator loss
    d_loss = (F.softplus(-real_pred) + F.softplus(fake_pred)).mean()
    
  
    
    # Generator loss
    g_loss = F.softplus(-discriminator(fake_images)).mean()
    
    return d_loss, g_loss

def get_all_urls():
    base_path = "gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets/laion400m/images/1.0.0"
    total_shards = 66256
    urls = [f"{base_path}/laion400m-train.tfrecord-{i:05d}-of-{total_shards:05d}" 
            for i in range(total_shards)]
    return urls



def train_tpu(index, args):
    # Setup TPU device and process
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    
    # Initialize cache
    global cache_path
    xm.master_print(f'[!]XLACACHE_PATH: {cache_path}')
    os.makedirs(cache_path, exist_ok=True)
    if not xla._XLAC._xla_computation_cache_is_initialized():
        xr.initialize_cache(cache_path, readonly=False)
    
    # Model setup
    # siglip_encoder = SigLIPEncoder(num_tokens=args.num_tokens, device=device)

    siglip_encoder = SigLIPVQEncoder(
        num_tokens=args.num_tokens,
        num_codebook_vectors=args.num_codebook_vectors,
        use_commitment=args.use_commitment,
        commitment_cost=args.commitment_cost,
        trainable=args.train_encoder,
        progressive_unfreeze=args.progressive_unfreeze,
        unfreeze_after_steps=args.unfreeze_after_steps,
        unfreeze_strategy=args.unfreeze_strategy,
        device=device,
        use_vq=(args.encoder_mode == 'vq')  # Set VQ mode based on encoder_mode
    )
    # Initialize codebook analyzer
    
    


    siglip_processor = siglip_encoder.processor
    if args.decoder_type == 'conv':
        vae = ConvDecoder(
            in_channels=1152,     # SigLIP hidden dimension
            out_resolution=args.resolution,
            # num_res_blocks=3
        ).to(device)
    else:  # 'vae' (default)
        vae = VAEDecoder(
            num_tokens=args.num_tokens, 
            output_resolution=args.resolution
        ).to(device)


    # vae = VAEDecoder(num_tokens=args.num_tokens, output_resolution=args.resolution).to(device)
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    
    discriminator = None
    if args.use_gan:
        discriminator = create_dinov2_discriminator(
            model_size=args.dino_size,
            img_size=args.resolution,
            use_augment=True
        ).to(device)
    
    # Get all shard files
    shard_files = get_all_urls()
    steps_per_epoch = calculate_steps_per_epoch(
        len(shard_files), args.batch_size, world_size
    )
    
    # Get optimizer parameters - include encoder if trainable
    optimizer_params = list(vae.parameters())
    if args.train_encoder:
        optimizer_params.extend(siglip_encoder.parameters())
    
    # Initialize optimizers
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )


    d_optimizer = None
    if args.use_gan:
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.disc_lr,
            betas=(0.0, 0.99)
        )
    
    # Initialize schedulers
    lr_scheduler_fn = get_scheduler_lambda(args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_fn)
    
    # Initialize wandb for master process
    if xm.get_ordinal() == 0:

        wandb.init(
            project="siglip-vae-tpu",
            config={**vars(args), "steps_per_epoch": steps_per_epoch},
            name=f"tpu_run_{args.run_name}"
        )
    
    # Training state
    state = TrainingState(args)
    
    # Load checkpoint if exists
    if args.resume or args.checkpoint_path:
        checkpoint_path = args.checkpoint_path or get_latest_checkpoint(args.save_dir)
        if checkpoint_path:
            if xm.get_ordinal() == 0:
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                
            # Load both VAE and discriminator (if using GAN) in a synchronized way
            state.global_step, state.epoch = load_checkpoint(
                model=vae,
                discriminator=discriminator if args.use_gan else None,
                optimizer=optimizer,
                d_optimizer=d_optimizer if args.use_gan else None,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path, 
                siglip_encoder=siglip_encoder
            )
        
    # Models and optimizers bundles
    models = (vae, discriminator, siglip_encoder, lpips_loss)
    optimizers = (optimizer, d_optimizer)
    
    # Training loop
    while state.global_step < args.max_steps:
        train_loader, num_samples = get_data_loader(
            index, world_size, state.epoch,
            shard_files, siglip_processor, args
        )
        
        for batch in train_loader:
            if state.global_step >= args.max_steps:
                break

            # Update encoder freeze status if using progressive unfreezing
            if args.progressive_unfreeze:
                siglip_encoder.update_freeze_status(state.global_step)
            
            # Training step
            losses, vae_images, recon_images, indices = train_one_step(
                batch, models, optimizers, state
                
            )

            # print("indices", indices, indices.shape)

            
            # Update codebook analysis



            scheduler.step()
            # print("1111111111111111111")
            # Logging
            if xm.get_ordinal() == 0:
                # print("22222222222222")
                log_dict = {
                    **losses,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": state.epoch,
                    "global_step": state.global_step,
                }
                
                # Only add VQ-specific metrics if in VQ mode
                if args.encoder_mode == 'vq':
                    log_dict.update({
                        "codebook_usage": siglip_encoder.vq.get_metrics()['usage_fraction'],
                        "codebook_perplexity": siglip_encoder.vq.get_metrics()['perplexity'],
                    })
                
                wandb.log(log_dict)

                
                # print("33333333333333")
                if state.global_step % args.sample_steps == 0:
                    with torch.no_grad():
                        
                        # print("why don't you save???")
                        samples = torch.cat([vae_images[:4], recon_images[:4]], dim=0)
                        samples = (samples + 1) / 2
                        # print("mid fuck")

                        wandb.log({
                            "samples": [wandb.Image(img) for img in samples.cpu()],
                            "global_step": state.global_step
                        })


                        # print("wtfffff")

            # print("4444444444444")
            # Checkpointing
            if (state.global_step+1) % args.save_step == 0:
                print(f"[Rank={xm.get_ordinal()}] Entering save at step={state.global_step}")

                save_checkpoint(
                    vae, discriminator, optimizer, d_optimizer, scheduler,
                    state.global_step,
                    f"{args.save_dir}/checkpoint_step_{state.global_step}.pt",
                    state.epoch, 
                    siglip_encoder
                )
                
            # print("55555555555")
            state.global_step += 1
            xm.mark_step()
        
        state.epoch += 1
    
    if xm.get_ordinal() == 0:
        wandb.finish()

def add_vq_args(parser):
    # VQ-specific arguments
    parser.add_argument("--num_codebook_vectors", type=int, default=16384,
                       help="Number of vectors in VQ codebook")
    parser.add_argument("--use_commitment", action="store_true",
                       help="Use commitment loss in VQ")
    parser.add_argument("--commitment_cost", type=float, default=0.25,
                       help="Weight of commitment loss if used")
    parser.add_argument("--analysis_steps", type=int, default=1000,
                       help="Steps between detailed codebook analysis")
                       
    # Encoder training arguments
    parser.add_argument("--train_encoder", action="store_true",
                       help="Allow encoder to be trained")
    parser.add_argument("--progressive_unfreeze", action="store_true",
                       help="Gradually unfreeze encoder")
    parser.add_argument("--unfreeze_after_steps", type=int, default=20000,
                       help="Steps before starting unfreezing")
    parser.add_argument("--unfreeze_strategy", type=str, default='all',
                       choices=['all'],
                       help="How to unfreeze encoder")
    parser.add_argument("--encoder_mode", type=str, choices=['vq', 'ae'], default='vq',
                   help="Encoder mode: 'vq' for vector quantization, 'ae' for regular autoencoder")


def main():
    parser = argparse.ArgumentParser(description="VAE Training Script for TPU")
    
    # Basic training arguments
    # Add decoder choice argument
    parser.add_argument("--decoder_type", type=str, choices=['vae', 'conv'], default='vae',
                       help="Type of decoder to use: 'vae' or 'conv'")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size per TPU core")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--max_steps", type=int, default=500000, help="Maximum number of training steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--save_step", type=int, default=5000, help="Save checkpoint every n steps")
    parser.add_argument("--sample_steps", type=int, default=500, help="Log sample images every n steps")
    
    # Model configuration
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--num_tokens", type=int, default=256, help="Number of tokens for SigLIP encoder")
    
    # GAN training arguments
    parser.add_argument("--use_gan", action="store_true", help="Enable GAN training")
    parser.add_argument("--gan_start_steps", type=int, default=20000, help="Start GAN training after n steps")
    parser.add_argument("--gan_rampup_steps", type=int, default=100000, help="Steps to ramp up GAN weight")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="Weight for GAN loss")
    parser.add_argument("--disc_lr", type=float, default=1e-5, help="Discriminator learning rate")
    parser.add_argument("--d_reg_every", type=int, default=2, help="Discriminator regularization every n steps")
    parser.add_argument("--r1_weight", type=float, default=10, help="Weight for R1 regularization")
    parser.add_argument("--dino_size", type=str, default="small", choices=["small", "base", "large", "giant"])
    
    # Loss weights and training options
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for KL divergence loss")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="Weight for LPIPS loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    
    # TPU specific arguments
    parser.add_argument("--tpu_cores", type=int, default=4, help="Number of TPU cores to use")
    parser.add_argument("--tpu_metrics_debug", action="store_true", help="Enable TPU metrics debugging")
    
    # Resume training arguments
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Specific checkpoint to resume from")

    # BF16 / Mixed precision
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision training on TPU")

    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="vae_run", help="Name of the run")

    # Add VQ arguments
    add_vq_args(parser)
    
    args = parser.parse_args()
    


   


    args.save_dir = f"ckpt_gcs/tokenizer/{args.run_name}_tpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start TPU training
    xmp.spawn(train_tpu, args=(args,), start_method='spawn')  # 8 TPU cores

if __name__ == "__main__":
    main()
