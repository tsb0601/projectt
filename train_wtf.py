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

xla._XLAC._xla_set_mat_mul_precision('highest') # set precision to high to assure accuracy

from rqvae.models.tokenizer.decoder import VAEDecoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    warmup_steps = 10000
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

def save_checkpoint(model, optimizer, scheduler, global_step, path, epoch):
    """Save checkpoint with proper distributed handling"""
    xm.rendezvous('pre_save')  # First sync point for all processes
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch
    }
    # print("1111111111111111111")
    # Only master saves
    if xm.is_master_ordinal():
        # print("222222222222222")
        xm.save(checkpoint, path)
        # print("3333333333333")
    # print("44444444444444444")
    xm.rendezvous('post_save')  # Second sync point for all processes
    # print(":55555555555555555555S")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint with proper distributed handling"""
    # Ensure all processes wait for the master to load checkpoint
    if xm.is_master_ordinal():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    xm.rendezvous('checkpoint_load')
    
    # Broadcast checkpoint from master to all processes
    checkpoint = xm.mesh_reduce('checkpoint', checkpoint, lambda x: x[0])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['global_step'], checkpoint['epoch']


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
    
    # Estimate number of samples (important for proper epoch handling)
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

    # First create a regular PyTorch DataLoader
    cpu_loader = DataLoader(
        dataset,
        batch_size=None,  # Already batched by WebDataset
        num_workers=args.num_workers
    )
    
    # Then wrap it with ParallelLoader for TPU
    device = xm.xla_device()
    loader = ParallelLoader(
        cpu_loader,
        [device],  # List of devices where data should be sent
        batchdim=0  # The dimension holding the batch size
    ).per_device_loader(device)  # Get the per-device loader
    
    return loader, estimated_samples

def calculate_steps_per_epoch(total_shards, batch_size, world_size):
    """Calculate steps per epoch for proper tracking"""
    samples_per_shard = 2500  # CC3M average
    total_samples = total_shards * samples_per_shard
    return total_samples // (batch_size * world_size)

def train_tpu(index, args):
    # Configure TPU settings
    # configure_tpu_settings()
    
    # TPU setup
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    
    # Model setup
    siglip_encoder = SigLIPEncoder(num_tokens=args.num_tokens, device=device)
    siglip_processor = siglip_encoder.processor
    vae = VAEDecoder(num_tokens=args.num_tokens, output_resolution=args.resolution).to(device)
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    
    # Get all shard files
    shard_files = get_cc3m_shards()
    steps_per_epoch = calculate_steps_per_epoch(len(shard_files), args.batch_size, world_size)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    lr_scheduler_fn = get_scheduler_lambda(args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_fn)
    
    # Initialize wandb for master process
    if xm.is_master_ordinal():
        wandb.init(
            project="siglip-vae-tpu",
            config={**vars(args), "steps_per_epoch": steps_per_epoch},
            name=f"tpu_run_{args.run_name}"
        )
    
    # Load checkpoint if exists
    global_step = 0
    start_epoch = 0
    # Handle checkpoint loading
    if args.resume or args.checkpoint_path:
        checkpoint_path = args.checkpoint_path if args.checkpoint_path else get_latest_checkpoint(args.save_dir)
        if checkpoint_path:
            if xm.is_master_ordinal():
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            global_step, start_epoch = load_checkpoint(vae, optimizer, scheduler, checkpoint_path)
            if xm.is_master_ordinal():
                logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
        else:
            if xm.is_master_ordinal():
                logger.warning("No checkpoint found, starting from scratch")
    # Training loop

    epoch = start_epoch
    while global_step < args.max_steps:
        # Create data loader for this epoch
        train_loader, num_samples = get_data_loader(
            index, world_size, epoch,
            shard_files, siglip_processor, args
        )
        
 
        # Train for one epoch
        vae.train()
        for batch in train_loader:
            if global_step >= args.max_steps:
                break
                
            siglip_images, vae_images = batch
            
            with torch.no_grad():
                siglip_embeddings = siglip_encoder.encode_image(siglip_images)
            
            recon_images = vae(siglip_embeddings)
            
            # Loss computation
            recon_loss = F.mse_loss(recon_images, vae_images)
            perceptual_loss = lpips_loss(recon_images, vae_images).mean()
            loss = recon_loss + args.perceptual_weight * perceptual_loss
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
            xm.optimizer_step(optimizer)
            scheduler.step()
            
            # Logging and checkpointing
            if xm.is_master_ordinal():
                print("12_12_12_12_12_12_12_12_12_")

                wandb.log({
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "perceptual_loss": perceptual_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "global_step": global_step,
                })
                print("13_13_13_13_13_13_13_13_13_13_13_13_13_")

                
                if global_step % args.sample_steps == 0:
                    with torch.no_grad():
                        samples = torch.cat([vae_images[:4], recon_images[:4]], dim=0)
                        samples = (samples + 1) / 2
                        wandb.log({
                            "samples": [wandb.Image(img) for img in samples.cpu()],
                            "global_step": global_step
                        })
                # print("Here???")
            if global_step % args.save_step == 0:

                save_checkpoint(
                    vae, optimizer, scheduler, global_step,
                    f"{args.save_dir}/checkpoint_step_{global_step}.pt",
                    epoch
                )

                    
            global_step += 1
            xm.mark_step()  # Important for TPU
            
        epoch += 1
        
    if xm.is_master_ordinal():
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="VAE Training Script for TPU")
    
    # Basic training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per TPU core")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--max_steps", type=int, default=32000, help="Maximum number of training steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--save_step", type=int, default=5000, help="Save checkpoint every n steps")
    parser.add_argument("--sample_steps", type=int, default=500, help="Log sample images every n steps")
    
    # Model configuration
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--num_tokens", type=int, default=256, help="Number of tokens for SigLIP encoder")
    
    # Loss weights and training options
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for KL divergence loss")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="Weight for LPIPS loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    
    # TPU specific arguments
    parser.add_argument("--tpu_cores", type=int, default=4, help="Number of TPU cores to use")
    parser.add_argument("--tpu_metrics_debug", action="store_true", help="Enable TPU metrics debugging")

    # Add resume training arguments
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Specific checkpoint to resume from")
    
    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="vae_run", help="Name of the run")
    args = parser.parse_args()
    
    args.save_dir = f"ckpt_gcs/tokenizer/{args.run_name}_tpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start TPU training
    xmp.spawn(train_tpu, args=(args,), start_method='fork')  # 8 TPU cores

if __name__ == "__main__":
    main()