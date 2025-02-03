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
from torch.nn.parallel import DistributedDataParallel as DDP

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
import rqvae.utils.dist as dist_utils
from rqvae.utils.setup import setup, setup_quick, wandb_dir
from rqvae.models.tokenizer.discriminator import create_dinov2_discriminator
# from rqvae.models.tokenizer.quantizer import CodebookAnalyzer
from rqvae.models.tokenizer.siglip_vq import SigLIPVQEncoder
from pathlib import Path
from rqvae.metrics.fid import InceptionWrapper, frechet_distance, Inception_Score, InceptionV3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
from torchvision import datasets

def create_inception_transform():
    """Create transform for inception model input"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
class SigLIPTransform:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)



def setup_val_loader(
    val_root: str = '/mnt/disks/boyang/datasets/ImageNet/',
    batch_size: int = 4,
    num_workers: int = 4,
    device = None,
    siglip_processor = None
):
    """
    Sets up validation loader using SigLIP processor
    """
    if device is None:
        device = xm.xla_device()
    

    
    # Create validation dataset using SigLIP processor
    val_dataset = datasets.ImageFolder(
        root=os.path.join(val_root, 'val'),
        transform=SigLIPTransform(siglip_processor)
    )
    
    # Setup distributed sampler
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    
    # Create basic loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        # Need to try this
        drop_last=False
    )
    
    # Wrap for TPU
    val_loader = ParallelLoader(
        val_loader, 
        [device]
    ).per_device_loader(device)
    

    
    return val_loader



def setup_incetpion_model():
    """
    Sets up validation loader using SigLIP processor
    """
    
    device = xm.xla_device()
    
    # Setup inception model
    inception_model = InceptionWrapper(
        [InceptionV3.BLOCK_INDEX_BY_DIM[2048]]
    ).to(device)
    inception_model.eval()
    
    return inception_model




def compute_rfid_and_is(
    vae,
    siglip_encoder,
    val_loader,
    inception_model,
    device=None,
    fid_gt_act=None
):
    """
    Compute both RFID score and Inception Score using all TPU cores
    """
    if device is None:
        device = xm.xla_device()
        
    inception_acts = []
    inception_logits = []  # For IS score
    
    # All ranks: compute inception features
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Computing RFID/IS", disable=not xm.is_master_ordinal())
        # counter = 0
        for siglip_images, _ in pbar:
            # print("11111111111")
            siglip_images = siglip_images.to(device)
            # print("222222222")
            # Get reconstructions through SigLIP-VAE pipeline
            quantized, _, _, _, _ = siglip_encoder(siglip_images)
            recon_images = vae(quantized)
            # print("333333333333")
            # Convert to float32 for consistent processing
            recon_images = recon_images.detach().clone().float()
            # print("44444444444444")
            # Efficient conversion to uint8 and back
            # recon_np = tensor_image_to_numpy(recon_images)
            # recon_processed = torch.from_numpy(recon_np).to(torch.float32).to(device) / 255.
            # recon_processed = recon_processed.permute(0, 3, 1, 2)  # BHWC -> BCHW
            # print("5555555555555")
            # Get inception features and logits
            incep_act, incep_logits = inception_model.get_logits(recon_images)
            inception_acts.append(incep_act)
            inception_logits.append(torch.nn.functional.softmax(incep_logits, dim=-1))
            # print("666666666666")
            xm.mark_step()

    print("before end of eval loop")
    xm.rendezvous("end_of_eval_loop")
    print("after end of eval loop")


    inception_acts = torch.cat(inception_acts, dim=0)
    inception_acts = xm.all_gather(inception_acts, pin_layout=True)
    # print("2222222222222")
    inception_logits = torch.cat(inception_logits, dim=0)
    inception_logits = xm.all_gather(inception_logits, pin_layout=True)

    inception_acts = inception_acts.cpu().numpy()
    inception_logits = inception_logits.cpu().float()
    
    print("before post_gather")
    xm.rendezvous("post_gather")
    print("after post_gather")

    # Compute FID statistics
    mu = np.mean(inception_acts, axis=0)
    sigma = np.cov(inception_acts, rowvar=False)
    
    mu_gt = np.mean(fid_gt_act, axis=0)
    sigma_gt = np.cov(fid_gt_act, rowvar=False)

    # print("55555555555")

    # Calculate FID
    fid = frechet_distance(mu_gt, sigma_gt, mu, sigma)

    
    print(f"FID score is {fid}")
  
    IS_value, IS_std = Inception_Score(inception_logits)

    print(f"IS Score is {IS_value}")
    

    return fid, IS_value, IS_std

    # return None

def tensor_image_to_numpy(tensor_im: torch.Tensor) -> np.ndarray:
    """
    Convert tensor image in range [-1, 1] to numpy array in range [0, 255]
    """
    len_shape = len(tensor_im.shape)
    if len_shape == 3:
        tensor_im = tensor_im.unsqueeze(0)
    # Direct conversion from [-1, 1] to [0, 255]
    tensor_im = torch.clamp(tensor_im * 127.5 + 128, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    if len_shape == 3:
        tensor_im = tensor_im.squeeze(0)
    return tensor_im


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
    """
    Find the latest checkpoint in the save directory.
    Returns tuple of (checkpoint_path, step_number).
    """
    checkpoints = glob.glob(f"{save_dir}/checkpoint_step_*.pt")
    if not checkpoints:
        return None, -1
        
    # Get step numbers for all checkpoints
    checkpoint_steps = [(ckpt, extract_step_from_checkpoint(ckpt)) for ckpt in checkpoints]
    # Filter out any checkpoints where we couldn't parse the step number
    valid_checkpoints = [(ckpt, step) for ckpt, step in checkpoint_steps if step >= 0]
    
    if not valid_checkpoints:
        return None, -1
        
    # Find checkpoint with highest step number
    latest_checkpoint, latest_step = max(valid_checkpoints, key=lambda x: x[1])
    return latest_checkpoint, latest_step



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
    if 'siglip_encoder_state_dict' in checkpoint:
        vq_params = ['vq.embeddings', 'vq.usage_count'] 
        state_dict = {k: v for k, v in checkpoint['siglip_encoder_state_dict'].items()
                        if not any(vq_param in k for vq_param in vq_params)}
        siglip_encoder.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    global_step = checkpoint['global_step']
    epoch = checkpoint['epoch']

    # If using a discriminator, load it too
    if discriminator is not None and d_optimizer is not None:
        try:
            d_path = checkpoint_path.replace('checkpoint', 'discriminator')
            if os.path.exists(d_path):
                d_checkpoint = torch.load(d_path, map_location='cpu')
                discriminator.load_state_dict(d_checkpoint['model_state_dict'])
                d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])
        except:
            print("didn't find matched discriminator")

    # Final sync
    xm.rendezvous('post_load')

    return global_step, epoch


# def get_latest_checkpoint(save_dir):
#     checkpoints = glob.glob(f"{save_dir}/checkpoint_step_*.pt")
#     if not checkpoints:
#         return None
#     latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#     return latest_checkpoint



def determine_checkpoint_to_load(save_dir, specified_checkpoint=None, rank=0):
   
    """
    Determine which checkpoint to load based on latest checkpoint and specified checkpoint.
    
    Args:
        save_dir: Directory where checkpoints are saved
        specified_checkpoint: Optional specific checkpoint path to consider
    
    Returns:
        tuple: (checkpoint_path_to_use, step_number)
    """
    # Get latest checkpoint and its step
    latest_checkpoint, latest_step = get_latest_checkpoint(save_dir)
    
    # If no checkpoint specified, use latest
    if not specified_checkpoint:
        return latest_checkpoint, latest_step
        
    # Get step number from specified checkpoint
    specified_step = extract_step_from_checkpoint(specified_checkpoint)
    
    # If we can't parse the specified checkpoint step, use it anyway
    if specified_step < 0:
        return specified_checkpoint, specified_step
        
    # If latest step is higher than specified, use latest
    if latest_step > specified_step:
        return latest_checkpoint, latest_step
        
    # Otherwise use specified checkpoint
    return specified_checkpoint, specified_step

import time

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
        seed_offset = int(time.time())  # or read from your checkpoint

        seed = 42 + epoch + seed_offset
        
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


def get_cc_data_loader(rank, world_size, epoch, urls, siglip_processor, args):
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
                # d_optimizer.step()
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
        # print(vae)
        xm.optimizer_step(optimizer)
        # optimizer.step()
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
            # "total_vq_loss": float(total_feat_loss.item()),
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

def dataparallel_and_sync(model, find_unused_parameters=True):
    """Unified function for DDP wrapping and parameter synchronization"""
    model = DDP(
        model,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True
    )
    # Broadcast parameters from rank 0
    for _, param in model.state_dict().items():
        dist.broadcast(param, 0)
   
    xm.mark_step()
    return model


def extract_step_from_checkpoint(checkpoint_path):
    """Extract step number from checkpoint path."""
    try:
        # Extract the step number from checkpoint path format "checkpoint_step_{step}.pt"
        step = int(checkpoint_path.split('checkpoint_step_')[-1].split('.pt')[0])
        return step
    except (ValueError, IndexError):
        return -1  # Return -1 if we can't parse the step number



def train_tpu(index, args):
    # Setup TPU device and process
    device = xm.xla_device()
    world_size = xm.xrt_world_size()

    os.environ['PJRT_DEVICE'] = 'TPU'


    # Initialize process group for DDP
    # torch.distributed.init_process_group(
    #     'xla',
    #     init_method='xla://',
    #     world_size=world_size,
    #     rank=xm.get_ordinal()
    # )
    
    # Initialize cache
    global cache_path
    xm.master_print(f'[!]XLACACHE_PATH: {cache_path}')
    os.makedirs(cache_path, exist_ok=True)
    if not xla._XLAC._xla_computation_cache_is_initialized():
        xr.initialize_cache(cache_path, readonly=False)


    scaled_lr = calculate_scaled_lr(
        base_lr=args.base_lr,
        base_batch_size=args.base_batch_size,
        current_batch_size=args.batch_size,
        world_size=world_size
    )

    print(f"The scaled lr is {scaled_lr}!")
    
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
        use_vq=(args.encoder_mode == 'vq'),  # Set VQ mode based on encoder_mode
        kmeans_path=args.kmeans_path,  # Pass kmeans path
        trainable_codebook=args.trainable_codebook,

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
        if args.decoder_type == "vae-huge":
            size = "huge"
        elif args.decoder_type == "vae-giant":
            size = "giant"
        elif args.decoder_type == "vae-enormous":  # New intermediate size
            size = "enormous"
        elif args.decoder_type == "vae-colossal":  # New intermediate size
            size = "colossal"
        else:
            size = "large"  # Default to "large"
        vae = VAEDecoder(
            size=size, 
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
    

    optimizer_params = list(vae.parameters())
    if args.train_encoder:
        # optimizer_params.extend(siglip_encoder.parameters())


        # Only add VQ parameters if codebook is trainable
        for name, param in siglip_encoder.named_parameters():
            if 'vq' not in name:  # Exclude VQ parameters
                optimizer_params.append(param)


    # Initialize optimizers
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=scaled_lr,
        # betas=(0.9, 0.999),
        betas=(0.5, 0.9),
        weight_decay=0.01
    )


    d_optimizer = None
    if args.use_gan:
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=scaled_lr/args.disc_lr,
            # betas=(0.0, 0.99)
            betas=(0.5, 0.9)

        )


    

    fid_gt_act = np.load(args.fid_gt_act_path)['act']
    
    inception_model = setup_incetpion_model()

       


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
        checkpoint_to_load, step = determine_checkpoint_to_load(
            args.save_dir,
            args.checkpoint_path,
            rank=xm.get_ordinal()
        )

        if checkpoint_to_load:
                
            # Load both VAE and discriminator (if using GAN) in a synchronized way
            state.global_step, state.epoch = load_checkpoint(
                model=vae,
                discriminator=discriminator if args.use_gan else None,
                optimizer=optimizer,
                d_optimizer=d_optimizer if args.use_gan else None,
                scheduler=scheduler,
                checkpoint_path=checkpoint_to_load, 
                siglip_encoder=siglip_encoder
            )
        else:
            if xm.get_ordinal() == 0:
                logger.warning("No checkpoint found to resume from")
        
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


                    # In training loop where we process batches
          


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
            # if (state.global_step+1) % args.save_step == 0:
            state.global_step += 1


            if (state.global_step+1) % args.eval_freq == 0:

            # if (state.global_step) % args.eval_freq == 0:

                print("Started online eval")
                val_loader = setup_val_loader(
                    siglip_processor=siglip_encoder.processor,  # Add this line
                    device=device
                )
                print("Got my loader!")

                rfid, is_score, is_std = compute_rfid_and_is(
                    vae=vae,
                    siglip_encoder=siglip_encoder,
                    val_loader=val_loader,
                    inception_model=inception_model,
                    device=device,
                    fid_gt_act=fid_gt_act
                )
                
                # Only master logs
                if xm.get_ordinal() == 0:
                    wandb.log({
                        "metrics/rfid": rfid,
                        "metrics/is_score": is_score,
                        "metrics/is_std": is_std,
                        "global_step": state.global_step
                    })
                
                # Make sure all ranks sync up after evaluation
                # xm.rendezvous("rfid_eval")
                print("Finished online eval")

            # state.global_step += 1

            if (state.global_step+1) % args.save_step == 0:
            # if (state.global_step) % args.save_step == 0:
            
                print(f"[Rank={xm.get_ordinal()}] Entering save at step={state.global_step}")

                save_checkpoint(
                    vae, discriminator, optimizer, d_optimizer, scheduler,
                    state.global_step,
                    f"{args.save_dir}/checkpoint_step_{state.global_step}.pt",
                    state.epoch, 
                    siglip_encoder
                )

            xm.mark_step()
        
        state.epoch += 1
    
    if xm.get_ordinal() == 0:
        wandb.finish()

def add_vq_args(parser):
    # VQ-specific arguments
    parser.add_argument("--num_codebook_vectors", type=int, default=65536,
                       help="Number of vectors in VQ codebook")
    parser.add_argument("--use_commitment", action="store_true",
                       help="Use commitment loss in VQ")
    parser.add_argument("--commitment_cost", type=float, default=0.25,
                       help="Weight of commitment loss if used")
    parser.add_argument("--analysis_steps", type=int, default=1000,
                       help="Steps between detailed codebook analysis")

    # New arguments for codebook initialization
    parser.add_argument("--kmeans_path", type=str, default=None,
                       help="Path to pre-trained kmeans centroids")
    parser.add_argument("--trainable_codebook", action="store_true",
                       help="Whether to train the codebook or keep it frozen")
                       
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

def calculate_scaled_lr(base_lr: float, base_batch_size: int, current_batch_size: int, world_size: int) -> float:
    """
    Calculate the scaled learning rate based on linear scaling rule.
    
    Args:
        base_lr: Base learning rate (e.g., 4.5e-6)
        base_batch_size: Base batch size (e.g., 12)
        current_batch_size: Current batch size per device
        world_size: Number of devices (TPU cores)
        
    Returns:
        float: Scaled learning rate
    """
    global_batch_size = current_batch_size * world_size
    scaled_lr = base_lr * (global_batch_size / base_batch_size)
    return scaled_lr


def main(index):
    parser = argparse.ArgumentParser(description="VAE Training Script for TPU")
    
    # Basic training arguments
    # Add decoder choice argument
    parser.add_argument("--decoder_type", type=str, choices=['vae', 'vae-huge', 'vae-giant', 'conv'], default='vae',
                       help="Type of decoder to use: 'vae' or 'conv'")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size per TPU core")
    parser.add_argument("--base_lr", type=float, default=4.5e-6, help="Base learning rate (default: 4.5e-6)")
    parser.add_argument("--base_batch_size", type=int, default=12, help="Base batch size for LR scaling")
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
    parser.add_argument("--disc_lr", type=float, default=10, help="Discriminator learning rate")
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

    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='xla', choices=['xla'],type=str, help='distributed backend')
    parser.add_argument('--timeout', type=int, default=120, help='time limit (s) to wait for other nodes in DDP')
    parser.add_argument('--seed', type=int, default=41)


    parser.add_argument("--eval_freq", type=int, default=1000, help="Run RFID evaluation every n steps")
    parser.add_argument("--fid_gt_act_path", type=str, default="val_256_act_norm.npz", 
                        help="Path to pre-computed inception features")
    parser.add_argument("--imagenet_val_path", type=str, default="/mnt/disks/boyang/datasets/ImageNet/val",
                        help="Path to ImageNet validation set")

    # Add VQ arguments
    add_vq_args(parser)
    
    args = parser.parse_args()
    
    args.save_dir = f"ckpt_gcs/tokenizer/{args.run_name}_tpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start TPU training
    train_tpu(index, args)

def _mp_fn(index):
    # cache init needs to happens inside the mp_fn.
    xr.initialize_cache(f'/tmp/xla_cache_{index}', readonly=False)
    main(index)
    
if __name__ == "__main__":
    torch_xla.launch(_mp_fn, args=())
