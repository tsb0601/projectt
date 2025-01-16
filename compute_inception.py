import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler
from rqvae.metrics.fid import InceptionWrapper, InceptionV3

def create_inception_transform():
    """Create transform that matches VAE's training distribution"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scales to [-1, 1] range
    ])

def setup_val_loader(val_root, batch_size, num_workers, device):
    """Sets up validation loader"""
    # Create validation dataset
    val_dataset = datasets.ImageFolder(
        root=val_root,
        transform=create_inception_transform()
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
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Wrap for TPU
    val_loader = pl.ParallelLoader(
        val_loader, 
        [device]
    ).per_device_loader(device)
    
    return val_loader

def setup_inception_model(device):
    """Sets up inception model"""
    inception_model = InceptionWrapper(
        [InceptionV3.BLOCK_INDEX_BY_DIM[2048]]
    ).to(device)
    inception_model.eval()
    return inception_model

def compute_inception_features(index, args):
    # Setup TPU device
    device = xm.xla_device()
    
    # Setup data loader
    val_loader = setup_val_loader(
        args.val_root,
        args.batch_size,
        args.num_workers,
        device
    )
    
    # Setup inception model
    inception_model = setup_inception_model(device)
    
    # Compute features
    all_features = []
    print("started!")
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Computing inception features"):
            features, _ = inception_model.get_logits(images)
            all_features.append(features)
            xm.mark_step()
    
    print("finished!")
    # Gather features from all TPU cores
    all_features = torch.cat(all_features, dim=0)
    all_features = xm.all_gather(all_features, pin_layout=True)
    print("gathered!")
    print(all_features.shape)
    # Only save on master process

# Save features for each rank
    rank = xm.get_ordinal()
    features_np = all_features.cpu().numpy()

    # Save rank-specific file
    rank_save_path = os.path.join(args.save_dir, f'val_256_act_rank{rank}.npz')
    print(f"Rank {rank}: Starting to save")
    np.savez(rank_save_path, act=features_np)
    print(f"Rank {rank}: Saved inception features to {rank_save_path}")
    print(f"Rank {rank}: Feature shape: {features_np.shape}")

def main(index):
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_root', type=str, default='/mnt/disks/boyang/datasets/ImageNet/val',
                       help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size per TPU core')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./inception_features',
                       help='Directory to save features')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Compute features
    compute_inception_features(index, args)

if __name__ == "__main__":
    xmp.spawn(main, args=())