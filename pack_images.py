from tqdm import tqdm
import numpy as np
from PIL import Image
import os
IM_SIZE = 256
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    imgs = os.listdir(sample_dir)
    #filter out non-image files
    img_suffix = ('.png', '.jpg', '.jpeg')
    imgs = [img for img in imgs if img.lower().endswith(img_suffix)]
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        img_path = os.path.join(sample_dir, imgs[i])
        sample_pil = Image.open(img_path).convert("RGB").resize((IM_SIZE, IM_SIZE), Image.BICUBIC)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python pack_images.py <sample_dir>")
        sys.exit(1)
    sample_dir = sys.argv[1]
    assert os.path.isdir(sample_dir), f"Invalid directory: {sample_dir}"
    create_npz_from_sample_folder(sample_dir)