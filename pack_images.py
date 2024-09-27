"""
modified from https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""

from tqdm import tqdm
import numpy as np
from PIL import Image
import os
IM_SIZE = 256
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    if pil_image.size == (image_size, image_size):
        return pil_image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    if sample_dir[-1] == '/':
        sample_dir = sample_dir[:-1] # remove trailing slash
    samples = []
    # get all imgs under sample_dir (recursively)
    imgs = []
    img_suffix = ('.png', '.jpg', '.jpeg')
    for root, dir , files in os.walk(sample_dir):
        for file in files:
            if file.lower().endswith(img_suffix):
                # append absolute path
                imgs.append(os.path.join(root, file))
    #filter out non-image files
    print(f"Found {len(imgs)} valid images in {sample_dir}.")
    num = min(num, len(imgs))
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        img_path = imgs[i]
        sample_pil = Image.open(img_path).convert("RGB")
        sample_pil = center_crop_arr(sample_pil, IM_SIZE)
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
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python pack_images.py <sample_dir>")
        sys.exit(1)
    sample_dir = sys.argv[1]
    im_size = 256 if len(sys.argv) == 2 else int(sys.argv[2])
    globals()['IM_SIZE'] = im_size
    assert os.path.isdir(sample_dir), f"Invalid directory: {sample_dir}"
    print(f"Creating .npz file from images in {sample_dir}, IM_SIZE={IM_SIZE}")
    create_npz_from_sample_folder(sample_dir)
    
if __name__ == "__main__":
    main()