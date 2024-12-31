import os
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
IMAGE_SIZE = 224
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
def get_all_image_under_dir(dir:str, img_ext: tuple = ('.png', '.jpg', '.jpeg')):
    """
    Get all images under the directory
    """
    img_list = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(img_ext):
                img_list.append(os.path.join(root, file))
    return img_list
target_img_dir = '/mnt/disks/storage/datasets/ImageNet/val'
save_dir = f'/mnt/disks/storage/datasets/ImageNet/val_{IMAGE_SIZE}'
def main():
    target_img_dir = '/mnt/disks/storage/datasets/ImageNet/val'
    save_dir = '/mnt/disks/storage/datasets/ImageNet/val_256'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_list = get_all_image_under_dir(target_img_dir)
    print(f'Found {len(img_list)} images in {target_img_dir}')
    for img_path in tqdm(img_list):
        img = Image.open(img_path).convert('RGB')
        img = center_crop_arr(img, IMAGE_SIZE)
        #img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_name = os.path.basename(img_path)
        #change suffix to png
        img_name = img_name.split('.')[0] + '.png'
        save_path = os.path.join(save_dir, img_name)
        img.save(save_path)
def process_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = center_crop_arr(img, IMAGE_SIZE)
    img_name = os.path.basename(img_path)
    #change suffix to png
    img_name = img_name.split('.')[0] + '.png'
    save_path = os.path.join(save_dir, img_name)
    img.save(save_path)
def multi_process_main():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_list = get_all_image_under_dir(target_img_dir)
    print(f'Found {len(img_list)} images in {target_img_dir}')
    with ProcessPoolExecutor() as executor:
        for _ in tqdm(executor.map(process_img, img_list), total=len(img_list)):  
            pass
if __name__ == '__main__':
    multi_process_main()