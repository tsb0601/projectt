import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
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

def main():
    npz_path = '/home/bytetriper/VAE-enhanced/eval/fid/VIRTUAL_imagenet256_labeled.npz'
    npz = np.load(npz_path)
    print(npz['arr_0'].shape)
    pil_image = Image.fromarray(npz['arr_0'][0])
    image_size = 256 #pil_image is 256x256
    cropped_image = center_crop_arr(pil_image, image_size)
    os.makedirs('./visuals/center_crop_arr', exist_ok=True)
    pil_image.save('./visuals/center_crop_arr/original.png')
    cropped_image.save('./visuals/center_crop_arr/cropped.png')

if __name__ == '__main__':
    main()