from PIL import Image
import torch
from torchvision.transforms import ToTensor
def get_default_image(im_size:tuple) -> Image.Image:
    #image_path = '/home/bytetriper/VAE-enhanced/test.png'
    image_path = '/home/bytetriper/VAE-enhanced/visuals/test_imagenet_orig.png'

    if len(im_size) >= 3:
        image = torch.randn(2, *im_size) # B, C, H, W
        # normalize to unit-norm
        image = image / image.norm(dim=1, keepdim=True)
    else:
        image = Image.open(image_path).resize(im_size).convert('RGB')
        #repeat 2 times to asssure model works with batch size > 1
        image = ToTensor()(image).unsqueeze(0).repeat(2,1,1,1)
    return image

