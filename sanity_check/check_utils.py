from PIL import Image
import torch
from torchvision.transforms import ToTensor
def get_default_image(im_size:tuple, single_image: bool = False) -> Image.Image:
    image_path = '/home/bytetriper/VAE-enhanced/test.png'
    #image_path = '/home/bytetriper/VAE-enhanced/visuals/test_imagenet_orig.png'
    #image_path = '/home/bytetriper/VAE-enhanced/visuals/bear/bear.png'
    bsz = 1 if single_image else 2
    if len(im_size) >= 3:
        image = torch.randn(bsz, *im_size) # B, C, H, W
        # normalize to unit-norm
        image = image / image.norm(dim=1, keepdim=True)
    else:
        image = Image.open(image_path).resize(im_size).convert('RGB')
        #repeat 2 times to asssure model works with batch size > 1
        image = ToTensor()(image).unsqueeze(0).repeat(bsz,1,1,1)
    image = torch.ones_like(image) 
    return image

