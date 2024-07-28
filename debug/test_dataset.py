import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.img_datasets.interfaces import LabeledImageData
from rqvae.img_datasets.imagenet import ImageNet_wImagepath, ImageNet_Fake
from PIL import Image
def test_imagenet():
    data_dir = '/home/bytetriper/VAE-enhanced/data/imagenet'
    dataset = ImageNet_wImagepath(data_dir, split='val')
    print(len(dataset))
    data:LabeledImageData = dataset[0]
    print(data.img)
    print(data.img_path)
    img = Image.open(data.img_path).resize((256, 256))
    data.img.save('./visuals/test_imagenet.png')
    img.save('./visuals/test_imagenet_orig.png')
    print(data.condition)

if __name__ == '__main__':
    test_imagenet()