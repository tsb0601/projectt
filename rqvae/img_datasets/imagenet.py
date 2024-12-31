from torchvision.datasets import ImageNet,ImageFolder
from .interfaces import LabeledImageData
class ImageNet_wImagepath(ImageNet):
    """
    a custom dataset class that returns image path along with the image
    """
    def __getitem__(self, index):
        img, target = super(ImageNet_wImagepath, self).__getitem__(index)
        path, _ = self.samples[index]
        return LabeledImageData(img=img, img_path=path, condition=target)

class ImageNet_Fake(ImageNet):
    """
    a custom dataset class that returns the Image and a index as img_path, all targets are set to 1000 (dropout)
    """
    def __getitem__(self, index):
        img, target = super(ImageNet_Fake, self).__getitem__(index)
        target = 1000 # unconditional
        return LabeledImageData(img=img, condition=target, img_path=f'{index}.png')