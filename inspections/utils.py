from rqvae.img_datasets.interfaces import LabeledImageData
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
def get_example_data(im_size:int = 256, bsz:int = 1)->LabeledImageData:
    image_path = 'visuals/dit_gen.png'
    image = Image.open(image_path).resize((im_size, im_size)).convert('RGB')
    #repeat 2 times to asssure model works with batch size > 1
    image = ToTensor()(image).unsqueeze(0).repeat(bsz,1,1,1)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    return data
