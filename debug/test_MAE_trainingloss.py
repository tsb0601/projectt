from transformers import ViTMAEForPreTraining, ViTImageProcessor, ViTMAEModel
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
ckpt_path = '../model_zoo/mae_base_256_ft'
model:ViTMAEForPreTraining = ViTMAEForPreTraining.from_pretrained(ckpt_path)
model.eval()
processor = ViTImageProcessor.from_pretrained(ckpt_path)
patch_num = (model.config.image_size // model.config.patch_size) ** 2
image_path = '/home/bytetriper/VAE-enhanced/test.png'
image_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
image_std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
print(f'Stage1MAE model loaded with mean {image_mean} and std {image_std}, mask_ratio {model.config.mask_ratio}')
image = Image.open(image_path).resize((model.config.image_size, model.config.image_size)).convert('RGB')
image = processor(image, return_tensors='pt')['pixel_values']
#repeat 20 times
image = image.repeat(32, 1, 1, 1)
#image = ToTensor()(image).unsqueeze(0)
print(image.shape, image.min(), image.max())
#image = (image - image_mean) / image_std
#image = (image * 2) - 1.
#noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
print(image.shape, image.min(), image.max())
outputs = model(image)
print(outputs.loss)
logits = outputs.logits
xs_recon = model.unpatchify(logits)
xs_recon = xs_recon * image_std + image_mean
print(xs_recon.shape, xs_recon.min(), xs_recon.max())
print((xs_recon - image).abs().mean())
recon_image = ToPILImage()(xs_recon[0].clamp(0., 1.))
recon_image.save('./recon.png')