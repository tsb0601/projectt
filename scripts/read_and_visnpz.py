import numpy as np
import torch
import os
import sys
from PIL import Image
npz_path = sys.argv[1]

assert os.path.isfile(npz_path), f'Invalid npz path {npz_path}'

npz = np.load(npz_path)
image = npz['image']
print(image.shape, image.dtype, image.min(), image.max())
image = (image + 1) / 2
image = image.squeeze()
image = Image.fromarray((image * 255).astype(np.uint8))
image.save('./visuals/npz_image.png')
print('Image saved to ./visuals/npz_image.png')