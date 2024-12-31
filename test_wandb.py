import wandb
from torch.utils.tensorboard import SummaryWriter
import os
# check if wandb is logged in
if wandb.api.api_key is None:
    print('Wandb is not logged in')
    exit()
# set run name to 'wanb-test'
wandb.init(project='VAE-enhanced', sync_tensorboard=True, dir='../test', name='wandb-test')
dir1 = os.path.join('../tmp/train')
dir2 = os.path.join('../tmp/valid')
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)
writer1 = SummaryWriter(dir1)
writer2 = SummaryWriter(dir2)
for i in range(10):
    writer1.add_scalar('train/loss', i, i)
    writer2.add_scalar('valid/loss', i, i)
# test images
import numpy as np
from PIL import Image
img = Image.open('test.png').resize((256, 312))
img = np.array(img)
img = np.transpose(img, (2, 0, 1))
writer1.add_image('train/image', img, 0)
writer2.add_image('valid/image', img, 0)
wandb.finish()