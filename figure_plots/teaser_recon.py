import numpy as np
from PIL import Image
from PIL import ImageDraw
from pyparsing import C
from utils import *
import torch
#images = [('./visuals/text.png', './visuals/text_vae.png', './visuals/text_mae.png'),
#            ('./visuals/crowd.png', './visuals/crowd_vae.png', './visuals/crowd_mae.png')]
images = [('./visuals/text2.png', './visuals/text2_vae.png', './visuals/text2_mae.png'),
            ('./visuals/crowd2.png', './visuals/crowd2_vae.png', './visuals/crowd2_mae.png'),
            ('./visuals/city.png', './visuals/city_vae.png', './visuals/city_mae.png')]
names = ['text2', 'crowd2', 'city']
crops_size = (50,50)
st = [(107,64),(190,140),(28,50)]
reshape_ratio = 2.
def get_crop(image, st, size):
    return image[st[0]:st[0]+size[0], st[1]:st[1]+size[1]]
def tag_image(image,st, size):
    # use a red rectangle to tag the crop
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    c = colors[2]
    draw.rectangle([st[1], st[0], st[1]+size[1], st[0]+size[0]], outline=c, width=2)
    # also tag the position used in paste_crop
    draw.rectangle([0, image.size[1]-size[0]*reshape_ratio, size[1]*reshape_ratio, image.size[1]], outline=c, width=2)
    return image

def paste_crop(image, crop):
    # paste the crop in the left-lower corner
    # resize crop by 2x
    crop = Image.fromarray(crop)
    crop = crop.resize((crop.size[0]*2, crop.size[1]*2), Image.BICUBIC)
    crop = np.array(crop)
    image[-crop.shape[0]:,:crop.shape[1]] = crop
    return image
def load_and_return_image_crops():
    global images, st, crops_size
    image_and_crops = []
    for k, group in enumerate(images):
        gt, vae, mae = group
        gt = Image.open(gt)
        vae = Image.open(vae)
        mae = Image.open(mae)
        if 'fence' in names[k]:
            # flip the image horizontally
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            vae = vae.transpose(Image.FLIP_LEFT_RIGHT)  
            mae = mae.transpose(Image.FLIP_LEFT_RIGHT)
        s, size = st[k], crops_size
        gt = tag_image(gt, s, size)
        vae = tag_image(vae, s, size)
        mae = tag_image(mae, s, size)
        #convert to numpy
        gt = np.array(gt)
        vae = np.array(vae)
        mae = np.array(mae)
        print(gt.shape, vae.shape, mae.shape)
        # do a crop at [st[k][0]:st[k][0]+crops_size[0], st[k][1]:st[k][1]+crops_size[1]]
        gt_crop = get_crop(gt, s, size)
        vae_crop = get_crop(vae, s, size)
        mae_crop = get_crop(mae, s, size)
        print(gt_crop.shape, vae_crop.shape, mae_crop.shape)
        # put the crop in left-lower corner
        gt = paste_crop(gt, gt_crop)
        vae = paste_crop(vae, vae_crop)
        mae = paste_crop(mae, mae_crop)
        # convert back to PIL
        gt = Image.fromarray(gt)
        vae = Image.fromarray(vae)
        mae = Image.fromarray(mae)
        image_and_crops.append((gt, vae, mae))
    return image_and_crops
svae_path = './visuals/teaser'
import os
os.makedirs(svae_path, exist_ok=True)
def main():
    image_and_crops = load_and_return_image_crops()
    for i, group in enumerate(image_and_crops):
        gt, vae, mae = group
        name = names[i]
        gt.save(f'{svae_path}/gt_{name}.png')
        vae.save(f'{svae_path}/vae_{name}.png')
        mae.save(f'{svae_path}/mae_{name}.png')
    print('saved to:', svae_path)
if __name__ == "__main__":
    main()