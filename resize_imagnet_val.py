import os
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
IMAGE_SIZE = 256

def get_all_image_under_dir(dir:str, img_ext: tuple = ('.png', '.jpg', '.jpeg')):
    """
    Get all images under the directory
    """
    img_list = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(img_ext):
                img_list.append(os.path.join(root, file))
    return img_list

def main():
    target_img_dir = '/mnt/disks/storage/datasets/ImageNet/val'
    save_dir = '/mnt/disks/storage/datasets/ImageNet/val_256'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_list = get_all_image_under_dir(target_img_dir)
    print(f'Found {len(img_list)} images in {target_img_dir}')
    for img_path in tqdm(img_list):
        img = Image.open(img_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, img_name)
        img.save(save_path)
        
if __name__ == '__main__':
    main()