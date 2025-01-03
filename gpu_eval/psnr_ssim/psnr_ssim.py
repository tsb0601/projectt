from multiprocessing.context import assert_spawning
import os
import abc
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tqdm import tqdm
from utils import calculate_psnr, calculate_ssim, calculate_ssim_pt
import sys
import torch

class BaseMetric(abc.ABC):
    """
    Base Class for evaluation metric
    """
    def __init__(self, dir_path1, dir_path2):
        self.dir_path1 = dir_path1
        self.dir_path2 = dir_path2
        if isinstance(dir_path1, np.ndarray):
            assert isinstance(dir_path2, np.ndarray), 'Both inputs should be numpy arrays'
            assert dir_path1.shape[0] == dir_path2.shape[0], 'The number of images should be the same'
    @abc.abstractmethod
    def evaluate(self, *args, **kwargs)->float:
        pass
    def _batch_evaluate_npz(self):
        #assert isinstance(self.dir_path1, np.ndarray), 'The input should be a numpy array'
        
        sum_loss = 0
        for i in tqdm(range(self.dir_path1.shape[0])):
            sum_loss += self.evaluate(self.dir_path1[i], self.dir_path2[i])
        return sum_loss/self.dir_path1.shape[0], self.dir_path1.shape[0], 0
    def batch_evaluate(self):
        """
        evaluate all images in the directory
        """
        if isinstance(self.dir_path1, np.ndarray):
            return self._batch_evaluate_npz()
        std_files = self.dir_path1
        #cmp_files = os.listdir(self.dir_path2)
        sum_loss = 0
        valid_cnt = 0
        invalid_cnt = 0
        tbar = tqdm(std_files)
        for std_img in tbar:
            if std_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                std_img_path = os.path.join(self.dir_path1, std_img)
                #change the suffix to png
                cmp_img_path = os.path.join(self.dir_path2, std_img)
                #see if the image is in the comparison directory
                if not os.path.isfile(cmp_img_path):
                    if std_img.startswith('0_'):
                        cmp_img_path = os.path.join(self.dir_path2, std_img[2:])
                    else:
                        cmp_img_path = os.path.join(self.dir_path2, '0_'+std_img) # another possible name
                if not os.path.isfile(cmp_img_path):
                    invalid_cnt += 1
                    print(f'{std_img} not found in comparison directory')
                    continue
                try:
                    std_img = Image.open(std_img_path)
                    cmp_img = Image.open(cmp_img_path)
                except Exception as e:
                    if isinstance(e,PIL.UnidentifiedImageError):
                        print('Identified an invalid image {}/{}'.format(std_img_path, cmp_img_path))
                        invalid_cnt +=1
                        continue
                    else:
                        raise e
                std_size = std_img.size
                cmp_size = cmp_img.size
                #check is square
                if std_size[0] != std_size[1]:
                    shorter_side = min(std_size[0], std_size[1])
                    std_img = std_img.crop((0, 0, shorter_side, shorter_side))
                    print(f'{std_img_path} is not square, crop to {shorter_side}x{shorter_side}')
                    #invalid_cnt += 1
                    #continue
                if cmp_size[0] != cmp_size[1]:
                    shorter_side = min(cmp_size[0], cmp_size[1])
                    cmp_img = cmp_img.crop((0, 0, shorter_side, shorter_side))
                    print(f'{cmp_img_path} is not square, crop to {shorter_side}x{shorter_side}')
                    #invalid_cnt += 1
                    #continue, 
                #resize to smaller size
                smaller_size = min(std_size[0], cmp_size[0])
                std_img = std_img.resize((smaller_size, smaller_size))
                cmp_img = cmp_img.resize((smaller_size, smaller_size))
                sum_loss = sum_loss + self.evaluate(std_img, cmp_img)
                valid_cnt += 1
        if valid_cnt == 0:
            return float('inf'), valid_cnt, invalid_cnt
        return sum_loss/valid_cnt, valid_cnt, invalid_cnt
    def run(self):
        avg_loss, valid_cnt, invalid_cnt = self.batch_evaluate()
        #print it in a table format
        """
        -------------------------------------
        |  Average Loss  |  Valid Count  |  Invalid Count  |
        -------------------------------------
        |      0.123     |      123      |       123       |
        """
        print('-------------------------------------------------')
        print(f'|  Average Loss  |  Valid Count  |  Invalid Count  |')
        print('-------------------------------------------------')
        #control the length of the float number
        print(f'|      {avg_loss:.3f}     |      {valid_cnt}      |       {invalid_cnt}       |')
import PIL
class PSNR(BaseMetric):
    def evaluate(self, std_img, cmp_img):
        std_img = np.array(std_img)
        cmp_img = np.array(cmp_img)
        assert std_img.shape == cmp_img.shape, (f'Image shapes are differnet: {std_img.shape}, {cmp_img.shape}.')
        psnr =  calculate_psnr(std_img, cmp_img, crop_border=4, input_order='HWC')
        if psnr == float('inf'):
            psnr = 100
        return psnr
    
class SSIM(BaseMetric):
    def evaluate(self, std_img, cmp_img):
        std_img = np.array(std_img)
        cmp_img = np.array(cmp_img)
        assert std_img.shape == cmp_img.shape, (f'Image shapes are differnet: {std_img.shape}, {cmp_img.shape}.')
        return calculate_ssim(std_img, cmp_img, crop_border=4, input_order='HWC')
def extract_from_path(path, suffix: tuple[str] = ('.png', '.jpg', '.jpeg')):
    if os.path.isdir(path):
        subdirs = [os.path.join(path, subdir) for subdir in os.listdir(path) if subdir.lower().endswith(suffix)]
        return subdirs
    elif os.path.isfile(path):
        if path.lower().endswith(suffix):
            return [path]
        elif path.lower().endswith('.npz'):
            file = np.load(path)['arr_0'] # num x C x H x W
            print(f'Loaded {file.shape[0]} images from {path}')
            return file
    raise ValueError(f'Invalid path: {path}')
import sys
def main():
    std_path = sys.argv[1]
    cmp_path = sys.argv[2]
    mode = sys.argv[3]
    assert os.path.exists(std_path), f'{std_path} is not a directory or does not exist'
    assert os.path.exists(cmp_path), f'{cmp_path} is not a directory or does not exist'
    std_images, cmp_images = extract_from_path(std_path), extract_from_path(cmp_path)
    if mode == 'psnr':
        metric = PSNR(std_images, cmp_images)
    elif mode == 'ssim':
        metric = SSIM(std_images, cmp_images)
    else:
        print(f'Invalid mode: {mode}')
        return
    metric.run()
if __name__ == '__main__':
    main()