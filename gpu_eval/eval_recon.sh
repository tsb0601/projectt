#!/bin/bash
file_name=$1
cd ~ # change to home directory
# download first
./download_file.sh /mnt/disks/storage/VAE-enhanced/ckpt/$file_name ./datas/$file_name
# run the evaluation
cd VAE-enhanced/gpu_eval/fid
python evaluator.py /mnt/data/boyang/datas/datas/val_256.npz /mnt/data/boyang/datas/datas/$file_name 2> /dev/null

cd ../psnr_ssim # run the evaluation

python psnr_ssim.py ~/ImageNets/val_256.npz ~/datas/$file_name psnr
python psnr_ssim.py ~/ImageNets/val_256.npz ~/datas/$file_name ssim