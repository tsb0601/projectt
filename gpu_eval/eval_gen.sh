#!/bin/bash
file_name=$1
cd ~ # change to home directory
# download first
# get the base directory from file_name
base_dir=$(dirname $file_name)
mkdir -p ./datas/$base_dir
./download_file.sh /mnt/disks/storage/VAE-enhanced/ckpt/$file_name ./datas/$file_name
# run the evaluation
cd VAE-enhanced/gpu_eval/fid
python evaluator.py VIRTUAL_imagenet256_labeled.npz /mnt/data/boyang/datas/datas/$file_name 2>/dev/null           