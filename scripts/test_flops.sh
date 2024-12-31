# !/bin/bash
#define a tuple ranging from 16 32... 1024
seqlen=(16 32 64 128 256 512 1024)
# run the script for seq in seqlen
# python sanity_check/stat_check_stage2.py  /home/bytetriper/VAE-enhanced/configs/imagenet256/stage2/pixel/length/seq.yaml seq, seq  

for seq in ${seqlen[@]}; do
    echo "Running for seq: $seq"
    python sanity_check/stat_check_stage2.py  /home/bytetriper/VAE-enhanced/configs/imagenet256/stage2/pixel/length/$seq.yaml $seq,$seq
done