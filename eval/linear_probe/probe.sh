export PJRG_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_linear_probe'
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 linear_probe.py \
    --accum_iter 4 \
    --batch_size 128 \
    --model_config $1 \
    --epochs 90 \
    --blr .1 \
    --weight_decay 0.0 \
    --cls_token \
    --dtype float32 \
    --image_size 224 \
    --num_workers 16 \
    --dist_eval --data_path /home/bytetriper/VAE-enhanced/data/imagenet