export PJRG_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_linear_probe'
ckpt_path='../../ckpt/linear_probe'
save_path=$2
time=$(date "+%Y%m%d-%H%M%S")
save_path=${ckpt_path}/${save_path}/${time}
log_path=${ckpt_path}/${save_path}/${time}/log/
mkdir -p $log_path
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 linear_probe.py \
    --accum_iter 8 \
    --batch_size 128 \
    --model_config $1 \
    --epochs 90 \
    --warmup_epochs 10 \
    --blr 1e-4 \
    --weight_decay 0.0 \
    --cls_token \
    --dtype float32 \
    --image_size 224 \
    --output_dir $save_path \
    --log_dir $log_path \
    --num_workers 16 \
    --dist_eval --data_path /home/bytetriper/VAE-enhanced/data/imagenet