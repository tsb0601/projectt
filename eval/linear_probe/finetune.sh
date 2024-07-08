export PJRG_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_finetune'
ckpt_path='../../ckpt/eval_finetune'
save_path=$2
time=$(date "+%Y%m%d-%H%M%S")
save_path=${ckpt_path}/${save_path}/${time}
log_path=${save_path}/log/
mkdir -p $log_path
python finetune.py \
    --accum_iter 16 \
    --batch_size 16 \
    --model_config $1 \
    --epochs 100 \
    --global_pool \
    --dtype float32 \
    --image_size 224 \
    --output_dir $save_path \
    --num_workers 16 \
    --log_dir $log_path \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --world_size 4 \
    --dist_eval  --data_path /home/bytetriper/VAE-enhanced/data/imagenet