export PJRG_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_linear_probe'
save_path=$2
time=$(date "+%Y%m%d-%H%M%S")
save_path=${save_path}/${time}
log_path=${save_path}/log/
mkdir -p ${log_path}
cp $0 ${log_path}/script.sh
cp $1 ${save_path}/model_config.yaml
cp linear_probe.py ${save_path}/
image_size=$3
bsz=$4
acc_iter=$5
world_size=$6
#torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 
python linear_probe.py \
    --accum_iter $acc_iter \
    --batch_size $bsz \
    --model_config $1 \
    --epochs 90 \
    --warmup_epochs 10 \
    --blr 3e-4 \
    --weight_decay 0.0 \
    --cls_token \
    --dtype float32 \
    --image_size $image_size \
    --output_dir $save_path \
    --log_dir $log_path \
    --num_workers 16 \
    --world_size $world_size \
    --dist_eval --data_path /home/bytetriper/VAE-enhanced/data/imagenet