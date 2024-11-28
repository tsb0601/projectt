export PJRG_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_linear_probe'
save_path=$2
time=$(date "+%Y%m%d-%H%M%S")
save_path=${save_path}/${time}
actual_log_path=${save_path}/log/
log_path='/home/bytetriper/tmp/'${time}
mkdir -p ${log_path}
mkdir -p ${save_path}
cp $0 ${log_path}/script.sh
cp $1 ${log_path}/model_config.yaml
cp linear_probe.py ${log_path}/
image_size=$3
bsz=$4
acc_iter=$5
hidden_dim=$6
world_size=$7
export WANDB_DIR=${log_path}
export WANDB_PROJECT='linear_probe'
#torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 
python linear_probe.py \
    --accum_iter $acc_iter \
    --batch_size $bsz \
    --model_config $1 \
    --epochs 90 \
    --warmup_epochs 10 \
    --blr .1 \
    --save_freq 90 \
    --weight_decay 0.0 \
    --cls_token \
    --dtype bfloat16 \
    --image_size $image_size \
    --output_dir $save_path \
    --log_dir $log_path \
    --num_workers 4 \
    --hidden_size $hidden_dim \
    --world_size $world_size \
    --dist_eval \
    --data_path /home/bytetriper/VAE-enhanced/data/imagenet

cp -r ${log_path} ${actual_log_path}