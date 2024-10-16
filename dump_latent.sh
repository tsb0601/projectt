config_path=$1
output_path=$2

python dump_latent.py \
    --config_path $config_path \
    --output_dir $output_path \
    --is_ddp