SAVE_DIR=$1
#assert save_dir is not empty
if [ -z "$SAVE_DIR" ]
then
    echo "Save dir is empty"
    exit 1
fi
mkdir -p $SAVE_DIR
echo "Save dir: $SAVE_DIR"
export PJRT_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_ft_test'
env | grep PJRT
env | grep DEBUG
model_config=$2
world_size=$3
python main_stage1.py \
    -m=$model_config \
    -r=$SAVE_DIR \
    --world_size=$world_size \
    --use_autocast \
    --cache_latent 