SAVE_DIR=$1
#assert save_dir is not empty
if [ -z "$SAVE_DIR" ]
then
    echo "Save dir is empty"
    exit 1
fi
mkdir -p $SAVE_DIR
echo "Save dir: $SAVE_DIR"
#export DEBUG=1
export PJRT_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_ft_test'
#echo "Running stage1 training"
#echo "setting env vars"
env | grep PJRT
env | grep DEBUG
load_path=$2
model_config=$3
world_size=$4
python main_stage1.py \
    --action gen \
    -m=$model_config \
    -r=$SAVE_DIR \
    -l=$load_path \
    --world_size=$world_size \
    --reload-batch-size 125 # default 125 for imagenet val