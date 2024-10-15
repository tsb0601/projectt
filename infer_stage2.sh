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
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/stage2_DiT_infer'
#echo "Running stage1 training"
#echo "setting env vars"
env | grep PJRT
env | grep DEBUG
load_path=$2
model_config=$3
world_size=$4
mkdir -p $SAVE_DIR
export XLA_DISABLE_FUNCTIONALIZATION=1
python main_stage2.py \
    --action gen \
    -m=$model_config \
    -r=$SAVE_DIR \
    -l=$load_path \
    --world_size=$world_size  \
    --use_autocast \
    --reload-batch-size 32 
    #--do_online_eval