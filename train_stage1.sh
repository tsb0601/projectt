EXP_NAME=$1
CKPT_DIR=$2 # ./ckpt_gcs
#assert save_dir is not empty
SAVE_DIR=${CKPT_DIR}/${EXP_NAME}
if [ -z "$SAVE_DIR" ]
then
    echo "Save dir is empty"
    exit 1
fi
mkdir -p $SAVE_DIR
echo "Save dir: $SAVE_DIR"
export PJRT_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_ft'
env | grep PJRT
env | grep DEBUG
model_config=$3
world_size=$4
export WANDB_DIR=$SAVE_DIR
export WANDB_PROJECT=$EXP_NAME
env | grep WANDB
python main_stage1.py \
    -m=$model_config \
    -r=$SAVE_DIR \
    --world_size=$world_size