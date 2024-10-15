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
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/stage2_DiT'
env | grep PJRT
env | grep DEBUG
model_config=$3
world_size=$4
EXP=$5
load_ckpt=$6
# if load_ckpt != '', add --resume to the command
if [ -z "$load_ckpt" ]
then
    echo "load_ckpt is empty"
else
    # append --resume to load_ckpt to $load_ckpt --resume
    load_ckpt="${load_ckpt} --resume"
    echo "load from: $load_ckpt"
fi
wandb_id=$7
if [ -z "$wandb_id" ]
then
    echo "wandb_id is empty"
else
    echo "wanb_id: $wandb_id"
    export WANDB_ID=$wandb_id
fi
export WANDB_DIR=$SAVE_DIR
export WANDB_PROJECT=$EXP_NAME
#env | grep WANDB
python main_stage2.py \
    -m=$model_config \
    -r=$SAVE_DIR \
    --world_size=$world_size \
    -l=$load_ckpt \
    --exp=$EXP \
    --do_online_eval
    #--use_autocast \