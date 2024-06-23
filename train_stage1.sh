SAVE_DIR=$1
#assert save_dir is not empty
if [ -z "$SAVE_DIR" ]
then
    echo "Save dir is empty"
    exit 1
fi
mkdir -p $SAVE_DIR
echo "Save dir: $SAVE_DIR"
export DEBUG=1
export PJRT_DEVICE=TPU
export XLACACHE_PATH='/home/bytetriper/.cache/xla_compile/MAE_256_ft_test'
echo "Running stage1 training"
echo "setting env vars"
env | grep PJRT
env | grep DEBUG
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 main_stage1.py \
    -m=configs/imagenet256/stage1/MAE.yaml \
    -r=$SAVE_DIR