SAVE_DIR=$1
#assert save_dir is not empty
if [ -z "$SAVE_DIR" ]
then
    echo "Save dir is empty"
    exit 1
fi
mkdir -p $SAVE_DIR
echo "Save dir: $SAVE_DIR"
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 main_stage1.py \
    -m=configs/imagenet256/stage1/in256-rqvae-8x8x4.yaml \
    -r=$SAVE_DIR