#!/bin/bash

# Hardcoded defaults for debugging
EXP_NAME="tokenizer"
CKPT_DIR="./ckpt_gcs"
MODEL_CONFIG="configs/imagenet256/stage1/klvae/f_16.yaml"
WORLD_SIZE=4
EXP="exp"
LOAD_CKPT=""
WANDB_ID=""
# Set up save directory
SAVE_DIR="${CKPT_DIR}/${EXP_NAME}"
mkdir -p $SAVE_DIR
echo "Save dir: $SAVE_DIR"
# Environment setup
export PJRT_DEVICE=TPU
export XLA_DISABLE_FUNCTIONALIZATION=1
export WANDB_DIR=$SAVE_DIR
export WANDB_PROJECT=$EXP_NAME
# Show debug info
echo "Running with:"
echo "Model config: $MODEL_CONFIG"
echo "World size: $WORLD_SIZE"
echo "Experiment: $EXP"
env | grep PJRT
env | grep DEBUG
# Run training
python main_stage1.py \
    -m=$MODEL_CONFIG \
    -r=$SAVE_DIR \
    --world_size=$WORLD_SIZE \
    --exp=$EXP \
    --do_online_eval
# To add checkpoint loading, uncomment and modify this line:
# python main_stage1.py -m=$MODEL_CONFIG -r=$SAVE_DIR --world_size=$WORLD_SIZE --exp=$EXP --do_online_eval -l=/path/to/checkpoint --resume