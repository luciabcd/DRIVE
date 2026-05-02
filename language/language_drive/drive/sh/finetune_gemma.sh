#!/bin/bash

export WANDB_MODE=offline
PROJECT_ROOT="/data1/.." 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
MASTER_PORT=29601


MODEL_PATH=".."
DATA_PATH=".."
CACHE_DIR="$PROJECT_ROOT/f.." 
BASE_OUTPUT_DIR="$PROJECT_ROOT/.."


BS=2
GRAD_ACC=32
LR=2e-5
EPOCHS=3
MAX_SEQ_LEN=1024


GAMMA=1
TAU=0.4587



set -e


if [ ! -f "$CACHE_DIR/ref_losses.npy" ] || \
   [ ! -f "$CACHE_DIR/ref_features.npy" ] || \
   [ ! -f "$CACHE_DIR/sigma_inv.pt" ]; then
    exit 1
fi

echo "$PROJECT_ROOT"
cd "$PROJECT_ROOT" || exit

mkdir -p "$BASE_OUTPUT_DIR"




echo "========================================================="

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT drive.py \
    --model "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --lambda_schedule "linear_decay" \
    --drive_gamma "$GAMMA" \
    --drive_tau "$TAU" \
    --batch_size "$BS" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --max_seq_length "$MAX_SEQ_LEN" \
    --finetune_type "full" \
    --project_name "drive-dynamic-decay"

MASTER_PORT=$(($MASTER_PORT + 1))



torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT drive.py \
    --model "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --lambda_schedule "linear_increase" \
    --drive_gamma "$GAMMA" \
    --drive_tau "$TAU" \
    --batch_size "$BS" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --max_seq_length "$MAX_SEQ_LEN" \
    --finetune_type "full" \
    --project_name "drive-dynamic-increase"


MASTER_PORT=$(($MASTER_PORT + 1))


