#!/bin/bash
# Ours best finetuning using Resnet18
set -e

# Define the list of datasets
datasets=('cifar10' 'cifar100' 'flowers102' 'caltech101' 'stanford_dogs' 'stanford_cars')

# Define the corresponding learning rates for each dataset
learning_rates=(0.005 0.01 0.05 0.005 0.001 0.05)

# Define other fixed parameters
MODEL="resnet18"
CHECKPOINT_BASE_DIR="./checkpoints/standard_ft/${MODEL}"
LOG_BASE_DIR="./logs/standard_ft/${MODEL}"

# Get the number of datasets
num_datasets=${#datasets[@]}

# Iterate over each dataset using its index to access corresponding temp and learning rate
for (( i=0; i<num_datasets; i++ ))
do
    dataset="${datasets[i]}"
    lr="${learning_rates[i]}"

    echo "========================================="
    echo "Starting training for dataset: $dataset"
    echo "Corresponding learning rate: $lr"
    echo "========================================="

    # Run the Python script with the current dataset, learning rate, and temp
    python standard_ft.py \
        --dataset "$dataset" \
        --model "$MODEL" \
        --lr "$lr" \
        --checkpoint-dir "$CHECKPOINT_BASE_DIR" \
        --log-dir "$LOG_BASE_DIR"

    echo "Completed training for dataset: $dataset"
    echo ""
done

echo "Training completed for all datasets."