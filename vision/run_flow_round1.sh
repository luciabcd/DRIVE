#!/bin/bash
# Step 1: Run our best run script, i.e., ours_bestrun.py
# Exit immediately if a command exits with a non-zero status
set -e

# Define the list of datasets
datasets=('cifar10' 'cifar100' 'flowers102' 'caltech101' 'stanford_dogs' 'stanford_cars')

# Define the corresponding temperature values for each dataset
temp=(0.1041 0.6626 0.0181 0.0543 0.2123 0.3832)

# Define the corresponding learning rates for each dataset
learning_rates=(0.001 0.005 0.05 0.01 0.005 0.01)

# Validate that all arrays have the same length
if [[ ${#datasets[@]} -ne ${#temp[@]} || ${#datasets[@]} -ne ${#learning_rates[@]} ]]; then
    echo "Error: The number of datasets, temp values, and learning rates must be the same."
    exit 1
fi

# Define other fixed parameters
MODEL="resnet18"
BASE_CHECKPOINT_DIR="./checkpoints/ours/${MODEL}"


# Get the number of datasets
num_datasets=${#datasets[@]}

# Iterate over each dataset using its index to access corresponding temp and learning rate
for (( i=0; i<num_datasets; i++ ))
do
    dataset="${datasets[i]}"
    lr="${learning_rates[i]}"
    temp_val="${temp[i]}"

    echo "========================================="
    echo "Starting training for dataset: $dataset"
    echo "Corresponding learning rate: $lr"
    echo "Temperature value: $temp_val"
    echo "========================================="

    # Define the checkpoint directory for the current dataset
    CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${dataset}"
    mkdir -p "$CHECKPOINT_DIR"

    # Run the Python script with the current dataset, learning rate, and temp
    python FLOW_ft.py \
        --dataset "$dataset" \
        --model "$MODEL" \
        --lr "$lr" \
        --temp "$temp_val" \
        --checkpoint-dir "$BASE_CHECKPOINT_DIR"

    echo "Completed training for dataset: $dataset"
    echo ""
done

echo "Training completed for all datasets."