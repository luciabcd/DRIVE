# Step2 : Run second round of linear probe after lpft + ours
set -e

# Define the list of datasets
datasets=('cifar10' 'cifar100' 'flowers102' 'caltech101' 'stanford_dogs' 'stanford_cars')


# Define the corresponding learning rates for each dataset
learning_rates=(0.005 0.005 0.05 0.01 0.005 0.05)

# Define other fixed parameters
MODEL="resnet18"
CHECKPOINT_BASE_DIR="./checkpoints/ours/${MODEL}"

# Create the base checkpoint directory if it doesn't exist
#mkdir -p "$CHECKPOINT_BASE_DIR"

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
    python FLOW_taskspecific_ft.py \
        --dataset "$dataset" \
        --model "$MODEL" \
        --lr "$lr" \
        --checkpoint-dir "$CHECKPOINT_BASE_DIR"

    echo "Completed training for dataset: $dataset"
    echo ""
done

echo "Training completed for all datasets."