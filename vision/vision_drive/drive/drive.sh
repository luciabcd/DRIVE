
set -e

datasets=('cifar10' 'cifar100' 'flowers102')
temp=(0.1041 0.6626 0.0181)
learning_rates=(0.001 0.005 0.05)
TARGET_MODES=(1)
LAMBDA_WEIGHTS=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

MODEL="resnet50"
BASE_CHECKPOINT_ROOT=".../${MODEL}"

if [[ ${#datasets[@]} -ne ${#temp[@]} || ${#datasets[@]} -ne ${#learning_rates[@]} ]]; then
    echo "Error: The number of datasets, temp values, and learning rates must be the same."
    exit 1
fi

num_datasets=${#datasets[@]}

for (( i=0; i<num_datasets; i++ ))
do
    dataset="${datasets[i]}"
    lr="${learning_rates[i]}"
    temp_val="${temp[i]}"

    echo "========================================="
    echo "Processing dataset: $dataset"
    echo "LR: $lr | Temp: $temp_val"
    echo "========================================="

    for mode in "${TARGET_MODES[@]}"
    do
        echo ">>> Running Mode: $mode"
        
        if [ "$mode" -eq 2 ]; then
            echo "    (Mode 2 detected: Iterating over fixed lambda weights)"
            
            for lambda_weight in "${LAMBDA_WEIGHTS[@]}"
            do
                CHECKPOINT_DIR="${BASE_CHECKPOINT_ROOT}/${dataset}/fixed_lambda_${lambda_weight}"
                mkdir -p "$CHECKPOINT_DIR"

                echo "    -> Running Fixed Lambda: $lambda_weight"
                python drive.py \
                    --dataset "$dataset" \
                    --model "$MODEL" \
                    --lr "$lr" \
                    --temp "$temp_val" \
                    --lambda-mode "$mode" \
                    --_lambda "$lambda_weight" \
                    --checkpoint-dir "$CHECKPOINT_DIR" \
                    --log-dir "./logs/drive/${MODEL}/${dataset}/fixed"
            done
            
        else
            if [ "$mode" -eq 0 ]; then
                mode_name="dynamic_0to1"
            else
                mode_name="dynamic_1to0"
            fi
            
            CHECKPOINT_DIR="${BASE_CHECKPOINT_ROOT}/${dataset}/${mode_name}"
            mkdir -p "$CHECKPOINT_DIR"
            
            echo "    -> Running Dynamic Mode: $mode_name"
            python drive.py \
                --dataset "$dataset" \
                --model "$MODEL" \
                --lr "$lr" \
                --temp "$temp_val" \
                --lambda-mode "$mode" \
                --_lambda 0.5 \
                --checkpoint-dir "$CHECKPOINT_DIR" \
                --log-dir "./logs/drive/${MODEL}/${dataset}/${mode_name}"
        fi
        
    done
    
    echo "Completed all modes for dataset: $dataset"
    echo "========================================="
    echo ""
done

echo "All training tasks completed."