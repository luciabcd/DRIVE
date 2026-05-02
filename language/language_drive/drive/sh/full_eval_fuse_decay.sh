#!/bin/bash

PROJECT_DIR="/data1/."
OUTPUT_ROOT="/data1/.."

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=0
export HF_ALLOW_CODE_EVAL="1"



MODELS_CONFIG=(
    "gemma-2-2b|.."
)


cd "$PROJECT_DIR" || exit

for config in "${MODELS_CONFIG[@]}"; do

    IFS='|' read -r MODEL_NAME ORIGINAL_BASE_MODEL EVAL_MODEL_DIR <<< "$config"
    CURRENT_OUTPUT_PATH="$OUTPUT_ROOT/$MODEL_NAME"
    

    echo "[Step 0] Tokenizer..."
    if [ -d "$ORIGINAL_BASE_MODEL" ] && [ -d "$EVAL_MODEL_DIR" ]; then
        cp -f "$ORIGINAL_BASE_MODEL"/tokenizer* "$EVAL_MODEL_DIR/" 2>/dev/null
        cp -f "$ORIGINAL_BASE_MODEL"/special_tokens_map.json "$EVAL_MODEL_DIR/" 2>/dev/null
        cp -f "$ORIGINAL_BASE_MODEL"/*.model "$EVAL_MODEL_DIR/" 2>/dev/null 
    else
        echo "❌"
        continue
    fi

    mkdir -p "$CURRENT_OUTPUT_PATH"

    MODEL_ARGS="pretrained=$EVAL_MODEL_DIR,trust_remote_code=True"



    echo ">>> [$MODEL_NAME] Running GSM8K..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 32 \
        --output_path "$CURRENT_OUTPUT_PATH/5shot_gsm8k" \
        --device cuda


    echo ">>> [$MODEL_NAME] Running MMLU..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 2 \
        --output_path "$CURRENT_OUTPUT_PATH/5shot_mmlu" \
        --device cuda

    echo ">>> [$MODEL_NAME] Running MBPP..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks mbpp \
        --num_fewshot 3 \
        --batch_size 16 \
        --output_path "$CURRENT_OUTPUT_PATH/3shot_mbpp" \
        --device cuda \
        --confirm_run_unsafe_code

    4. Commonsense (0-shot)--batch_size 8
    echo ">>> [$MODEL_NAME] Running Commonsense..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks hellaswag,piqa,social_iqa,openbookqa,arc_easy,arc_challenge \
        --num_fewshot 0 \
        --batch_size 8 \
        --output_path "$CURRENT_OUTPUT_PATH/0shot_commonsense" \
        --device cuda

    # 5. HumanEval (0-shot)--batch_size 16
    echo ">>> [$MODEL_NAME] Running HumanEval..."
    lm_eval --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size 16 \
        --output_path "$CURRENT_OUTPUT_PATH/0shot_humaneval" \
        --device cuda \
        --confirm_run_unsafe_code

    echo "✅"
    echo ""
done

echo "========================================================="
echo "$OUTPUT_ROOT"