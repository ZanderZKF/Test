#!/bin/bash

# Activate Conda Environment
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate mamba2

# Common Settings
DATA_ROOT="/root/autodl-tmp/"
EPOCHS=100 # Reduced for faster ablation, adjust as needed
BATCH_SIZE=10
TRAIN_STEPS=40000

# Function to run experiment
run_exp() {
    EXP_NAME=$1
    USE_EXPERTS=$2
    USE_SSIM=$3
    
    echo "========================================================"
    echo "Running Experiment: $EXP_NAME"
    echo "Experts: $USE_EXPERTS, SSIM Loss: $USE_SSIM"
    echo "========================================================"
    
    python train_test_eval.py \
        --Training True \
        --Testing True \
        --Evaluation True \
        --data_root $DATA_ROOT \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --train_steps $TRAIN_STEPS \
        --use_experts_highfreq $USE_EXPERTS \
        --use_grad_ssim_loss $USE_SSIM \
        --save_model_dir "checkpoint/$EXP_NAME/" \
        --save_test_path_root "preds/$EXP_NAME/" \
        --save_dir "Evaluation/Result/$EXP_NAME"
        
    echo "Finished Experiment: $EXP_NAME"
    echo ""
}

# 1. Proposed Method (All Enabled)
run_exp "Proposed_Method" "True" "True"

# 2. Ablation: No Heterogeneous Experts (Back to Symmetric IGM)
run_exp "Ablation_NoExperts" "False" "True"

# 3. Ablation: No Gradient SSIM Loss
run_exp "Ablation_NoSSIM" "True" "False"

# 4. Baseline (No Experts, No SSIM)
run_exp "Baseline" "False" "False"

echo "All Ablation Experiments Completed."
