#!/bin/bash

# Activate Conda Environment
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate mamba2

# Common Settings
DATA_ROOT="/root/autodl-tmp/"
EPOCHS=800          # Scale up to full training epochs
BATCH_SIZE=10       # Standard batch size (adjusted from 12 if needed for memory safety)
TRAIN_STEPS=40000   # Total training iterations

# Model Configuration
PRETRAINED_MODEL="/root/autodl-tmp/STAMF-main/pretrained_model/80.7_T2T_ViT_t_14.pth.tar"

# Experiment Name (Used for checkpoints and logs)
EXP_NAME="DualStream_Experts_AFF_Final"

echo "========================================================"
echo "Starting Formal Training: $EXP_NAME"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, Steps: $TRAIN_STEPS"
echo "Features: Heterogeneous Experts + AFF Fusion + Grad SSIM"
echo "========================================================"

# Run Training, Testing, and Evaluation
python train_test_eval.py \
    --Training True \
    --Testing True \
    --Evaluation True \
    --data_root $DATA_ROOT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_steps $TRAIN_STEPS \
    --pretrained_model $PRETRAINED_MODEL \
    --use_experts_highfreq True \
    --use_grad_ssim_loss True \
    --save_model_dir "checkpoint/$EXP_NAME/" \
    --save_test_path_root "preds/$EXP_NAME/" \
    --save_dir "Evaluation/Result/$EXP_NAME" \
    --trainset "USOD10KK/TR" \
    --test_paths "USOD10KK/TE"

echo "Training Completed for $EXP_NAME."
