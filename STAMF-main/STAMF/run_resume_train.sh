#!/bin/bash

# Activate Conda Environment
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate mamba2

# Common Settings
DATA_ROOT="/root/autodl-tmp/"
EPOCHS=200          # Reduced Epochs because we are resuming from half-way or fine-tuning
BATCH_SIZE=10       # Keep consistent
TRAIN_STEPS=40000   # Keep consistent

# Resume Configuration
# User provided checkpoint path
RESUME_PATH="/root/autodl-tmp/STAMF-main/STAMF/checkpoint/DualStream_Experts_AFF_Final/IGMamba_half_20260131_0315.pth"
PRETRAINED_MODEL="/root/autodl-tmp/STAMF-main/pretrained_model/80.7_T2T_ViT_t_14.pth.tar"

# Experiment Name
EXP_NAME="DualStream_Experts_AFF_Resume_Adam"

echo "========================================================"
echo "Resuming Training with Adam + StepLR"
echo "Checkpoint: $RESUME_PATH"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "========================================================"

# Run Training, Testing, and Evaluation
# Added --optimizer adam and --scheduler step
python train_test_eval.py \
    --Training True \
    --Testing True \
    --Evaluation True \
    --data_root $DATA_ROOT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_steps $TRAIN_STEPS \
    --resume $RESUME_PATH \
    --pretrained_model $PRETRAINED_MODEL \
    --use_experts_highfreq True \
    --use_grad_ssim_loss True \
    --optimizer adam \
    --scheduler step \
    --stepvalue1 10000 \
    --stepvalue2 20000 \
    --save_model_dir "checkpoint/$EXP_NAME/" \
    --save_test_path_root "preds/$EXP_NAME/" \
    --save_dir "Evaluation/Result/$EXP_NAME" \
    --trainset "USOD10KK/TR" \
    --test_paths "USOD10KK/TE"

echo "Resume Training Completed for $EXP_NAME."
