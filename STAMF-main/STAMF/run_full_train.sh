#!/bin/bash

# Activate environment
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate mamba2

# Run training with all modules enabled and weights loaded
# --pretrained_model loads the T2T-ViT weights for BOTH RGB and Depth backbones (default behavior)
# --use_experts_highfreq enables the GradientDetailExperts
# --low_pass_enabled enables the TinyViM-inspired low-frequency processing
# --fusion_type aff enables the Attention Feature Fusion

python train_test_eval.py \
    --Training True \
    --Testing True \
    --Evaluation True \
    --data_root /root/autodl-tmp/ \
    --trainset USOD10KK/TR \
    --epochs 100 \
    --batch_size 12 \
    --lr 0.00005 \
    --pretrained_model /root/autodl-tmp/STAMF-main/pretrained_model/80.7_T2T_ViT_t_14.pth.tar \
    --save_model_dir checkpoint/DualStream_Experts_AFF_Full \
    --save_dir Evaluation/Result/DualStream_Experts_AFF_Full \
    --use_experts_highfreq True \
    --low_pass_enabled True \
    --fusion_type aff \
    --optimizer adamw \
    --scheduler step
