#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate mamba2

# Minimal training verification
# Epochs: 1
# Train Steps: 20 (Simulate short training)
# Batch Size: 8 (Small enough)
# Pretrained Model: Empty (Skip loading)
# Optimizer: Adam (Faster convergence for check)
# Scheduler: Step (Simple)

python train_test_eval.py --Training True --Testing True --Evaluation True \
    --trainset USOD10KK/TR \
    --epochs 1 \
    --batch_size 8 \
    --train_steps 20 \
    --save_model_dir checkpoint/Verify_Run \
    --save_dir Evaluation/Result/Verify_Run \
    --pretrained_model '' \
    --optimizer adam \
    --scheduler step \
    --stepvalue1 10 \
    --stepvalue2 15
