#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Use the configured conda environment
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate mamba2

python verify_train.py
