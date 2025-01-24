#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1

source .venv/bin/activate

module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

TRAIN_PREFIX=$1
MODEL=$2
IDENTITY_DIM=$3
MAX_TOTAL_STEPS=$4
VALIDATE_ITER=$5

python -m graphsage.unsupervised_train \
  --train_prefix "$TRAIN_PREFIX" \
  --model "$MODEL" \
  --identity_dim "$IDENTITY_DIM" \
  --max_total_steps "$MAX_TOTAL_STEPS" \
  --validate_iter "$VALIDATE_ITER"