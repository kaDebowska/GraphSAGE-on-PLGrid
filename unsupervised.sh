#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1

source .venv/bin/activate

module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

python -m graphsage.utils.py data/amazon-G.json data/amazon-walks.txt

python -m graphsage.unsupervised_train --train_prefix ./data/amazon --model graphsage_mean --identity_dim 128 --max_total_steps 1000 --validate_iter 10