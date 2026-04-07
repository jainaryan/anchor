#!/bin/bash
#SBATCH --job-name=mindmate_dpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err

# Usage:
#   sbatch run_dpo.slurm              # trains llama DPO adapter (default)
#   sbatch --export=ALL,MODEL=qwen run_dpo.slurm

cd ~/projects/mindmate
mkdir -p logs

source mindmatenv/bin/activate

MODEL=${MODEL:-llama}

echo "=========================================="
echo "  MindMate DPO Training — ${MODEL^^}"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

python finetuning/CUDA_train_dpo.py --model "$MODEL" --steps 800

echo "DPO training complete."
