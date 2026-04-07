#!/bin/bash
#SBATCH --job-name=mindmate_dpo_datagen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dpo_datagen_%j.out
#SBATCH --error=logs/dpo_datagen_%j.err

cd ~/projects/mindmate
mkdir -p logs

source mindmatenv/bin/activate

echo "=========================================="
echo "  MindMate DPO Data Generation"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

# Use Qwen3-72B 4-bit (requires H100/H200 for ~40GB VRAM)
TEACHER_MODEL=72b python synthetic/dpo_pipeline.py

echo "Data generation complete."
