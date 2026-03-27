#!/bin/bash
#SBATCH --job-name=mindmate-finetune
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=24:00:00

unset VIRTUAL_ENV
source mindmatenv/bin/activate

# Ensure training dependencies are installed
uv pip install peft trl bitsandbytes transformers accelerate datasets pandas wandb

echo "Starting CUDA finetuning pipeline..."
mindmatenv/bin/python finetuning/CUDA_run_pipeline.py

echo "Finetuning complete!"
