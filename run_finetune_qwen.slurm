#!/bin/bash
#SBATCH --job-name=mindmate-qwen
#SBATCH --output=logs/finetune_qwen_%j.log
#SBATCH --error=logs/finetune_qwen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=24:00:00

set -euo pipefail

unset VIRTUAL_ENV

if [ ! -f "mindmatenv/bin/activate" ]; then
  echo "ERROR: virtual environment not found at mindmatenv/bin/activate"
  exit 1
fi
source mindmatenv/bin/activate

uv pip install peft trl bitsandbytes transformers accelerate datasets pandas wandb

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  fi
fi

echo "Starting Qwen3-1.7B finetuning pipeline..."
mindmatenv/bin/python finetuning/CUDA_run_pipeline_qwen.py

echo "Finetuning complete!"
