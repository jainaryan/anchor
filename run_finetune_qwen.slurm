#!/bin/bash
#SBATCH --job-name=mindmate-qwen-v2
#SBATCH --output=/home/a/aryanj/logs/finetune_qwen_v2_%j.log
#SBATCH --error=/home/a/aryanj/logs/finetune_qwen_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
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

cd ~/projects/mindmate

echo "Starting Qwen3-1.7B finetuning pipeline (v2)..."
mindmatenv/bin/python finetuning/CUDA_run_pipeline_qwen.py

echo "Finetuning complete!"
