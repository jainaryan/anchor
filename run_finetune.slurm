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

set -euo pipefail

unset VIRTUAL_ENV

if [ ! -f "mindmatenv/bin/activate" ]; then
  echo "ERROR: virtual environment not found at mindmatenv/bin/activate"
  exit 1
fi
source mindmatenv/bin/activate

# Ensure training dependencies are installed
uv pip install peft trl bitsandbytes transformers accelerate datasets pandas wandb

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Export HF_TOKEN before submitting this job."
  exit 1
fi

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo "Running training preflight checks..."
  mindmatenv/bin/python finetuning/preflight_cuda_train.py \
    --model models/CUDA_llama-3.2-3b-instruct \
    --samples-per-split 64
  echo "Preflight passed."
else
  echo "Skipping preflight checks (SKIP_PREFLIGHT=1)."
fi

echo "Starting CUDA finetuning pipeline..."
mindmatenv/bin/python finetuning/CUDA_run_pipeline.py

echo "Finetuning complete!"
