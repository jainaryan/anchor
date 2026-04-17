#!/bin/bash
#SBATCH --job-name=mindmate-gemma4-e2b
#SBATCH --output=/home/a/aryanj/logs/finetune_gemma4_e2b_%j.log
#SBATCH --error=/home/a/aryanj/logs/finetune_gemma4_e2b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=48:00:00

set -euo pipefail

unset VIRTUAL_ENV

if [ ! -f "mindmatenv/bin/activate" ]; then
  echo "ERROR: virtual environment not found at mindmatenv/bin/activate"
  exit 1
fi
source mindmatenv/bin/activate

uv pip install peft trl bitsandbytes transformers accelerate datasets pandas

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  fi
fi

# Gemma 4 requires accepting the license on HuggingFace — ensure HF_TOKEN is set
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Gemma 4 is a gated model — download may fail."
fi

cd ~/projects/mindmate

echo "Starting Gemma 4 E2B IT finetuning pipeline..."
mindmatenv/bin/python finetuning/CUDA_run_pipeline_gemma4_e2b.py

echo "Finetuning complete!"
