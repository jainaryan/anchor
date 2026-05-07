#!/bin/bash
#SBATCH --job-name=mindmate-casual
#SBATCH --output=logs/casual_%j.log
#SBATCH --error=logs/casual_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=36:00:00

unset VIRTUAL_ENV

if [ ! -d "mindmatenv" ]; then
    uv venv mindmatenv --python 3.12
fi

source mindmatenv/bin/activate

uv pip install torch transformers bitsandbytes tqdm accelerate

# A100-80 has 80GB VRAM — Gemma 4 26B A4B bfloat16 uses ~52GB, fits fine
export USE_4BIT=0
export TEACHER_MODEL=gemma4

# Gemma 4 is gated — HF_TOKEN required
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set — Gemma 4 download will fail (gated model)" && exit 1
fi

echo "Starting casual conversation data pipeline..."
mindmatenv/bin/python synthetic/casual_pipeline.py

echo "Pipeline finished!"
