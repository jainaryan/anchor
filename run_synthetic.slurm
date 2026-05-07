#!/bin/bash
#SBATCH --job-name=mindmate-synthetic
#SBATCH --output=logs/synthetic_%j.log
#SBATCH --error=logs/synthetic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=72:00:00

# Force clear any inherited virtual environment from the login node
unset VIRTUAL_ENV

echo "Creating environment and checking dependencies..."
if [ ! -d "mindmatenv" ]; then
    uv venv mindmatenv --python 3.12
fi

source mindmatenv/bin/activate

# Use uv pip (not pip) since uv venv doesn't install pip
uv pip install torch transformers bitsandbytes tqdm accelerate

echo "Clearing old outputs..."
> data/synthetic_train.jsonl
> synthetic/outputs/scenarios.jsonl
> synthetic/outputs/dialogues_raw.jsonl

# Gemma 4 26B A4B bfloat16 uses ~52GB — fits fine on H100-96
export USE_4BIT=0
export TEACHER_MODEL=gemma4

# Gemma 4 is gated — HF_TOKEN required
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set — Gemma 4 download will fail (gated model)" && exit 1
fi

echo "Starting synthetic data pipeline..."
mindmatenv/bin/python synthetic/pipeline.py

echo "Pipeline finished!"
