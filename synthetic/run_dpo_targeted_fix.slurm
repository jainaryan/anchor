#!/bin/bash
#SBATCH --job-name=mindmate-dpo-targeted-fix
#SBATCH --output=logs/dpo_targeted_fix_%j.log
#SBATCH --error=logs/dpo_targeted_fix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=08:00:00

unset VIRTUAL_ENV

if [ ! -d "mindmatenv" ]; then
    uv venv mindmatenv --python 3.12
fi

source mindmatenv/bin/activate

uv pip install torch transformers bitsandbytes tqdm accelerate

# A100-80: full bfloat16 — no 4-bit quantisation
export USE_4BIT=0

echo "Starting DPO targeted fix pipeline (help_mode + memory_recall)..."
mindmatenv/bin/python synthetic/dpo_targeted_fix_pipeline.py

echo "Pipeline finished!"
