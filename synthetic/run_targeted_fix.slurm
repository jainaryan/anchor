#!/bin/bash
#SBATCH --job-name=mindmate-targeted-fix
#SBATCH --output=logs/targeted_fix_%j.log
#SBATCH --error=logs/targeted_fix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=10:00:00

unset VIRTUAL_ENV

if [ ! -d "mindmatenv" ]; then
    uv venv mindmatenv --python 3.12
fi

source mindmatenv/bin/activate

uv pip install torch transformers bitsandbytes tqdm accelerate

# A100-80: full bfloat16 for higher quality outputs
export USE_4BIT=0

echo "Starting targeted fix data pipeline (help_mode + memory_recall)..."
mindmatenv/bin/python synthetic/targeted_fix_pipeline.py

echo "Pipeline finished!"
