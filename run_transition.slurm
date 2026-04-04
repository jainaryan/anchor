#!/bin/bash
#SBATCH --job-name=mindmate-transition
#SBATCH --output=logs/transition_%j.log
#SBATCH --error=logs/transition_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=48:00:00

unset VIRTUAL_ENV

if [ ! -d "mindmatenv" ]; then
    uv venv mindmatenv --python 3.12
fi

source mindmatenv/bin/activate

uv pip install torch transformers bitsandbytes tqdm accelerate

# A100-80 has 80GB VRAM — run full bfloat16 for higher quality outputs
export USE_4BIT=0

echo "Starting transition dialogue data pipeline..."
mindmatenv/bin/python synthetic/transition_pipeline.py

echo "Pipeline finished!"
