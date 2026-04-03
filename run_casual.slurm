#!/bin/bash
#SBATCH --job-name=mindmate-casual
#SBATCH --output=logs/casual_%j.log
#SBATCH --error=logs/casual_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=36:00:00

unset VIRTUAL_ENV

if [ ! -d "mindmatenv" ]; then
    uv venv mindmatenv --python 3.12
fi

source mindmatenv/bin/activate

uv pip install torch transformers bitsandbytes tqdm accelerate

echo "Starting casual conversation data pipeline..."
mindmatenv/bin/python synthetic/casual_pipeline.py

echo "Pipeline finished!"
