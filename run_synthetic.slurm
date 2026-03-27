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

echo "Starting synthetic data pipeline..."
mindmatenv/bin/python synthetic/pipeline.py

echo "Pipeline finished!"
