#!/bin/bash
#SBATCH --job-name=mindmate-synthetic
#SBATCH --output=logs/synthetic_%j.log
#SBATCH --error=logs/synthetic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Load necessary modules (Adjust these based on NUS cluster available modules)
# Example for NUS HPC/SoC Cluster:
module load cuda/12.1
module load python/3.12.1  # Make sure Python >= 3.12 is loaded

# Use uv for fast dependency management and execution
# If uv is not installed globally, you can install it locally:
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.cargo/env

echo "Creating environment and checking dependencies..."
uv venv mindmatenv --python 3.12
source mindmatenv/bin/activate
pip install -e . --break-system-packages

echo "Starting synthetic data pipeline..."
# Run the pipeline
uv run python synthetic/pipeline.py

echo "Pipeline finished!"
