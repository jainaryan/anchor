#!/bin/bash
#SBATCH --job-name=mindmate-synthetic
#SBATCH --output=logs/synthetic_%j.log
#SBATCH --error=logs/synthetic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=04:00:00

# Load necessary modules (Adjust these based on NUS cluster available modules)
# Example for NUS HPC/SoC Cluster:
module load cuda/12.1
module load python/3.12.1  # Make sure Python >= 3.12 is loaded

# Force clear any inherited virtual environment from the login node
unset VIRTUAL_ENV

echo "Creating environment and checking dependencies..."
if [ ! -d "mindmatenv" ]; then
    uv venv mindmatenv --python 3.12
fi

source mindmatenv/bin/activate
mindmatenv/bin/pip install -e . --break-system-packages

echo "Starting synthetic data pipeline..."
# Run the pipelne explicitly using the precise python binary
mindmatenv/bin/python synthetic/pipeline.py

echo "Pipeline finished!"
