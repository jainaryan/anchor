#!/bin/bash
#SBATCH --job-name=download-qwen
#SBATCH --output=logs/download_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Activate environment
source mindmatenv/bin/activate

# Execute the download CLI
echo "Starting download..."
HF_HUB_DISABLE_CONCURRENT_DOWNLOADS=1 huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507
echo "Download complete!"
