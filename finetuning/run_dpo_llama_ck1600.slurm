#!/bin/bash
#SBATCH --job-name=mindmate-dpo-llama-ck1600
#SBATCH --output=/home/a/aryanj/logs/dpo_llama_ck1600_%j.log
#SBATCH --error=/home/a/aryanj/logs/dpo_llama_ck1600_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=24:00:00

set -euo pipefail

unset VIRTUAL_ENV

if [ ! -f "mindmatenv/bin/activate" ]; then
  echo "ERROR: virtual environment not found at mindmatenv/bin/activate"
  exit 1
fi
source mindmatenv/bin/activate

uv pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install peft bitsandbytes transformers accelerate datasets trl

cd ~/projects/mindmate

echo "=========================================="
echo "  MindMate DPO — Llama 3.2 3B (ck1600)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

python finetuning/CUDA_train_dpo.py --model llama_ck1600 --steps 1200 --data v2

echo "DPO training complete."
