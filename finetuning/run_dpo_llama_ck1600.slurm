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

cd ~/projects/mindmate
source mindmatenv/bin/activate

echo "=========================================="
echo "  MindMate DPO — Llama 3.2 3B (genzv2 ck1600)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Data: dpo_train.jsonl (5750 pairs, incl. help_mode + memory_recall)"
echo "=========================================="

python -u finetuning/CUDA_train_dpo.py --model llama_ck1600 --steps 1200 --data v1

echo "DPO training complete."
