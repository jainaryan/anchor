#!/bin/bash
#SBATCH --job-name=mindmate-sft-v3
#SBATCH --output=/home/a/aryanj/logs/sft_v3_%j.out
#SBATCH --error=/home/a/aryanj/logs/sft_v3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=24:00:00

set -euo pipefail

cd ~/projects/mindmate
source mindmatenv/bin/activate

echo "=========================================="
echo "  MindMate SFT v3 — Llama 3.2 3B (fresh)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Started: $(date)"
echo "=========================================="
echo ""
echo "Data mix: v3"
echo "  transition      2500  25%"
echo "  targeted_fix    2000  20%"
echo "  therapist       1500  15%"
echo "  biometric       1500  15%"
echo "  friend          1500  15%"
echo "  casual          1000  10%"
echo "  gold (fixed)      65  bonus"
echo "  Total:         ~10065"
echo ""

# Step 1: Build dataset
echo "[Step 1] Building dataset (v3 mix)..."
python -u finetuning/build_dataset.py --model v3

# Step 2: Clean dataset
echo "[Step 2] Cleaning dataset..."
python -u finetuning/clean_dataset.py \
    --train-in  data/conversations_raw_v3/mindmate_train.jsonl \
    --val-in    data/conversations_raw_v3/mindmate_val.jsonl \
    --train-out data/conversations_cleaned_v3/mindmate_train.jsonl \
    --val-out   data/conversations_cleaned_v3/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup

# Step 3: Train
echo "[Step 3] Training (1600 steps, fresh LoRA)..."
python -u finetuning/CUDA_train_qlora.py \
    --iters    1600 \
    --data-dir data/conversations_cleaned_v3 \
    --out-dir  adapters/genzv3

echo ""
echo "=========================================="
echo "  SFT v3 complete: $(date)"
echo "  Output: adapters/genzv3"
echo "=========================================="
