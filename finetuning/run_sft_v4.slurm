#!/bin/bash
#SBATCH --job-name=mindmate-sft-v4
#SBATCH --output=/home/a/aryanj/logs/sft_v4_%j.out
#SBATCH --error=/home/a/aryanj/logs/sft_v4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=24:00:00

set -euo pipefail

cd ~/projects/mindmate
source mindmatenv/bin/activate

echo "=========================================="
echo "  MindMate SFT v4 — Llama 3.2 3B (fresh)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Started: $(date)"
echo "=========================================="
echo ""
echo "Data mix: v4 (~21,529 total)"
echo "  targeted_fix    6000  28%"
echo "  transition      4000  19%"
echo "  friend          3000  14%"
echo "  therapist       2500  11%"
echo "  casual          2500  12%"
echo "  biometric       2348  11%"
echo "  grief           1000   5%"
echo "  gold (fixed)     181  bonus"
echo "  Steps: 2400 (target Goldilocks ck400-800)"
echo ""

# Step 1: Build dataset
echo "[Step 1] Building dataset (v4 mix)..."
python -u finetuning/build_dataset.py --model v4

# Step 2: Clean dataset
echo "[Step 2] Cleaning dataset..."
python -u finetuning/clean_dataset.py \
    --train-in  data/conversations_raw_v4/mindmate_train.jsonl \
    --val-in    data/conversations_raw_v4/mindmate_val.jsonl \
    --train-out data/conversations_cleaned_v4/mindmate_train.jsonl \
    --val-out   data/conversations_cleaned_v4/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup

# Step 3: Train
echo "[Step 3] Training (2400 steps, fresh LoRA)..."
python -u finetuning/CUDA_train_qlora.py \
    --iters    2400 \
    --data-dir data/conversations_cleaned_v4 \
    --out-dir  adapters/genzv4

echo ""
echo "=========================================="
echo "  SFT v4 complete: $(date)"
echo "  Output: adapters/genzv4"
echo "=========================================="
