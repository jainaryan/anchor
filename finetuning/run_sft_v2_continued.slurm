#!/bin/bash
#SBATCH --job-name=mindmate-sft-v2c
#SBATCH --output=/home/a/aryanj/logs/sft_v2c_%j.out
#SBATCH --error=/home/a/aryanj/logs/sft_v2c_%j.err
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
echo "  MindMate SFT v2_continued — genzv2 ck1600 base"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Started: $(date)"
echo "=========================================="
echo ""
echo "Data mix: v2_continued"
echo "  targeted_fix    1500  30%"
echo "  biometric       1250  25%"
echo "  transition       750  15%  (replay)"
echo "  therapist        600  12%  (replay)"
echo "  casual           500  10%  (replay)"
echo "  friend           250   5%  (replay)"
echo "  gold (fixed)      65  bonus"
echo "  Total:          ~4915"
echo ""
echo "Base adapter: adapters/genz/checkpoint-1600"
echo ""

# Step 1: Build dataset
echo "[Step 1] Building dataset (v2_continued mix)..."
python -u finetuning/build_dataset.py --model v2_continued

# Step 2: Clean dataset
echo "[Step 2] Cleaning dataset..."
python -u finetuning/clean_dataset.py \
    --train-in  data/conversations_raw_v2_continued/mindmate_train.jsonl \
    --val-in    data/conversations_raw_v2_continued/mindmate_val.jsonl \
    --train-out data/conversations_cleaned_v2_continued/mindmate_train.jsonl \
    --val-out   data/conversations_cleaned_v2_continued/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup

# Step 3: Continue training from genzv2 ck1600
echo "[Step 3] Continued training (500 steps, loading genzv2 ck1600 adapter)..."
python -u finetuning/CUDA_train_qlora.py \
    --iters        500 \
    --data-dir     data/conversations_cleaned_v2_continued \
    --out-dir      adapters/genzv2_continued \
    --adapter-path adapters/genz/checkpoint-1600

echo ""
echo "=========================================="
echo "  SFT v2_continued complete: $(date)"
echo "  Output: adapters/genzv2_continued"
echo "=========================================="
