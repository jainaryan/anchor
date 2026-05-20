#!/bin/bash
#SBATCH --job-name=mindmate-sft-v5
#SBATCH --output=/home/a/aryanj/logs/sft_v5_%j.out
#SBATCH --error=/home/a/aryanj/logs/sft_v5_%j.err
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
export PYTHONUNBUFFERED=1

# torch 2.11.0+cu130 requires CUDA driver ≥ 13.0. Not all cluster nodes have this
# (e.g. xgph6 has an older driver). Install cu126 (CUDA 12.6 runtime) which works
# on any node with a driver supporting CUDA ≥ 12.6 — covers all current cluster nodes.
echo "[setup] Installing compatible PyTorch (cu126)..."
pip install -q --index-url https://download.pytorch.org/whl/cu126 torch
echo "[setup] torch version: $(python -c 'import torch; print(torch.__version__)')"

echo "=========================================="
echo "  MindMate SFT v5 — Llama 3.2 3B (fresh)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Started: $(date)"
echo "=========================================="
echo ""
echo "Data mix: v5 (~13,988 total)"
echo "  targeted_fix    4000  29%  weight=2.0"
echo "  friend          2000  14%"
echo "  transition      1500  11%"
echo "  biometric       2651  19%  (1500 Gemma4 + 993+80+78 Qwen)  weight=1.5"
echo "  conv_memory     1441  10%  (167+299+335+316+324)  weight=1.5"
echo "  casual          1000   7%"
echo "  therapist        800   6%"
echo "  help_mode_qwen   115   1%  weight=3.0"
echo "  grief            300   2%"
echo "  gold (fixed)     181   1%  weight=3.0"
echo "  Steps: 2000 (~1 epoch, checkpoints every 200)"
echo ""

# Step 1: Build dataset
echo "[Step 1] Building dataset (v5 mix)..."
python -u finetuning/build_dataset.py --model v5

# Step 2: Clean dataset
echo "[Step 2] Cleaning dataset..."
python -u finetuning/clean_dataset.py \
    --train-in  data/conversations_raw_v5/mindmate_train.jsonl \
    --val-in    data/conversations_raw_v5/mindmate_val.jsonl \
    --train-out data/conversations_cleaned_v5/mindmate_train.jsonl \
    --val-out   data/conversations_cleaned_v5/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup

# Step 3: Train
echo "[Step 3] Training (2000 steps, fresh LoRA from base)..."
python -u finetuning/CUDA_train_qlora.py \
    --iters    2000 \
    --data-dir data/conversations_cleaned_v5 \
    --out-dir  adapters/genzv5

echo ""
echo "=========================================="
echo "  SFT v5 complete: $(date)"
echo "  Output: adapters/genzv5"
echo "=========================================="
