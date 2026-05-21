#!/bin/bash
#SBATCH --job-name=mindmate-sft-v5
#SBATCH --output=/home/a/aryanj/logs/sft_v5_%j.out
#SBATCH --error=/home/a/aryanj/logs/sft_v5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=24:00:00

# A100-40 FALLBACK VERSION — no torch reinstall needed.
#
# genzv2/v3 used A100-40 with torch+cu130 from the venv and it just worked.
# The A100-40 nodes have CUDA 13.0 drivers, so the venv's torch 2.11.0+cu130
# is fully compatible — no reinstall required, no driver mismatch.
#
# Use this if A100-80 runs keep failing due to driver or torchvision issues.
# The 3B QLoRA model needs ~8-10GB VRAM; A100-40 (40GB) has plenty of headroom.

set -euo pipefail

PROJ=~/projects/mindmate
cd $PROJ
source mindmatenv/bin/activate
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  MindMate SFT v5 — Llama 3.2 3B (A100-40, cu130)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Started: $(date)"
echo "=========================================="
echo ""
echo "Data mix: v5 (~14,373 total)"
echo "  targeted_fix    4000  28%  weight=2.0"
echo "  friend          2000  14%"
echo "  transition      1500  10%"
echo "  biometric       2860  20%  (1500 Gemma4 + 993+80+78+91+57+61 Qwen)  weight=1.5"
echo "  conv_memory     1617  11%  (167+299+335+316+324+44+28+21+24+33+26)  weight=1.5"
echo "  casual          1000   7%"
echo "  therapist        800   6%"
echo "  help_mode_qwen   115   1%  weight=3.0"
echo "  grief            300   2%"
echo "  gold (fixed)     181   1%  weight=3.0"
echo "  Steps: 2000 (~1 epoch, checkpoints every 200)"
echo ""

# There is a logging.py in ~/projects/mindmate/ that shadows stdlib's logging
# when Python runs with CWD=mindmate ('' in sys.path finds the local file).
# WORKAROUND: run all python steps from /tmp with absolute paths.
# Note: this is a CWD issue, not a torch issue — it affects all python steps.

# Step 1: Build dataset
echo "[Step 1] Building dataset (v5 mix)..."
cd /tmp && python -u $PROJ/finetuning/build_dataset.py --model v5
cd $PROJ

# Step 2: Clean dataset
echo "[Step 2] Cleaning dataset..."
cd /tmp && python -u $PROJ/finetuning/clean_dataset.py \
    --train-in  $PROJ/data/conversations_raw_v5/mindmate_train.jsonl \
    --val-in    $PROJ/data/conversations_raw_v5/mindmate_val.jsonl \
    --train-out $PROJ/data/conversations_cleaned_v5/mindmate_train.jsonl \
    --val-out   $PROJ/data/conversations_cleaned_v5/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup
cd $PROJ

# Step 3: Train
echo "[Step 3] Training (2000 steps, fresh LoRA from base)..."
cd /tmp && python -u $PROJ/finetuning/CUDA_train_qlora.py \
    --iters    2000 \
    --data-dir $PROJ/data/conversations_cleaned_v5 \
    --out-dir  $PROJ/adapters/genzv5

echo ""
echo "=========================================="
echo "  SFT v5 complete: $(date)"
echo "  Output: adapters/genzv5"
echo "=========================================="
