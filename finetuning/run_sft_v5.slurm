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
#
# CRITICAL: use `python -m pip`, NOT bare `pip`. Even after `source mindmatenv/bin/activate`,
# bare `pip` resolves to miniconda3's pip (/home/a/aryanj/miniconda3/bin/pip), which installs
# to miniconda3's site-packages instead of the venv. This caused 4 jobs (617977, 618080, 618086,
# 618200) to fail: pip said cu126 installed, but venv's Python kept loading the original cu130.
# `python -m pip` forces pip to use the venv's site-packages.
# The venv was originally created without pip (`python -m venv --without-pip`),
# so `python -m pip` fails with ModuleNotFoundError. Bootstrap pip first.
echo "[setup] Bootstrapping pip into venv (if missing)..."
python -m ensurepip --upgrade 2>&1 | tail -3 || python -m ensurepip 2>&1 | tail -3
echo "[setup] Verifying pip is now in venv:"
python -c "import pip; print('  pip location:', pip.__file__); print('  pip version:', pip.__version__)"
echo "[setup] Uninstalling existing torch from venv..."
python -m pip uninstall -y torch 2>&1 | tail -3 || true
# Belt-and-suspenders: nuke any leftover torch directory in venv (--force-reinstall
# is known to leave dist-info but not always replace package files cleanly).
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "[setup] Wiping any remnant torch dirs in $VENV_SITE..."
rm -rf "$VENV_SITE/torch" "$VENV_SITE"/torch-*.dist-info 2>/dev/null || true
echo "[setup] Installing PyTorch cu126 into venv..."
python -m pip install -q --index-url https://download.pytorch.org/whl/cu126 torch
# Run from /tmp to avoid logging.py in project root shadowing stdlib logging module
echo "[setup] torch version (must be cu126):"
cd /tmp && python -c 'import torch; print("  version:", torch.__version__); print("  cuda:", torch.version.cuda); print("  file:", torch.__file__)' && cd ~/projects/mindmate
# Fail fast if cu126 didn't actually install
python -c 'import torch; assert "cu126" in torch.__version__, f"Expected cu126, got {torch.__version__}"' || { echo "[setup] FATAL: cu126 install failed"; exit 1; }

echo "=========================================="
echo "  MindMate SFT v5 — Llama 3.2 3B (fresh)"
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
