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
# The venv was originally created with `uv venv` (without pip). Bootstrap pip if missing.
# Skip ensurepip if pip is already present (which it is, after a successful bootstrap).
echo "[setup] Ensuring pip is in venv..."
if python -c "import pip" 2>/dev/null; then
    echo "[setup] pip already present in venv"
else
    echo "[setup] Installing pip via ensurepip..."
    python -m ensurepip --upgrade
fi
echo "[setup] Verifying pip is in venv:"
python -c "import pip; print('  pip location:', pip.__file__); print('  pip version:', pip.__version__)"
echo "[setup] Uninstalling existing torch + torchvision from venv..."
python -m pip uninstall -y torch torchvision 2>&1 | tail -5 || true
# Belt-and-suspenders: nuke any leftover package directories in venv.
# torchvision 0.26.0 compiled for torch 2.11.0 crashes on torch 2.12 with
# "operator torchvision::nms does not exist" (job 618371).
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "[setup] Wiping any remnant torch/torchvision dirs in $VENV_SITE..."
rm -rf "$VENV_SITE/torch" "$VENV_SITE"/torch-*.dist-info 2>/dev/null || true
rm -rf "$VENV_SITE/torchvision" "$VENV_SITE"/torchvision-*.dist-info 2>/dev/null || true
echo "[setup] Installing PyTorch cu126 + torchvision into venv..."
# torchvision must be upgraded alongside torch — torchvision 0.26.0 (compiled for
# torch 2.11.0) fails on torch 2.12 with "operator torchvision::nms does not exist"
# which cascades through transformers' lazy loader as ModuleNotFoundError (job 618371).
python -m pip install -q --index-url https://download.pytorch.org/whl/cu126 torch torchvision
# Run from /tmp to avoid logging.py in project root shadowing stdlib logging module.
# IMPORTANT: both the version print AND the assert must run from /tmp — the assert was
# previously back in ~/projects/mindmate and hit the same logging.py shadow (job 618209).
echo "[setup] torch version (must be cu126):"
cd /tmp && python -c '
import torch, torchvision
print("  torch version:", torch.__version__)
print("  cuda:", torch.version.cuda)
print("  torch file:", torch.__file__)
print("  torchvision version:", torchvision.__version__)
assert "cu126" in torch.__version__, f"Expected cu126, got {torch.__version__}"
print("  [OK] cu126 confirmed")
' || { echo "[setup] FATAL: cu126/torchvision install failed"; exit 1; }
cd ~/projects/mindmate

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

# There is a logging.py in ~/projects/mindmate/ that shadows stdlib's logging.
# Python adds '' (CWD) to sys.path for -c and -u invocations, which causes
# `import logging` inside transformers/torch to resolve to the local file instead
# of stdlib, producing AttributeError / ModuleNotFoundError cascades.
# WORKAROUND: run all python steps from /tmp with absolute paths so that
# ~/projects/mindmate is never the CWD when Python imports stdlib.
PROJ=~/projects/mindmate

# Step 1: Build dataset (run from /tmp to avoid logging.py shadow)
echo "[Step 1] Building dataset (v5 mix)..."
cd /tmp && python -u $PROJ/finetuning/build_dataset.py --model v5
cd ~/projects/mindmate

# Step 2: Clean dataset (run from /tmp)
echo "[Step 2] Cleaning dataset..."
cd /tmp && python -u $PROJ/finetuning/clean_dataset.py \
    --train-in  $PROJ/data/conversations_raw_v5/mindmate_train.jsonl \
    --val-in    $PROJ/data/conversations_raw_v5/mindmate_val.jsonl \
    --train-out $PROJ/data/conversations_cleaned_v5/mindmate_train.jsonl \
    --val-out   $PROJ/data/conversations_cleaned_v5/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup
cd ~/projects/mindmate

# Step 3: Train (run from /tmp — same reason)
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
