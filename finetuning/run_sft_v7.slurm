#!/bin/bash
#SBATCH --job-name=mindmate-sft-v7
#SBATCH --output=/home/a/aryanj/logs/sft_v7_%j.out
#SBATCH --error=/home/a/aryanj/logs/sft_v7_%j.err
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
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "[setup] Wiping any remnant torch/torchvision dirs in $VENV_SITE..."
rm -rf "$VENV_SITE/torch" "$VENV_SITE"/torch-*.dist-info 2>/dev/null || true
rm -rf "$VENV_SITE/torchvision" "$VENV_SITE"/torchvision-*.dist-info 2>/dev/null || true
echo "[setup] Installing PyTorch cu126 + torchvision into venv..."
python -m pip install -q --index-url https://download.pytorch.org/whl/cu126 torch torchvision
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
echo "  MindMate SFT v7 — Llama 3.2 3B (fresh)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Started: $(date)"
echo "=========================================="
echo ""
echo "Data mix: v7 (= v6 + recovered biometric_qwen_s0, ~13,722 caps)"
echo "  targeted_fix       3500  27%  weight=2.0"
echo "  friend             2000  15%"
echo "  transition         2000  15%"
echo "  biometric          2348  18%  (Gemma4)              weight=1.5"
echo "  biometric_qwen_s0   993   8%  (recovered, salvaged) weight=1.5"
echo "  casual             1000   8%"
echo "  therapist          1000   8%"
echo "  grief               500   4%"
echo "  gold (fixed)        181   1%  weight=3.0"
echo "  help_mode_qwen      200   2%  weight=3.0  (markdown-stripped)"
echo "  Steps: 2000 (~1.24 epochs, checkpoints every 200)"
echo ""

PROJ=~/projects/mindmate

# Step 1: Build dataset
echo "[Step 1] Building dataset (v7 mix)..."
cd /tmp && python -u $PROJ/finetuning/build_dataset.py --model v7
cd ~/projects/mindmate

# Step 2: Clean dataset
echo "[Step 2] Cleaning dataset..."
cd /tmp && python -u $PROJ/finetuning/clean_dataset.py \
    --train-in  $PROJ/data/conversations_raw_v7/mindmate_train.jsonl \
    --val-in    $PROJ/data/conversations_raw_v7/mindmate_val.jsonl \
    --train-out $PROJ/data/conversations_cleaned_v7/mindmate_train.jsonl \
    --val-out   $PROJ/data/conversations_cleaned_v7/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup
cd ~/projects/mindmate

# Step 3: Train
echo "[Step 3] Training (2000 steps, fresh LoRA from base)..."
cd /tmp && python -u $PROJ/finetuning/CUDA_train_qlora.py \
    --iters    2000 \
    --data-dir $PROJ/data/conversations_cleaned_v7 \
    --out-dir  $PROJ/adapters/genzv7

echo ""
echo "=========================================="
echo "  SFT v7 complete: $(date)"
echo "  Output: adapters/genzv7"
echo "=========================================="
