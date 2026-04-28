#!/bin/bash
#SBATCH --job-name=mindmate-dpo-genz-v2
#SBATCH --output=/home/a/aryanj/logs/dpo_genz_ck1600_v2_%j.log
#SBATCH --error=/home/a/aryanj/logs/dpo_genz_ck1600_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=24:00:00

set -euo pipefail

unset VIRTUAL_ENV

if [ ! -f "mindmatenv/bin/activate" ]; then
  echo "ERROR: virtual environment not found at mindmatenv/bin/activate"
  exit 1
fi
source mindmatenv/bin/activate

# Pin torch to avoid CUDA breakage from auto-upgrade
uv pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install peft bitsandbytes transformers accelerate datasets trl

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

cd ~/projects/mindmate

echo "=========================================="
echo "  MindMate DPO Training — GenZ ck1600 v2"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Base: adapters/genz/checkpoint-1600"
echo "  Out:  adapters/genz_ck1600_dpo_v2"
echo "  beta=0.2 | max_length=2048 | steps=1200 | LoRA r=16 | GPU: H100-47"
echo "=========================================="

# Preflight: require at least 5000 training pairs
TRAIN_LEN=$(wc -l < data/dpo_train_v2.jsonl 2>/dev/null || echo 0)
echo "Training pairs found: $TRAIN_LEN"
if [ "$TRAIN_LEN" -lt 5000 ]; then
  echo "ERROR: Fewer than 5000 training pairs ($TRAIN_LEN). Run run_dpo_datagen_v2.slurm first."
  exit 1
fi

echo "Starting DPO training..."
mindmatenv/bin/python finetuning/CUDA_train_dpo.py --model genz_ck1600 --steps 1200

echo "DPO training complete!"
echo "Adapter saved to: adapters/genz_ck1600_dpo_v2/"
