#!/bin/bash
#SBATCH --job-name=mindmate-crisis-help-qwen
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=48:00:00

PROJECT_DIR=~/projects/mindmate

echo "======================================="
echo "  MindMate Crisis + Help Mode Pipeline"
echo "  Teacher: Qwen3-30B-A3B"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "  PIPELINE_MODE: ${PIPELINE_MODE:-crisis}"
echo "======================================="

source $PROJECT_DIR/mindmatenv/bin/activate
export PYTHONUNBUFFERED=1
echo "[Setup] Python: $(which python) -- $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Fix broken torch/torchvision on H100 nodes.
# NFS home dirs can have stale file handles (Errno 116), blocking venv reinstalls.
# Install torch + torchvision to /tmp (local disk) and prepend to PYTHONPATH.
# torchvision must come from the same install as torch — venv torchvision imports
# torch C++ extensions that break if torch comes from a different path.
python -c "import torch; import torchvision" 2>/dev/null || {
  echo "[Setup] torch/torchvision import failed — installing to /tmp..."
  TORCH_TMP=/tmp/torch_fix_$$
  python -m pip install --target="$TORCH_TMP" \
    --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.11.0" "torchvision==0.26.0" \
    || { echo "[Setup] FATAL: pip install to /tmp failed"; exit 1; }
  export PYTHONPATH="$TORCH_TMP:${PYTHONPATH:-}"
  python -c "import torch; import torchvision; assert torch.cuda.is_available(), 'CUDA not available'; print('[Setup] torch', torch.__version__, '/ torchvision', torchvision.__version__, '/ CUDA', torch.cuda.get_device_name(0), 'OK')" \
    || { echo "[Setup] FATAL: torch/torchvision/CUDA broken after /tmp install"; exit 1; }
  echo "[Setup] PYTHONPATH=$PYTHONPATH"
}

cd $PROJECT_DIR
export TEACHER_MODEL=30b
export OUT_LABEL=qwen

# USE_VLLM=0: vLLM fails on H100 nodes (CUDA fork issue).
# HF backend proven stable on H100-96 (conv-memory jobs 614942-614947 ran with it).
export USE_VLLM=0
echo "[Setup] USE_VLLM=$USE_VLLM"

# PIPELINE_MODE is passed via --export=ALL,PIPELINE_MODE=crisis (or help_mode).
# Defaults to 'crisis' if not set.
export PIPELINE_MODE=${PIPELINE_MODE:-crisis}

# SHARD_IDX support — pass via --export=ALL,PIPELINE_MODE=crisis,SHARD_IDX=0
if [ -n "${SHARD_IDX:-}" ]; then
  echo "[Setup] Running as shard $SHARD_IDX"
fi

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set" && exit 1
fi

echo "Starting ${PIPELINE_MODE} pipeline (Qwen3-30B-A3B, HF backend)..."
python synthetic/crisis_help_pipeline.py

echo "Pipeline finished at $(date)"
