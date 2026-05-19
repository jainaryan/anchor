#!/bin/bash
#SBATCH --job-name=mindmate-crisis-help-qwen
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
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

cd $PROJECT_DIR
export TEACHER_MODEL=30b
export OUT_LABEL=qwen

# vLLM is NOT usable on this cluster — H100-96 nodes run CUDA driver 12090
# (12.0.90), which is too old for torch.accelerator used by vLLM V1 engine.
# HF backend (USE_VLLM=0) is the only viable option.
export USE_VLLM=0
export PHASE1_BATCH_SIZE=8
echo "[Setup] USE_VLLM=$USE_VLLM  PHASE1_BATCH_SIZE=$PHASE1_BATCH_SIZE"

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
