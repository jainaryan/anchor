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
echo "[Setup] Python: $(which python) -- $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

cd $PROJECT_DIR
export TEACHER_MODEL=30b
export OUT_LABEL=qwen

# vLLM re-enabled on H100-96 (2026-05-19). Root cause of the CUDA fork error was
# vLLM V1 EngineCore using fork — fixed by forcing V0 engine (in-process, no fork).
# VLLM_USE_V1 must be set before python runs; utils.py also defaults to V0 if not set.
export USE_VLLM=1
export VLLM_USE_V1=0
export PHASE1_BATCH_SIZE=8
echo "[Setup] USE_VLLM=$USE_VLLM  VLLM_USE_V1=$VLLM_USE_V1  PHASE1_BATCH_SIZE=$PHASE1_BATCH_SIZE"

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
