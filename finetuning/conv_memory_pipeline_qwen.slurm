#!/bin/bash
#SBATCH --job-name=mindmate-conv-memory-qwen
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=72:00:00

PROJECT_DIR=~/projects/mindmate

echo "======================================="
echo "  MindMate Conversation+Memory Pipeline"
echo "  Teacher: Qwen3-30B-A3B"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "======================================="

source $PROJECT_DIR/mindmatenv/bin/activate
echo "[Setup] Python: $(which python) -- $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

cd $PROJECT_DIR
export TEACHER_MODEL=30b
export OUT_LABEL=qwen

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set" && exit 1
fi

echo "Starting conversation+memory data pipeline (Qwen3-30B-A3B)..."
python synthetic/conversation_memory_pipeline.py

echo "Pipeline finished at $(date)"
