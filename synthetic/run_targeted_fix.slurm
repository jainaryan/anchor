#!/bin/bash
#SBATCH --job-name=mindmate-targeted-fix
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=24:00:00

PROJECT_DIR=~/projects/mindmate

echo "======================================="
echo "  MindMate Targeted Fix Data Generation"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "======================================="

source $PROJECT_DIR/mindmatenv/bin/activate
echo "[Setup] Python: $(which python) -- $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

cd $PROJECT_DIR
export USE_4BIT=0
export TEACHER_MODEL=gemma4

# Gemma 4 is gated — HF_TOKEN required
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set — Gemma 4 download will fail (gated model)" && exit 1
fi

echo "Starting targeted fix data pipeline (help_mode + memory_recall)..."
python synthetic/targeted_fix_pipeline.py

echo "Pipeline finished at $(date)"
