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

# USE_VLLM=1 enables vLLM backend (PagedAttention + FlashAttention2, ~1.5-2x faster)
export USE_VLLM=1
pip install -q vllm 2>&1 | tail -1
echo "[Setup] USE_VLLM=$USE_VLLM"

# Shard support — pass SHARD_IDX via --export=ALL,SHARD_IDX=N when submitting.
# Without it, runs as a single unsharded job (writes to *_qwen.jsonl).
# With SHARD_IDX=0,1,2 → three parallel jobs writing to *_qwen_s0.jsonl etc.
if [ -n "${SHARD_IDX:-}" ]; then
  echo "[Setup] Running as shard $SHARD_IDX"
fi

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set" && exit 1
fi

echo "Starting conversation+memory data pipeline (Qwen3-30B-A3B)..."
python synthetic/conversation_memory_pipeline.py

echo "Pipeline finished at $(date)"
