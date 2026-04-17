#!/bin/bash
#SBATCH --job-name=mindmate_dpo_datagen_v2
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/a/aryanj/logs/dpo_datagen_v2_%j.out
#SBATCH --error=/home/a/aryanj/logs/dpo_datagen_v2_%j.err

cd ~/projects/mindmate

unset VIRTUAL_ENV
source mindmatenv/bin/activate

# Safety check: verify system_prompt.txt is the Anchor prompt, not the old 150-line one
PROMPT_LEN=$(wc -c < system_prompt.txt)
echo "System prompt length: $PROMPT_LEN chars"
if [ "$PROMPT_LEN" -gt 1000 ]; then
  echo "ERROR: system_prompt.txt appears to be the old long prompt ($PROMPT_LEN chars). Aborting."
  exit 1
fi

# Set FRESH_RUN=1 for first job in a chain to clear old partial data
# Leave unset (or =0) for resume jobs — pipeline auto-resumes from dpo_pairs_partial.jsonl
FRESH_RUN=${FRESH_RUN:-0}
if [ "$FRESH_RUN" = "1" ]; then
  echo "Fresh run: removing any existing v2 partial/train/val files"
  rm -f data/dpo_pairs_partial_v2.jsonl data/dpo_train_v2.jsonl data/dpo_val_v2.jsonl
  echo "Done. Old v1 files (dpo_train.jsonl etc.) are untouched."
fi

echo "=========================================="
echo "  MindMate DPO Data Generation v2"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  FRESH_RUN=$FRESH_RUN"
echo "  Target: 8000 pairs"
echo "=========================================="

export HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

TEACHER_MODEL=30b python synthetic/dpo_pipeline.py

echo "Data generation complete."
echo "Pair counts:"
wc -l data/dpo_train_v2.jsonl data/dpo_val_v2.jsonl 2>/dev/null || echo "(files not found)"
