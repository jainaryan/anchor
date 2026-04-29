#!/bin/bash
#SBATCH --job-name=mindmate-dpo-tfix
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=24:00:00

# ── DPO Targeted Fix Data Generation (help_mode + memory_recall) ─────────────
#
# Generates preference pairs for the two worst benchmark failure modes:
#   help_mode    — user asks for help → chosen names ONE coping technique,
#                  rejected asks a question or hallucinates shared history
#   memory_recall — user mentions someone/event from past session →
#                  chosen references it naturally, rejected ignores or hallucinates
#
# Runs until 22h timeout (2h buffer before 24h wall). On SIGTERM, pipeline
# saves gracefully. Appends pairs to:
#   data/dpo_train.jsonl + data/dpo_val.jsonl
# Partial recovery file:
#   data/dpo_targeted_fix_partial.jsonl
#
# Requires A100-80 (Qwen3-30B-A3B-Instruct-2507 teacher in bfloat16, ~60GB VRAM)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR=~/projects/mindmate
SYNTHETIC_DIR=$PROJECT_DIR/synthetic

echo "========================================"
echo "  MindMate DPO Targeted Fix Datagen"
echo "  help_mode + memory_recall pairs"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "========================================"

source $PROJECT_DIR/mindmatenv/bin/activate
echo "[Setup] Python: $(which python) — $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

cd $SYNTHETIC_DIR
export USE_4BIT=0

echo "[Pipeline] Generating DPO pairs (help_mode + memory_recall)..."
echo "  Output: $PROJECT_DIR/data/dpo_train.jsonl + dpo_val.jsonl"
echo "  Partial: $PROJECT_DIR/data/dpo_targeted_fix_partial.jsonl"
echo ""

# 22h timeout — 2h buffer before 24h wall-time
timeout 79200 python -u dpo_targeted_fix_pipeline.py || true

echo ""
echo "========================================"
echo "  Pipeline complete: $(date)"
TRAIN_LINES=$(wc -l < $PROJECT_DIR/data/dpo_train.jsonl 2>/dev/null || echo 0)
VAL_LINES=$(wc -l < $PROJECT_DIR/data/dpo_val.jsonl 2>/dev/null || echo 0)
PARTIAL_LINES=$(wc -l < $PROJECT_DIR/data/dpo_targeted_fix_partial.jsonl 2>/dev/null || echo 0)
echo "  dpo_train.jsonl:              $TRAIN_LINES pairs"
echo "  dpo_val.jsonl:                $VAL_LINES pairs"
echo "  dpo_targeted_fix_partial:     $PARTIAL_LINES pairs"
echo "========================================"
