#!/bin/bash
#SBATCH --job-name=mindmate-biometric
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err

# ── Biometric Data Generation (SFT + DPO) ────────────────────────────────────
#
# Runs both biometric_sft_pipeline.py and biometric_dpo_pipeline.py
# sequentially. Both use Qwen3-30B-A3B-Instruct-2507 teacher in bfloat16
# (~60GB VRAM → requires A100-80).
#
# SFT pipeline generates:
#   data/synthetic_train_biometric.jsonl
#   Modes: biometric_relevant (references health data once) +
#          biometric_irrelevant (does not inject health data)
#
# DPO pipeline generates preference pairs appended to:
#   data/dpo_train.jsonl + data/dpo_val.jsonl
#   Pair types:
#     A — biometric_relevant_ignore  (chosen=references, rejected=ignores)
#     B — biometric_relevant_obsess  (chosen=once+moves on, rejected=harps on it)
#     C — biometric_irrelevant_inject (chosen=on-topic, rejected=injects health)
#
# Both pipelines run until wall-time (SIGTERM triggers graceful save).
# SFT runs for 12h, DPO for 10h (leaving 2h buffer for save + cleanup).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR=~/projects/mindmate
SYNTHETIC_DIR=$PROJECT_DIR/synthetic

echo "========================================"
echo "  MindMate Biometric Data Generation"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "========================================"

# Activate venv
source ~/projects/mindmate/mindmatenv/bin/activate
echo "[Setup] Python: $(which python) — $(python --version)"

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

cd $SYNTHETIC_DIR

# ── Phase 1: SFT data generation (12 hours) ──────────────────────────────────
echo ""
echo "[Phase 1] Biometric SFT pipeline — running for 12h..."
echo "  Modes: biometric_relevant + biometric_irrelevant"
echo "  Output: $PROJECT_DIR/data/synthetic_train_biometric.jsonl"
echo ""

# Run with a 12h timeout then SIGTERM (pipeline handles graceful shutdown)
timeout 43200 python biometric_sft_pipeline.py || true

echo ""
echo "[Phase 1] SFT pipeline complete at $(date)"
SFT_LINES=$(wc -l < $PROJECT_DIR/data/synthetic_train_biometric.jsonl 2>/dev/null || echo 0)
echo "[Phase 1] Generated $SFT_LINES SFT examples"

# Cool-down between phases
sleep 60

# ── Phase 2: DPO data generation (10 hours) ───────────────────────────────────
echo ""
echo "[Phase 2] Biometric DPO pipeline — running for 10h..."
echo "  Pair types: A (ignore) · B (obsess) · C (inject)"
echo "  Output: appended to $PROJECT_DIR/data/dpo_train.jsonl + dpo_val.jsonl"
echo ""

timeout 36000 python biometric_dpo_pipeline.py || true

echo ""
echo "[Phase 2] DPO pipeline complete at $(date)"
TRAIN_LINES=$(wc -l < $PROJECT_DIR/data/dpo_train.jsonl 2>/dev/null || echo 0)
VAL_LINES=$(wc -l < $PROJECT_DIR/data/dpo_val.jsonl 2>/dev/null || echo 0)
echo "[Phase 2] DPO dataset: $TRAIN_LINES train + $VAL_LINES val pairs"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Job complete: $(date)"
echo "  SFT examples:  $SFT_LINES"
echo "  DPO train:     $TRAIN_LINES"
echo "  DPO val:       $VAL_LINES"
echo "========================================"
