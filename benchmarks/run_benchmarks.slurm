#!/bin/bash
#SBATCH --job-name=mindmate-bench
#SBATCH --output=/home/a/aryanj/logs/benchmark_%j.log
#SBATCH --error=/home/a/aryanj/logs/benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=04:00:00

# ── MindMate Benchmark Runner ─────────────────────────────────────────────────
#
# Runs the full benchmark suite against one or more checkpoints.
# Results land in benchmarks/results/<model>_<timestamp>.{json,md}
#
# Usage:
#   sbatch benchmarks/run_benchmarks.slurm
#   sbatch benchmarks/run_benchmarks.slurm --export=MODEL=llama_dpo_ck1600
#   sbatch benchmarks/run_benchmarks.slurm --export=MODEL=llama_ck1600,CATEGORY=MEMORY_USE
#
# Environment overrides (via --export):
#   MODEL      one of: llama_ck1600 (default), llama_dpo_ck1600, llama_ck200, qwen25_3b
#   CATEGORY   filter to one category (optional)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${MODEL:-llama_ck1600}"
ADAPTER="${ADAPTER:-}"
LABEL="${LABEL:-}"
CATEGORY_ARG=""
if [ -n "${CATEGORY:-}" ]; then
    CATEGORY_ARG="--category $CATEGORY"
fi

PROJECT_DIR=~/projects/mindmate
cd "$PROJECT_DIR"

echo "========================================"
echo "  MindMate Benchmark"
if [ -n "$ADAPTER" ]; then
echo "  Adapter: $ADAPTER"
echo "  Label:   ${LABEL:-$(basename $ADAPTER)}"
else
echo "  Model:  $MODEL"
fi
echo "  Job:    $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start:  $(date)"
echo "========================================"

source mindmatenv/bin/activate

if [ -n "$ADAPTER" ]; then
    LABEL_ARG=""
    if [ -n "$LABEL" ]; then LABEL_ARG="--label $LABEL"; fi
    python -u benchmarks/run_benchmarks.py \
        --adapter "$ADAPTER" \
        $LABEL_ARG \
        $CATEGORY_ARG
else
    python -u benchmarks/run_benchmarks.py \
        --model "$MODEL" \
        $CATEGORY_ARG
fi

echo ""
echo "Done: $(date)"
