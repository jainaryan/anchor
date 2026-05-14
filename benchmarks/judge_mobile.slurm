#!/bin/bash
#SBATCH --job-name=mindmate-judge-mobile
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=4:00:00

# Judge a mobile-generated raw results file using Gemma4 26B on the cluster.
#
# Submit with:
#   sbatch --gres=gpu:a100-80:1 \
#       --export=ALL,MOBILE_RESULTS=<path/to/raw.json> \
#       benchmarks/judge_mobile.slurm
#
# Optional env vars:
#   OUTPUT_DIR     — where to write judged output (default: benchmarks/results)
#   LOG_LEVEL      — TRACE/DEBUG/INFO/WARN/ERROR (default: INFO)

PROJECT_DIR=~/projects/mindmate

echo "======================================="
echo "  Judge mobile results"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "======================================="

if [ -z "${MOBILE_RESULTS:-}" ]; then
    echo "ERROR: MOBILE_RESULTS env var not set"
    echo "Submit with: sbatch --export=ALL,MOBILE_RESULTS=path/to/raw.json benchmarks/judge_mobile.slurm"
    exit 1
fi

# Resolve to absolute path relative to project dir if not already absolute
case "$MOBILE_RESULTS" in
    /*) INPUT="$MOBILE_RESULTS" ;;
    *)  INPUT="$PROJECT_DIR/$MOBILE_RESULTS" ;;
esac

if [ ! -f "$INPUT" ]; then
    echo "ERROR: file not found: $INPUT"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/benchmarks/results}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "[Setup] Input:      $INPUT"
echo "[Setup] Output dir: $OUTPUT_DIR"
echo "[Setup] Log level:  $LOG_LEVEL"

source $PROJECT_DIR/mindmatenv/bin/activate
echo "[Setup] Python: $(which python) -- $(python --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

cd $PROJECT_DIR

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set (needed for Gemma4 — gated model)"
    exit 1
fi

set -euo pipefail

python -m benchmarks.judge_mobile_results \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --log-level "$LOG_LEVEL"

echo ""
echo "Judging finished at $(date)"
