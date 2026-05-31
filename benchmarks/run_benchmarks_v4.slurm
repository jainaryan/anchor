#!/bin/bash
#SBATCH --job-name=mindmate-bench-v4
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=4:00:00

# v4 benchmark runner — uses shared scenarios.json + JudgeService + EvalLogger.
#
# Submit with:
#   # NF4 adapter (default — fast, tests training)
#   sbatch --gres=gpu:a100-80:1 \
#       --export=ALL,MODEL=genzv5_ck200 \
#       benchmarks/run_benchmarks_v4.slurm
#
#   # GGUF Q4_K_M (tests what actually ships)
#   sbatch --gres=gpu:a100-80:1 \
#       --export=ALL,GGUF_PATH=exports/mindmate_genzv2_ck1200_q4_k_m.gguf,LABEL=genzv2_ck1200_gguf \
#       benchmarks/run_benchmarks_v4.slurm
#
#   # Category-only or scripted-only (skip dynamic — Qwen3 simulator is heavy)
#   sbatch --gres=gpu:a100-80:1 \
#       --export=ALL,MODEL=genzv5_ck200,TYPE=single,scripted_multiturn \
#       benchmarks/run_benchmarks_v4.slurm
#
# Env vars:
#   MODEL          named shortcut (genzv2_ck1200, llama_base, etc.)
#   ADAPTER        path to adapter dir (alternative to MODEL)
#   GGUF_PATH      path to .gguf file (set BACKEND=gguf automatically)
#   BACKEND        nf4 (default) | gguf
#   LABEL          override the label used in the output dir/JSON
#   CATEGORY       run only scenarios in this category
#   TYPE           comma-separated types to include
#   IDS            comma-separated scenario IDs
#   SYSPROMPT      full (default) | preamble_only | none
#   NO_JUDGE       set to 1 for inference-only (skip Gemma4 judge; judge later
#                  via benchmarks/replay.slurm). Use for large eval models that
#                  don't co-fit with the judge on one A100-80 (e.g. qwen3_30b).
#   TEMPERATURE    eval model temp for single/scripted (default 0.0)
#   TEMPERATURE_DYNAMIC  eval model temp for dynamic (default 0.7)
#   LOG_LEVEL      TRACE/DEBUG/INFO/WARN/ERROR (default INFO)
#   OUTPUT_DIR     where to write the run dir (default benchmarks/results)

PROJECT_DIR=~/projects/mindmate

echo "======================================="
echo "  MindMate Benchmark v4"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "======================================="

source $PROJECT_DIR/mindmatenv/bin/activate
export PYTHONUNBUFFERED=1
echo "[Setup] Python: $(which python) -- $(python --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Do NOT cd into $PROJECT_DIR — logging.py in that directory shadows stdlib's logging
# module when Python uses CWD=mindmate ('' in sys.path finds the local file first),
# causing `import torch` to fail with AttributeError: module 'logging' has no attribute 'getLogger'.
# WORKAROUND: stay in /tmp and pass PROJECT_DIR via PYTHONPATH so `python -m benchmarks.*`
# can find the package without making mindmate the CWD.
cd /tmp
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set (needed for gated models)"
    exit 1
fi

# Build argument list from env vars
ARGS=()

if [ -n "${GGUF_PATH:-}" ]; then
    ARGS+=(--gguf-path "$GGUF_PATH" --backend gguf)
elif [ -n "${MODEL:-}" ]; then
    ARGS+=(--model "$MODEL")
elif [ -n "${ADAPTER:-}" ]; then
    ARGS+=(--adapter "$ADAPTER")
else
    echo "ERROR: must set MODEL, ADAPTER, or GGUF_PATH"
    exit 1
fi

[ -n "${BACKEND:-}" ] && ARGS+=(--backend "$BACKEND")
[ -n "${LABEL:-}" ] && ARGS+=(--label "$LABEL")
[ -n "${CATEGORY:-}" ] && ARGS+=(--category "$CATEGORY")
[ -n "${TYPE:-}" ] && ARGS+=(--type "$TYPE")
[ -n "${IDS:-}" ] && ARGS+=(--ids "$IDS")
[ -n "${SYSPROMPT:-}" ] && ARGS+=(--sysprompt "$SYSPROMPT")
[ -n "${NO_JUDGE:-}" ] && ARGS+=(--no-judge)
[ -n "${TEMPERATURE:-}" ] && ARGS+=(--temperature "$TEMPERATURE")
[ -n "${TEMPERATURE_DYNAMIC:-}" ] && ARGS+=(--temperature-dynamic "$TEMPERATURE_DYNAMIC")
[ -n "${LOG_LEVEL:-}" ] && ARGS+=(--log-level "$LOG_LEVEL")
[ -n "${OUTPUT_DIR:-}" ] && ARGS+=(--output-dir "$OUTPUT_DIR")

echo "[Setup] Running: python -m benchmarks.run_benchmarks_v4 ${ARGS[*]}"
echo ""

set -euo pipefail
python -m benchmarks.run_benchmarks_v4 "${ARGS[@]}"

echo ""
echo "Finished at $(date)"
