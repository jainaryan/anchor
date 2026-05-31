#!/bin/bash
#SBATCH --job-name=mindmate-replay
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=4:00:00

# Replay-judge an existing benchmark run — loads ONLY the Gemma4 judge (no eval
# model), re-judges every transcript in <RUN_DIR>/transcripts/, and writes a
# fresh judged.json under <RUN_DIR>/replay_<timestamp>/.
#
# Pairs with NO_JUDGE=1 inference runs: a large eval model (e.g. qwen3_30b) that
# can't co-reside with the judge produces transcripts in phase 1, then this job
# judges them in phase 2 with the judge owning the whole GPU.
#
# Submit with:
#   sbatch --gres=gpu:a100-80:1 \
#       --export=ALL,RUN_DIR=benchmarks/results/cluster_qwen3_30b_<ts> \
#       benchmarks/replay.slurm
#
# Env vars:
#   RUN_DIR      run directory containing transcripts/ (relative to PROJECT_DIR
#                or absolute). Either this or RESULTS_GLOB is required.
#   RESULTS_GLOB glob (relative to PROJECT_DIR) — picks the NEWEST matching dir
#                that contains a transcripts/ subdir. Handy with --dependency,
#                where the phase-1 run dir is timestamped and unknown at submit
#                time, e.g. RESULTS_GLOB='benchmarks/results/cluster_qwen3_30b_*'
#   JUDGE_MODEL  override judge model id (default: replay.py's Gemma4 default)
#   LOG_LEVEL    TRACE/DEBUG/INFO/WARN/ERROR (default INFO)

PROJECT_DIR=~/projects/mindmate

echo "======================================="
echo "  MindMate Replay Judge"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Run dir: ${RUN_DIR:-<unset>}"
echo "  Started: $(date)"
echo "======================================="

if [ -z "${RUN_DIR:-}" ] && [ -z "${RESULTS_GLOB:-}" ]; then
    echo "ERROR: must set RUN_DIR or RESULTS_GLOB"
    exit 1
fi

source $PROJECT_DIR/mindmatenv/bin/activate
export PYTHONUNBUFFERED=1
echo "[Setup] Python: $(which python) -- $(python --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Same logging.py-shadowing workaround as run_benchmarks_v4.slurm: stay out of
# PROJECT_DIR and reach the package via PYTHONPATH.
cd /tmp
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set (needed for gated models)"
    exit 1
fi

# Resolve the run directory.
if [ -n "${RESULTS_GLOB:-}" ]; then
    # Newest dir matching the glob that actually has transcripts/ to judge.
    RUN_DIR_ABS=""
    for d in $(ls -dt $PROJECT_DIR/$RESULTS_GLOB 2>/dev/null); do
        if [ -d "$d/transcripts" ]; then RUN_DIR_ABS="$d"; break; fi
    done
    if [ -z "$RUN_DIR_ABS" ]; then
        echo "ERROR: no dir matching RESULTS_GLOB='$RESULTS_GLOB' contains transcripts/"
        exit 1
    fi
    echo "[Setup] RESULTS_GLOB resolved to: $RUN_DIR_ABS"
else
    case "$RUN_DIR" in
        /*) RUN_DIR_ABS="$RUN_DIR" ;;
        *)  RUN_DIR_ABS="$PROJECT_DIR/$RUN_DIR" ;;
    esac
fi

ARGS=("$RUN_DIR_ABS" --judge-only)
[ -n "${JUDGE_MODEL:-}" ] && ARGS+=(--judge-model "$JUDGE_MODEL")
[ -n "${LOG_LEVEL:-}" ] && ARGS+=(--log-level "$LOG_LEVEL")

echo "[Setup] Running: python -m benchmarks.replay ${ARGS[*]}"
echo ""

set -euo pipefail
python -m benchmarks.replay "${ARGS[@]}"

echo ""
echo "Finished at $(date)"
