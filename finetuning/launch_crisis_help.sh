#!/bin/bash
# Launch both crisis and help_mode pipelines in parallel.
# Usage: bash finetuning/launch_crisis_help.sh
#
# Each pipeline runs as a single 48h job (no sharding needed — targets ~200-400
# examples each, which is achievable in a single job at ~2 min/example).
# Shard if you want more volume or faster turnaround.

set -euo pipefail
SCRIPT=finetuning/crisis_help_pipeline_qwen.slurm

echo "Submitting crisis pipeline..."
CRISIS_JOB=$(sbatch --export=ALL,PIPELINE_MODE=crisis "$SCRIPT" | awk '{print $NF}')
echo "  crisis → job $CRISIS_JOB"

echo "Submitting help_mode pipeline..."
HELP_JOB=$(sbatch --export=ALL,PIPELINE_MODE=help_mode "$SCRIPT" | awk '{print $NF}')
echo "  help_mode → job $HELP_JOB"

echo ""
echo "Both submitted. Monitor with: squeue -u aryanj"
echo "Logs: ~/logs/mindmate-crisis-help-qwen_${CRISIS_JOB}.out"
echo "      ~/logs/mindmate-crisis-help-qwen_${HELP_JOB}.out"
echo ""
echo "Output files (after jobs finish):"
echo "  data/synthetic_train_crisis_qwen.jsonl"
echo "  data/synthetic_train_help_mode_qwen.jsonl"
