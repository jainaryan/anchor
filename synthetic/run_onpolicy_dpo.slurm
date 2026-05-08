#!/bin/bash
#SBATCH --job-name=mindmate-onpolicy-dpo
#SBATCH --output=/home/a/aryanj/logs/onpolicy_dpo_%j.out
#SBATCH --error=/home/a/aryanj/logs/onpolicy_dpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=12:00:00

# On-policy DPO pair generation for MindMate.
#
# Phase 1: policy model (genzv2_ck1200, ~10GB 4-bit) samples N responses per prompt.
#          Saves raw responses to data/dpo_onpolicy_raw.jsonl, then unloads.
# Phase 2: teacher (Qwen3-30B bf16, ~60GB) judges + generates hybrid chosen responses.
#          Writes data/dpo_onpolicy_train.jsonl + data/dpo_onpolicy_val.jsonl.
#
# Expected time: ~2h phase 1 + ~4h phase 2 = ~6h (within 8h limit).
# A100-80 required: phase 2 teacher ~60GB VRAM.
#
# After completion, add dpo_onpolicy_train.jsonl to CUDA_train_dpo.py DATA argument.
# Recommended DPO config: beta=0.2, 600-800 steps, base=genzv2_ck1200.

set -euo pipefail

cd ~/projects/mindmate
source mindmatenv/bin/activate

echo "=========================================="
echo "  MindMate On-policy DPO Datagen"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "=========================================="
echo ""

# Override these via --export if needed:
#   sbatch --export=N_PROMPTS=3000,N_SAMPLES=3 synthetic/run_onpolicy_dpo.slurm
N_PROMPTS=${N_PROMPTS:-2000}
N_SAMPLES=${N_SAMPLES:-3}
POLICY_TEMP=${POLICY_TEMP:-1.0}
ADAPTER=${ADAPTER:-adapters/genz/checkpoint-1200}
SEED=${SEED:-42}

echo "  Config:"
echo "    Adapter    : $ADAPTER"
echo "    N prompts  : $N_PROMPTS"
echo "    N samples  : $N_SAMPLES"
echo "    Policy temp: $POLICY_TEMP"
echo ""

echo "--- Phase 1: Policy sampling ---"
python -u synthetic/onpolicy_dpo_pipeline.py \
    --phase 1 \
    --adapter "$ADAPTER" \
    --n-prompts "$N_PROMPTS" \
    --n-samples "$N_SAMPLES" \
    --policy-temp "$POLICY_TEMP" \
    --seed "$SEED" \
    --resume

echo ""
echo "--- Phase 2: Judge + pair ---"
python -u synthetic/onpolicy_dpo_pipeline.py \
    --phase 2 \
    --teacher-temp 0.75 \
    --val-split 0.10

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "  Outputs:"
echo "    data/dpo_onpolicy_raw.jsonl        (phase 1 checkpoint)"
echo "    data/dpo_onpolicy_train.jsonl      (DPO training pairs)"
echo "    data/dpo_onpolicy_val.jsonl        (DPO validation pairs)"
echo "=========================================="
